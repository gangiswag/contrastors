import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
from contrastors.dataset.text_text_loader import collate_fn, get_local_query_document_dataloader
from contrastors.distributed import gather_with_grad
from contrastors.loss import clip_loss, grad_cache_loss
from contrastors.models import DualEncoderText, DualEncoderTextConfig, LogitScale

from .text_text import TextTextTrainer


class QueryDocumentTrainer(TextTextTrainer):
    def __init__(self, config, dtype):
        super(TextTextTrainer, self).__init__(config, dtype)
        self.use_grad_cache = config.train_args.grad_cache
        self.matryoshka_dims = config.train_args.matryoshka_dims
        if self.matryoshka_dims:
            self.matryoshka_loss_weights = (
                config.train_args.matryoshka_loss_weights
                if config.train_args.matryoshka_dims and config.train_args.matryoshka_loss_weights
                else [1] * len(config.train_args.matryoshka_dims)
            )
        else:
            self.matryoshka_loss_weights = None

    def get_encoder_tokenizer(self, config):
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
        tokenizer.model_max_length = config.seq_len

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.cls_token is None:
            tokenizer.add_special_tokens({"cls_token": "<s>"})

        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<mask>"})

        return tokenizer

    def get_tokenizer(self, config):
        self.query_tokenizer = self.get_encoder_tokenizer(config.query_model_args)
        self.document_tokenizer = self.get_encoder_tokenizer(config.document_model_args)

        return None

    def get_model(self, config):
        model_config = DualEncoderTextConfig(config)
        model =  DualEncoderText(model_config)

        has_trainable_params = sum(p.requires_grad for p in model.parameters()) > 0
        model = model.to(f"cuda:{self.process_index}")

        if self.distributed and not self.deepspeed and has_trainable_params:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dist.get_rank()],
            )

        scale = LogitScale(model_config)

        if self.distributed and not self.deepspeed:
            scale = scale.to("cuda")
            if sum(p.requires_grad for p in scale.parameters()) > 0:
                scale = torch.nn.parallel.DistributedDataParallel(
                    scale,
                    device_ids=[self.process_index],
                )

        return {"model": model, "logit_scale": scale}

    def get_dataloaders(self, config, epoch=0):
        train_args = config.train_args
        data_config = config.data_args
        model_args = config.model_args
        gradient_accumulation_steps = train_args.gradient_accumulation_steps        
        # config defines global batch size
        if data_config.batch_size % self.num_processes != 0:
            raise ValueError(
                f"Batch size {data_config.batch_size} must be divisible by accelerator.num_processes {self.num_processes}"
            )

        batch_size = int(data_config.batch_size / self.num_processes)
        train_dataloader = get_local_query_document_dataloader(
            data_config.input_shards,
            batch_size,
            query_tokenizer=self.query_tokenizer,
            document_tokenizer=self.document_tokenizer,
            seed=data_config.seed,
            num_negatives=model_args.num_negatives,
            add_prefix=model_args.add_prefix,
            num_workers=data_config.workers,
            epoch=0,
        )
        self.total_num_steps = int(
            len(train_dataloader.dataset) / gradient_accumulation_steps // data_config.batch_size
        )

        return {"train": train_dataloader, "val": None, "test": None}

    def save_model(self, output_dir):
        super().save_model(output_dir)
        if self.global_rank == 0:
            logit_scale = self.model.get("logit_scale", None)
            if isinstance(logit_scale, (nn.Module, nn.DataParallel, nn.parallel.DistributedDataParallel)) and any(
                p.requires_grad for p in logit_scale.parameters()
            ):
                unwrapped_scale = self.unwrap(logit_scale)
                torch.save(unwrapped_scale.state_dict(), f"{output_dir}/logit_scale.pt")

    def clip_gradients(self, max_grad_norm):
        super().clip_gradients(max_grad_norm)

    def forward_step(self, model, inputs, logit_scale, **kwargs):
        model.train()
        if self.use_grad_cache:
            loss = self._grad_cache_forward_step(model, inputs, logit_scale, **kwargs)
        else:
            loss = self._forward_step(
                model,
                inputs,
                logit_scale,
                matryoshka_dims=self.matryoshka_dims,
                matroyshka_loss_weights=self.matryoshka_loss_weights,
                **kwargs,
            )

        return loss

    def backward(self, loss):
        if self.deepspeed:
            self.engine.backward(loss)
            self.engine.step()
        else:
            # grad cache backprops in the loss function, becomes a noop
            if not self.use_grad_cache:
                loss.backward()

    def _grad_cache_forward_step(self, model, batch, logit_scale, **kwargs):
        # TODO: could pass this to grad cache loss and log?
        batch.pop("dataset_name")
        kwargs.pop("step")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        query_inputs = {k.replace("query_", ""): v for k, v in batch.items() if "query" in k}
        document_inputs = {k.replace("document_", ""): v for k, v in batch.items() if "document" in k}
        loss = grad_cache_loss(
            tower1=model.query,
            tower2=model.document,
            t1_inputs=query_inputs,
            t2_inputs=document_inputs,
            chunk_size=self.config.train_args.chunk_size,
            logit_scale=logit_scale,
            **kwargs,
        )
        return loss
    
    def _forward_step(self, model, batch, logit_scale, matryoshka_dims=None, matroyshka_loss_weights=None, **kwargs):
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        query_outputs, document_outputs = model(inputs)
        dataset_name = ""
        queries = query_outputs["embedding"]
        all_documents = gather_with_grad(document_outputs["embedding"])

        loss = clip_loss(
                query=queries,
                document=all_documents,
                logit_scale=logit_scale,
                tracker=self.tracker,
                dataset=dataset_name,
                **kwargs,
            )

        return loss

    def training_step(
        self, model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps
    ):
        loss = super().training_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            train_args=train_args,
            total_num_steps=total_num_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        if train_args.clamp_logits:
            with torch.no_grad():
                self.model["scale"].module.logit_scale.clamp_(0, np.log(train_args.logit_max))

        return loss

    def eval_loop(self, model, dataloader, step):
        raise NotImplementedError("CLIP Trainer does not support evaluation")
