from contrastors.read import read_config
from contrastors.models import BiEncoderConfig, BiEncoder
from transformers import AutoTokenizer
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    return parser.parse_args()

def get_tokenizer(config): 
        tokenizer = AutoTokenizer.from_pretrained(config.model_args.tokenizer_name, remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.cls_token is None:
            tokenizer.add_special_tokens({"cls_token": "<s>"})

        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<mask>"})

        return tokenizer

if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config)
    model_config = BiEncoderConfig.from_pretrained(args.checkpoint_dir)
    if config.model_args.projection_dim is not None:
        model_config.projection_dim = config.model_args.projection_dim
    model = BiEncoder.from_pretrained(args.checkpoint_dir, config=model_config)

    tokenizer = get_tokenizer(config)
    tokenizer.save_pretrained(args.output_dir)

    model.trunk.save_pretrained(args.output_dir)
