import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import PreTrainedModel

from contrastors.distributed import gather_with_grad
from contrastors.models.biencoder import BiEncoder, LogitScale

class DualEncoderText(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.document = BiEncoder(config.document_model_args)
        document_model_args = config.document_model_args

        self.query = BiEncoder(config.query_model_args)

        self.logit_scale = LogitScale(config.query_model_args)

    def encode_query(self, text, normalize=True):
        query_outputs = self.text(**text, normalize=normalize)

        return query_outputs["embedding"]
    
    def encode_document(self, text, normalize=True):
        document_outputs = self.text(**text, normalize=normalize)

        return document_outputs["embedding"] 
    
    def forward(self, inputs):

        query_outputs = self.query(inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"], normalize=False)

        document_outputs = self.document(inputs["document_input_ids"], attention_mask=inputs["document_attention_mask"], normalize=False)

        return query_outputs, document_outputs

        
