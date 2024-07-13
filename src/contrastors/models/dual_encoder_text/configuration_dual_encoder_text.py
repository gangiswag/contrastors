from typing import Any, Dict

from transformers.configuration_utils import PretrainedConfig

from contrastors.models.biencoder import BiEncoderConfig


class DualEncoderTextConfig(PretrainedConfig):
    def __init__(
        self,
        config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if config:
            self.query_model_args = BiEncoderConfig(**config.query_model_args.dict())
            self.document_model_args = BiEncoderConfig(**config.document_model_args.dict())
            self.logit_scale = config.model_args.logit_scale
            self.trainable_logit_scale = config.model_args.trainable_logit_scale
        else:
            self.query_model_args = BiEncoderConfig()
            self.document_model_args = BiEncoderConfig()

        self.projection_dim = self.query_model_args.projection_dim
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> PretrainedConfig:
        if kwargs.get("return_unused_kwargs", False):
            config, _ = super().from_dict(config_dict, **kwargs)
        else:
            config = super().from_dict(config_dict, **kwargs)

        for modality in ["query_model_args", "document_model_args"]:
            if config_dict.get(modality):
                cleaned_config = config_dict[modality]
                setattr(config, modality, BiEncoderConfig(**cleaned_config))
            else:
                setattr(config, modality, None)

        if kwargs.get("return_unused_kwargs", False):
            return config, {}
        else:
            return config
