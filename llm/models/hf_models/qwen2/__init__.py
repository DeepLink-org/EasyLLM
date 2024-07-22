from ..utils.model_utils import get_model_cfg
from .modeling_qwen2 import Qwen2ForCausalLM
from .tokenization_qwen2 import Qwen2Tokenizer
from llm.utils.general.registry_factory import TOKENIZER_REGISTRY
from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY


def build_model(**cfg):
    cfg["model_type"] = "qwen2"
    model_cfg = get_model_cfg(cfg)
    model_path = model_cfg.pop("model_path")
    from ..qwen2.configuration_qwen2 import Qwen2Config
    config = Qwen2Config.from_pretrained(model_path)
    model = Qwen2ForCausalLM.from_pretrained(model_path, config=config, torch_dtype=model_cfg['torch_dtype'])

    if cfg.get("model_parallel", False):
        model.is_parallelizable = True
        model.model_parallel = True

    return model


MODULE_ZOO_REGISTRY.register("QWen2ForCausalLM", build_model)
TOKENIZER_REGISTRY.register("QWen2Tokenizer", Qwen2Tokenizer)
