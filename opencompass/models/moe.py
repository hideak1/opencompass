import torch

from opencompass.registry import MODELS
from opencompass.models.huggingface import HuggingFaceCausalLM


@MODELS.register_module()
class LlamaMoECausalLM(HuggingFaceCausalLM):
    def _load_model(self, path: str, model_kwargs: dict, peft_path: str | None = None):
        from smoe.models.llama_moefication import LlamaMoEForCausalLM

        model_kwargs.setdefault('torch_dtype', torch.float16)
        self.model = LlamaMoEForCausalLM.from_pretrained(path, **model_kwargs)
        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
