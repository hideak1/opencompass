import torch

from opencompass.registry import MODELS
from opencompass.models.huggingface import HuggingFaceCausalLM


@MODELS.register_module()
class LlamaMoECausalLM(HuggingFaceCausalLM):
    def init(self):
        print('init')

    def callback(self):
        print('callback')
        self.model.callback()

    def _load_model(self, path: str, model_kwargs: dict, peft_path: str | None = None):
        from smoe.models.llama_moe import LlamaMoEForCausalLM

        model_kwargs.setdefault('torch_dtype', torch.float16)
        # model_kwargs.setdefault('torch_dtype', torch.float32)
        self.model = LlamaMoEForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.eval()
