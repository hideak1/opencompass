from opencompass.models import LlamaMoECausalLM


models = [
    # LLaMA 13B
    dict(
        type=LlamaMoECausalLM,
        abbr='llama-moe-3b',
        path="/data/hongxuan/runs/LLaMA-MoE-v1-3_0B-2_16/",
        tokenizer_path='/data/hongxuan/runs/LLaMA-MoE-v1-3_0B-2_16',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
