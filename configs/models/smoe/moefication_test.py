from opencompass.models import LlamaMoECausalLM


models = [
    # LLaMA 13B
    dict(
        type=LlamaMoECausalLM,
        abbr='llama-moefication-13b',
        path="/mnt/petrelfs/share_data/quxiaoye/model_back/13b_moe_first100Btoken_models/checkpoint-2500/",
        tokenizer_path='/mnt/petrelfs/share_data/quxiaoye/model_back/13b_moe_first100Btoken_models/checkpoint-2500/',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
