from opencompass.models import LlamaMoECausalLM


models = [
    # LLaMA 13B
    dict(
        type=LlamaMoECausalLM,
        abbr='llama-moe-3.5b',
        # path="/mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_scale4_112gpus_dynamic_data/outputs/cpt-llama2_random_scale4_112gpus_dynamic_data-2356022/checkpoint-13600/",
        # tokenizer_path='/mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_scale4_112gpus_dynamic_data/outputs/cpt-llama2_random_scale4_112gpus_dynamic_data-2356022/checkpoint-13600/',
        path="/data/hongxuan/runs/LLaMA-MoE-v1-3_5B-4_16/",
        tokenizer_path='/data/hongxuan/runs/LLaMA-MoE-v1-3_5B-4_16/',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=2,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
