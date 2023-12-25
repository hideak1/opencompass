from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        abbr="opt-2.7b",  # Model abbreviation
        type=HuggingFaceCausalLM,
        # the folowing are HuggingFaceCausalLM init parameters
        path="/mnt/petrelfs/share_data/quxiaoye/models/opt-2.7b",
        tokenizer_path="/mnt/petrelfs/share_data/quxiaoye/models/opt-2.7b",
        tokenizer_kwargs=dict(
            padding_side="left",
            truncation_side="left",
            proxies=None,
            trust_remote_code=True,
        ),
        model_kwargs=dict(device_map="auto"),
        max_seq_len=2048,
        # the folowing are not HuggingFaceCausalLM init parameters
        max_out_len=100,  # Maximum number of generated tokens
        batch_size=8,
        run_cfg=dict(
            num_gpus=1
        ),  # Run configuration for specifying resource requirements
    )
]
