# python run.py --datasets gsm8k_gen \
#     --hf-path /mnt/petrelfs/share_data/quxiaoye/models/open_llama_3b_v2/ \
#     --model-kwargs device_map='auto' \
#     --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False \
#     --max-out-len 100 \
#     --max-seq-len 2048 \
#     --batch-size 8 \
#     --no-batch-padding \
#     --num-gpus 1

{
    export MKL_THREADING_LAYER=GNU
    # python run.py --datasets nq_gen_32shot \
    #     --hf-path /mnt/petrelfs/share_data/quxiaoye/models/pythia-2.8b/ \
    #     --model-kwargs device_map='auto' \
    #     --tokenizer-kwargs padding_side='left' truncation='left' use_fast=True \
    #     --max-out-len 100 \
    #     --max-seq-len 2048 \
    #     --batch-size 8 \
    #     --no-batch-padding \
    #     --num-gpus 1 \
    #     --slurm \
    #     --partition "MoE"

    python run.py --datasets nq_gen_32shot \
        --models llama_moe_3b \
        --model-kwargs device_map='auto' \
        --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False \
        --max-out-len 100 \
        --max-seq-len 2048 \
        --batch-size 8 \
        --no-batch-padding \
        --num-gpus 1 \
        --slurm \
        --partition "MoE"
}

# /mnt/petrelfs/share_data/quxiaoye/models/open_llama_3b_v2/
# /mnt/petrelfs/share_data/quxiaoye/models/opt-2.7b/
# /mnt/petrelfs/share_data/quxiaoye/models/pythia-2.8b/
# /mnt/petrelfs/share_data/quxiaoye/models/Sheared-LLaMA-2.7B/
# /mnt/petrelfs/share_data/quxiaoye/models/INCITE-Base-3B/
# llama_moe_3.5b
# llama_moe_3b
