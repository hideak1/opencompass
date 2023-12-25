import time
import subprocess
from pathlib import Path

from loguru import logger


def run_command(command):
    try:
        logger.info(f"Running cmd: {command}")
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred: {e}")


def listen(folder):
    logger.info(f"Listening to {folder}")
    folder_path = Path(folder)
    while not folder_path.exists():
        time.sleep(60)
    logger.info(f"Folder {folder} exists, wait for dumping")
    time.sleep(120)
    run_command("bash run.sh")
    logger.info("Start NQ evaluating")


# nohup srun -p MoE -n1 -N1 --gres=gpu:1 --quotatype=auto --output=logs/sheared_fluency_8_2-13260-arc.log --error=logs/sheared_fluency_8_2-13260-arc.log python main.py --model='llama-moe-causal' --model_args='pretrained=/mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_split_112gpus_8_2/outputs/cpt-llama2_random_split_112gpus_8_2-2377343/checkpoint-13260,use_accelerate=True' --tasks='arc_challenge' --num_fewshot=25 --batch_size=2 --no_cache --output_path='results/sheared_fluency_8_2/13260-arc_challenge-25shot.json' --device='cuda:0' 1>logs/sheared_fluency_8_2-13260-arc.log 2>&1 &
# nohup srun -p MoE -n1 -N1 --gres=gpu:1 --quotatype=auto --output=logs/sheared_fluency_8_2-13260-hellaswag.log --error=logs/sheared_fluency_8_2-13260-hellaswag.log python main.py --model='llama-moe-causal' --model_args='pretrained=/mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_split_112gpus_8_2/outputs/cpt-llama2_random_split_112gpus_8_2-2377343/checkpoint-13260,use_accelerate=True' --tasks='hellaswag' --num_fewshot=10 --batch_size=2 --no_cache --output_path='results/sheared_fluency_8_2/13260-hellaswag-10shot.json' --device='cuda:0' 1>logs/sheared_fluency_8_2-13260-hellaswag.log 2>&1 &


if __name__ == "__main__":
    listen(
        "/mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_split_112gpus_8_2/outputs/cpt-llama2_random_split_112gpus_8_2-2377343/checkpoint-13600/"
    )
