models=(
    # "incite_base_3b"
    # "open_llama_3b_v2"
    # "opt_2.7b"
    # "pythia_2.8b"
    "sheared_llama_2.7b"
)

for model_dir in ${models[@]}; do
    echo $model_dir
    nohup bash run.sh $model_dir 1>logs/nq_$model_dir.log 2>&1 &
done
