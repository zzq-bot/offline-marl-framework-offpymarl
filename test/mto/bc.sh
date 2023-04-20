export CUDA_VISIBLE_DEVICES=$1
python src/main.py --mto --config=mt_bc --env-config=sc2_offline --task-config=$2 --customized_quality=$3 --seed=$4 --t_max=40000 --test_interval=250 --log_interval=250 --runner_log_interval=250 --learner_log_interval=250 --save_model_interval=10000
