# bash test/mto/sto.sh 0 mt_qmix_cql/mt_matd3_bc/mt_bc 3m/2s3z/... expert/medium/medium-replay 0/1/...
export CUDA_VISIBLE_DEVICES=$1
python src/main.py --mto --config=$2 --env-config=sc2_offline --task-config=sc2_single_task --st=$3 --customized_quality=$4 --seed=$5 --t_max=40000 --test_interval=250 --log_interval=250 --runner_log_interval=250 --learner_log_interval=250 --save_model_interval=10000
