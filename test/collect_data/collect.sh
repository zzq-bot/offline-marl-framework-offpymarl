## bash xx/medium.sh 0 qmix 2s1z_vs_3z medium 0.6
export CUDA_VISIBLE_DEVICES=$1
python src/main.py --collect --config=$2 --env-config=sc2_collect --map_name=$3 --offline_data_quality=$4 --save_replay_buffer=False --num_episodes_collected=4000 --stop_winrate=$5 --seed=1 