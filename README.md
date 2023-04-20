# OffPyMARL
> 🚧 This repo is not ready for release, benchmarking is ongoing. 🚧

OffPyMARL provides unofficial and benchmarked PyTorch implementations for selected Offline MARL algorithms, including:

- BC
- VDN+CQL
- QMIX+CQL
- MATD3+BC
- MAICQ

we also implement selected Multi-Task versions to tackle with the population-invariante issue for BC, QMIX+CQL and MATD+BC, Multi-Task versions for other algorithms are under developing.

## Installation

```bash
conda create -n offpymarl python=3.10 -y
conda activate offpymarl
pip install -r requirements.txt
bash install_sc2.sh # if you have not installed StarCraftII on your machine
bash install_smac.sh
```

## Collect Data
```bash
python src/main.py --collect --config=<alg> --env-config=sc2_collect with env_args.map_name=<map_name> offline_data_quality=<quality> save_replay_buffer=<whether_to_save_replay>
num_episodes_collected=<num_episodes_per_collection> stop_winrate=<stop_winrate> --seed=<seed>
```
quality is optinal in ['random', 'medium', 'expert', 'full'].

if assign save_replay_buffer, will generate 'medium_replay', 'expert_replay' offline data with adequate offline_data_quality param.

see debug/collect_data/evaluate.sh if you want to use trained checkpoint to collect data.

## Offline Training
```bash
python src/main.py --offline --config=<alg_name> --env-config=sc2_offline --map_name=<sc2_map>  --offline_data_quality=<data_quality> --seed=<seed> --t_max=40000 --test_interval=250 --log_interval=250 --runner_log_interval=250 --learner_log_interval=250 --save_model_interval=100001 
```
see test/offline_train for more information.

## MultiTask Offline Training
```bash
python src/main.py --mto --config=<alg_name> --env-config=sc2_offline --task-config=<task_name> --customized_quality=<data_quality> --seed=<seed> --t_max=40000 --test_interval=250 --log_interval=250 --runner_log_interval=250 --learner_log_interval=250 --save_model_interval=10000
```
see test/mto for more information.

## Citing OffPyMARL

If you use OfflineRL-Lib in your work, please use the following bibtex

```tex
@misc{OffPyMARL,
  author = {Ziqian Zhang},
  title = {OffPyMARL: Benchmarked Implementations of Offline Reinforcement Learning Algorithms},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zzq-bot/offpymarl}},
}
```

## Acknowledgements
We thank [ODIS](https://github.com/LAMDA-RL/ODIS) for providing data collection related code.
