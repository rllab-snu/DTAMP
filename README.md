# Diffused Task-Agnostic Milestone Planner

This is an official GitHub Repository for the paper:
- Mineui Hong, Minjae Kang, and Songhwai Oh, "Diffused Task-Agnostic Milestone Planner," in Proc. of the 37th Conference on Neural Information Processing Systems (NeurIPS 2023), Dec. 2023.

## How to run experiments

### 1. Requirements

Please note that the installation of the D4RL environments and CALVIN benchmark are not included in `requirements.txt`.  
We recommend you to install the D4RL environments from [D4RL repo](https://github.com/Farama-Foundation/D4RL),
and CALVIN benchmark from [CALVIN repo](https://github.com/mees/calvin).  
We also note that we utilize dataset provided by [TACO-RL repo](https://github.com/ErickRosete/tacorl) for CALVIN experiments,
which has slightly different training/validation split.

### 2. Training

Before running the scripts, you should set environment variable ```PYTHONPATH```.
```bash
export PYTHONPATH=$PYTHONPATH:/{path}/{to}/{dtamp}
```
To train DTAMP for the D4RL environments, run:
```bash
python scripts/d4rl/train_dtamp.py --env {env_name}
```
To train DTAMP for the CALVIN benchmark, first run ```preprocess_calvin_data.py``` for preprocessing the dataset:
```bash
python scripts/calvin/preprocess_calvin_data.py --source_data_dir {where}/{tacorl_data}/{saved} --target_data_dir {where}/{to save}/{processed data}
```
