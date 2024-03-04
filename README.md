# Diffused Task-Agnostic Milestone Planner

This is an official GitHub Repository for the paper:
- Mineui Hong, Minjae Kang, and Songhwai Oh, "Diffused Task-Agnostic Milestone Planner," in Proc. of the 37th Conference on Neural Information Processing Systems (NeurIPS 2023), Dec. 2023.

## How to run experiments

### 1. Requirements

Please note that the installation of the D4RL environments and CALVIN benchmark are not included in `requirements.txt`.
We recommend you to install the D4RL environments from [D4RL repo](https://github.com/Farama-Foundation/D4RL),
and CALVIN benchmark from [CALVIN repo](https://github.com/mees/calvin).
However, installing D4RL environments and CALVIN environment in the same virtual environment might cause conflict of dependencies.
We recommend to make two separated virtual environments for D4RL and CALVIN experiments.
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
python scripts/calvin/preprocess_calvin_data.py --source_data_dir {where}/{tacorl_data}/{saved} --target_data_dir {where}/{to_save}/{processed_data}
```
Then, you should train [PlayLMP](https://learning-from-play.github.io/) model first to learn skill representations:
```bash
python scripts/calvin/train_lmp.py --data_dir {where}/{processed_data}/{saved}; python scripts/calvin/add_skills_to_calvin_dataset.py --data_dir {where}/{processed_data}/{saved}
```
Now you can finally train DTAMP:
```bash
python scripts/calvin/train_dtamp.py --data_dir {where}/{processed_data}/{saved} --lmp_dir {where}/{lmp_checkpoint}/{saved}
```

### 3. Evaluation

To evaluate the trained model, run:
```bash
python scripts/d4rl/evaluate_dtamp.py --env {env_name} --checkpoint_dir {checkpoint}/{dir}
```
or
```bash
python scripts/calvin/evaluate_dtamp.py --calvin_dir {calvin_env}/{root}/{pth} --data_dir {where}/{data}/{saved} --checkpoint_dir {checkpoint}/{dir} --tasks_per_rollout {1 or 2 or 3}
```

## Reference
```bash
@inproceedings{hong2023dtamp,
author={Mineui Hong and Minjae Kang and Songhwai Oh},
title={Diffused Task-Agnostic Milestone Planner},
journal={Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS)},
year={2023}
}
```

## Contact
If you have any problem, please contact to <mineui.hong@rllab.snu.ac.kr>.

## Acknowledgements
The codebase of diffusion model is based on [decision-diffuser repo](https://github.com/anuragajay/decision-diffuser/tree/main/code) and [diffuser repo](https://github.com/jannerm/diffuser/).
