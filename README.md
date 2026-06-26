# rl-research-suite-on-isaac-lab

## Overview
RL Research Suite on Isaac Lab is a tool that helps to easily run simulations by allowing one to set nearly every configuration using YAML config files. Hence, it eliminates the need to go through hundreds of lines of code each time, trying to find a single hyperparameter to change its value. The platform ensures a stable simulation pipeline, with numerous mock tests, and a strict structure for the algorithms to follow. It has three built-in RL algorithms: PPO, SAC, and TD3, which were successfully tested and run on the Franka Arm.

The project consists of seven main folders:

* **configs/** stores the YAML configurations for the project, including environments (`envs/`), reward managers (`reward/`), algorithms (`algo/`), and the `experiment/` folder, which aggregates all configurations together for a single run.
* **algorithms/** includes the abstract Trainer class, from which all algorithms must inherit to ensure compatibility with the research suite, alongside the developed PPO, SAC, and TD3 algorithms.
* **encoders/** includes the FrozenResNet18 encoder that compresses complex images into a compact and stable vector with 512 features.
* **envs/** includes the main environment wrapper, ensuring gymnasium-style spaces for compatibility with external policies and normalizing pixel data. It also contains a `managers/` folder for reward, observation, and termination management, as well as the `scene_cfg.py`, which is responsible for setting up the simulation scene.
* **policies/** includes the BasePolicy abstract class, from which all the built-in policies must inherit, and the `from_hub.py` responsible for loading policies from SB3. In the `custom/` subfolder, custom policies are defined for the built-in algorithms.
* **thirdparty/** includes the IsaacLab, brought to the project as a git submodule.
* **checkpoints/** stores the trained algorithms at specific checkpoints.

The main folder also includes `train.py`, the main script for running the project, and `eval.py`, which is needed for collecting success metrics and WandB logging.

## Prerequisites
Firstly, clone the repository:
```bash
git clone https://github.com/VladimirGHG/rl-research-suite-on-isaac-lab.git
```

After that, ensure that all the prerequisites are installed:
### Python dependencies:
Once the repository is cloned, before doing anything else, create a virtual environment and activate it:
```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate.bat
```
After that, install the Isaac Lab submodule, the link to which is already added to the `thirdparty/` folder (Run from the root directory):
```bash
git submodule update --init --recursive
```
Finally, install all the required packages, using the requirements.txt:
```bash
pip install -r requirements.txt
```

The launch point of the project is the train.py script. To run it, use:
```bash
python train.py num_envs=XXX algo_cfg=XXX 
```
#### You can directly add other flags to override the default settings, such as `total_timesteps`, `checkpoint_interval`, etc.
