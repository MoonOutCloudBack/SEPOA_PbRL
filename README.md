# S-EPOA: Overcoming the Indistinguishability of Segments with Skill-Driven Preference-Based Reinforcement Learning

This is the official implementation of S-EPOA (IJCAI 2025, [arxiv](https://arxiv.org/abs/2408.12130)).

## Requirements

### Install MuJoCo 210

```bash
sudo apt update
sudo apt install -y unzip gcc libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libegl1 libopengl0
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
mkdir ~/.mujoco
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -zxvf mujoco210-linux-x86_64.tar.gz
rm -f mujoco210-linux-x86_64.tar.gz
```

Then, include the following lines in the `~/.bashrc` file:

```bash
export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin
export PATH="$LD_LIBRARY_PATH:$PATH"
```

Finally, run `source ~/.bashrc`

### Install dependencies

```bash
conda env create -f sepoa_conda_env.yaml
conda activate sepoa
pip install git+https://github.com/rlworkgroup/metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

You could run `python -c "import mujoco_py; print(mujoco_py.__version__)"` to check if `mujoco-py` is installed properly. 


## Run experiments under non-ideal teachers

Run S-EPOA for `Quadruped_run` task with $\epsilon=0.3$ (seed=1, GPU `cuda:0`):

```bash
# skill-based pretraining
bash ./scripts/pretrain_sdpbrl.sh  {agent: aps_sac / diayn_sac / cic_sac} quadruped cuda:0 1
# online PbRL
bash ./scripts/run_sdpbrl.sh  {agent: aps_sac / diayn_sac / cic_sac} quadruped_run 2000 200 0.3 1 cuda:0 1000000
```

Run S-EPOA for `Door_open` task with $\epsilon=0.2$:

```bash
# skill-based pretraining
bash ./scripts/metaworld/pretrain_metaworld_sdpbrl.sh  door_open cuda:0 1
# online PbRL
bash ./scripts/metaworld/run_metaworld_sdpbrl.sh  door_open 0.2 1000000 cuda:0 1
```

Run SURF for `Quadruped_run` task with $\epsilon=0.3$:

```bash
# PEBBLE's pretraining
bash ./scripts/pretrain_pebble.sh  quadruped cuda:0 1
# online PbRL
bash ./scripts/run_surf.sh  quadruped_run 0.3 2000 cuda:0 1
```

Run PEBBLE for `Door_open` task with $\epsilon=0.2$:

```bash
# PEBBLE's pretraining
bash ./scripts/metaworld_scripts/pretrain_metaworld_pebble.sh  door_open cuda:0 1
# online PbRL
bash ./scripts/metaworld_scripts/run_metaworld_pebble.sh  door_open 0.2 cuda:0 1
```


## Acknowledgement

This repo benefits from [URLB](https://github.com/rll-research/url_benchmark), [BPref](https://github.com/rll-research/BPref), [SURF](https://github.com/alinlab/SURF), [RUNE](https://github.com/rll-research/rune), and [RIME](https://github.com/CJReinforce/RIME_ICML2024). Thanks for their wonderful work.

## Citation

If you find this project helpful, please consider citing the following paper:

```bibtex
@inproceedings{mu2025sepoa,
  title     = {S-EPOA: Overcoming the Indistinguishability of Segments with Skill-Driven Preference-Based Reinforcement Learning},
  author    = {Mu, Ni and Luan, Yao and Yang, Yiqin and Xu, Bo and Jia, Qing-Shan},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  pages     = {5967--5975},
  year      = {2025},
  doi       = {10.24963/ijcai.2025/664}
}
```


