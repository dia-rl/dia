# DIA: Denoising Intermediate Advantage

[[Project page](https://dia-rl.github.io)]&nbsp;&nbsp;[[Built on DPPO](https://github.com/irom-princeton/dppo)]

Arjun Sohal<sup>1</sup>, Yuchi Zhao<sup>1,2</sup>, Miroslav Bogdanovic<sup>1,2,3</sup>, Alan Aspuru-Guzik<sup>1,2,3,4,5</sup>

<sup>1</sup>University of Toronto&nbsp;&nbsp;<sup>2</sup>Vector Institute for Artificial Intelligence

> DIA is a policy-gradient method for fine-tuning diffusion policies. It learns a
> value function over *partially denoised* actions and turns it into a per-denoising-step
> advantage, so the optimizer can tell which intermediate decisions in the denoising
> chain contributed to the return.

## The method

Policy-gradient fine-tuning of a diffusion policy treats the problem as two nested
MDPs: an outer environment MDP, and an inner MDP whose steps are the denoising
iterations that produce one action chunk. DPPO and related methods compute a single
environment-level advantage per action and broadcast it unchanged across every
denoising step, so every step of the chain receives identical credit.

DIA adds a second, denoising-level signal on top of that. Writing `x_k` for the
partially denoised action at step `k` of a `K`-step chain and `a_0` for the action the
chain finally emits:

1. **A Q critic** `Q(s, a)` is trained on environment transitions by 1-step TD, as an
   `n_heads` ensemble whose heads are averaged. A target copy is refreshed each
   iteration at `target_ema_rate`.
2. **An inner value** `V_in(s, x_k, k)` is regressed onto the target Q at the emitted
   action, `Q_target(s, a_0)`. The regression target is one scalar per chain, shared by
   every `k`, but the *inputs* vary along the chain, so `V_in` learns how much `x_k`
   already reveals about the value of the action the chain will produce.
3. **An inner advantage** is a GAE along `k` over the differences in that value, with no
   environment reward inside a chain and no within-chain discount:

   ```
   d_k     = V_in(s, x_{k+1}, k+1) - V_in(s, x_k, k)
   A_in[k] = d_k + lambda_in * A_in[k+1],    A_in[K] = 0
   ```

4. **Scale matching** rescales it to the outer advantage's spread,
   `A_in_scaled = A_in * (sigma_out / sigma_in)`, so that `alpha` below is a mixing
   weight rather than a scale knob.
5. **The combined advantage** fed to the PPO objective is

   ```
   adv[t, k] = A_out[t] + alpha * A_in_scaled[t, k]
   ```

   With `alpha = 0` this reduces exactly to DPPO. The outer advantage stays dominant;
   `alpha` controls how much denoising-level credit is layered on top.

Nothing about the actor changes. `V_in` and the Q ensemble exist only to build the
advantage, and at evaluation time a DIA policy is an ordinary diffusion policy.

The knobs that matter live under `train:` in each config: `v_inner_alpha` (the `alpha`
above), `v_inner_lambda` (`lambda_in`), `q_aggregation`, `critic_q_lr`,
`target_ema_rate`, and `n_critic_warmup_itr`.

## Relationship to DPPO

This repository is a fork of [DPPO](https://github.com/irom-princeton/dppo)
([paper](https://arxiv.org/abs/2409.00588)), and DPPO is both the codebase it is built
on and the baseline it is compared against. The environment wrappers, diffusion policy,
PPO update, dataset handling and configuration layout are DPPO's. DIA adds:

- `agent/finetune/train_ppo_dia_diffusion_agent.py` and
  `model/diffusion/diffusion_ppo_dia.py`, the agent and model implementing the above
- `CriticObsInnerState` in `model/common/critic.py`, the inner value network
- one `ft_dia_diffusion_mlp.yaml` per task under `cfg/`
- the ablation agents under `agent/finetune/train_ablation_*.py`
- a residual-RL baseline under `agent/residual/`

The DPPO baseline in every comparison is this repository's own
`ft_ppo_diffusion_mlp.yaml`, unchanged from upstream apart from the settings a fair
comparison requires.

## Reproducing the results

Each task has one DIA config and one matching DPPO config, both of which are the ones
behind the reported numbers:

```console
# Robomimic - lift/can/square/transport
python script/run.py --config-name=ft_dia_diffusion_mlp \
    --config-dir=cfg/robomimic/finetune/transport
# Franka Kitchen - complete/mixed/partial
python script/run.py --config-name=ft_dia_diffusion_mlp \
    --config-dir=cfg/gym/finetune/kitchen-mixed-v0
# D3IL avoid - m1/m2/m3
python script/run.py --config-name=ft_dia_diffusion_mlp \
    --config-dir=cfg/d3il/finetune/avoid_m1
# FurnitureBench - one_leg_med/lamp_med
python script/run.py --config-name=ft_dia_diffusion_mlp \
    --config-dir=cfg/furniture/finetune/lamp_med
```

Swap `ft_dia_diffusion_mlp` for `ft_ppo_diffusion_mlp` to run the DPPO baseline on the
same task.

**Franka Kitchen needs the checkpoints shipped here.** Kitchen observations contain
many near-constant dimensions, and upstream's normalization divides by their range,
which is ~0. `script/dataset/get_d4rl_dataset.py` in this repository floors that range
and clips the result, and the kitchen base policies were pretrained on data regenerated
that way. Those policies are committed under `pretrained/kitchen-*/`, and the kitchen
configs point at them, so do not substitute the checkpoints from upstream's download
link: they predate the fix and will not reproduce the kitchen results.

## Installation 

1. Clone the repository
```console
git clone git@github.com:dia-rl/dia.git
cd dia
```

2. Install core dependencies with a conda environment (if you do not plan to use Furniture-Bench, a higher Python version such as 3.10 can be installed instead) on a Linux machine with a Nvidia GPU.
```console
conda create -n dppo python=3.8 -y
conda activate dppo
pip install -e .
```

3. Install specific environment dependencies (Gym / Kitchen / Robomimic / D3IL / Furniture-Bench) or all dependencies (except for Kitchen, which has dependency conflicts with other tasks).
```console
pip install -e .[gym] # or [kitchen], [robomimic], [d3il], [furniture]
pip install -e .[all] # except for Kitchen
```

4. [Install MuJoCo for Gym and/or Robomimic](installation/install_mujoco.md). [Install D3IL](installation/install_d3il.md). [Install IsaacGym and Furniture-Bench](installation/install_furniture.md)

5. Set environment variables for data and logging directory (default is `data/` and `log/`), and set WandB entity (username or team name)
```
source script/set_path.sh
```

## Usage - Pre-training

**Note**: You may skip pre-training if you would like to use the default checkpoint (available for download) for fine-tuning.

<!-- ### Prepare pre-training data

First create a directory as the parent directory of the pre-training data and set the environment variable for it.
```console
export DPPO_DATA_DIR=/path/to/data -->
<!-- ``` -->

Pre-training data for all tasks are pre-processed and can be found at [here](https://drive.google.com/drive/folders/1AXZvNQEKOrp0_jk1VLepKh_oHCg_9e3r?usp=drive_link). Pre-training script will download the data (including normalization statistics) automatically to the data directory.
<!-- The data path follows `${DPPO_DATA_DIR}/<benchmark>/<task>/train.npz`, e.g., `${DPPO_DATA_DIR}/gym/hopper-medium-v2/train.npz`. -->

### Run pre-training with data
All the configs can be found under `cfg/<env>/pretrain/`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.
<!-- To run pre-training, first set your WandB entity (username or team name) and the parent directory for logging as environment variables. -->
<!-- ```console
export DPPO_WANDB_ENTITY=<your_wandb_entity>
export DPPO_LOG_DIR=<your_prefered_logging_directory>
``` -->
```console
# Gym - hopper/walker2d/halfcheetah
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/gym/pretrain/hopper-medium-v2
# Robomimic - lift/can/square/transport
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/robomimic/pretrain/can
# D3IL - avoid_m1/m2/m3
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/d3il/pretrain/avoid_m1
# Furniture-Bench - one_leg/lamp/round_table_low/med
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/furniture/pretrain/one_leg_low
```

See [here](cfg/pretraining.md) for details of the experiments in the paper.

## Usage - Fine-tuning

<!-- ### Set up pre-trained policy -->

<!-- If you did not set the environment variables for pre-training, we need to set them here for fine-tuning. 
```console
export DPPO_WANDB_ENTITY=<your_wandb_entity>
export DPPO_LOG_DIR=<your_prefered_logging_directory>
``` -->
<!-- First create a directory as the parent directory of the downloaded checkpoints and set the environment variable for it.
```console
export DPPO_LOG_DIR=/path/to/checkpoint
``` -->

Pre-trained policies used in the paper can be found [here](https://drive.google.com/drive/folders/1ZlFqmhxC4S8Xh1pzZ-fXYzS5-P8sfpiP?usp=drive_link). Fine-tuning script will download the default checkpoint automatically to the logging directory.
 <!-- or you may manually download other ones (different epochs) or use your own pre-trained policy if you like. -->

 <!-- e.g., `${DPPO_LOG_DIR}/gym-pretrain/hopper-medium-v2_pre_diffusion_mlp_ta4_td20/2024-08-26_22-31-03_42/checkpoint/state_0.pt`. -->

<!-- The checkpoint path follows `${DPPO_LOG_DIR}/<benchmark>/<task>/.../<run>/checkpoint/state_<epoch>.pt`. -->

### Fine-tuning pre-trained policy

All the configs can be found under `cfg/<env>/finetune/`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.
<!-- Running them will download the default pre-trained policy. -->
<!-- Running the script will download the default pre-trained policy checkpoint specified in the config (`base_policy_path`) automatically, as well as the normalization statistics, to `DPPO_LOG_DIR`.  -->
```console
# Gym - hopper/walker2d/halfcheetah
python script/run.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/gym/finetune/hopper-v2
# Robomimic - lift/can/square/transport
python script/run.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/robomimic/finetune/can
# D3IL - avoid_m1/m2/m3
python script/run.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/d3il/finetune/avoid_m1
# Furniture-Bench - one_leg/lamp/round_table_low/med
python script/run.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/furniture/finetune/one_leg_low
```

**Note**: In Gym, Robomimic, and D3IL tasks, we run 40, 50, and 50 parallelized MuJoCo environments on CPU, respectively. If you would like to use fewer environments (given limited CPU threads, or GPU memory for rendering), you can reduce `env.n_envs` and increase `train.n_steps`, so the total number of environment steps collected in each iteration (n_envs x n_steps x act_steps) remains roughly the same. Try to set `train.n_steps` a multiple of `env.max_episode_steps / act_steps`, and be aware that we only count episodes finished within an iteration for eval. Furniture-Bench tasks run IsaacGym on a single GPU.

To fine-tune your own pre-trained policy instead, override `base_policy_path` to your own checkpoint, which is saved under `checkpoint/` of the pre-training directory. You can set `base_policy_path=<path>` in the command line when launching fine-tuning.

<!-- **Note**: If you did not download the pre-training [data](https://drive.google.com/drive/folders/1AXZvNQEKOrp0_jk1VLepKh_oHCg_9e3r?usp=drive_link), you need to download the normalization statistics from it for fine-tuning, e.g., `${DPPO_DATA_DIR}/furniture/round_table_low/normalization.pkl`. -->

See [here](cfg/finetuning.md) for details of the experiments in the paper.


### Visualization
* Furniture-Bench tasks can be visualized in GUI by specifying `env.specific.headless=False` and `env.n_envs=1` in fine-tuning configs.
* D3IL environment can be visualized in GUI by `+env.render=True`, `env.n_envs=1`, and `train.render.num=1`. There is a basic script at `script/test_d3il_render.py`.
* Videos of trials in Robomimic tasks can be recorded by specifying `env.save_video=True`, `train.render.freq=<iterations>`, and `train.render.num=<num_video>` in fine-tuning configs.

## Usage - Evaluation
Pre-trained or fine-tuned policies can be evaluated without running the fine-tuning script now. Some example configs are provided under `cfg/{gym/robomimic/furniture}/eval}` including ones below. Set `base_policy_path` to override the default checkpoint, and `ft_denoising_steps` needs to match fine-tuning config (otherwise assumes `ft_denoising_steps=0`, which means evaluating the pre-trained policy).
```console
python script/run.py --config-name=eval_diffusion_mlp \
    --config-dir=cfg/gym/eval/hopper-v2 ft_denoising_steps=?
python script/run.py --config-name=eval_{diffusion/gaussian}_mlp_{?img} \
    --config-dir=cfg/robomimic/eval/can ft_denoising_steps=?
python script/run.py --config-name=eval_diffusion_mlp \
    --config-dir=cfg/furniture/eval/one_leg_low ft_denoising_steps=?
```

## DPPO implementation

Our diffusion implementation is mostly based on [Diffuser](https://github.com/jannerm/diffuser) and at [`model/diffusion/diffusion.py`](model/diffusion/diffusion.py) and [`model/diffusion/diffusion_vpg.py`](model/diffusion/diffusion_vpg.py). PPO specifics are implemented at [`model/diffusion/diffusion_ppo.py`](model/diffusion/diffusion_ppo.py). The main training script is at [`agent/finetune/train_ppo_diffusion_agent.py`](agent/finetune/train_ppo_diffusion_agent.py) that follows [CleanRL](https://github.com/vwxyzjn/cleanrl).

### Key configurations
* `denoising_steps`: number of denoising steps (should always be the same for pre-training and fine-tuning regardless the fine-tuning scheme)
* `ft_denoising_steps`: number of fine-tuned denoising steps
* `horizon_steps`: predicted action chunk size (should be the same as `act_steps`, executed action chunk size, with MLP. Can be different with UNet, e.g., `horizon_steps=16` and `act_steps=8`)
* `model.gamma_denoising`: denoising discount factor
* `model.min_sampling_denoising_std`: <img src="https://latex.codecogs.com/gif.latex?\epsilon^\text{exp}_\text{min} "/>, minimum amount of noise when sampling at a denoising step
* `model.min_logprob_denoising_std`: <img src="https://latex.codecogs.com/gif.latex?\epsilon^\text{prob}_\text{min} "/>, minimum standard deviation when evaluating likelihood at a denoising step
* `model.clip_ploss_coef`: PPO clipping ratio
* `train.batch_size`: you may notice the batch size is rather large --- this is due to the PPO update being in expectation over both environment steps and denoising steps (new in v0.6).

### DDIM fine-tuning

To use DDIM fine-tuning, set `denoising_steps=100` in pre-training and set `model.use_ddim=True`, `model.ddim_steps` to the desired number of total DDIM steps, and `ft_denoising_steps` to the desired number of fine-tuned DDIM steps. In our Furniture-Bench experiments we use `denoising_steps=100`, `model.ddim_steps=5`, and `ft_denoising_steps=5`.

## Adding your own dataset/environment

### Pre-training data
Pre-training script is at [`agent/pretrain/train_diffusion_agent.py`](agent/pretrain/train_diffusion_agent.py). The pre-training dataset [loader](agent/dataset/sequence.py) assumes a npz file containing numpy arrays `states`, `actions`, `images` (if using pixel; img_h = img_w and a multiple of 8) and `traj_lengths`, where `states` and `actions` have the shape of num_total_steps x obs_dim/act_dim, `images` num_total_steps x C (concatenated if multiple images) x H x W, and `traj_lengths` is a 1-D array for indexing across num_total_steps.
<!-- One pre-processing example can be found at [`script/process_robomimic_dataset.py`](script/process_robomimic_dataset.py). -->

<!-- **Note:** The current implementation does not support loading history observations (only using observation at the current timestep). If needed, you can modify [here](agent/dataset/sequence.py#L130-L131). -->

#### Observation history
In our experiments we did not use any observation from previous timesteps (state or pixel), but it is implemented. You can set `cond_steps=<num_state_obs_step>` (and `img_cond_steps=<num_img_obs_step>`, no larger than `cond_steps`) in pre-training, and set the same when fine-tuning the newly pre-trained policy.

### Fine-tuning environment
We follow the Gym format for interacting with the environments. The vectorized environments are initialized at [make_async](env/gym_utils/__init__.py#L10) (called in the parent fine-tuning agent class [here](agent/finetune/train_agent.py#L38-L39)). The current implementation is not the cleanest as we tried to make it compatible with Gym, Robomimic, Furniture-Bench, and D3IL environments, but it should be easy to modify and allow using other environments. We use [multi_step](env/gym_utils/wrapper/multi_step.py) wrapper for history observations and multi-environment-step action execution. We also use environment-specific wrappers such as [robomimic_lowdim](env/gym_utils/wrapper/robomimic_lowdim.py) and [furniture](env/gym_utils/wrapper/furniture.py) for observation/action normalization, etc. You can implement a new environment wrapper if needed.

## Known issues
* IsaacGym simulation can become unstable at times and lead to NaN observations in Furniture-Bench. The current env wrapper does not handle NaN observations.

## License
This repository is released under the MIT license. See [LICENSE](LICENSE).

## Acknowledgement
* [DPPO, Ren et al.](https://github.com/irom-princeton/dppo): the codebase this repository is built on, and the baseline DIA is compared against
* [Diffuser, Janner et al.](https://github.com/jannerm/diffuser): general code base and DDPM implementation
* [Diffusion Policy, Chi et al.](https://github.com/real-stanford/diffusion_policy): general code base especially the env wrappers
* [CleanRL, Huang et al.](https://github.com/vwxyzjn/cleanrl): PPO implementation
* [IBRL, Hu et al.](https://github.com/hengyuan-hu/ibrl): ViT implementation
* [D3IL, Jia et al.](https://github.com/ALRhub/d3il): D3IL benchmark
* [Robomimic, Mandlekar et al.](https://github.com/ARISE-Initiative/robomimic): Robomimic benchmark
* [Furniture-Bench, Heo et al.](https://github.com/clvrai/furniture-bench): Furniture-Bench benchmark
* [AWR, Peng et al.](https://github.com/xbpeng/awr): DAWR baseline (modified from AWR)
* [DIPO, Yang et al.](https://github.com/BellmanTimeHut/DIPO): DIPO baseline
* [IDQL, Hansen-Estruch et al.](https://github.com/philippe-eecs/IDQL): IDQL baseline
* [DQL, Wang et al.](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL): DQL baseline
* [QSM, Psenka et al.](https://www.michaelpsenka.io/qsm/): QSM baseline
* [Score SDE, Song et al.](https://github.com/yang-song/score_sde_pytorch/): diffusion exact likelihood