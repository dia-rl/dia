# ResiP: residual RL on a frozen diffusion policy

Baseline port of

> Lars Ankile, Anthony Simeonov, Idan Shenfeld, Marcel Torne, Pulkit Agrawal.
> *From Imitation to Refinement: Residual RL for Precise Assembly.* arXiv 2407.16677.
> <https://github.com/ankile/robust-rearrangement>

The base policy is frozen and plays an action chunk open loop; a small Gaussian
residual is queried every environment step and corrects the action about to run:

```
a = a_base + action_scale * a_residual        # normalized action space
residual input = [normalized obs (clipped to +-3), a_base]
```

## Design: subclass DPPO, change only the algorithm

Built the way DIA is. `ResidualDiffusion` extends `VPGDiffusion` (the class
`PPODiffusion` also extends) and `TrainResidualPPOAgent` extends
`TrainPPODiffusionAgent`, so everything that is not the algorithm is DPPO's own code
and cannot drift from it.

Most importantly, **`forward` is not overridden**. The base policy is sampled by
DPPO's sampler, so the exploration noise floor (`min_sampling_denoising_std`), the
DDPM/DDIM branch, the deterministic evaluation branch and the initial-latent
randomization are shared rather than reimplemented, so with the same weights and seed
the two models produce **bitwise identical** samples.

`ResidualDiffusion` subclasses `VPGDiffusion`, the **sibling** of DPPO's
`PPODiffusion`, not `PPODiffusion` itself. DIA extends `PPODiffusion` because it
reuses DPPO's actor update and only adds critic terms; ResiP freezes the chain, so
that loss would be dead code.

**Inert keys.** Because each config is a literal copy of the DPPO one, it carries four
`PPODiffusion`-only keys that have **no effect** here: `gamma_denoising`,
`clip_ploss_coef`, `clip_ploss_coef_base`, `clip_ploss_coef_rate`. They are accepted by
`VPGDiffusion`'s `**kwargs` and ignored, because the loss that reads them is never
called. The `critic` block (DPPO's `CriticObs`) is likewise built but never trained or
read; ResiP's value function is `residual_policy.critic`. Changing any of them changes
nothing. They are kept so the config stays comparable to the DPPO one key for key.

`ft_denoising_steps: 0` is what freezes the base. `VPGDiffusion.p_mean_var` computes
`ft_indices = where(t < ft_denoising_steps)`, empty at 0, so every denoising step uses
the frozen `actor`.

| file | contents |
| --- | --- |
| `residual_policy.py` | their `src/models/residual.py`: Gaussian actor, value critic, initialization |
| `residual_diffusion.py` | `ResidualDiffusion(VPGDiffusion)`: chunk queue and composition |
| `train_residual_ppo_agent.py` | `TrainResidualPPOAgent(TrainPPODiffusionAgent)`: their PPO update |
| `gen_configs.py` | writes each config as the DPPO config plus a `resip` block |

Nothing outside this directory is modified. Deleting `agent/residual/` and
`cfg/**/ft_residual_ppo_diffusion_mlp.yaml` removes the baseline entirely.

## What is ResiP's

The algorithm and its tuned hyperparameters, from their `base_residual_rl.yaml` and
`actor/residual_diffusion.yaml`, all under `train.resip` and `model.residual_policy`:

| | |
| --- | --- |
| residual actor | 2x[256] ReLU, output layer orthogonal std 0, no bias |
| residual critic | 2x[256] ReLU, output std 0.25, bias 0.25 |
| log std | fixed at -1.0 (`learn_std: false`) |
| `action_scale` | 0.1 |
| discount, GAE lambda | 0.999 **per environment step**, 0.95 |
| clip coef, value clipping | 0.2, off |
| entropy, value coef | 0.0, 1.0 |
| grad norm, target KL | 1.0, 0.1 |
| epochs, minibatches | 50, 1 |
| lr actor, critic | 3e-4, 5e-3, AdamW eps 1e-5 wd 1e-6, cosine with warmup |

`action_head_std: 0` and `learn_std: false` are why a run starts exactly at the frozen
base: the residual mean is identically zero at initialization.

## What is DPPO's

Everything else, so the arms differ only in the learning algorithm:

- base policy sampling, inherited (verified bitwise)
- environment, wrappers, observation, frozen BC checkpoint, normalization, seeds
- `n_train_itr`, `val_freq`, `save_model_freq`, and the environment steps per iteration
- `gamma`, `reward_scale_running`, `reward_scale_const`
- `n_critic_warmup_itr`, which holds the actor while the critic settles
- **terminal handling**: the GAE masks on `terminated` only, so a time-limit truncation
  is bootstrapped through, as DPPO does. ResiP's config sets `truncation_as_done: true`,
  which cuts the bootstrap; on robomimic, where the wrapper never sets `terminated`,
  that alone biases every late-episode value target low. It reads the harness, not the
  algorithm, so it is matched and the flag is not exposed.
- episode statistics and the success-rate formula, computed by DPPO's code on a
  chunk-aggregated view of the rollout, so `max(reward)/act_steps` means the same thing
- reward scaling: DPPO's `RunningRewardScaler`, updated at DPPO's chunk granularity so
  its divisor is the same number, then applied to the per-step rewards

Two structural consequences, inherent to closed-loop control rather than chosen:

- the wrapper executes one environment step per policy call, so `train.n_steps` counts
  environment steps and is `DPPO n_steps x act_steps`. Environment steps per iteration
  match exactly.
- ResiP stores one transition per environment step, so its PPO buffer is `act_steps`
  times DPPO's. Rollout cost is matched; update cost is not, and it is each method's
  own budget (ResiP takes 50 gradient steps per iteration, DPPO 200-355).

The discount is left at ResiP's 0.999 per environment step, which is a shorter
effective horizon than DPPO's 0.999 per chunk. That is their design and is kept.

## Running

```console
python agent/residual/gen_configs.py     # regenerate configs from the DPPO ones
python script/run.py --config-name=ft_residual_ppo_diffusion_mlp \
    --config-dir=cfg/robomimic/finetune/can seed=42
```

Configs: robomimic can/lift/square/transport, kitchen complete/partial/mixed,
furniture one_leg_med/lamp_med.
