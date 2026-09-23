"""
Run N independent deterministic eval rollouts of a final policy and report
mean +/- std of success rate and avg episode reward.

Each "eval round" = one rollout of train.n_steps (matches what the finetune
log calls an "eval": SR/R over all episodes completed in that rollout, ~200
episodes for transport at n_steps=400). Rounds differ only by the env-reset
seed-set; pass the SAME --seed-base to two runs for a PAIRED (same start
states) comparison across policies.

Loads the run's OWN saved .hydra/config.yaml so ft_denoising_steps matches
how the policy was trained (faithful to each method).

Usage:
  python script/eval_final_policy.py \
    --cfg <run>/.hydra/config.yaml \
    --ckpt <run>/checkpoint/state_109.pt \
    --rounds 10 --seed-base 7000 \
    --label dual_s42 --out /tmp/transport_runs/eval_dual.json
"""
import os, sys, math, json, argparse, logging, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from omegaconf import OmegaConf
import hydra

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--label", default="policy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="override train.n_steps (for cheap validation; default uses training value)")
    ap.add_argument("--n-envs", type=int, default=None,
                    help="override env.n_envs (e.g. 1 for a single-env eval)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.wandb = None
    if args.n_steps is not None:
        cfg.train.n_steps = args.n_steps
    if args.n_envs is not None:
        cfg.env.n_envs = args.n_envs
    # Concrete logdir avoids the ${now} resolver; nothing is written we care about.
    cfg.logdir = tempfile.mkdtemp(prefix="eval_final_")
    OmegaConf.resolve(cfg)

    agent = hydra.utils.get_class(cfg._target_)(cfg)
    d = torch.load(args.ckpt, weights_only=True, map_location=agent.device)
    agent.model.load_state_dict(d["model"], strict=False)
    log.info(f"Loaded {args.ckpt} (itr={d.get('itr')})")

    n = agent.n_envs
    n_steps = agent.n_steps
    act_steps = agent.act_steps
    thresh = agent.best_reward_threshold_for_success
    K_ft = agent.model.ft_denoising_steps
    log.info(
        f"=== eval_final_policy: label={args.label} n_envs={n} n_steps={n_steps} "
        f"act_steps={act_steps} ft_denoising_steps={K_ft} "
        f"rounds={args.rounds} seed_base={args.seed_base} ==="
    )

    agent.model.eval()
    round_sr, round_R, round_nep = [], [], []

    for r in range(args.rounds):
        seeds = [args.seed_base + r * 1000 + i for i in range(n)]
        prev_obs = agent.reset_env_all(options_venv=[{"seed": s} for s in seeds])
        firsts = np.zeros((n_steps + 1, n))
        firsts[0] = 1
        reward_trajs = np.zeros((n_steps, n))

        for step in range(n_steps):
            with torch.no_grad():
                cond = {"state": torch.from_numpy(prev_obs["state"]).float().to(agent.device)}
                samples = agent.model(cond=cond, deterministic=True, return_chain=True)
                output_venv = samples.trajectories.cpu().numpy()
            action_venv = output_venv[:, :act_steps]
            obs_venv, reward_venv, term_venv, trunc_venv, _ = agent.venv.step(action_venv)
            reward_trajs[step] = reward_venv
            firsts[step + 1] = term_venv | trunc_venv
            prev_obs = obs_venv

        # Episode accounting — identical to train_ppo_diffusion_agent.run()
        eps = []
        for e in range(n):
            edges = np.where(firsts[:, e] == 1)[0]
            for i in range(len(edges) - 1):
                s0, s1 = edges[i], edges[i + 1]
                if s1 - s0 > 1:
                    eps.append((e, s0, s1 - 1))
        if eps:
            splits = [reward_trajs[s0:s1 + 1, e] for e, s0, s1 in eps]
            ep_R = np.array([np.sum(x) for x in splits])
            ep_best = np.array([np.max(x) / act_steps for x in splits])
            sr = float(np.mean(ep_best >= thresh))
            avgR = float(np.mean(ep_R))
            nep = len(splits)
        else:
            sr, avgR, nep = 0.0, 0.0, 0
        round_sr.append(sr); round_R.append(avgR); round_nep.append(nep)
        log.info(f"  round {r:2d} | seeds[{seeds[0]}..{seeds[-1]}] | "
                 f"n_ep={nep:4d} | SR={sr:.4f} | R={avgR:8.3f}")

    sr_a, R_a = np.array(round_sr), np.array(round_R)
    log.info("=== Summary ===")
    log.info(f"  SR : mean={sr_a.mean():.4f} std={sr_a.std():.4f} min={sr_a.min():.4f} max={sr_a.max():.4f}")
    log.info(f"  R  : mean={R_a.mean():.3f} std={R_a.std():.3f} min={R_a.min():.3f} max={R_a.max():.3f}")
    log.info(f"  total episodes: {int(np.sum(round_nep))}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "label": args.label, "ckpt": args.ckpt, "itr": int(d.get("itr", -1)),
                "n_envs": n, "n_steps": n_steps, "ft_denoising_steps": K_ft,
                "seed_base": args.seed_base,
                "round_sr": round_sr, "round_R": round_R, "round_nep": round_nep,
                "sr_mean": float(sr_a.mean()), "sr_std": float(sr_a.std()),
                "R_mean": float(R_a.mean()), "R_std": float(R_a.std()),
            }, f, indent=2)
        log.info(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
