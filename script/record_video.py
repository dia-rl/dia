"""Record a video of one deterministic rollout from a saved ckpt."""
import os, sys, math, argparse, logging, tempfile, shutil
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--ep-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.wandb = None
    cfg.logdir = tempfile.mkdtemp(prefix="vid_")
    cfg.env.n_envs = args.n_envs
    cfg.env.save_video = True
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.train.render = {"freq": 1, "num": args.n_envs}
    cfg.train.n_steps = args.ep_steps
    OmegaConf.resolve(cfg)
    agent = hydra.utils.get_class(cfg._target_)(cfg)
    d = torch.load(args.ckpt, weights_only=True, map_location=agent.device)
    agent.model.load_state_dict(d["model"], strict=False)
    # skip reward_scaler load — irrelevant for eval rendering

    # rollout with video_path option
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    options_venv = [{"video_path": out if i == 0 else None} for i in range(args.n_envs)]
    prev = agent.reset_env_all(options_venv=options_venv)
    agent.model.eval()
    success = False
    total_r = 0.0
    for step in range(args.ep_steps):
        with torch.no_grad():
            cond = {"state": torch.from_numpy(prev["state"]).float().to(agent.device)}
            samples = agent.model(cond=cond, deterministic=True, return_chain=False)
            out_a = samples.trajectories.cpu().numpy()
        action = out_a[:, : agent.act_steps]
        obs, rew, term, trunc, info = agent.venv.step(action)
        total_r += float(rew[0])
        if rew[0] >= 1:
            success = True
        prev = obs
        if term[0] or trunc[0]:
            break
    log.info(f"video saved -> {out}, success={success}, total_r={total_r:.2f}")


if __name__ == "__main__":
    main()
