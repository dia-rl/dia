"""Derive the outer-target ablation configs from the DIA configs, mechanically.

The arm must be DIA in every respect except the agent class, so the config is
generated rather than written: it copies the DIA config and edits exactly two lines,
the target class and the name.

    python script/gen_ablation_outertarget_configs.py            # write
    python script/gen_ablation_outertarget_configs.py --check    # verify, write nothing
"""
import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIA = "agent.finetune.train_ppo_dia_diffusion_agent.TrainPPODIADiffusionAgent"
VARIANTS = {
    "outertarget_raw": "agent.finetune.train_ablation_outer_target_raw_agent.TrainAblationOuterTargetRawAgent",
}
SOURCES = [
    "cfg/d3il/finetune/avoid_m1/ft_dia_diffusion_mlp.yaml",
    "cfg/robomimic/finetune/can/ft_dia_diffusion_mlp.yaml",
    "cfg/robomimic/finetune/transport/ft_dia_diffusion_mlp.yaml",
    "cfg/gym/finetune/kitchen-mixed-v0/ft_dia_diffusion_mlp.yaml",
    "cfg/gym/finetune/kitchen-complete-v0/ft_dia_diffusion_mlp.yaml",
    "cfg/gym/finetune/kitchen-partial-v0/ft_dia_diffusion_mlp.yaml",
]


def derive(text, tag):
    if text.count(DIA) != 1:
        raise SystemExit(f"expected exactly one {DIA}")
    out = text.replace(DIA, VARIANTS[tag])
    m = re.search(r"^name: (.+)$", out, re.M)
    if not m:
        raise SystemExit("no name: line")
    return out[: m.end(1)] + f"_{tag}" + out[m.end(1):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    bad = 0
    for src in SOURCES:
      for tag in VARIANTS:
        dst = src.replace(".yaml", f"_{tag}.yaml")
        want = derive(open(os.path.join(REPO, src)).read(), tag)
        p = os.path.join(REPO, dst)
        if args.check:
            ok = os.path.exists(p) and open(p).read() == want
            print(f"{'ok  ' if ok else 'DRIFT'} {dst}")
            bad += not ok
        else:
            open(p, "w").write(want)
            print(f"wrote {dst}")
    if args.check and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
