"""Derive the alpha=0 control config from a DIA config, mechanically.

alpha=0 zeroes the inner advantage, so the policy update reduces to DPPO's while the Q
and V_inner critics are still trained. It is the compute-matched null. The bundled a0
runs set it with a CLI override (train.v_inner_alpha=0); doing it in a config instead
keeps the job script free of algorithmic overrides, so the arm is fully described by
its config.

Edits exactly two things: v_inner_alpha, and name.

    python script/gen_ablation_alpha0_configs.py            # write
    python script/gen_ablation_alpha0_configs.py --check    # verify, write nothing
"""
import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = [
    "cfg/d3il/finetune/avoid_m1/ft_dia_diffusion_mlp.yaml",
    "cfg/robomimic/finetune/can/ft_dia_diffusion_mlp.yaml",
    "cfg/gym/finetune/kitchen-mixed-v0/ft_dia_diffusion_mlp.yaml",
]


def derive(text):
    m = re.search(r"^(\s*)v_inner_alpha:\s*\S+", text, re.M)
    if not m:
        raise SystemExit("no v_inner_alpha line")
    out = text[: m.start()] + f"{m.group(1)}v_inner_alpha: 0.0" + text[m.end():]
    n = re.search(r"^name: (.+)$", out, re.M)
    if not n:
        raise SystemExit("no name: line")
    return out[: n.end(1)] + "_a0" + out[n.end(1):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    bad = 0
    for src in SOURCES:
        dst = src.replace(".yaml", "_a0.yaml")
        want = derive(open(os.path.join(REPO, src)).read())
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
