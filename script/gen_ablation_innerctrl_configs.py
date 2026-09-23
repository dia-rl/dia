"""Derive the inner-advantage control config from a DIA config, mechanically.

The control arm has to be the DIA arm in every respect except the agent class, so
the config is generated from DIA's rather than written by hand: it copies the file
verbatim and edits exactly three things.

  _target_                     -> the control agent
  name                         -> same, plus the mode, so logdirs stay distinct
  train.inner_control_mode     -> added

    python script/gen_ablation_innerctrl_configs.py            # write the configs
    python script/gen_ablation_innerctrl_configs.py --check    # verify, write nothing

Run --check after touching a DIA config; it fails if the pair has drifted apart.
"""

import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT = "agent.finetune.train_ablation_inner_control_agent.TrainAblationInnerControlAgent"
DIA_AGENT = "agent.finetune.train_ppo_dia_diffusion_agent.TrainPPODIADiffusionAgent"

# DIA configs to derive a control arm from. One config is written per mode, so a job
# script never has to pass an algorithmic override: the arm is the config.
SOURCES = [
    "cfg/robomimic/finetune/can/ft_dia_diffusion_mlp.yaml",
    "cfg/gym/finetune/kitchen-mixed-v0/ft_dia_diffusion_mlp.yaml",
    "cfg/robomimic/finetune/transport/ft_dia_diffusion_mlp.yaml",
]
# the two controls we run: one destroys the signal in the increments, the other
# keeps every marginal exact and destroys only the advantage-to-transition pairing
MODES = ("delta_gaussian", "shuffle_cross_sample")


def pairs():
    for src in SOURCES:
        for mode in MODES:
            yield src, src.replace(".yaml", f"_innerctrl_{mode}.yaml"), mode


def derive(src_text, mode):
    if src_text.count(DIA_AGENT) != 1:
        raise SystemExit(f"expected exactly one {DIA_AGENT} line")
    out = src_text.replace(DIA_AGENT, AGENT)

    m = re.search(r"^name: (.+)$", out, re.M)
    if not m:
        raise SystemExit("no name: line")
    out = out[: m.end(1)] + f"_innerctrl_{mode}" + out[m.end(1) :]

    # put the knob at the top of the train block so it is impossible to miss
    m = re.search(r"^train:\n", out, re.M)
    if not m:
        raise SystemExit("no train: block")
    knob = (
        "  # Control arm: the inner advantage is replaced by something of the same\n"
        "  # size carrying less of its structure. See\n"
        "  # agent/finetune/train_ablation_inner_control_agent.py for what each mode keeps.\n"
        f"  inner_control_mode: {mode}\n"
    )
    return out[: m.end()] + knob + out[m.end() :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    bad = 0
    for src, dst, mode in pairs():
        sp, dp = os.path.join(REPO, src), os.path.join(REPO, dst)
        want = derive(open(sp).read(), mode)
        if args.check:
            have = open(dp).read() if os.path.exists(dp) else None
            ok = have == want
            print(f"{'ok  ' if ok else 'DRIFT'} {dst}")
            bad += not ok
        else:
            with open(dp, "w") as f:
                f.write(want)
            print(f"wrote {dst}")
    if args.check and bad:
        print(f"\n{bad} control config(s) no longer match their DIA source; rerun "
              f"without --check")
        sys.exit(1)


if __name__ == "__main__":
    main()
