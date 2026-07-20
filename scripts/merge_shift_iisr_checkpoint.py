#!/usr/bin/env python
# Copyright (c) 2026 Yunpeng Hua
# Licensed under the NTU S-Lab License 1.0.

import argparse
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.shift_iisr_checkpoint import build_shift_iisr_checkpoint


def load_component(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        checkpoint_step = checkpoint.get("iters_start")
    else:
        state_dict = checkpoint
        checkpoint_step = None
    return state_dict, checkpoint_step


def get_parser():
    parser = argparse.ArgumentParser(
        description="Merge released Shift-IISR GRM component checkpoints."
    )
    parser.add_argument(
        "--grm_feature_extractor",
        default="weights/grm_feature_extractor_18000.pth",
        help="Legacy GRM feature-extractor checkpoint.",
    )
    parser.add_argument(
        "--grm_projector",
        default="weights/private_proj_18000.pth",
        help="Legacy GRM projector checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="weights/shift_iisr.pth",
        help="Output Shift-IISR checkpoint.",
    )
    parser.add_argument("--checkpoint_step", type=int, default=18000)
    parser.add_argument("--lsr_strength", type=float, default=1.0)
    return parser


def main():
    args = get_parser().parse_args()
    extractor_state, extractor_step = load_component(args.grm_feature_extractor)
    projector_state, projector_step = load_component(args.grm_projector)

    component_steps = {
        step for step in (extractor_step, projector_step) if step is not None
    }
    if component_steps and component_steps != {args.checkpoint_step}:
        raise ValueError(
            f"Component checkpoint steps {sorted(component_steps)} do not match "
            f"--checkpoint_step={args.checkpoint_step}."
        )

    checkpoint = build_shift_iisr_checkpoint(
        extractor_state,
        projector_state,
        checkpoint_step=args.checkpoint_step,
        lsr_strength=args.lsr_strength,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Saved {output_path}")
    print(f"GRM feature extractor tensors: {len(extractor_state)}")
    print(f"GRM projector tensors: {len(projector_state)}")


if __name__ == "__main__":
    main()
