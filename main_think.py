import sys, os, subprocess
VENV = r"C:\Users\Blinvo\Desktop\发现\模型训练\project\venv\Scripts\python.exe"
if "venv" not in sys.executable.lower() and os.path.exists(VENV):
    subprocess.run([VENV, __file__] + sys.argv[1:])
    sys.exit(0)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Acheron Thinking Demo — Biomimetic deliberation for Gridman.

Usage:
    python main_think.py                  # Interactive chat (focused mode)
    python main_think.py --mode divergent # Divergent thinking
    python main_think.py --mode reflective# Reflective thinking
    python main_think.py --query "明月几时有"  # Single query

The model enters a genuine deliberative phase where internal state evolves
through self-referential attractor dynamics — no Chain-of-Thought prompting.
"""

import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

import torch

from naxi.v_0d1.gridman.config import RUNNING_CONFIG, Config
from naxi.v_0d1.gridman.core import Gridman
from naxi.v_0d1.gridman.tools import load_checkpoint, print_model_parameters
from naxi.v_0d1.gridman.thinking import AcheronThinking, AcheronConfig, acheron_chat


def build_parser():
    p = argparse.ArgumentParser(
        description="Acheron Thinking — Biomimetic deliberation for Ouro/Gridman"
    )
    p.add_argument("--model", type=str, default="medium",
                   choices=["mini", "small", "medium", "large"],
                   help="Model size (default: medium)")
    p.add_argument("--mode", type=str, default="focused",
                   choices=["focused", "divergent", "reflective"],
                   help="Thinking mode (default: focused)")
    p.add_argument("--sft", action="store_true", default=True,
                   help="Load SFT checkpoint (default)")
    p.add_argument("--pretrain", dest="sft", action="store_false",
                   help="Load pretrain checkpoint")
    p.add_argument("--max-steps", type=int, default=50,
                   help="Max thinking steps (default: 50)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Convergence threshold override")
    p.add_argument("--temp", type=float, default=0.7,
                   help="Generation temperature (default: 0.7)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress trajectory output")
    p.add_argument("--query", type=str, default=None,
                   help="Single query (non-interactive mode)")
    return p


MODEL_MAP = {
    "mini":   ("gridman_mini_v_0d1",),
    "small":  ("gridman_small_v_0d1",),
    "medium": ("gridman_medium_v_0d1",),
    "large":  ("gridman_large_v_0d1",),
}


def load_model(args) -> tuple[Gridman, Config]:
    """Load Gridman model with specified config."""
    from naxi.v_0d1.gridman.config import (
        GRIDMAN_MINI, GRIDMAN_SMALL, GRIDMAN_MEDIUM, GRIDMAN_LARGE
    )
    config_map = {
        "mini": GRIDMAN_MINI, "small": GRIDMAN_SMALL,
        "medium": GRIDMAN_MEDIUM, "large": GRIDMAN_LARGE,
    }
    config = config_map[args.model]
    device = config.device

    print(f"\n  Loading Gridman-{args.model.capitalize()}...")
    model = Gridman(config).to(device)
    print_model_parameters(model)

    load_checkpoint(model, is_sft=args.sft, config=config)
    model.eval()
    return model, config


def main():
    args = build_parser().parse_args()

    model, config = load_model(args)

    ac = AcheronConfig(
        max_steps=args.max_steps,
        generation_temperature=args.temp,
        verbose=not args.quiet,
    )
    if args.threshold is not None:
        ac.energy_threshold = args.threshold

    thinker = AcheronThinking(model, config, ac)

    if args.query:
        # Single-shot mode
        result = thinker.think(args.query, mode=args.mode)
        if args.quiet:
            print(result.output_text)
        else:
            thinker.show_trajectory(result)
    else:
        # Interactive mode
        acheron_chat(model, config, ac)


if __name__ == "__main__":
    main()
