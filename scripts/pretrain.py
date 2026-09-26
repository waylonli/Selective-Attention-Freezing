"""Run an archived scientific configuration without a cluster scheduler."""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(arm, out, world_size=None):
    overrides = dict(arm["overrides"], out_dir=str(out))
    ranks = arm.get("world_size", 1) if world_size is None else world_size
    if ranks < 1 or overrides["gradient_accumulation_steps"] % ranks:
        raise ValueError("The global accumulation must be divisible by the GPU count")
    argv = [sys.executable]
    if ranks > 1:
        argv += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ranks}"]
    argv += ["nanogpt/train.py", arm["config"]]
    return argv + [f"--{key}={value!r}" for key, value in overrides.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    arms = json.loads(args.manifest.read_text())["arms"]
    if args.list:
        for arm in arms:
            print(arm["run_id"], "parent:", arm.get("resume_from") or "none")
        return
    matches = [arm for arm in arms if arm["run_id"] == args.run_id]
    if len(matches) != 1:
        parser.error("Choose exactly one --run-id, using --list first")
    arm = matches[0]
    out = (args.out or ROOT / arm["result_dir"]).resolve()
    parent = args.parent or (ROOT / arm["resume_from"] if arm.get("resume_from") else None)
    if args.parent and not arm.get("resume_from"):
        parser.error("This is a from-scratch arm; it does not take --parent")
    argv = command(arm, out, args.world_size)
    print(shlex.join(argv), flush=True)
    if args.dry_run:
        return
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    if parent is not None and not parent.is_file():
        raise FileNotFoundError(f"A training checkpoint, including optimiser state, is required: {parent}")
    out.mkdir(parents=True)
    if parent is not None:
        shutil.copyfile(parent, out / "ckpt.pt")
    (out / "launch.json").write_text(json.dumps({"argv": argv, "arm": arm}, indent=2) + "\n")
    subprocess.run(argv, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
