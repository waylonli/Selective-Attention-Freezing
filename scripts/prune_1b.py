"""Pack and continue the single-seed, four-GPU 1B gate-Taylor comparison."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.export_checkpoint import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if sha256(args.parent) != args.expected_sha256:
        raise ValueError("Parent checkpoint identity mismatch")
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/pruning_1b.json").read_text())
    for name, expected in config["required_data"].items():
        path = args.data / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Data identity mismatch: {name}")
    args.out.mkdir(parents=True)
    manifest = dict(config, source_hashes={}, data_dir=str(args.data.resolve()),
                    parent={"path": str(args.parent.resolve()), "sha256": args.expected_sha256})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from supplement.scaleup_pruning import prepare
    training = args.out.resolve() / "training"
    prepare(args.out.resolve(), training)
    settings = dict(config["training"], dataset=str(args.data.resolve()), out_dir=str(training))
    argv = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
            "nanogpt/train.py", *[f"--{k}={v!r}" for k, v in settings.items()]]
    (args.out / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
    if not args.prepare_only:
        subprocess.run(argv, cwd=root, check=True)


if __name__ == "__main__":
    main()
