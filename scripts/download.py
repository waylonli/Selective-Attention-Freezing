"""Download released evaluation assets and verify their SHA256 checksums."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download, snapshot_download

MODEL_REPO = "waylonli/Selective-Attention-Freezing"
DATA_REPO = "waylonli/Selective-Attention-Freezing-data"
DATA_REVISION = "86db20743dc7ace0313fa293f80799409bac9bc7"


def check_file(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Invalid release path")
    return path


def download_data(out):
    folder = Path(snapshot_download(DATA_REPO, repo_type="dataset",
                                    revision=DATA_REVISION, local_dir=out))
    checksums = json.loads((folder / "SHA256SUMS.json").read_text())
    for name, digest in checksums.items():
        check_file(folder / safe_name(name), digest)
    return folder


def download_checkpoint(identifier, out):
    manifest = hf_hub_download(MODEL_REPO, "checkpoints.json")
    matches = [r for r in json.loads(Path(manifest).read_text())["models"]
               if r["id"] == identifier]
    if len(matches) != 1:
        raise ValueError(f"Unknown checkpoint: {identifier}")
    record = matches[0]
    if record.get("status") != "uploaded":
        raise ValueError(f"Checkpoint is not yet published: {identifier}")
    name = safe_name(record["upload_path"])
    folder = Path(snapshot_download(MODEL_REPO, revision=record["hf_revision"],
                                    allow_patterns=[str(name.parent / "*")], local_dir=out))
    model = folder / name
    check_file(model, record["export_sha256"])
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--data", action="store_true")
    mode.add_argument("--checkpoint", metavar="ID")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    path = download_data(args.out) if args.data else download_checkpoint(args.checkpoint, args.out)
    print(path)


if __name__ == "__main__":
    main()
