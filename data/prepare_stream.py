"""
Stream a HuggingFace text dataset, GPT-2-tokenize it, and write capped
train.bin / val.bin for nanoGPT — without downloading the whole dataset.

Used to build a larger CODE corpus (codeparrot/codeparrot-clean, Python, non-
gated, parquet) for the 124M experiments, where the-stack-smol Python (~37M
tokens) is too small to avoid heavy re-epoching.

Single-process and streamed: low source-disk, RAM bounded by incremental writes.
At ~1M tokens/s this handles a ~1.5B-token corpus in well under an hour.

Matches the existing data/*/prepare.py tokenisation exactly:
  enc = tiktoken gpt2 ; ids = encode_ordinary(text) + [eot] ; dtype uint16.

Usage (on a node with internet, e.g. the login node):
  python data/prepare_stream.py \
      --hf_id codeparrot/codeparrot-clean --text_col content \
      --out_name codeparrot_py --max_tokens 1_500_000_000
"""

import argparse
import os
import pathlib
import time

import numpy as np
import tiktoken
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--text_col", default="content")
    ap.add_argument("--out_name", required=True, help="dir under data/ to write to")
    ap.add_argument("--max_tokens", type=int, default=1_500_000_000)
    ap.add_argument("--val_tokens", type=int, default=3_000_000)
    ap.add_argument("--min_chars", type=int, default=64)
    args = ap.parse_args()

    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(k) == "1":
            print(f"WARNING: {k}=1 set; unsetting so streaming can reach HF.", flush=True)
            os.environ[k] = "0"

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    out_dir = repo_root / "data" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    if train_path.exists() and val_path.exists():
        print(f"{train_path} and {val_path} already exist — skipping.", flush=True)
        return

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    kwargs = dict(streaming=True, split="train")
    if args.config:
        kwargs["name"] = args.config
    print(f"Streaming {args.hf_id} ({kwargs}) ...", flush=True)
    ds = load_dataset(args.hf_id, **kwargs)

    train_f = open(train_path, "wb")
    val_buf, val_filled, total, n_docs = [], 0, 0, 0
    t0 = time.time()
    for ex in ds:
        txt = ex.get(args.text_col)
        if not txt or not isinstance(txt, str) or len(txt) < args.min_chars:
            continue
        ids = np.array(enc.encode_ordinary(txt) + [eot], dtype=np.uint16)
        n = len(ids)
        n_docs += 1
        if val_filled < args.val_tokens:
            take = min(args.val_tokens - val_filled, n)
            val_buf.append(ids[:take])
            val_filled += take
            if take < n:
                train_f.write(ids[take:].tobytes())
                total += n - take
        else:
            train_f.write(ids.tobytes())
            total += n
        if n_docs % 20000 == 0:
            rate = total / max(time.time() - t0, 1e-9) / 1e6
            print(f"  {n_docs} docs, {total/1e9:.2f}B train tokens "
                  f"({rate:.2f}M tok/s) ...", flush=True)
            train_f.flush()
        if total + val_filled >= args.max_tokens:
            break
    train_f.close()
    np.concatenate(val_buf).tofile(val_path)
    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f} min: {n_docs} docs", flush=True)
    print(f"  train: {total/1e9:.2f}B tokens -> {train_path}", flush=True)
    print(f"  val:   {val_filled/1e6:.2f}M tokens -> {val_path}", flush=True)


if __name__ == "__main__":
    main()
