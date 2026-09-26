"""
Streaming corpus pipeline for the prior-corpus experiments (lmfuser_data).

Nothing is downloaded up front: each corpus is a MANIFEST — a text file with
one shard URL per line (built from the HF Hub file listing) — and
`lmfuser_data.DataLoader` streams shards on demand inside a worker process:

    shard URL --(scanner)--> rows --(map_fn: tokenize)--> ids
              --(flow_fn: pack)--> (seq_len + 1)-token windows --> (B, T+1) batches

Deterministic and disjoint: with shuffle=False / one worker the row order is
the manifest order, and consuming ONE iterator for the extraction span and
then the evaluation span yields provably non-overlapping token spans.

Corpora (the NL / math / code trio of the project):
    fineweb_edu  HuggingFaceFW/fineweb-edu  sample/10BT/*.parquet   (2.1 GB shards)
    openwebmath  open-web-math/open-web-math data/*.parquet          (236 MB shards)
    the_stack    bigcode/the-stack-smol     data/python/data.json    (gated; HF token)

Usage:
    from prior_corpus_data import corpus_batches, write_manifest
    it = corpus_batches("openwebmath", "Qwen/Qwen3-4B", seq_len=2048, batch_size=2)
    extract = list(itertools.islice(it, 32))   # first 32 batches
    evaluate = list(itertools.islice(it, 64))  # next 64, disjoint
"""
from __future__ import annotations

import functools
import io
import json
import os
from typing import Iterator

import numpy as np
import requests
import torch
from lmfuser_data import DataLoader
from lmfuser_data.scanners.parquet import ParquetScanner, _HTTP_TIMEOUT
from lmfuser_data.scanners.interface import Scanner
from lmfuser_data.utils import retry

CORPORA = {
    # name: (hf dataset id, path prefix inside the repo, text column)
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample/10BT/", "text"),
    "openwebmath": ("open-web-math/open-web-math", "data/", "text"),
    "the_stack": ("bigcode/the-stack-smol", "data/python/", "content"),
}
DEFAULT_MANIFEST_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "manifests")


def _auth_headers():
    """Bearer token for gated repos (the-stack); harmless on public ones."""
    try:
        from huggingface_hub import get_token
        tok = get_token()
    except Exception:
        tok = None
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _fetch(path):
    if path.startswith("http"):
        resp = requests.get(path, headers=_auth_headers(), timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.content
    with open(path, "rb") as f:
        return f.read()


class HFParquetScanner(ParquetScanner):
    """ParquetScanner that sends the HF token (lmfuser_data's own sends none)."""

    @retry(tries=10, delay=0.1, backoff=2)
    def _load_data(self):
        if self._rows is not None:
            return
        import pandas as pd
        self._rows = pd.read_parquet(io.BytesIO(_fetch(self.path))).to_dict(orient="records")


class HFJsonScanner(Scanner):
    """Whole-file JSON / JSONL shard (the-stack-smol ships data/python/data.json)."""

    def __init__(self, path, **kw):
        super().__init__(path, **kw)
        self._rows = None

    @retry(tries=10, delay=0.1, backoff=2)
    def _load_data(self):
        if self._rows is not None:
            return
        raw = _fetch(self.path).decode("utf-8")
        try:
            obj = json.loads(raw)
            rows = obj if isinstance(obj, list) else obj.get("data", [obj])
        except json.JSONDecodeError:                 # JSONL
            rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
        self._rows = rows

    def __len__(self):
        self._load_data()
        return len(self._rows)

    def __getitem__(self, i):
        self._load_data()
        return self._rows[i]

    @classmethod
    def check_file(cls, path):
        return path.lower().endswith((".json", ".jsonl"))


def shard_urls(name):
    """List the corpus' shard URLs from the HF Hub (no download)."""
    from huggingface_hub import list_repo_files
    repo, prefix, _ = CORPORA[name]
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset")
                   if f.startswith(prefix) and f.endswith((".parquet", ".json", ".jsonl")))
    return [f"https://huggingface.co/datasets/{repo}/resolve/main/{f}" for f in files]


def write_manifest(name, manifest_dir=DEFAULT_MANIFEST_DIR, max_shards=None):
    """Write data/manifests/<name>.txt (one shard URL per line); returns path."""
    os.makedirs(manifest_dir, exist_ok=True)
    path = os.path.join(manifest_dir, f"{name}.txt")
    urls = shard_urls(name)
    if max_shards:
        urls = urls[:max_shards]
    with open(path, "w") as f:
        f.write("\n".join(urls) + "\n")
    return path


# ---- map / flow functions (module level: they run inside worker processes) ----

_TOKENIZERS = {}


def _tokenizer(name):
    if name not in _TOKENIZERS:
        from transformers import AutoTokenizer
        _TOKENIZERS[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZERS[name]


def tokenize_row(row, tokenizer_name, text_col, max_chars):
    text = str(row.get(text_col, ""))[:max_chars]
    tok = _tokenizer(tokenizer_name)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id
    if eos is not None:
        ids = ids + [eos]
    return {"ids": np.asarray(ids, dtype=np.int64)}


def pack_windows(rows, seq_len):
    """Concatenate documents (eos-separated) and cut (seq_len + 1)-token windows
    with stride seq_len: window[:-1] is the input, window[1:] the labels."""
    buf = np.empty(0, dtype=np.int64)
    for row in rows:
        buf = np.concatenate([buf, row["ids"]]) if buf.size else row["ids"]
        while buf.size >= seq_len + 1:
            yield {"tokens": buf[: seq_len + 1].copy()}
            buf = buf[seq_len:]


def corpus_batches(name, tokenizer_name, seq_len, batch_size, *,
                   manifest_dir=DEFAULT_MANIFEST_DIR, max_chars=100_000,
                   seed=0, qps=None) -> Iterator[torch.Tensor]:
    """Yield (batch_size, seq_len + 1) int64 token tensors, streamed from the
    corpus' shards in manifest order (deterministic; call once and slice the
    iterator for disjoint spans)."""
    manifest = os.path.join(manifest_dir, f"{name}.txt")
    if not os.path.exists(manifest):
        write_manifest(name, manifest_dir)
    _, _, text_col = CORPORA[name]
    with open(manifest) as f:
        first = f.readline().strip()
    scanner = HFJsonScanner if first.endswith((".json", ".jsonl")) else HFParquetScanner
    loader = DataLoader(
        batch_size=batch_size, path_list=[manifest], scanner_type=scanner,
        seed=seed, shuffle=False, pre_fetch_factor=4, infinite=False,
        map_fn=functools.partial(tokenize_row, tokenizer_name=tokenizer_name,
                                 text_col=text_col, max_chars=max_chars),
        flow_fn=functools.partial(pack_windows, seq_len=seq_len),
        ignore_error=True, qps=qps, num_workers=1, num_ranks=1, rank_idx=0,
    )
    for batch in loader:
        t = batch["tokens"]
        yield t if torch.is_tensor(t) else torch.as_tensor(np.stack(t))


if __name__ == "__main__":
    import argparse, itertools, time
    ap = argparse.ArgumentParser(description="smoke test: stream a few batches")
    ap.add_argument("--corpus", default="openwebmath")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    it = corpus_batches(args.corpus, args.tokenizer, args.seq_len, args.batch_size)
    a = list(itertools.islice(it, args.n))
    b = list(itertools.islice(it, args.n))
    print(f"{args.corpus}: {len(a)}+{len(b)} batches of {tuple(a[0].shape)} {a[0].dtype} "
          f"in {time.time()-t0:.0f}s; spans disjoint: {not torch.equal(a[0], b[0])}")
    tok = _tokenizer(args.tokenizer)
    print("  first window decodes to:", repr(tok.decode(a[0][0, :40].tolist()))[:160])
