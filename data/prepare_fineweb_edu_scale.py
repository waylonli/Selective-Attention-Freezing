#!/usr/bin/env python3
"""Build a provenance-rich FineWeb-Edu binary for the approximately 1B run."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from datasets import load_dataset
from huggingface_hub import HfApi
import numpy as np
import tiktoken


_ENCODER = None


def init_encoder():
    global _ENCODER
    _ENCODER = tiktoken.get_encoding("gpt2")


def encode_text(text):
    if _ENCODER is None:
        init_encoder()
    return np.asarray(
        _ENCODER.encode_ordinary(text) + [_ENCODER.eot_token], dtype=np.uint16)


def write_tokens(handle, digest, tokens, remaining):
    take = min(len(tokens), remaining)
    if take:
        payload = tokens[:take].tobytes()
        handle.write(payload)
        digest.update(payload)
    return take


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_and_mark_complete(out_dir, train_path, validation_path, metadata_path,
                             metadata, requested_train, requested_validation):
    expected = {
        train_path: (
            int(metadata["train_bytes"]), metadata["train_sha256"],
            requested_train * np.dtype(np.uint16).itemsize),
        validation_path: (
            int(metadata["validation_bytes"]), metadata["validation_sha256"],
            int(metadata["validation_tokens"]) * np.dtype(np.uint16).itemsize),
    }
    verified = {}
    for path, (metadata_bytes, metadata_sha, token_bytes) in expected.items():
        actual_bytes = path.stat().st_size
        if actual_bytes != metadata_bytes or actual_bytes != token_bytes:
            raise RuntimeError(
                f"size mismatch for {path}: actual={actual_bytes}, "
                f"metadata={metadata_bytes}, token_count={token_bytes}")
        actual_sha = sha256_file(path)
        if actual_sha != metadata_sha:
            raise RuntimeError(
                f"SHA256 mismatch for {path}: {actual_sha} != {metadata_sha}")
        verified[path.name] = {"bytes": actual_bytes, "sha256": actual_sha}
    completion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "requested_train_tokens": requested_train,
        "requested_validation_tokens": requested_validation,
        "metadata_sha256": sha256_file(metadata_path),
        "files": verified,
    }
    target = out_dir / "COMPLETE"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, target)
    return completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-id", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-100BT")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/fineweb_edu_20b"))
    parser.add_argument("--train-tokens", type=int, default=20_100_000_000)
    parser.add_argument("--validation-tokens", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--document-batch", type=int, default=4096)
    parser.add_argument("--min-characters", type=int, default=64)
    args = parser.parse_args()
    if args.train_tokens <= 0 or args.validation_tokens <= 0:
        raise ValueError("token targets must be positive")
    if args.workers <= 0 or args.document_batch <= 0:
        raise ValueError("workers and document-batch must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.bin"
    validation_path = args.out_dir / "val.bin"
    metadata_path = args.out_dir / "metadata.json"
    if train_path.exists() or validation_path.exists() or metadata_path.exists():
        if train_path.exists() and validation_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            requested_train = metadata.get(
                "requested_train_tokens", metadata.get("train_tokens"))
            requested_validation = metadata.get(
                "requested_validation_tokens", metadata.get("validation_tokens"))
            if (requested_train == args.train_tokens and
                    requested_validation == args.validation_tokens and
                    metadata["train_tokens"] == args.train_tokens and
                    metadata["validation_tokens"] >= args.validation_tokens):
                completion = verify_and_mark_complete(
                    args.out_dir, train_path, validation_path, metadata_path,
                    metadata, args.train_tokens, args.validation_tokens)
                print(json.dumps(completion, indent=2, sort_keys=True))
                return
        raise FileExistsError(
            f"refusing to overwrite existing or partial final files in {args.out_dir}")

    train_partial = args.out_dir / "train.bin.partial"
    validation_partial = args.out_dir / "val.bin.partial"
    train_digest = hashlib.sha256()
    validation_digest = hashlib.sha256()
    train_written = validation_written = documents = 0
    start = time.time()
    resolved_revision = HfApi().dataset_info(
        args.hf_id, revision=args.revision).sha
    dataset = load_dataset(
        args.hf_id, name=args.config, split="train", streaming=True,
        revision=resolved_revision)

    # Final-file names are never exposed until both targets are complete.
    with validation_partial.open("wb") as validation_handle, \
            train_partial.open("wb") as train_handle, \
            ProcessPoolExecutor(max_workers=args.workers,
                                initializer=init_encoder) as pool:
        texts = []
        for example in dataset:
            text = example.get(args.text_column)
            if not isinstance(text, str) or len(text) < args.min_characters:
                continue
            texts.append(text)
            if len(texts) < args.document_batch:
                continue
            for tokens in pool.map(encode_text, texts, chunksize=32):
                documents += 1
                if validation_written < args.validation_tokens:
                    # Keep validation and training document-disjoint. The
                    # validation count can exceed its requested minimum by at
                    # most one document; training begins at the next document.
                    validation_written += write_tokens(
                        validation_handle, validation_digest, tokens,
                        len(tokens))
                    continue
                if train_written < args.train_tokens:
                    train_written += write_tokens(
                        train_handle, train_digest, tokens,
                        args.train_tokens - train_written)
                if train_written >= args.train_tokens:
                    break
            texts.clear()
            if documents and documents % 20_000 == 0:
                elapsed = max(time.time() - start, 1e-9)
                print(json.dumps({
                    "documents": documents,
                    "train_tokens": train_written,
                    "validation_tokens": validation_written,
                    "million_tokens_per_second":
                        (train_written + validation_written) / elapsed / 1e6,
                }), flush=True)
            if train_written >= args.train_tokens:
                break
        if train_written < args.train_tokens:
            for tokens in pool.map(encode_text, texts, chunksize=32):
                documents += 1
                if validation_written < args.validation_tokens:
                    validation_written += write_tokens(
                        validation_handle, validation_digest, tokens,
                        len(tokens))
                    continue
                if train_written < args.train_tokens:
                    train_written += write_tokens(
                        train_handle, train_digest, tokens,
                        args.train_tokens - train_written)
                if train_written >= args.train_tokens:
                    break
        train_handle.flush()
        validation_handle.flush()
        os.fsync(train_handle.fileno())
        os.fsync(validation_handle.fileno())

    if (train_written != args.train_tokens or
            validation_written < args.validation_tokens):
        raise RuntimeError(
            f"dataset exhausted: train={train_written}, val={validation_written}")
    os.replace(train_partial, train_path)
    os.replace(validation_partial, validation_path)
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hf_id": args.hf_id,
        "config": args.config,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "split": "train",
        "streaming": True,
        "text_column": args.text_column,
        "minimum_characters": args.min_characters,
        "tokenizer": "tiktoken:gpt2",
        "document_terminator": "<|endoftext|>",
        "partition": (
            "Complete documents from the start of the deterministic stream are "
            "assigned to val.bin until validation_tokens is met or exceeded. "
            "Training starts at the next document and is truncated at exactly "
            "train_tokens. The two files are document-disjoint contiguous "
            "portions of the revision-pinned stream."),
        "documents_consumed": documents,
        "requested_validation_tokens": args.validation_tokens,
        "requested_train_tokens": args.train_tokens,
        "validation_tokens": validation_written,
        "train_tokens": train_written,
        "validation_sha256": validation_digest.hexdigest(),
        "train_sha256": train_digest.hexdigest(),
        "validation_bytes": validation_path.stat().st_size,
        "train_bytes": train_path.stat().st_size,
        "elapsed_seconds": time.time() - start,
        "workers": args.workers,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    verify_and_mark_complete(
        args.out_dir, train_path, validation_path, metadata_path, metadata,
        args.train_tokens, args.validation_tokens)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
