# saves a subset of HuggingFaceFW/fineweb-edu to binary files for training.
# FineWeb-Edu is a filtered subset of FineWeb containing high-quality educational web content.
# We use the "sample-10BT" subset (~10B tokens) to keep things manageable.
#
# Usage:
#   python data/fineweb_edu/prepare.py
#
# Requirements:
#   pip install datasets tiktoken tqdm

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

num_proc = 8

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    out_dir = os.path.dirname(__file__)
    train_path = os.path.join(out_dir, 'train.bin')
    val_path = os.path.join(out_dir, 'val.bin')

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"  {train_path} and {val_path} already exist, skipping.")
        raise SystemExit(0)

    # sample-10BT is ~10B GPT-2 tokens, a manageable subset of the full 1.3T token dataset.
    # Other available subsets: "sample-100BT", "sample-350BT", or the full "default".
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        num_proc=num_proc,
    )

    # create a val split (0.05%, same as openwebtext)
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized = split_dataset.map(
        process,
        remove_columns=split_dataset['train'].column_names,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
