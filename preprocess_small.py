import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

num_proc = 8

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    data_files = {
        'train': 'dataset/webtext.train.jsonl',
        'val': 'dataset/webtext.valid.jsonl'
    }
    dataset = load_dataset('json', data_files=data_files, num_proc=num_proc)

    def process(example):
        ids = enc.encode_ordinary(example['text']) 
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # --- THIS IS THE KEY MODIFICATION ---
    # We are creating a small subset for rapid testing.
    tokenized['train'] = tokenized['train'].select(range(50000))
    tokenized['val'] = tokenized['val'].select(range(5000))
    print(f"Using small dataset: {len(tokenized['train'])} training examples, {len(tokenized['val'])} validation examples.")
    # ------------------------------------

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        # We add "_small" to the output filename to avoid overwriting the full dataset
        filename = os.path.join(os.path.dirname(__file__), f'{split}_small.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Shard the new, smaller dataset
            if len(dset) < total_batches:
                # If the dataset is smaller than the batches, adjust logic
                if batch_idx < len(dset):
                     batch = dset.shard(num_shards=len(dset), index=batch_idx, contiguous=True).with_format('numpy')
                else:
                     continue # Skip empty batches
            else:
                 batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')

            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
