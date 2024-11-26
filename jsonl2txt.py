import json
import os
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

def convert_single_jsonl_file(jsonl_file, counter, output_dir):
    output_file = output_dir / f'input{counter}.txt'
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as infile:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    data = json.loads(line.strip())
                    # Extract only the text field and write it directly
                    if 'text' in data:
                        text = data['text'].strip()
                        # Add a newline between samples for better separation
                        outfile.write(text + '\n\n')
        return f'Converted {jsonl_file.name} to {output_file.name}'
    except Exception as e:
        return f'Error processing {jsonl_file.name}: {str(e)}'

def convert_jsonl_files_parallel():
    dataset_dir = Path('dataset')
    output_dir = Path('text_data')
    output_dir.mkdir(exist_ok=True)
    
    jsonl_files = list(dataset_dir.glob('*.jsonl'))
    
    num_cores = mp.cpu_count()
    with ThreadPool(processes=num_cores) as thread_pool:
        with mp.Pool(processes=num_cores) as process_pool:
            with tqdm(total=len(jsonl_files), desc='Converting files', unit='file') as pbar:
                futures = []
                for counter, jsonl_file in enumerate(jsonl_files, start=1):
                    future = thread_pool.apply_async(
                        process_pool.apply_async, 
                        (convert_single_jsonl_file, (jsonl_file, counter, output_dir))
                    )
                    futures.append(future)
                
                for future in futures:
                    result = future.get().get()
                    print(result)
                    pbar.update(1)

if __name__ == '__main__':
    convert_jsonl_files_parallel()