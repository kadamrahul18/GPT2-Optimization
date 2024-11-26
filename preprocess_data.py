import os
import tarfile
import lzma
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import argparse

def extract_single_tar(tar_path, extract_dir):
    """
    Extracts a single .tar file into a unique subdirectory within extract_dir with a security filter.
    """
    # Derive a unique subdirectory name from the tar file name
    tar_filename = os.path.basename(tar_path)
    tar_name_without_ext = os.path.splitext(tar_filename)[0]
    target_dir = os.path.join(extract_dir, tar_name_without_ext)
    os.makedirs(target_dir, exist_ok=True)

    def safe_extract_filter(member, path):
        """
        Filter function to ensure safe extraction of tar files.
        Prevents files from being extracted outside the target directory.
        """
        # Prevent absolute paths and directory traversal
        if os.path.isabs(member.name) or '..' in member.name.split(os.sep):
            return None
        # You can add additional security checks here if needed
        return member

    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=target_dir, filter=safe_extract_filter)
        return True, tar_path
    except Exception as e:
        return False, tar_path, str(e)

def extract_tar_files(tar_dir, extract_dir):
    """
    Extracts all .tar files from tar_dir into unique subdirectories within extract_dir using multiprocessing.
    """
    os.makedirs(extract_dir, exist_ok=True)
    tar_files = [os.path.join(tar_dir, f) for f in os.listdir(tar_dir) if f.endswith('.tar')]

    print(f"Found {len(tar_files)} .tar files to extract.")

    if not tar_files:
        print("No .tar files found. Exiting extraction phase.")
        return

    # Use multiprocessing Pool to extract tar files in parallel
    with Pool(processes=cpu_count()) as pool:
        func = partial(extract_single_tar, extract_dir=extract_dir)
        results = list(tqdm(pool.imap(func, tar_files), total=len(tar_files), desc="Extracting .tar files"))

    # Handle extraction results
    failed = [res for res in results if not res[0]]
    if failed:
        print(f"Failed to extract {len(failed)} .tar files:")
        for fail in failed:
            print(f" - {fail[1]}: {fail[2]}")
    else:
        print("All .tar files extracted successfully.")

def decompress_single_xz(xz_path, text_dir):
    """
    Decompresses a single .xz file and saves the text file to text_dir with a .txt extension.
    """
    try:
        filename = os.path.basename(xz_path)
        text_filename = filename.replace('.xz', '.txt')  # Append .txt extension
        text_path = os.path.join(text_dir, text_filename)

        with lzma.open(xz_path, 'rt') as f_in, open(text_path, 'w', encoding='utf-8') as f_out:
            shutil.copyfileobj(f_in, f_out)

        return True, xz_path
    except Exception as e:
        return False, xz_path, str(e)

def decompress_xz_files(extract_dir, text_dir):
    """
    Decompresses all .xz files in extract_dir and saves the text files to text_dir using multiprocessing.
    """
    os.makedirs(text_dir, exist_ok=True)
    xz_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.xz'):
                xz_files.append(os.path.join(root, file))

    print(f"Found {len(xz_files)} .xz files to decompress.")

    if not xz_files:
        print("No .xz files found. Exiting decompression phase.")
        return

    # Use multiprocessing Pool to decompress xz files in parallel
    with Pool(processes=cpu_count()) as pool:
        func = partial(decompress_single_xz, text_dir=text_dir)
        results = list(tqdm(pool.imap(func, xz_files), total=len(xz_files), desc="Decompressing .xz files"))

    # Handle decompression results
    failed = [res for res in results if not res[0]]
    if failed:
        print(f"Failed to decompress {len(failed)} .xz files:")
        for fail in failed:
            print(f" - {fail[1]}: {fail[2]}")
    else:
        print("All .xz files decompressed successfully.")

def main():
    parser = argparse.ArgumentParser(description='Preprocess data by extracting and decompressing .tar and .xz files.')
    parser.add_argument('--tar_dir', type=str, required=True, help='Directory containing .tar files.')
    parser.add_argument('--extract_dir', type=str, default='extracted/', help='Directory to extract .tar files.')
    parser.add_argument('--text_dir', type=str, default='text_data/', help='Directory to store decompressed text files.')
    args = parser.parse_args()

    extract_tar_files(args.tar_dir, args.extract_dir)
    decompress_xz_files(args.extract_dir, args.text_dir)

    print("Preprocessing completed successfully.")

if __name__ == '__main__':
    main()
