#!/bin/bash

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir data
fi

# Change to data directory
cd data

echo "Downloading files into data directory..."

for i in $(seq -f "%02g" 0 20)
do
    echo "Downloading subset ${i}..."
    curl -L -O "https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset${i}.tar"
    echo "Completed downloading subset ${i}"
done

echo "All downloads completed in data directory"