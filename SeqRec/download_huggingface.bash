#!/bin/bash

# Download datasets from huggingface，please check the profile before running this script

ENDPOINT="https://hf-mirror.com"
REPO="McAuley-Lab/Amazon-Reviews-2023"

rec_type='Movies_and_TV'
BASE_SAVE_DIR="data_${rec_type}"

META_CATEGORY="raw_meta_${rec_type}"
META_TOTAL_SHARDS="0001"
META_MAX_INDEX=1

BENCHMARK_PATH="benchmark/5core/last_out"
CSV_FILES=("${rec_type}.test.csv" "${rec_type}.train.csv" "${rec_type}.valid.csv")

mkdir -p "$BASE_SAVE_DIR"

echo "Start downloading metadata (Parquet shards)..."
for i in $(seq -f "%05g" 0 $META_MAX_INDEX); do
    FILENAME="full-$i-of-$META_TOTAL_SHARDS.parquet"
    URL="${ENDPOINT}/datasets/${REPO}/resolve/main/${META_CATEGORY}/${FILENAME}"
    
    echo "Downloading metadata shard: $FILENAME"
    wget -c "$URL" -P "$BASE_SAVE_DIR" --no-check-certificate --tries=5
done

echo "--------------------------------------"
echo "Start downloading Benchmark data (CSV)..."
for CSV in "${CSV_FILES[@]}"; do
    URL="${ENDPOINT}/datasets/${REPO}/resolve/main/${BENCHMARK_PATH}/${CSV}"
    
    echo "Downloading Benchmark file: $CSV"
    wget -c "$URL" -P "$BASE_SAVE_DIR" --no-check-certificate --tries=5
done

echo "--------------------------------------"
echo "✅ All tasks completed! Data is stored in: $BASE_SAVE_DIR"
ls -lh "$BASE_SAVE_DIR"