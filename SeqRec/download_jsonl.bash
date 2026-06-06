# mkdir -p ../data_CDs/

# using wget to download
wget -c "https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Movies_and_TV.jsonl" \
     -P data_Movies_and_TV/ \
     --no-check-certificate \
     --tries=10