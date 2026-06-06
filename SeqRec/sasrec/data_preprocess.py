import os
import os.path
import gzip
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import random
import pickle
from datasets import load_dataset

    
def preprocess_raw_5core(fname):
    
    random.seed(0)
    np.random.seed(0)
    
    local_path = f"../data_{fname}/"

    print("Loading local Meta Data...")
    meta_dict = {}
    if 'CD' not in fname and 'Movies' not in fname:
        # --- Parquet path ---
        meta_files = os.path.join(local_path, "full-*-of-*.parquet")
        meta_dataset = load_dataset("parquet", data_files={"full": meta_files})
        
        print("Load Meta Data")
        for l in tqdm(meta_dataset['full'], desc="Processing Parquet meta"):
            meta_dict[l['parent_asin']] = [l.get('title', ''), l.get('description', '')]
        del meta_dataset

    else:
        # --- JSONL path：directly create meta_dict instead of meta_dataset ---
        meta_file = os.path.join(local_path, f"meta_{fname}.jsonl")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Meta file not found: {meta_file}")
        
        with open(meta_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing JSONL meta"):
                if line.strip():
                    try:
                        item = json.loads(line)
                        parent_asin = item.get('parent_asin')
                        if parent_asin is not None:
                            title = str(item.get('title', ''))
                            description = str(item.get('description', ''))
                            meta_dict[parent_asin] = [title, description]
                    except Exception:
                        continue  # skip malformed lines 
    
    data_files = {
        "train": os.path.join(local_path, f"{fname}.train.csv"),
        "valid": os.path.join(local_path, f"{fname}.valid.csv"),
        "test": os.path.join(local_path, f"{fname}.test.csv")
    }
    print("Loading local 5core CSV split files...")
    dataset = load_dataset("csv", data_files=data_files)

    
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = defaultdict(list)
    User_s = {'train': defaultdict(list), 'valid': defaultdict(list), 'test': defaultdict(list)}
    id2asin = dict()
    time_dict = defaultdict(dict)
    for t in ['train', 'valid', 'test']:
        d = dataset[t]

        for l in tqdm(d):
            
            user_id = l['user_id']
            asin = l['parent_asin']
            
            
            if user_id in usermap:
                userid = usermap[user_id]
            else:
                usernum += 1
                userid = usernum
                usermap[user_id] = userid
            
            if asin in itemmap:
                itemid = itemmap[asin]
            else:
                itemnum += 1
                itemid = itemnum
                itemmap[asin] = itemid
                
            User[userid].append(itemid)
            User_s[t][userid].append(itemid)
            id2asin[itemid] = asin
            time_dict[itemid][userid] = l['timestamp']
            
    
    sample_size = int(len(User.keys()))
    print('num users raw', sample_size)
    sample_rate = {
        'Movies_and_TV': 0.05,
        'Electronics': 0.05,
        'Industrial_and_Scientific': 1.0,
        'CDs_and_Vinyl':0.33,
        'Books': 0.075,
        'Sport': 0.1,
    }
    
    sample_ratio = sample_rate[fname]
    use_key = random.sample(list(User.keys()), int(sample_size*sample_ratio))#Movies 0.05 Industrial 은 바로 사용 Electronics 0.05 Books 0.075 Sport 0.1 Beauty 0.075 Software 0.33
    # use_key = list(User.keys())#Movies 0.05 Grocery_and_Gourmet_Food 0.1

    print('num sample user', len(use_key))
    
    CountU = defaultdict(int)
    CountI = defaultdict(int)
    
    usermap_final = dict()
    itemmap_final = dict()
    usernum_final = 0
    itemnum_final = 0
    use_key_dict = defaultdict(int)
    use_train_dict = defaultdict(int)
    for key in use_key:
        use_key_dict[key] = 1

        for t in ['train', 'valid', 'test']:
            for i_ in User_s[t][key]:
                CountI[i_] +=1
                CountU[key] +=1
        
    text_dict = {'time':defaultdict(dict), 'description':{}, 'title':{}}
    for t in ['train', 'valid', 'test']:
        d = dataset[t]
        use_id = defaultdict(int)
        f = open(f'./../data_{fname}/{fname}_{t}.txt', 'w')
        for l in tqdm(d):
            
            user_id = l['user_id']
            asin = l['parent_asin']
            user_id_ = usermap[user_id]
            if use_id[user_id_] == 0:
                use_id[user_id_] = 1
                pass
            else:
                continue
            # if user_id_ in use_key:
            if use_key_dict[user_id_] == 1 and CountU[user_id_] >4:
                
                use_items = []
                for it in User_s[t][user_id_]:
                    if CountI[it] >4:
                        use_items.append(it)
                if t == 'train':
                    if len(use_items) > 4:
                        use_train_dict[user_id_] = 1
                        if user_id_ in usermap_final:
                            userid = usermap_final[user_id_]
                        else:
                            usernum_final +=1
                            userid = usernum_final
                            usermap_final[user_id_] = userid
                        for it in use_items:
                            if it in itemmap_final:
                                itemid = itemmap_final[it]
                            else:
                                itemnum_final +=1
                                itemid = itemnum_final
                                itemmap_final[it] = itemid
                            
                            d = meta_dict[id2asin[it]][1]
                            if d == None:
                                text_dict['description'][itemid] = 'Empty description'
                            elif len(d) == 0:
                                text_dict['description'][itemid] = 'Empty description'
                            else:
                                text_dict['description'][itemid] = d[0]
                            texts = meta_dict[id2asin[it]][0]
                            
                            if texts ==None:
                                text_dict['title'][itemid] = 'Empty title'
                            elif len(texts) == 0:
                                text_dict['title'][itemid] = 'Empty title'
                            else:
                                texts_ = texts
                                text_dict['title'][itemid] = texts_
                            text_dict['time'][itemid][userid] = time_dict[it][user_id_]
                        
                            f.write('%d %d\n' % (userid, itemid))
                else:
                    if use_train_dict[user_id_] ==1:
                        
                        for it in User_s[t][user_id_]:
                            if CountI[it] >4:
                                if user_id_ in usermap_final:
                                    userid = usermap_final[user_id_]
                                else:
                                    usernum_final +=1
                                    userid = usernum_final
                                    usermap_final[user_id_] = userid
                                if it in itemmap_final:
                                    itemid = itemmap_final[it]
                                else:
                                    itemnum_final +=1
                                    itemid = itemnum_final
                                    itemmap_final[it] = itemid
                            
                                d = meta_dict[id2asin[it]][1]
                                if d == None:
                                    text_dict['description'][itemid] = 'Empty description'
                                elif len(d) == 0:
                                    text_dict['description'][itemid] = 'Empty description'
                                else:
                                    text_dict['description'][itemid] = d[0]
                                texts = meta_dict[id2asin[it]][0]
                                
                                if texts ==None:
                                    text_dict['title'][itemid] = 'Empty title'
                                elif len(texts) == 0:
                                    text_dict['title'][itemid] = 'Empty title'
                                else:
                                    texts_ = texts
                                    text_dict['title'][itemid] = texts_
                                text_dict['time'][itemid][userid] = time_dict[it][user_id_]
                            
                                f.write('%d %d\n' % (userid, itemid))
        f.close()
        with open(f'./../data_{fname}/text_name_dict.json.gz', 'wb') as tf:
            pickle.dump(text_dict, tf)
    
    del text_dict
    del meta_dict
    del dataset