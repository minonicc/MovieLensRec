import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import pandas as pd
import numpy as np
import pickle
import faiss
import random
from tqdm import tqdm
from src.model import DualTowerModel

def get_mixed_history(full_seq, pos, recent_len=50, random_len=10):
    history_all = full_seq[:pos]
    if not history_all: return [0] * (recent_len + random_len)
    recent = history_all[-recent_len:]
    rem = history_all[:-recent_len]
    rand = random.sample(rem, 10) if len(rem) > 10 else rem
    combined = recent + rand
    return combined + [0] * (recent_len + random_len - len(combined))

def get_hist_genres(hist_ids, movie_genres, max_len=100):
    genres = []
    for i in hist_ids:
        if i == 0: continue
        genres.extend(movie_genres.get(i, []))
    return (genres[:max_len] + [0]*max_len)[:max_len]

def evaluate(model_path, k_list=[10, 50, 100]):
    device = torch.device("cpu")
    faiss.omp_set_num_threads(1)
    print(f"评估设备: {device} (稳定模式)")

    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device); model.eval()

    # 1. 导出电影池
    print("生成增强版电影向量池...")
    item_count = meta['item_count']
    all_item_indices = torch.arange(item_count)
    all_genres = []
    all_stats = []
    for i in range(item_count):
        g = meta['movie_genres'].get(i, [])
        all_genres.append((g[:6] + [0]*6)[:6])
        f = meta['item_feat_map'].get(i, {})
        all_stats.append([f.get('year_bucket', 0), f.get('rating_bucket', 0), f.get('count_bucket', 0)])
    
    all_genres_t = torch.tensor(all_genres)
    all_stats_t = torch.tensor(all_stats)

    with torch.no_grad():
        item_embs_list = []
        batch_size = 4096
        for i in range(0, item_count, batch_size):
            embs = model.forward_item(all_item_indices[i:i+batch_size], all_genres_t[i:i+batch_size], all_stats_t[i:i+batch_size])
            item_embs_list.append(embs.numpy().astype('float32'))
        item_pool_np = np.vstack(item_embs_list)

    index = faiss.IndexFlatIP(item_pool_np.shape[1])
    index.add(item_pool_np)

    # 2. 评估测试用户
    test_df = pd.read_csv('data/processed/test.csv')
    user_sequences = meta['user_sequences']
    user_feat_map = meta['user_feat_map']
    movie_genres = meta['movie_genres']
    
    hits, ndcgs = {k: 0 for k in k_list}, {k: 0.0 for k in k_list}
    total = len(test_df)

    print(f"正在对 {total} 个用户进行多特征召回评估...")
    u_batch_size = 1024
    with torch.no_grad():
        for i in tqdm(range(0, total, u_batch_size)):
            batch_df = test_df.iloc[i : i + u_batch_size]
            u_ids = torch.tensor(batch_df['user_idx'].values)
            
            u_hists = [get_mixed_history(user_sequences.get(u, []), pos) for u, pos in zip(batch_df['user_idx'], batch_df['pos'])]
            u_hgenres = [get_hist_genres(h, movie_genres) for h in u_hists]
            u_stats = [[user_feat_map.get(u, {}).get('rating_bucket', 0), user_feat_map.get(u, {}).get('count_bucket', 0)] for u in batch_df['user_idx']]
            
            u_vecs = model.forward_user(u_ids, torch.tensor(u_hists), torch.tensor(u_hgenres), torch.tensor(u_stats)).numpy().astype('float32')
            _, topk_indices = index.search(u_vecs, max(k_list))
            
            targets = batch_df['item_idx'].values
            for j in range(len(targets)):
                target = targets[j]
                if target in topk_indices[j]:
                    rank = np.where(topk_indices[j] == target)[0][0]
                    for k in k_list:
                        if rank < k:
                            hits[k] += 1
                            ndcgs[k] += 1.0 / np.log2(rank + 2)

    print("\n--- V2.2 特征增强版评估报告 ---")
    for k in k_list:
        print(f"Top-{k:<3} | Hit Rate: {hits[k]/total:.4f} | NDCG: {ndcgs[k]/total:.4f}")

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else "tower_model_v2_epoch_5.pth"
    evaluate(model_file)
