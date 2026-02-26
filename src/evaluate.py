import os
# --- 环境冲突修复 ---
# 解决 macOS 上多个 OpenMP 库竞争导致的崩溃问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import pandas as pd
import numpy as np
import pickle
import faiss
import random
from tqdm import tqdm
from src.model import DualTowerModel

# --- 全局配置 ---
# InfoNCE Loss 下，每个正样本额外采样的全局随机负样本数
GLOBAL_NEG_RATIO = 2048 

def get_mixed_history(full_seq, pos, recent_len=50, random_len=10):
    """
    构造混合历史特征：50个最近点击 + 10个远期随机点击。
    必须与训练逻辑严格对齐。
    """
    history_all = full_seq[:pos]
    if not history_all: 
        return [0] * (recent_len + random_len)
    
    recent_part = history_all[-recent_len:]
    remaining_part = history_all[:-recent_len]
    
    if len(remaining_part) > random_len:
        random_part = random.sample(remaining_part, random_len)
    else:
        random_part = remaining_part
        
    combined = recent_part + random_part
    max_len = recent_len + random_len
    if len(combined) < max_len:
        combined = combined + [0] * (max_len - len(combined))
    return combined

def get_hist_genres(hist_ids, movie_genres, max_len=100):
    """
    获取用户历史序列中所有电影的题材，并展平。
    """
    genres = []
    for i in hist_ids:
        if i == 0: continue
        genres.extend(movie_genres.get(i, []))
    if len(genres) > max_len: genres = genres[:max_len]
    return genres + [0] * (max_len - len(genres))

def evaluate(model_path, k_list=[10, 50, 100]):
    """
    全量召回评估：强制使用 CPU 以确保在 Mac M系列芯片上的稳定性。
    """
    # --- 稳定性优化：评估阶段回归 CPU ---
    device = torch.device("cpu")
    faiss.omp_set_num_threads(1)
    
    print(f"正在使用的评估设备: {device} (已开启稳定性保护)")

    # 1. 加载元数据
    with open('data/processed/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    # 2. 初始化模型并加载权重
    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()

    # 3. 生成全量物品向量池
    print("正在生成全量电影向量池...")
    item_count = meta['item_count']
    all_item_indices = torch.arange(item_count).to(device)
    
    # 准备所有 item 的 genres 和 stats (从 meta.pkl 中获取)
    all_genres = []
    all_stats = []
    for i in range(item_count):
        g = meta["movie_genres"].get(i, [])
        all_genres.append((g[:6] + [0]*6)[:6]) # max_genre_len=6
        f = meta["item_feat_map"].get(i, {})
        all_stats.append([f.get("year_bucket", 0), f.get("rating_bucket", 0), f.get("count_bucket", 0)])
    
    all_genres_t = torch.tensor(all_genres).to(device)
    all_stats_t = torch.tensor(all_stats).to(device)

    with torch.no_grad():
        item_embs_list = []
        batch_size = 4096 
        for i in range(0, item_count, batch_size):
            embs = model.forward_item(all_item_indices[i : i + batch_size], 
                                    all_genres_t[i : i + batch_size], 
                                    all_stats_t[i : i + batch_size])
            item_embs_list.append(embs.numpy().astype('float32'))
        item_pool_np = np.vstack(item_embs_list)

    # 4. 构建 Faiss 索引
    print(f"正在构建 Faiss 索引 (维度: {item_pool_np.shape[1]})...")
    index = faiss.IndexFlatIP(item_pool_np.shape[1])
    index.add(item_pool_np)

    # 5. 加载测试集
    print("正在加载测试数据...")
    test_df = pd.read_csv('data/processed/test.csv')
    user_sequences = meta['user_sequences']
    user_feat_map = meta['user_feat_map']
    movie_genres = meta['movie_genres']
    
    hits = {k: 0 for k in k_list}
    ndcgs = {k: 0.0 for k in k_list}
    total = len(test_df)

    # 6. 执行全量召回检索
    print(f"正在为 {total} 个测试用户进行 Top-K 检索评估 (Recall/NDCG)...")
    u_batch_size = 1024
    with torch.no_grad():
        for i in tqdm(range(0, total, u_batch_size)):
            batch_df = test_df.iloc[i : i + u_batch_size]
            
            u_ids = torch.tensor(batch_df['user_idx'].values).to(device)
            # 构造混合历史特征
            u_hists = []
            u_hgenres = []
            u_stats = []
            for u, pos in zip(batch_df['user_idx'], batch_df['pos']):
                full_seq = user_sequences.get(u, [])[:pos]
                recent = full_seq[-50:]
                rem = full_seq[:-50]
                rand = random.sample(rem, 10) if len(rem) > 10 else rem
                h = (recent + rand + [0]*60)[:60] # max_hist_len=60
                u_hists.append(h)
                
                # 题材展平
                hg = get_hist_genres(h, movie_genres)
                u_hgenres.append(hg)

                # 用户侧统计特征
                f = user_feat_map.get(u, {})
                u_stats.append([f.get("rating_bucket", 0), f.get("count_bucket", 0)])
            
            u_vecs = model.forward_user(u_ids, torch.tensor(u_hists), torch.tensor(u_hgenres), torch.tensor(u_stats)).numpy().astype('float32')
            
            # 向量检索
            max_k = max(k_list)
            _, topk_indices = index.search(u_vecs, max_k)
            
            targets = batch_df['item_idx'].values
            for j in range(len(targets)):
                target = targets[j]
                user_topk = topk_indices[j]
                
                if target in user_topk:
                    rank = np.where(user_topk == target)[0][0]
                    for k in k_list:
                        if rank < k:
                            hits[k] += 1
                            ndcgs[k] += 1.0 / np.log2(rank + 2)

    # 7. 报告
    print("\n" + "="*40)
    print(f"召回评估报告 (正样本阈值: 3.0)")
    print("="*40)
    for k in k_list:
        hr = hits[k] / total
        ndcg = ndcgs[k] / total
        print(f"Top-{k:<3} | Hit Rate: {hr:.4f} | NDCG: {ndcg:.4f}")
    print("="*40)

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else "best_tower_model.pth" # 默认加载最佳模型
    evaluate(model_file)
