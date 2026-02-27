import os
# --- 环境冲突修复 ---
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
    """
    构造混合历史特征：50个最近点击 + 10个远期随机点击。
    """
    history_all = full_seq[:pos]
    if not history_all: return [0] * (recent_len + random_len)
    recent = history_all[-recent_len:]
    rem = history_all[:-recent_len]
    rand = random.sample(rem, min(len(rem), random_len)) if len(rem) > 0 else []
    combined = recent + rand
    return (combined + [0]*60)[:60]

def get_hist_genres(hist_ids, movie_genres, max_len=100):
    genres = []
    for i in hist_ids:
        if i != 0: genres.extend(movie_genres.get(i, []))
    return (genres[:max_len] + [0]*max_len)[:max_len]

def evaluate_quality(model_path, k=100):
    """
    质量分析主函数：按分值区间 [0-3, 3-4, 4-5] 统计召回命中率。
    """
    device = torch.device("cpu")
    faiss.omp_set_num_threads(1)
    print(f"正在进行召回质量分析: {model_path} (K={k})")

    # 1. 加载元数据和带分值的测试集
    with open('data/processed/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    test_df = pd.read_csv('data/processed/test.csv')
    
    if 'rating' not in test_df.columns:
        print("错误: test.csv 中缺少 rating 列。请先重新运行 python src/data_preprocess.py")
        return

    # 2. 初始化模型
    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device); model.eval()

    # 3. 生成全量电影向量池
    print("生成电影向量池...")
    item_count = meta['item_count']
    all_item_indices = torch.arange(item_count)
    all_genres, all_stats = [], []
    for i in range(item_count):
        g = meta['movie_genres'].get(i, [])
        all_genres.append((g[:6] + [0]*6)[:6])
        f = meta['item_feat_map'].get(i, {})
        all_stats.append([f.get('year_bucket', 0), f.get('rating_bucket', 0), f.get('count_bucket', 0)])
    
    all_genres_t, all_stats_t = torch.tensor(all_genres), torch.tensor(all_stats)
    with torch.no_grad():
        item_embs = []
        for i in range(0, item_count, 4096):
            embs = model.forward_item(all_item_indices[i:i+4096], all_genres_t[i:i+4096], all_stats_t[i:i+4096])
            item_embs.append(embs.numpy().astype('float32'))
        item_pool = np.vstack(item_embs)

    index = faiss.IndexFlatIP(item_pool.shape[1])
    index.add(item_pool)

    # 4. 执行按分值区间的召回评估
    bin_labels = ['[0.0, 3.0) 分', '[3.0, 4.0) 分', '[4.0, 5.0] 分']
    bin_stats = {label: {'total': 0, 'hits': 0} for label in bin_labels}

    user_sequences = meta['user_sequences']
    user_feat_map = meta['user_feat_map']
    movie_genres = meta['movie_genres']
    
    print(f"执行 Top-{k} 检索并按分值归类统计...")
    u_batch_size = 1024
    with torch.no_grad():
        for i in tqdm(range(0, len(test_df), u_batch_size)):
            batch_df = test_df.iloc[i : i + u_batch_size]
            u_ids = batch_df['user_idx'].values
            
            # 构造特征
            u_hists = [get_mixed_history(user_sequences.get(u, []), pos) for u, pos in zip(u_ids, batch_df['pos'])]
            u_hgenres = [get_hist_genres(h, movie_genres) for h in u_hists]
            u_stats = [[user_feat_map.get(u, {}).get('rating_bucket', 0), user_feat_map.get(u, {}).get('count_bucket', 0)] for u in u_ids]
            
            u_vecs = model.forward_user(torch.tensor(u_ids), torch.tensor(u_hists), torch.tensor(u_hgenres), torch.tensor(u_stats)).numpy().astype('float32')
            _, topk_indices = index.search(u_vecs, k)
            
            targets = batch_df['item_idx'].values
            ratings = batch_df['rating'].values
            
            for j in range(len(targets)):
                score = ratings[j]
                if score < 3.0: label = bin_labels[0]
                elif score < 4.0: label = bin_labels[1]
                else: label = bin_labels[2]
                
                bin_stats[label]['total'] += 1
                if targets[j] in topk_indices[j]:
                    bin_stats[label]['hits'] += 1

    print("\n" + "="*60)
    print(f"召回质量分析报告 (按测试集原始评分, K={k})")
    print("="*60)
    for label in bin_labels:
        total = bin_stats[label]['total']
        hits = bin_stats[label]['hits']
        hr = hits / total if total > 0 else 0
        print(f"{label:<15} | 样本数: {total:>6} | 命中数: {hits:>4} | 组内 HitRate: {hr:.2%}")
    print("="*60)

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else "best_tower_model_3.0.pth"
    evaluate_quality(model_file)
