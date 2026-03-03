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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import re

def get_mixed_history(full_seq, pos):
    h = full_seq[:pos]
    return (h[-50:] + [0]*60)[:60]

def get_hist_genres(hist_ids, movie_genres):
    genres = []
    for i in hist_ids:
        if i != 0: genres.extend(movie_genres.get(i, []))
    return (genres[:100] + [0]*100)[:100]

def evaluate_batch_fusion(batch_df, tower_model_path, meta, item_sim_matrix, id2year, year_hot_map, item_counts):
    """
    单个进程执行的任务：执行 150+150+50 的优先级补位融合召回。
    """
    device = torch.device("cpu")
    with open('data/processed/meta.pkl', 'rb') as f: meta_local = pickle.load(f)
    model = DualTowerModel(meta_local['user_count'], meta_local['item_count'], meta_local['genre_count'])
    model.load_state_dict(torch.load(tower_model_path, map_location='cpu'))
    model.eval()

    # 重建 Faiss
    item_count = meta_local['item_count']
    all_item_indices = torch.arange(item_count)
    all_genres, all_stats = [], []
    for i in range(item_count):
        g = meta_local['movie_genres'].get(i, [])
        all_genres.append((g[:6] + [0]*6)[:6])
        f = meta_local['item_feat_map'].get(i, {})
        all_stats.append([f.get('year_bucket', 0), f.get('rating_bucket', 0), f.get('count_bucket', 0)])
    
    with torch.no_grad():
        item_pool = []
        for i in range(0, item_count, 4096):
            embs = model.forward_item(all_item_indices[i:i+4096], torch.tensor(all_genres[i:i+4096]), torch.tensor(all_stats[i:i+4096]))
            item_pool.append(embs.numpy().astype('float32'))
        index = faiss.IndexFlatIP(np.vstack(item_pool).shape[1])
        index.add(np.vstack(item_pool))

    results = []
    user_sequences = meta_local['user_sequences']
    movie_genres = meta_local['movie_genres']
    user_feat_map = meta_local['user_feat_map']

    for _, row in batch_df.iterrows():
        u_idx, target, pos = int(row['user_idx']), int(row['item_idx']), int(row['pos'])
        history_list = user_sequences.get(u_idx, [])[:pos]
        history_set = set(history_list)
        random.seed(u_idx)

        final_recall_pool = []
        seen_items = set()

        # --- 1. 双塔路：定额 150 ---
        u_hist = get_mixed_history(user_sequences.get(u_idx, []), pos)
        u_hgen = get_hist_genres(u_hist, movie_genres)
        f = user_feat_map.get(u_idx, {})
        u_stat = [f.get('rating_bucket', 0), f.get('count_bucket', 0)]
        with torch.no_grad():
            u_vec = model.forward_user(torch.tensor([u_idx]), torch.tensor([u_hist]), torch.tensor([u_hgen]), torch.tensor([u_stat])).numpy().astype('float32')
        _, tower_topk = index.search(u_vec, 150) # 直接取前 150
        
        for m in tower_topk[0]:
            if m not in seen_items:
                final_recall_pool.append(m)
                seen_items.add(m)

        # --- 2. ItemCF 路：补位 150 ---
        # 允许 ItemCF 检索范围稍大，以确保去重后能补齐 150 个新增
        item_cf_rank = defaultdict(float)
        for h_item in history_list:
            if h_item in item_sim_matrix:
                for neighbor, score in item_sim_matrix[h_item]:
                    if neighbor not in history_set: item_cf_rank[neighbor] += score
        
        cf_candidates = [x[0] for x in sorted(item_cf_rank.items(), key=lambda x: (-x[1], x[0]))[:400]]
        cf_count = 0
        for m in cf_candidates:
            if m not in seen_items:
                final_recall_pool.append(m)
                seen_items.add(m)
                cf_count += 1
                if cf_count >= 150: break

        # --- 3. 年份热门路：补位 50 ---
        ref_year = id2year.get(history_list[-1], 2000) if history_list else 2000
        y_cand = []
        for y in range(ref_year-2, ref_year+3): y_cand.extend(year_hot_map.get(y, []))
        y_cand = sorted(list(set(y_cand)), key=lambda x: item_counts.get(x, 0), reverse=True)
        
        y_count = 0
        for m in y_cand:
            if m not in history_set and m not in seen_items:
                final_recall_pool.append(m)
                seen_items.add(m)
                y_count += 1
                if y_count >= 50: break

        # 记录命中情况
        results.append(target in seen_items)

    return results

def run_final_fusion():
    print("正在启动最终多路召回融合评估 (定额补位方案)...")
    print("规则：双塔(150) + ItemCF(补150) + 年份热门(补50) = 总池子 350")

    # 1. 加载所有基础数据
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    with open('data/processed/item_cf_sim.pkl', 'rb') as f: item_sim_matrix = pickle.load(f)
    test_df = pd.read_csv('data/processed/test.csv')
    movies = pd.read_csv('data/ml-32m/movies.csv')
    
    movie2id = {id: i+1 for i, id in enumerate(movies['movieId'].unique())}
    movies['item_idx'] = movies['movieId'].map(movie2id)
    
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else 0
    movies['year'] = movies['title'].apply(extract_year)
    id2year = movies.set_index('item_idx')['year'].to_dict()

    item_counts = defaultdict(int)
    for seq in meta['user_sequences'].values():
        for i in seq: item_counts[i] += 1
    
    year_hot_map = {y: movies[movies['year']==y].assign(pop=movies['item_idx'].map(item_counts)).sort_values('pop', ascending=False)['item_idx'].tolist()[:100] for y in movies['year'].unique() if y != 0}

    # 2. 多进程配置
    num_workers = max(1, os.cpu_count() - 1)
    tower_model_path = 'best_tower_model_all.pth'
    print(f"启动 {num_workers} 个并行 worker 执行全量融合...")
    batches = np.array_split(test_df, num_workers)

    all_hits = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(evaluate_batch_fusion, b, tower_model_path, meta, item_sim_matrix, id2year, year_hot_map, item_counts) for b in batches]
        for future in tqdm(futures, desc="融合处理进度"):
            all_hits.extend(future.result())

    # 3. 输出汇总报告
    total = len(all_hits)
    hit_count = sum(all_hits)
    
    print("" + "="*60)
    print(f"多路召回融合最终评估报告 (全量用户: {total})")
    print("="*60)
    print(f"融合池子总规模: 350 (固定配额补位)")
    print(f"总命中用户数: {hit_count}")
    print(f"最终 Hit Rate @ 350: {hit_count/total:.4f}")
    print("="*60)
    print("注：该结果直接决定了下一阶段排序模型（Ranking）的输入上限。")

if __name__ == "__main__":
    run_final_fusion()
