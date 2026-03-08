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
from ..retrieval.model import DualTowerModel
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import re

# --- 全局配置 ---
SNAPSHOT_INTERVAL = 10    # 训练集每隔 10 个样本点跑一次召回
TOWER_MODEL_PATH = 'best_tower_model_all.pth'
USER_PACK_PATH = 'data/ranking_processed/ranking_user_pack.pkl'
OUTPUT_PATH = 'data/ranking_processed/multi_snapshot_pools.pkl'

def get_mixed_history(full_seq, pos):
    """
    构造混合历史特征：50个最近 + 10个随机。
    """
    h = full_seq[:pos]
    return (h[-50:] + [0]*60)[:60]

def build_snapshot_pools_batch(batch_uids, meta, item_sim_matrix, id2year, year_hot_map, item_counts, user_sequences):
    """
    核心计算单元：为每个用户生成 训练(分段)、验证(n-2)、测试(n-1) 的所有召回池。
    """
    device = torch.device("cpu")
    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count'])
    model.load_state_dict(torch.load(TOWER_MODEL_PATH, map_location='cpu'))
    model.eval()

    # 1. 重建 Faiss 索引 (各子进程独立维护，确保线程安全)
    item_count = meta['item_count']
    all_genres, all_stats = [], []
    for i in range(item_count):
        g = meta['movie_genres'].get(i, [])
        all_genres.append((g[:6] + [0]*6)[:6])
        f = meta['item_feat_map'].get(i, {})
        all_stats.append([f.get('year_bucket', 0), f.get('rating_bucket', 0), f.get('count_bucket', 0)])
    
    with torch.no_grad():
        item_pool = []
        for i in range(0, item_count, 4096):
            embs = model.forward_item(torch.arange(i, min(i+4096, item_count)), 
                                    torch.tensor(all_genres[i:i+4096]), 
                                    torch.tensor(all_stats[i:i+4096]))
            item_pool.append(embs.numpy().astype('float32'))
        index = faiss.IndexFlatIP(np.vstack(item_pool).shape[1])
        index.add(np.vstack(item_pool))

    user_feat_map = meta['user_feat_map']
    movie_genres = meta['movie_genres']
    batch_snapshot_results = {}

    for u_idx in batch_uids:
        full_seq = user_sequences[u_idx]
        n = len(full_seq)
        if n < 3: continue
        
        # --- 确定所有需要召回的关键位置 ---
        # A. 训练点：最近 100 个训练样本 (排除倒数两个)，每 10 个设一快照
        train_indices = list(range(max(0, n-102), n-2, SNAPSHOT_INTERVAL))
        # B. 验证点：n-2
        val_index = n - 2
        # C. 测试点：n-1
        test_index = n - 1
        
        all_target_indices = sorted(list(set(train_indices + [val_index, test_index])))
        
        user_pools = {} # {pos: [candidates_350]}

        for pos in all_target_indices:
            history_list = full_seq[:pos]
            history_set = set(history_list)
            random.seed(u_idx + pos)

            # --- 执行标准四路补位召回 (150+150+50) ---
            candidates = []
            seen = set()

            # 1. 双塔召回
            u_hist = get_mixed_history(full_seq, pos)
            u_hgen = []
            for h in u_hist:
                if h != 0: u_hgen.extend(movie_genres.get(h, []))
            u_hgen_truncated = (u_hgen[:100] + [0]*100)[:100]
            f_u = user_feat_map.get(u_idx, {})
            u_stat = [f_u.get('rating_bucket', 0), f_u.get('count_bucket', 0)]
            with torch.no_grad():
                u_vec = model.forward_user(torch.tensor([u_idx]), torch.tensor([u_hist]), 
                                         torch.tensor([u_hgen_truncated]), torch.tensor([u_stat])).numpy().astype('float32')
            _, tower_topk = index.search(u_vec, 350)
            for m in tower_topk[0]:
                if m not in seen: candidates.append(m); seen.add(m)

            # 2. ItemCF 补位
            item_cf_rank = defaultdict(float)
            for h_item in history_list:
                if h_item in item_sim_matrix:
                    for neighbor, score in item_sim_matrix[h_item]:
                        if neighbor not in history_set: item_cf_rank[neighbor] += score
            cf_list = [x[0] for x in sorted(item_cf_rank.items(), key=lambda x: (-x[1], x[0]))[:500]]
            cf_c = 0
            for m in cf_list:
                if m not in seen:
                    candidates.append(m); seen.add(m); cf_c += 1
                    if cf_c >= 150: break

            # 3. 年份热门补位
            ref_year = id2year.get(history_list[-1], 2000) if history_list else 2000
            y_cand = []
            for y in range(ref_year-2, ref_year+3): y_cand.extend(year_hot_map.get(y, []))
            y_cand = sorted(list(set(y_cand)), key=lambda x: item_counts.get(x, 0), reverse=True)
            y_c = 0
            for m in y_cand:
                if m not in history_set and m not in seen:
                    candidates.append(m); seen.add(m); y_c += 1
                    if y_c >= 50: break

            # 保存该快照下的前 350 名候选人
            user_pools[pos] = candidates[:350]
            
        batch_snapshot_results[u_idx] = user_pools
        
    return batch_snapshot_results

def run_prep_pools():
    """
    主运行函数：实现大规模并行化的快照生成。
    """
    print("正在启动全方位召回快照生成程序 (训练+验证+测试一次性闭环)...")
    with open(USER_PACK_PATH, 'rb') as f: user_pack = pickle.load(f)
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    with open('data/processed/item_cf_sim.pkl', 'rb') as f: item_sim_matrix = pickle.load(f)
    movies = pd.read_csv('data/ml-32m/movies.csv')
    
    # 预处理电影年份与流行度榜单
    movie2id = {id: i+1 for i, id in enumerate(movies['movieId'].unique())}
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title); return int(match.group(1)) if match else 0
    movies['year'] = movies['title'].apply(extract_year)
    id2year = movies.set_index(movies['movieId'].map(movie2id))['year'].to_dict()
    item_counts = defaultdict(int)
    for seq in meta['user_sequences'].values():
        for i in seq: item_counts[i] += 1
    # 预存年度热门 Top-100
    year_hot_map = {y: movies[movies['year']==y].assign(pop=movies['movieId'].map(movie2id).map(item_counts)).sort_values('pop', ascending=False)['movieId'].map(movie2id).dropna().tolist()[:100] for y in movies['year'].unique() if y != 0}

    num_workers = max(1, os.cpu_count() - 1)
    user_ids = user_pack['user_ids']
    user_sequences = user_pack['user_sequences']
    
    print(f"启动 {num_workers} 个并行进程计算 {len(user_ids)} 个用户的全量时序快照...")
    batches = np.array_split(user_ids, num_workers)
    
    final_pools = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(build_snapshot_pools_batch, b, meta, item_sim_matrix, id2year, year_hot_map, item_counts, user_sequences) for b in batches]
        for f in tqdm(futures, desc="召回进度"):
            final_pools.update(f.result())

    print(f"所有时序快照生成完成，正在保存至 {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(final_pools, f)
    print(">>> 全链路召回负样本池已就绪，后续步骤可直接从中提取数据。")

if __name__ == "__main__":
    run_prep_pools()
