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

def evaluate_batch(batch_df, tower_model_path, meta, item_sim_matrix, id2year, global_hot_list, year_hot_map, item_counts, k):
    device = torch.device("cpu")
    with open('data/processed/meta.pkl', 'rb') as f: meta_local = pickle.load(f)
    model = DualTowerModel(meta_local['user_count'], meta_local['item_count'], meta_local['genre_count'])
    model.load_state_dict(torch.load(tower_model_path, map_location='cpu'))
    model.eval()

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
    # 存储命中记录：(tower, cf, global_pop, year_pop)
    
    user_sequences = meta_local['user_sequences']
    movie_genres = meta_local['movie_genres']
    user_feat_map = meta_local['user_feat_map']

    for _, row in batch_df.iterrows():
        u_idx, target, pos = int(row['user_idx']), int(row['item_idx']), int(row['pos'])
        history_list = user_sequences.get(u_idx, [])[:pos]
        history_set = set(history_list)
        random.seed(u_idx)

        # 1. 双塔路
        u_hist = get_mixed_history(user_sequences.get(u_idx, []), pos)
        u_hgen = get_hist_genres(u_hist, movie_genres)
        f = user_feat_map.get(u_idx, {})
        u_stat = [f.get('rating_bucket', 0), f.get('count_bucket', 0)]
        with torch.no_grad():
            u_vec = model.forward_user(torch.tensor([u_idx]), torch.tensor([u_hist]), torch.tensor([u_hgen]), torch.tensor([u_stat])).numpy().astype('float32')
        _, tower_topk = index.search(u_vec, k)
        tower_set = set(tower_topk[0])

        # 2. ItemCF路
        item_cf_rank = defaultdict(float)
        for h_item in history_list:
            if h_item in item_sim_matrix:
                for neighbor, score in item_sim_matrix[h_item]:
                    if neighbor not in history_set: item_cf_rank[neighbor] += score
        cf_list = [x[0] for x in sorted(item_cf_rank.items(), key=lambda x: (-x[1], x[0]))[:k]]
        cf_set = set(cf_list)

        # 3. 热门路分拆
        ref_year = id2year.get(history_list[-1], 2000) if history_list else 2000
        # A. 全局热门 (Top-50)
        g_pop_list = []
        for m in global_hot_list:
            if m not in history_set:
                g_pop_list.append(m)
                if len(g_pop_list) >= 50: break
        g_pop_set = set(g_pop_list)
        
        # B. 年份热门 (Top-50)
        y_pop_list = []
        y_cand = []
        for y in range(ref_year-2, ref_year+3): y_cand.extend(year_hot_map.get(y, []))
        y_cand = sorted(list(set(y_cand)), key=lambda x: item_counts.get(x, 0), reverse=True)
        for m in y_cand:
            if m not in history_set:
                y_pop_list.append(m)
                if len(y_pop_list) >= 50: break
        y_pop_set = set(y_pop_list)
        
        # 命中判断
        results.append({
            'tower_hit': target in tower_set,
            'cf_hit': target in cf_set,
            'g_pop_hit': target in g_pop_set,
            'y_pop_hit': target in y_pop_set,
            # 候选集重叠计算 (当前用户的重叠比例)
            'overlap_t_cf': len(tower_set & cf_set) / k,
            'overlap_t_gpop': len(tower_set & g_pop_set) / len(g_pop_set) if g_pop_set else 0,
            'overlap_t_ypop': len(tower_set & y_pop_set) / len(y_pop_set) if y_pop_set else 0,
            'overlap_cf_gpop': len(cf_set & g_pop_set) / len(g_pop_set) if g_pop_set else 0,
            'overlap_cf_ypop': len(cf_set & y_pop_set) / len(y_pop_set) if y_pop_set else 0
        })

    return results

def analyze_all_recall(tower_model_path, k=100):
    print(f"正在准备全量三路召回『精细重叠度』分析 (K={k})...")
    
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
    
    global_hot_list = movies.assign(pop=movies['item_idx'].map(item_counts)).sort_values('pop', ascending=False)['item_idx'].tolist()[:500]
    year_hot_map = {y: movies[movies['year']==y].assign(pop=movies['item_idx'].map(item_counts)).sort_values('pop', ascending=False)['item_idx'].tolist()[:100] for y in movies['year'].unique() if y != 0}

    num_workers = max(1, os.cpu_count() - 1)
    print(f"并行计算启动 (Worker核心: {num_workers})...")
    batches = np.array_split(test_df, num_workers)

    all_data = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(evaluate_batch, b, tower_model_path, meta, item_sim_matrix, id2year, global_hot_list, year_hot_map, item_counts, k) for b in batches]
        for future in tqdm(futures, desc="全量数据吞吐中"):
            all_data.extend(future.result())

    res_df = pd.DataFrame(all_data)
    total = len(res_df)

    # --- 深度精细报告 ---
    print("\n" + "="*60)
    print(f"全量三路召回精细重叠度权威报告 (N={total})")
    print("="*60)
    
    print(f"1. 候选集平均重叠分布 (对齐到各路长度):")
    print(f"   [双塔]   vs [ItemCF] : {res_df['overlap_t_cf'].mean():.2%}")
    print(f"   [双塔]   vs [全局热门]: {res_df['overlap_t_gpop'].mean():.2%}")
    print(f"   [双塔]   vs [年份热门]: {res_df['overlap_t_ypop'].mean():.2%}")
    print(f"   [ItemCF] vs [全局热门]: {res_df['overlap_cf_gpop'].mean():.2%}")
    print(f"   [ItemCF] vs [年份热门]: {res_df['overlap_cf_ypop'].mean():.2%}")

    print(f"\n2. 命中用户精细统计 (命中重合情况):")
    tower = res_df['tower_hit']
    cf = res_df['cf_hit']
    gpop = res_df['g_pop_hit']
    ypop = res_df['y_pop_hit']
    anypop = gpop | ypop
    
    def count_str(arr): return f"{np.sum(arr):>6} (HR: {np.sum(arr)/total:.4f})"

    print(f"   双塔 ∩ 全局热门命中: {count_str(tower & gpop)}")
    print(f"   双塔 ∩ 年份热门命中: {count_str(tower & ypop)}")
    print(f"   ItemCF ∩ 全局热门命中: {count_str(cf & gpop)}")
    print(f"   ItemCF ∩ 年份热门命中: {count_str(cf & ypop)}")
    
    print(f"\n3. 热门召回路效果分析:")
    print(f"   全局热门路命中人数: {count_str(gpop)}")
    print(f"   年份热门路命中人数: {count_str(ypop)}")
    print(f"   热门路合并后总命中: {count_str(anypop)}")
    
    print(f"\n4. 系统最终收益:")
    final_hits = tower | cf | anypop
    print(f"   全路合并最终 HR@100: {np.sum(final_hits)/total:.4f}")
    
    # 纯增量：只有热门路中了，其他都没中
    pure_pop_gain = anypop & ~tower & ~cf
    print(f"   热门路带来的绝对纯收益: {count_str(pure_pop_gain)}")
    print("="*60)

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else "best_tower_model_all.pth"
    analyze_all_recall(model_file)
