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
import multiprocessing
import re

# --- 配置 ---
VAL_SAMPLE_SIZE = 10000   # 验证集固定抽取 1万 用户
TEST_FRACTION = 0.2       # 测试集采样比例
OUTPUT_DIR = 'data/ranking_processed/'
MODEL_PATH = 'best_tower_model_all.pth' # 召回阶段产生的全量模型

def get_mixed_history(full_seq, pos):
    h = full_seq[:pos]
    return (h[-50:] + [0]*60)[:60]

def build_exam_batch(batch_df, meta, item_sim_matrix, id2year, year_hot_map, item_counts):
    """
    单个进程执行的任务：为一批用户执行 150+150+50 补位召回，并强制补入 GT。
    """
    device = torch.device("cpu")
    # 每个进程内部重建召回环境
    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count'])
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # 构造 Faiss 索引
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

    user_sequences = meta['user_sequences']
    movie_genres = meta['movie_genres']
    user_feat_map = meta['user_feat_map']
    
    batch_results = []
    for _, row in batch_df.iterrows():
        u_idx, target, pos = int(row['user_idx']), int(row['item_idx']), int(row['pos'])
        history_list = user_sequences.get(u_idx, [])[:pos]
        history_set = set(history_list)
        random.seed(u_idx)

        candidates = []
        seen = set()

        # 1. 双塔路 (150)
        u_hist = get_mixed_history(user_sequences.get(u_idx, []), pos)
        u_hgen = []
        for h in u_hist:
            if h != 0: u_hgen.extend(movie_genres.get(h, []))
        u_hgen_truncated = (u_hgen[:100] + [0]*100)[:100]
        f_u = user_feat_map.get(u_idx, {})
        u_stat = [f_u.get('rating_bucket', 0), f_u.get('count_bucket', 0)]
        
        with torch.no_grad():
            u_vec = model.forward_user(torch.tensor([u_idx]), torch.tensor([u_hist]), 
                                     torch.tensor([u_hgen_truncated]), torch.tensor([u_stat])).numpy().astype('float32')
        _, tower_topk = index.search(u_vec, 150)
        for m in tower_topk[0]:
            if m not in seen: candidates.append(m); seen.add(m)

        # 2. ItemCF 补位 (150)
        item_cf_rank = defaultdict(float)
        for h_item in history_list:
            if h_item in item_sim_matrix:
                for neighbor, score in item_sim_matrix[h_item]:
                    if neighbor not in history_set: item_cf_rank[neighbor] += score
        cf_list = [x[0] for x in sorted(item_cf_rank.items(), key=lambda x: (-x[1], x[0]))[:400]]
        cf_c = 0
        for m in cf_list:
            if m not in seen:
                candidates.append(m); seen.add(m); cf_c += 1
                if cf_c >= 150: break

        # 3. 年份热门补位 (50)
        ref_year = id2year.get(history_list[-1], 2000) if history_list else 2000
        y_cand = []
        for y in range(ref_year-2, ref_year+3): y_cand.extend(year_hot_map.get(y, []))
        y_cand = sorted(list(set(y_cand)), key=lambda x: item_counts.get(x, 0), reverse=True)
        y_c = 0
        for m in y_cand:
            if m not in history_set and m not in seen:
                candidates.append(m); seen.add(m); y_c += 1
                if y_c >= 50: break

        # --- 核心逻辑：强制补入 Ground Truth (解耦评估) ---
        if target not in seen:
            if len(candidates) >= 350:
                candidates[349] = target # 替换第 350 名
            else:
                candidates.append(target)
        
        # 兜底填充
        while len(candidates) < 350:
            rand_m = random.randint(1, meta['item_count']-1)
            if rand_m not in seen: candidates.append(rand_m); seen.add(rand_m)

        batch_results.append({
            'user_idx': u_idx, 'item_idx': target, 'pos': pos, 'candidates': candidates[:350]
        })
    return batch_results

def run_generate_exams():
    print(f"正在启动排序考卷生成程序 (V100 模式准备)...")
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    with open('data/processed/item_cf_sim.pkl', 'rb') as f: item_sim_matrix = pickle.load(f)
    val_raw = pd.read_csv('data/processed/val.csv')
    test_raw = pd.read_csv('data/processed/test.csv')
    movies = pd.read_csv('data/ml-32m/movies.csv')
    
    # 预处理
    movie2id = {id: i+1 for i, id in enumerate(movies['movieId'].unique())}
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title); return int(match.group(1)) if match else 0
    movies['year'] = movies['title'].apply(extract_year)
    id2year = movies.set_index(movies['movieId'].map(movie2id))['year'].to_dict()
    item_counts = defaultdict(int)
    for seq in meta['user_sequences'].values():
        for i in seq: item_counts[i] += 1
    year_hot_map = {y: movies[movies['year']==y].assign(pop=movies['movieId'].map(movie2id).map(item_counts)).sort_values('pop', ascending=False)['movieId'].map(movie2id).dropna().tolist()[:100] for y in movies['year'].unique() if y != 0}

    num_workers = max(1, os.cpu_count() - 1)
    
    # --- 1. 生成 10,000 人 Val Exams ---
    print("[1/3] 正在生成 1万用户验证集考卷...")
    val_sampled = val_raw.sample(n=min(VAL_SAMPLE_SIZE, len(val_raw)), random_state=42)
    batches = np.array_split(val_sampled, num_workers)
    val_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(build_exam_batch, b, meta, item_sim_matrix, id2year, year_hot_map, item_counts) for b in batches]
        for f in tqdm(futures, desc="Val进度"): val_results.extend(f.result())
    pd.DataFrame(val_results).to_parquet(os.path.join(OUTPUT_DIR, 'val_exams_10k.parquet'))

    # --- 2. 生成采样版 Test Exams (如 20%) ---
    print(f"[2/3] 正在生成 {TEST_FRACTION*100}% 采样版测试集考卷...")
    test_sampled = test_raw.sample(frac=TEST_FRACTION, random_state=42)
    batches = np.array_split(test_sampled, num_workers)
    test_sampled_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(build_exam_batch, b, meta, item_sim_matrix, id2year, year_hot_map, item_counts) for b in batches]
        for f in tqdm(futures, desc="TestSampled进度"): test_sampled_results.extend(f.result())
    pd.DataFrame(test_sampled_results).to_parquet(os.path.join(OUTPUT_DIR, f'test_exams_sampled_{TEST_FRACTION}.parquet'))

    # --- 3. 生成全量版 Test Exams (可选) ---
    # 这里我们先注释掉全量，如果您需要跑 20万人的，取消注释运行即可
    # print("[3/3] 正在生成 100% 全量测试集考卷...")
    # batches = np.array_split(test_raw, num_workers)
    # test_full_results = []
    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [executor.submit(build_exam_batch, b, meta, item_sim_matrix, id2year, year_hot_map, item_counts) for b in batches]
    #     for f in tqdm(futures, desc="TestFull进度"): test_full_results.extend(f.result())
    # pd.DataFrame(test_full_results).to_parquet(os.path.join(OUTPUT_DIR, 'test_exams_full.parquet'))

    print(f"考卷全部生成完成！输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_generate_exams()
