import pandas as pd
import numpy as np
import pickle
import os
import random
import re
from tqdm import tqdm
from collections import defaultdict

# --- 全局配置 ---
# 负采样配比
NEG_PER_POS = 3
# 特征统计截断长度 (题材浓度计算)
FEAT_SNAPSHOT_MAX = 200
# DIN 专用历史序列长度：正式扩展为 100
DIN_HIST_LEN = 100
# 分块落盘行数
CHUNK_SIZE_LIMIT = 100000
OUTPUT_DIR = 'data/ranking_processed/'
USER_PACK_PATH = 'data/ranking_processed/ranking_user_pack.pkl'
TRAIN_POOLS_PATH = 'data/ranking_processed/training_pools_only.pkl'

def calculate_genre_punch(target_item_genres, snapshot_history_items, item_idx2genres, snapshot_history_ratings):
    """
    计算题材组合拳特征。保持 200 次快照回溯深度。
    """
    if not target_item_genres or not snapshot_history_items:
        return 0.0, 0.0, 0.6
    
    h_items = snapshot_history_items[-FEAT_SNAPSHOT_MAX:]
    h_ratings = snapshot_history_ratings[-FEAT_SNAPSHOT_MAX:]
    
    genre_counts = defaultdict(int)
    genre_total_score = defaultdict(float)
    for item, rating in zip(h_items, h_ratings):
        for g in item_idx2genres.get(item, []):
            genre_counts[g] += 1; genre_total_score[g] += rating
            
    total_g_apps = sum(genre_counts.values())
    density = sum(genre_counts[g] for g in target_item_genres) / (total_g_apps + 1e-8)
    
    recent_5_g = set()
    for item in snapshot_history_items[-5:]:
        recent_5_g.update(item_idx2genres.get(item, []))
    recent_match = 1.0 if (set(target_item_genres) & recent_5_g) else 0.0
    
    relevant_scores = [genre_total_score[g] / (genre_counts[g] + 1e-8) for g in target_item_genres if g in genre_counts]
    rating_bias = np.mean(relevant_scores) / 5.0 if relevant_scores else 0.6
    
    return density, recent_match, rating_bias

def process_single_user_din(u_idx, user_sequences, user_ratings_seq, training_pools, meta):
    """
    精细化处理单个用户，生产包含 100 维长序列的样本。
    """
    item_idx2genres = meta['movie_genres']
    item_feat_map = meta['item_feat_map']
    user_feat_map = meta['user_feat_map']
    
    full_seq = user_sequences.get(u_idx, [])
    full_ratings = user_ratings_seq.get(u_idx, [])
    u_train_pools = training_pools.get(u_idx, {})
    
    n = len(full_seq)
    if n < 3 or not u_train_pools: return []
    
    # 严格锁定 n-3 训练边界
    train_limit = n - 2
    start_pos = max(0, train_limit - 100)
    pos_indices = list(range(start_pos, train_limit))
    u_stats = user_feat_map.get(u_idx, {})
    u_r_b, u_c_b = u_stats.get('rating_bucket', 0), u_stats.get('count_bucket', 0)

    # 增量去重
    current_history_set = set(full_seq[:start_pos])
    user_results = []

    for pos in pos_indices:
        target_item = full_seq[pos]
        current_history_set.add(target_item)
        
        def build_row(t_item, label, c_pos):
            i_f = item_feat_map.get(t_item, {})
            snap_it, snap_rt = full_seq[:c_pos], full_ratings[:c_pos]
            
            # --- DIN 核心改动：截取最近 100 个点击 ---
            h_ids = (snap_it[-DIN_HIST_LEN:] + [0]*DIN_HIST_LEN)[:DIN_HIST_LEN]
            
            # 题材序列同样同步到 100，提供更丰富的语义背景
            h_gs = []
            for h in h_ids:
                if h != 0: h_gs.extend(item_idx2genres.get(h, []))
            h_gs_trun = (h_gs[-100:] + [0]*100)[:100] # 保持 100 题材截断
            
            d, rm, rb = calculate_genre_punch(item_idx2genres.get(t_item, []), snap_it, item_idx2genres, snap_rt)
            
            return {
                'user_idx': u_idx, 'item_idx': t_item, 'label': label,
                'year_bucket': i_f.get('year_bucket', 0), 'item_rating_bucket': i_f.get('rating_bucket', 0),
                'item_count_bucket': i_f.get('count_bucket', 0), 'user_rating_bucket': u_r_b,
                'user_count_bucket': u_c_b, 'hist_movie_ids': h_ids, 'hist_genre_ids': h_gs_trun,
                'genre_density': d, 'genre_recent_match': rm, 'genre_rating_bias': rb
            }

        # 正样本行
        user_results.append(build_row(target_item, 1, pos))
        
        # 负采样补位 (1:3)
        valid_snaps = [s for s in u_train_pools.keys() if s <= pos]
        if valid_snaps:
            candidates = u_train_pools[max(valid_snaps)]
            clean_pool = [m for m in candidates if m not in current_history_set]
            if clean_pool:
                num_to_sample = min(len(clean_pool), NEG_PER_POS)
                for neg_item in random.sample(clean_pool, num_to_sample):
                    user_results.append(build_row(neg_item, 0, pos))
                    
    return user_results

def generate_ranking_dataset_din():
    """
    单线程流式生产 DIN 专用训练集。
    """
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    random.seed(42); np.random.seed(42)
    
    print(f"正在加载预计算资源进行 DIN 数据准备 (序列长度: {DIN_HIST_LEN})...")
    with open(USER_PACK_PATH, 'rb') as f: user_pack = pickle.load(f)
    with open(TRAIN_POOLS_PATH, 'rb') as f: training_pools = pickle.load(f)
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    
    user_ids = user_pack['user_ids']
    u_seqs = user_pack['user_sequences']
    u_rts = user_pack['user_ratings_seq']
    
    all_dfs = []
    rows_buffer = []
    total_count = 0
    
    output_file = os.path.join(OUTPUT_DIR, 'train_din.parquet')

    for u_idx in tqdm(user_ids, desc="DIN加工进度"):
        user_samples = process_single_user_din(u_idx, u_seqs, u_rts, training_pools, meta)
        rows_buffer.extend(user_samples)
        
        if len(rows_buffer) >= CHUNK_SIZE_LIMIT:
            all_dfs.append(pd.DataFrame(rows_buffer))
            total_count += len(rows_buffer)
            rows_buffer = []
            
    if rows_buffer:
        all_dfs.append(pd.DataFrame(rows_buffer))
        total_count += len(rows_buffer)
            
    print(f"加工结束！总样本数: {total_count:,}")
    print("正在执行合并与保存为 train_din.parquet...")
    
    df = pd.concat(all_dfs, ignore_index=True)
    df.to_parquet(output_file, index=False)
    print(f">>> DIN 训练集生产成功：{output_file}")

if __name__ == "__main__":
    generate_ranking_dataset_din()
