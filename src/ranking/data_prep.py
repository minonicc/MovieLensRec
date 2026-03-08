import pandas as pd
import numpy as np
import pickle
import os
import random
import re
from tqdm import tqdm
from collections import defaultdict

# --- 全局配置 ---
# 正负比例控制：为每一个正样本配比 3 个来自『召回池』的负样本名额
NEG_PER_POS = 3
# 特征计算时的快照截断长度：计算题材浓度等特征时，回溯用户最近 200 条高分行为
FEAT_SNAPSHOT_MAX = 200
# 数据路径配置
OUTPUT_DIR = 'data/ranking_processed/'
USER_PACK_PATH = 'data/ranking_processed/ranking_user_pack.pkl'
TRAIN_POOLS_PATH = 'data/ranking_processed/training_pools_only.pkl'

def calculate_genre_punch(target_item_genres, snapshot_history_items, item_idx2genres, snapshot_history_ratings):
    """
    精排强特征工厂：计算题材『组合拳』数值特征。
    利用真实的 3.0-5.0 评分数据，实现高分辨率的用户偏好建模。
    """
    if not target_item_genres or not snapshot_history_items:
        return 0.0, 0.0, 0.6 # 默认中性值
    
    # 截取该时刻最新的行为片段进行统计，兼顾时效性与性能
    h_items = snapshot_history_items[-FEAT_SNAPSHOT_MAX:]
    h_ratings = snapshot_history_ratings[-FEAT_SNAPSHOT_MAX:]
    
    genre_counts = defaultdict(int)
    genre_total_score = defaultdict(float)
    
    for item, rating in zip(h_items, h_ratings):
        for g in item_idx2genres.get(item, []):
            genre_counts[g] += 1
            genre_total_score[g] += rating
            
    total_g_apps = sum(genre_counts.values())
    
    # 1. 浓度 (Density)：当前题材在用户最近历史中的频次占比，反映用户此时的『领域偏好高度』
    density = sum(genre_counts[g] for g in target_item_genres) / (total_g_apps + 1e-8)
    
    # 2. 近期匹配 (Recent Match)：判断当前电影是否命中用户最近 5 个行为的题材（捕捉即时兴致）
    recent_5_g = set()
    for item in snapshot_history_items[-5:]:
        recent_5_g.update(item_idx2genres.get(item, []))
    recent_match = 1.0 if (set(target_item_genres) & recent_5_g) else 0.0
    
    # 3. 评分偏好 (Rating Bias)：历史中用户对该题材的平均打分，反映喜爱强度
    relevant_scores = [genre_total_score[g] / (genre_counts[g] + 1e-8) for g in target_item_genres if g in genre_counts]
    rating_bias = np.mean(relevant_scores) / 5.0 if relevant_scores else 0.6
    
    return density, recent_match, rating_bias

def process_single_user(u_idx, user_sequences, user_ratings_seq, training_pools, meta):
    """
    精细化处理单个黄金用户：生产其所有的正向训练样本及其对应的召回池负样本。
    确保每一行样本的特征都严格基于该行为发生时的『历史瞬间』。
    """
    item_idx2genres = meta['movie_genres']
    item_feat_map = meta['item_feat_map']
    user_feat_map = meta['user_feat_map']
    
    # 获取该黄金用户的原始序列与评分快照
    full_seq = user_sequences.get(u_idx, [])
    full_ratings = user_ratings_seq.get(u_idx, [])
    u_train_pools = training_pools.get(u_idx, {})
    
    n = len(full_seq)
    if n < 3 or not u_train_pools: return []
    
    # --- 核心时序隔离：训练集绝对不允许越过 n-2 (索引 n-3 为止) ---
    # 理由：n-2 是验证目标，n-1 是测试目标，模型必须在完全不感知『未来答案』的情况下训练。
    train_limit = n - 2
    # 取最近 100 个点击作为正样本锚点
    start_pos = max(0, train_limit - 100)
    pos_indices = list(range(start_pos, train_limit))
    
    u_stats = user_feat_map.get(u_idx, {})
    user_results = []

    # --- 性能优化：增量维护已看集合，避免在循环内重复全量构造 history_set (解决卡顿的关键) ---
    # 初始禁选名单：该用户在进入精排正样本采样范围之前的所有交互记录
    current_history_set = set(full_seq[:start_pos])

    for pos in pos_indices:
        target_item = full_seq[pos]
        # 实时更新该时刻的禁选名单：包含当前正样本及其之前的所有交互
        current_history_set.add(target_item)
        
        # 内部闭包：构造具备快照特性的行记录
        def build_row(t_item, label, c_pos):
            i_f = item_feat_map.get(t_item, {})
            # 严格截取该点击时刻之前的历史快照 (零穿越防护)
            s_items = full_seq[:c_pos]
            s_ratings = full_ratings[:c_pos]
            # 时序特征：50 个 ID 序列
            h_ids = (s_items[-50:] + [0]*50)[:50]
            # 时序特征：100 个题材序列
            h_gs = []
            for h in h_ids:
                if h != 0: h_gs.extend(item_idx2genres.get(h, []))
            h_gs_trun = (h_gs[-100:] + [0]*100)[:100]
            
            # 计算强特征组合拳数值
            d, rm, rb = calculate_genre_punch(item_idx2genres.get(t_item, []), s_items, item_idx2genres, s_ratings)
            
            return {
                'user_idx': u_idx, 'item_idx': t_item, 'label': label,
                'year_bucket': i_f.get('year_bucket', 0),
                'item_rating_bucket': i_f.get('rating_bucket', 0),
                'item_count_bucket': i_f.get('count_bucket', 0),
                'user_rating_bucket': u_stats.get('rating_bucket', 0),
                'user_count_bucket': u_stats.get('count_bucket', 0),
                'hist_movie_ids': h_ids, 'hist_genre_ids': h_gs_trun,
                'genre_density': d, 'genre_recent_match': rm, 'genre_rating_bias': rb
            }

        # A. 记录正样本行 (Label=1)
        user_results.append(build_row(target_item, 1, pos))
        
        # B. 记录召回池补位负样本行 (Label=0)
        # 寻找该正样本时刻之前最近的一个召回快照
        valid_snaps = [s for s in u_train_pools.keys() if s <= pos]
        if valid_snaps:
            latest_snap = max(valid_snaps)
            raw_candidates = u_train_pools[latest_snap]
            
            # 使用增量维护的 current_history_set 执行 O(1) 的实时去重，防止正负样本冲突
            clean_pool = [m for m in raw_candidates if m not in current_history_set]
            
            if clean_pool:
                num_to_sample = min(len(clean_pool), NEG_PER_POS)
                # 随机采样 3 个干扰项作为负样本
                for neg_item in random.sample(clean_pool, num_to_sample):
                    user_results.append(build_row(neg_item, 0, pos))
                    
    return user_results

def generate_ranking_dataset():
    """
    单线程高响应主程序：逐个用户生产具备『时序绝对隔离』属性的精排样本集。
    """
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    random.seed(42); np.random.seed(42)
    
    print("正在加载预计算资源 (读取 4万 黄金用户包与精简训练池)...")
    # ranking_user_pack.pkl 已在 prep_sampling_users.py 阶段完成了用户抽样
    with open(USER_PACK_PATH, 'rb') as f: user_pack = pickle.load(f)
    # training_pools_only.pkl 已在 split_large_pools.py 阶段剔除了未来数据
    with open(TRAIN_POOLS_PATH, 'rb') as f: training_pools = pickle.load(f)
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    
    user_ids = user_pack['user_ids']
    u_seqs = user_pack['user_sequences']
    u_rts = user_pack['user_ratings_seq']
    
    print(f"启动精排数据生产管线 (用户总数: {len(user_ids)} | 步长限制: n-3)...")
    final_rows = []
    
    # 采用单线程循环，避免大型字典在多进程间复制导致的挂起与内存爆炸
    for u_idx in tqdm(user_ids, desc="加工进度"):
        user_samples = process_single_user(u_idx, u_seqs, u_rts, training_pools, meta)
        final_rows.extend(user_samples)
            
    print(f"\n加工流程结束！共产出严谨训练样本: {len(final_rows):,}")
    df = pd.DataFrame(final_rows)
    # 持久化为 Parquet (列式存储)，极大提升后续 DeepFM 的训练吞吐量
    output_file = os.path.join(OUTPUT_DIR, 'train.parquet')
    df.to_parquet(output_file, index=False)
    print(f">>> Ranking 纯净训练集生产成功：{output_file}")

if __name__ == "__main__":
    generate_ranking_dataset()
