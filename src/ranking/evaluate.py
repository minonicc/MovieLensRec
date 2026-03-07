import os
# --- 环境冲突修复 ---
# 解决 macOS 上由于多个库冲突导致的 OpenMP 重复初始化崩溃问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import pandas as pd
import numpy as np
import pickle
import math
from tqdm import tqdm
from .model import DeepFM
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score

# 特征快照计算的最大截断长度 (需与 data_prep 保持严格一致)
FEAT_SNAPSHOT_MAX = 200

def calculate_genre_punch_eval(target_item_genres, snap_items, item_idx2genres, snap_ratings):
    """
    评估阶段的特征实时工厂：根据测试瞬间的用户快照，为候选物品计算题材组合拳数值。
    """
    if not target_item_genres or not snap_items:
        return 0.0, 0.0, 0.5
    
    # 截取该时刻最新的行为片段
    h_items = snap_items[-FEAT_SNAPSHOT_MAX:]
    h_ratings = snap_ratings[-FEAT_SNAPSHOT_MAX:]
    
    g_counts = defaultdict(int)
    g_total_score = defaultdict(float)
    g_occur = defaultdict(int)
    for it, rt in zip(h_items, h_ratings):
        gs = item_idx2genres.get(it, [])
        for g in gs:
            g_counts[g] += 1; g_total_score[g] += rt; g_occur[g] += 1
            
    # 计算三个维度的偏好分数
    total_g = sum(g_counts.values())
    density = sum(g_counts[g] for g in target_item_genres) / (total_g + 1e-8)
    
    recent_5_g = set()
    for it in snap_items[-5:]: recent_5_g.update(item_idx2genres.get(it, []))
    recent_match = 1.0 if (set(target_item_genres) & recent_5_g) else 0.0
    
    scores = [g_total_score[g] / (g_occur[g] + 1e-8) for g in target_item_genres if g in g_occur]
    rating_bias = np.mean(scores) / 5.0 if scores else 0.6
    
    return density, recent_match, rating_bias

def evaluate_ranking_batch(batch_df, model_path, feature_dims, meta, hit_k_list, ndcg_k):
    """
    并行计算单元：为一批用户在 350 个候选物品组成的考卷上进行精排性能打分。
    """
    device = torch.device("cpu")
    # 子进程独立加载模型，确保内存与逻辑隔离
    model = DeepFM(feature_dims, embed_dim=16)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 提取多进程共享的基础索引
    item_idx2genres = meta['movie_genres']
    item_feat_map = meta['item_feat_map']
    user_feat_map = meta['user_feat_map']
    user_sequences = meta['user_sequences']
    user_ratings_seq = meta.get('user_ratings_seq', {}) 
    
    g_list, n_list, m_list = [], [], []
    h_res = {k: [] for k in hit_k_list}

    for _, row in batch_df.iterrows():
        u_idx, target, candidates, pos = int(row['user_idx']), int(row['item_idx']), row['candidates'], int(row['pos'])
        
        # --- 1. 恢复行为快照逻辑 ---
        u_full_seq = user_sequences.get(u_idx, [])
        u_full_ratings = user_ratings_seq.get(u_idx, [4.0]*len(u_full_seq))
        
        snap_items = u_full_seq[:pos]
        snap_ratings = u_full_ratings[:pos]
        
        # 序列特征：50 ID + 100 题材
        hist_ids = (snap_items[-50:] + [0]*50)[:50]
        hist_gs = []
        for h in hist_ids:
            if h != 0: hist_gs.extend(item_idx2genres.get(h, []))
        hist_gs_truncated = (hist_gs[-100:] + [0]*100)[:100]
        u_stats = user_feat_map.get(u_idx, {})

        # --- 2. 批量特征化：将 350 个候选物品转化为模型可识别的张量矩阵 ---
        c_cat = defaultdict(list)
        c_seq = {'hist_movie_ids': [], 'hist_genre_ids': []}
        c_num = {'genre_density': [], 'genre_recent_match': [], 'genre_rating_bias': []}

        for c_id in candidates:
            i_f = item_feat_map.get(c_id, {})
            # A. 基础属性
            c_cat['user_idx'].append(u_idx); c_cat['item_idx'].append(c_id)
            c_cat['year_bucket'].append(i_f.get('year_bucket', 0))
            c_cat['item_rating_bucket'].append(i_f.get('rating_bucket', 0))
            c_cat['item_count_bucket'].append(i_f.get('count_bucket', 0))
            c_cat['user_rating_bucket'].append(u_stats.get('rating_bucket', 0))
            c_cat['user_count_bucket'].append(u_stats.get('count_bucket', 0))
            # B. 行为序列
            c_seq['hist_movie_ids'].append(hist_ids); c_seq['hist_genre_ids'].append(hist_gs_truncated)
            # C. 实时题材组合拳特征
            d, rm, rb = calculate_genre_punch_eval(item_idx2genres.get(c_id, []), snap_items, item_idx2genres, snap_ratings)
            c_num['genre_density'].append(d); c_num['genre_recent_match'].append(rm); c_num['genre_rating_bias'].append(rb)

        # --- 3. 极速打分推理 ---
        with torch.no_grad():
            t_cat = {k: torch.tensor(v, dtype=torch.long) for k, v in c_cat.items()}
            t_seq = {k: torch.tensor(v, dtype=torch.long) for k, v in c_seq.items()}
            t_num = {k: torch.tensor(v, dtype=torch.float32) for k, v in c_num.items()}
            # 推算点击概率
            scores = torch.sigmoid(model(t_cat, t_seq, t_num)).numpy()

        # --- 4. 结算排序指标 ---
        labels = np.zeros(len(candidates))
        try:
            # 定位 Ground Truth
            gt_idx = candidates.tolist().index(target) if isinstance(candidates, np.ndarray) else candidates.index(target)
            labels[gt_idx] = 1
            # A. 计算 GAUC 分量
            g_list.append(roc_auc_score(labels, scores))
            # B. 计算命中位置
            rank_idx = np.argsort(-scores, kind='stable')
            rank_in = np.where(rank_idx == gt_idx)[0][0]
            # C. 记录各档位 HitRate 与 NDCG
            m_list.append(1.0 / (rank_in + 1))
            for k in hit_k_list: h_res[k].append(1.0 if rank_in < k else 0.0)
            if rank_in < ndcg_k: n_list.append(1.0 / math.log2(rank_in + 2))
            else: n_list.append(0.0)
        except: continue

    return g_list, h_res, n_list, m_list

def run_ranking_evaluation(model_path='best_deepfm_model.pth'):
    """
    精排模型全量性能报告生成主函数。
    """
    print(f"--- 精排深度评估启动 ---")
    print(f"待评权重: {model_path}")
    
    # 1. 准备全局元数据
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    # 默认路径：由 generate_ranking_exams.py 产生的 20% 采样测试考卷
    TEST_PATH = 'data/ranking_processed/test_exams_sampled_0.2.parquet'
    if not os.path.exists(TEST_PATH):
        print(f"错误：未找到测试考卷文件 {TEST_PATH}。")
        return
    test_df = pd.read_parquet(TEST_PATH)
    
    # 2. 配置模型特征空间词表
    feature_dims = {
        'user_idx': meta['user_count'], 'item_idx': meta['item_count'],
        'year_bucket': 12, 'item_rating_bucket': 12, 'item_count_bucket': 12,
        'user_rating_bucket': 12, 'user_count_bucket': 12,
        'hist_movie_ids': meta['item_count'], 'hist_genre_ids': meta['genre_count']
    }
    hit_k_list, ndcg_k = [10, 20, 50, 100], 50
    num_workers = max(1, os.cpu_count() - 1)
    
    print(f"启动 {num_workers} 个并行 worker 处理 {len(test_df)} 个测试考卷...")
    batches = np.array_split(test_df, num_workers)
    all_g, all_n, all_m = [], [], []
    all_h = {k: [] for k in hit_k_list}

    # 执行并行评估
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(evaluate_ranking_batch, b, model_path, feature_dims, meta, hit_k_list, ndcg_k) for b in batches]
        for f in tqdm(futures, desc="评估进度"):
            g, h, n, m = f.result()
            all_g.extend(g); all_n.extend(n); all_m.extend(m)
            for k in hit_k_list: all_h[k].extend(h[k])

    # 3. 产出最终性能看板
    print("\n" + "="*60)
    print(f"排序模型深度评估报告 (GAUC + NDCG@50 + HitRate)")
    print("="*60)
    print(f"GAUC (用户平均排序精度): {np.mean(all_g):.4f}")
    print(f"MRR (平均倒数排名):      {np.mean(all_m):.4f}")
    print(f"NDCG @ {ndcg_k}:             {np.mean(all_n):.4f}")
    print("-" * 40)
    for k in hit_k_list:
        print(f"Hit Rate @ {k:<3}:        {np.mean(all_h[k]):.2%}")
    print("="*60)
    print("注：该结果反映了模型在 350 个召回候选物品中精准定位『真命天子』的能力。")

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else "best_deepfm_model.pth"
    run_ranking_evaluation(model_file)
