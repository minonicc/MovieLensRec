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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score

# 动态加载所有待评精排模型定义
from .model import DeepFM
from .model_dcn import DCNv2

# 特征快照计算的最大截断长度 (必须与 data_prep 保持严格一致)
FEAT_SNAPSHOT_MAX = 200

def calculate_genre_punch_eval(target_item_genres, snap_items, item_idx2genres, snap_ratings):
    """
    评估阶段特征工厂：基于当前快照实时计算题材组合拳数值。
    利用真实的评分序列，衡量用户对当前电影题材的匹配度、即时热度与评价偏好。
    """
    if not target_item_genres or not snap_items:
        return 0.0, 0.0, 0.6
    
    # 截取该时刻最新的行为流水片段
    h_it = snap_items[-FEAT_SNAPSHOT_MAX:]
    h_rt = snap_ratings[-FEAT_SNAPSHOT_MAX:]
    
    # 统计题材分布与累计分值
    g_counts, g_scores = defaultdict(int), defaultdict(float)
    for it, rt in zip(h_it, h_rt):
        for g in item_idx2genres.get(it, []): 
            g_counts[g] += 1; g_scores[g] += rt
            
    # 计算三个维度的偏好分数
    total_g = sum(g_counts.values())
    # 1. 题材浓度 (Density)
    density = sum(g_counts[g] for g in target_item_genres) / (total_g + 1e-8)
    
    # 2. 最近匹配 (Recent Match)：判断是否命中最近 5 个好评记录
    r5_g = set()
    for it in snap_items[-5:]: r5_g.update(item_idx2genres.get(it, []))
    recent_match = 1.0 if (set(target_item_genres) & r5_g) else 0.0
    
    # 3. 评分偏好 (Rating Bias)：历史评价均值
    bias_scores = [g_scores[g] / (g_counts[g] + 1e-8) for g in target_item_genres if g in g_counts]
    rating_bias = np.mean(bias_scores) / 5.0 if bias_scores else 0.6
    
    return density, recent_match, rating_bias

def evaluate_ranking_batch(batch_df, model_path, feature_dims, meta, hit_k_list, ndcg_k, model_type='DeepFM'):
    """
    子进程评估单元：执行 350 个候选物品的打分与指标计算。
    支持自动识别 DeepFM 或 DCNv2 模型架构。
    """
    # 评估阶段推荐使用 CPU 模式以换取更高的样本吞吐量，并规避多进程下的显存竞争
    device = torch.device("cpu")
    
    # 根据指定的模型类型进行结构初始化
    if model_type == 'DCNv2':
        model = DCNv2(feature_dims, embed_dim=16, cross_layers=2)
    else:
        model = DeepFM(feature_dims, embed_dim=16)
        
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 提取多进程共享的基础索引
    item_idx2genres = meta['movie_genres']
    item_feat_map = meta['item_feat_map']
    user_feat_map = meta['user_feat_map']
    user_sequences = meta['user_sequences']
    user_ratings_seq = meta['user_ratings_seq']
    
    g_list, n_list, m_list = [], [], []
    h_res = {k: [] for k in hit_k_list}

    for _, row in batch_df.iterrows():
        u_idx, target, candidates, pos = int(row['user_idx']), int(row['item_idx']), row['candidates'], int(row['pos'])
        
        # --- 1. 严格恢复行为快照 (防止特征穿越) ---
        u_full_seq = user_sequences.get(u_idx, [])
        u_full_ratings = user_ratings_seq.get(u_idx, [4.0]*len(u_full_seq))
        
        snap_items = u_full_seq[:pos]
        snap_ratings = u_full_ratings[:pos]
        
        # 构造序列特征
        hist_ids = (snap_items[-50:] + [0]*50)[:50]
        hist_gs = []
        for h in hist_ids:
            if h != 0: hist_gs.extend(item_idx2genres.get(h, []))
        h_gs_trun = (hist_gs[-100:] + [0]*100)[:100]
        hist_gs_set = set(hist_gs)
        u_stats = user_feat_map.get(u_idx, {})

        # --- 2. 批量构造 350 个 Candidate 的特征矩阵 ---
        c_cat = defaultdict(list)
        c_seq = {'hist_movie_ids': [], 'hist_genre_ids': []}
        c_num = {'genre_density': [], 'genre_recent_match': [], 'genre_rating_bias': []}

        for c_id in candidates:
            i_f = item_feat_map.get(c_id, {})
            # ID 与分桶
            c_cat['user_idx'].append(u_idx); c_cat['item_idx'].append(c_id)
            c_cat['year_bucket'].append(i_f.get('year_bucket', 0))
            c_cat['item_rating_bucket'].append(i_f.get('rating_bucket', 0))
            c_cat['item_count_bucket'].append(i_f.get('count_bucket', 0))
            c_cat['user_rating_bucket'].append(u_stats.get('rating_bucket', 0))
            c_cat['user_count_bucket'].append(u_stats.get('count_bucket', 0))
            # 序列
            c_seq['hist_movie_ids'].append(hist_ids); c_seq['hist_genre_ids'].append(h_gs_trun)
            # 组合拳
            d, rm, rb = calculate_genre_punch_eval(item_idx2genres.get(c_id, []), snap_items, item_idx2genres, snap_ratings)
            c_num['genre_density'].append(d); c_num['genre_recent_match'].append(rm); c_num['genre_rating_bias'].append(rb)

        # --- 3. 执行推理与概率转化 ---
        with torch.no_grad():
            t_cat = {k: torch.tensor(v, dtype=torch.long) for k, v in c_cat.items()}
            t_seq = {k: torch.tensor(v, dtype=torch.long) for k, v in c_seq.items()}
            t_num = {k: torch.tensor(v, dtype=torch.float32) for k, v in c_num.items()}
            scores = torch.sigmoid(model(t_cat, t_seq, t_num)).numpy()

        # --- 4. 结算全维度精排指标 ---
        labels = np.zeros(len(candidates))
        try:
            # 确定 Ground Truth 位置
            gt_idx = candidates.tolist().index(target) if isinstance(candidates, np.ndarray) else candidates.index(target)
            labels[gt_idx] = 1
            # GAUC: 分用户评估区分度
            g_list.append(roc_auc_score(labels, scores))
            # 排序排位计算
            rank_idx = np.argsort(-scores, kind='stable')
            rank_in = np.where(rank_idx == gt_idx)[0][0]
            # MRR 与各档位 HitRate
            m_list.append(1.0 / (rank_in + 1))
            for k in hit_k_list: h_res[k].append(1.0 if rank_in < k else 0.0)
            # NDCG@50
            if rank_in < ndcg_k: n_list.append(1.0 / math.log2(rank_in + 2))
            else: n_list.append(0.0)
        except: continue

    return g_list, h_res, n_list, m_list

def run_ranking_evaluation(model_path='best_deepfm_model.pth'):
    """
    通用精排性能评估入口：支持多模型对比。
    """
    # 根据文件名关键词自动路由模型类型
    model_type = 'DCNv2' if 'dcn' in model_path.lower() else 'DeepFM'
    print(f"--- 精排深度评估启动 | 目标模型: {model_type} ---")
    
    # 1. 加载元数据并注入真实评分序列
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    with open('data/ranking_processed/ranking_user_pack.pkl', 'rb') as f:
        user_pack = pickle.load(f)
        meta['user_ratings_seq'] = user_pack['user_ratings_seq']

    # 路径配置：读取由 generate_ranking_exams.py 产生的采样版测试考卷
    TEST_PATH = 'data/ranking_processed/test_exams_sampled_0.2.parquet'
    if not os.path.exists(TEST_PATH):
        print(f"错误：考卷文件 {TEST_PATH} 不存在。")
        return
    test_df = pd.read_parquet(TEST_PATH)
    
    # 配置模型特征空间
    feature_dims = {
        'user_idx': meta['user_count'], 'item_idx': meta['item_count'],
        'year_bucket': 12, 'item_rating_bucket': 12, 'item_count_bucket': 12,
        'user_rating_bucket': 12, 'user_count_bucket': 12,
        'hist_movie_ids': meta['item_count'], 'hist_genre_ids': meta['genre_count']
    }
    hit_k_list, ndcg_k = [10, 20, 50, 100], 50
    num_workers = max(1, os.cpu_count() - 1)
    
    print(f"启动 {num_workers} 个并行 Worker 处理 {len(test_df)} 个测试用户...")
    batches = np.array_split(test_df, num_workers)
    all_g, all_n, all_m = [], [], []
    all_h = {k: [] for k in hit_k_list}

    # 执行并行指标结算
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(evaluate_ranking_batch, b, model_path, feature_dims, meta, hit_k_list, ndcg_k, model_type) for b in batches]
        for f in tqdm(futures, desc="指标采集进度"):
            g, h, n, m = f.result()
            all_g.extend(g); all_n.extend(n); all_m.extend(m)
            for k in hit_k_list: all_h[k].extend(h[k])

    # 2. 产出正式对比报告
    print("\n" + "="*60)
    print(f"【{model_type}】精排模型性能报告")
    print("="*60)
    print(f"GAUC (用户平均排序精度): {np.mean(all_g):.4f}")
    print(f"MRR (平均倒数排名):      {np.mean(all_m):.4f}")
    print(f"NDCG @ {ndcg_k}:             {np.mean(all_n):.4f}")
    print("-" * 40)
    for k in hit_k_list:
        print(f"Hit Rate @ {k:<3}:        {np.mean(all_h[k]):.2%}")
    print("="*60)
    print("注：该结果衡量的是模型在 350 个候选电影中精准锁定 Ground Truth 的能力。")

if __name__ == "__main__":
    import sys
    # 允许通过命令行指定不同权重的路径
    model_file = sys.argv[1] if len(sys.argv) > 1 else "best_deepfm_model.pth"
    run_ranking_evaluation(model_file)
