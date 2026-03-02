import os
# --- 环境冲突修复 ---
# 解决 macOS 上多个库竞争导致的 libomp 重复初始化崩溃问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import faiss
import random
from tqdm import tqdm
from src.model import DualTowerModel
from collections import defaultdict

def get_mixed_history(full_seq, pos, recent_len=50, random_len=10):
    """
    构造混合历史特征：50个最近点击 + 10个远期随机点击。
    必须与训练逻辑严格对齐，确保特征一致性。
    """
    pos = int(pos)
    history_all = full_seq[:pos]
    if not history_all: return [0] * (recent_len + random_len)
    
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
    将用户历史序列中所有电影涉及到的题材进行展平聚合。
    """
    genres = []
    for i in hist_ids:
        if i != 0: genres.extend(movie_genres.get(i, []))
    if len(genres) > max_len: genres = genres[:max_len]
    return genres + [0] * (max_len - len(genres))

def evaluate_fusion(tower_model_path, k_list=[10, 50, 100, 150, 200]):
    """
    双路召回融合评估：将双塔模型召回与 ItemCF 召回结果进行去重合并。
    采用交替填充（Interleaving）策略平衡两路贡献，并加入逐用户种子保证可复现性。
    """
    # 强制回归 CPU 模式以确保在 Mac M系列芯片上的绝对稳定性
    device = torch.device("cpu")
    faiss.omp_set_num_threads(1)
    
    print(f"正在加载双塔模型: {tower_model_path}")
    # 1. 加载元数据、ItemCF 相似度矩阵和测试集索引
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    with open('data/processed/item_cf_sim.pkl', 'rb') as f: item_sim_matrix = pickle.load(f)
    test_df = pd.read_csv('data/processed/test.csv')
    
    # 初始化双塔模型结构并加载训练好的权重
    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count'])
    model.load_state_dict(torch.load(tower_model_path, map_location='cpu'))
    model.to(device); model.eval()

    # 2. 生成全量电影向量池 (Item Vector Pool)
    # 将库中 8.7 万部电影一次性转为向量，方便后续进行大规模向量检索
    print("生成全量电影向量池...")
    item_count = meta['item_count']
    all_item_indices = torch.arange(item_count).to(device)
    
    # 准备题材特征与侧向统计特征桶
    all_genres, all_stats = [], []
    for i in range(item_count):
        g = meta['movie_genres'].get(i, [])
        all_genres.append((g[:6] + [0]*6)[:6])
        f = meta['item_feat_map'].get(i, {})
        all_stats.append([f.get('year_bucket', 0), f.get('rating_bucket', 0), f.get('count_bucket', 0)])
    
    all_genres_t = torch.tensor(all_genres).to(device)
    all_stats_t = torch.tensor(all_stats).to(device)

    with torch.no_grad():
        item_embs_list = []
        batch_size = 4096
        for i in range(0, item_count, batch_size):
            embs = model.forward_item(all_item_indices[i:i+4096], all_genres_t[i:i+4096], all_stats_t[i:i+4096])
            item_embs_list.append(embs.cpu().numpy().astype('float32'))
        item_pool_np = np.vstack(item_embs_list)

    # 构建 Faiss 索引（内积模式，对应余弦相似度检索）
    index = faiss.IndexFlatIP(item_pool_np.shape[1])
    index.add(item_pool_np)

    # 3. 准备融合评估相关元数据
    user_sequences = meta['user_sequences']
    user_feat_map = meta['user_feat_map']
    movie_genres = meta['movie_genres']
    
    fusion_hits = {k: 0 for k in k_list}
    total = len(test_df)
    RECALL_EACH = 100 # 每路召回的基础数量

    print(f"执行双路召回融合评估 (测试用户: {total})...")
    u_batch_size = 512
    with torch.no_grad():
        for i in tqdm(range(0, total, u_batch_size)):
            batch_df = test_df.iloc[i : i + u_batch_size]
            u_ids = batch_df['user_idx'].values
            targets = batch_df['item_idx'].values
            
            # --- A. 双塔路召回：执行向量检索 ---
            batch_u_vecs = []
            for idx_in_batch in range(len(u_ids)):
                u_idx = u_ids[idx_in_batch]
                pos = batch_df.iloc[idx_in_batch]['pos']
                
                # 设置逐用户种子：确保混合历史构造（那 10 个随机物品）完全可复现
                random.seed(u_idx)
                np.random.seed(u_idx)
                
                # 构造并对齐特征
                u_hist = get_mixed_history(user_sequences.get(u_idx, []), pos)
                u_hgenre = get_hist_genres(u_hist, movie_genres)
                f = user_feat_map.get(u_idx, {})
                u_stat = [f.get('rating_bucket', 0), f.get('count_bucket', 0)]
                
                # 生成单个用户向量
                u_vec = model.forward_user(torch.tensor([u_idx]), torch.tensor([u_hist]), 
                                         torch.tensor([u_hgenre]), torch.tensor([u_stat]))
                batch_u_vecs.append(u_vec.cpu().numpy().astype('float32'))
            
            u_vecs_np = np.vstack(batch_u_vecs)
            _, tower_topk_all = index.search(u_vecs_np, RECALL_EACH)
            
            # --- B. 融合计算 ---
            for j in range(len(targets)):
                u_idx = u_ids[j]
                target = targets[j]
                # 重新设置种子以确保融合阶段的排序一致性（应对同分情况）
                random.seed(u_idx)
                np.random.seed(u_idx)
                
                current_pos = int(batch_df.iloc[j]['pos'])
                user_hist_list = user_sequences.get(u_idx, [])[:current_pos]
                history_set = set(user_hist_list)
                
                # 1. 提取 ItemCF 候选：基于全量历史触发，并执行稳定排序
                item_cf_rank = defaultdict(float)
                for h_item in user_hist_list:
                    if h_item in item_sim_matrix:
                        for neighbor, score in item_sim_matrix[h_item]:
                            if neighbor not in history_set:
                                item_cf_rank[neighbor] += score
                # 使用 (-score, item_id) 双重排序解决平票波动
                item_cf_topk = [x[0] for x in sorted(item_cf_rank.items(), key=lambda x: (-x[1], x[0]))[:RECALL_EACH]]
                
                # 2. 提取双塔候选结果
                tower_topk = tower_topk_all[j].tolist()
                
                # 3. 融合策略：交替填充 + 去重合并 (Interleaving)
                combined_recall = []
                seen_items = set()
                for idx in range(RECALL_EACH):
                    if idx < len(tower_topk):
                        item_t = tower_topk[idx]
                        if item_t not in seen_items:
                            combined_recall.append(item_t); seen_items.add(item_t)
                    if idx < len(item_cf_topk):
                        item_c = item_cf_topk[idx]
                        if item_c not in seen_items:
                            combined_recall.append(item_c); seen_items.add(item_c)
                
                # 4. 统计不同 K 档位下的命中率
                for k in k_list:
                    if target in combined_recall[:k]: fusion_hits[k] += 1

    # 5. 输出融合评估报告
    print("\n" + "="*50)
    print(f"双路召回融合评估报告 (各路取 {RECALL_EACH} 后交替合并, 可复现版)")
    print("="*50)
    for k in k_list:
        hr = fusion_hits[k] / total
        print(f"Fusion Top-{k:<3} | Hit Rate: {hr:.4f}")
    print("="*50)

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else "best_tower_model.pth"
    evaluate_fusion(model_file)
