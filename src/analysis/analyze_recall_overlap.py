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
from ..model import DualTowerModel
from collections import defaultdict

def get_mixed_history(full_seq, pos, recent_len=50, random_len=10):
    """
    构造混合历史特征：50个最近点击 + 10个远期随机点击。
    """
    pos = int(pos)
    history_all = full_seq[:pos]
    if not history_all: return [0] * (recent_len + random_len)
    recent = history_all[-recent_len:]; rem = history_all[:-recent_len]
    rand = random.sample(rem, min(len(rem), random_len)) if len(rem) > 0 else []
    combined = (recent + rand + [0]*60)[:60]
    return combined

def get_hist_genres(hist_ids, movie_genres, max_len=100):
    """
    展平用户历史题材。
    """
    genres = []
    for i in hist_ids:
        if i != 0: genres.extend(movie_genres.get(i, []))
    if len(genres) > max_len: genres = genres[:max_len]
    return genres + [0] * (max_len - len(genres))

def analyze_overlap(tower_model_path, k=100):
    """
    分析双塔召回与 ItemCF 召回的重合度及互补性。
    """
    device = torch.device("cpu")
    faiss.omp_set_num_threads(1)
    
    print(f"正在加载元数据与模型进行分析 (K={k})...")
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    with open('data/processed/item_cf_sim.pkl', 'rb') as f: item_sim_matrix = pickle.load(f)
    test_df = pd.read_csv('data/processed/test.csv')
    
    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count'])
    model.load_state_dict(torch.load(tower_model_path, map_location='cpu'))
    model.to(device); model.eval()

    # 1. 生成全量电影向量池
    print("导出双塔电影向量池...")
    item_count = meta['item_count']
    all_item_indices = torch.arange(item_count).to(device)
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
        for i in range(0, item_count, 4096):
            embs = model.forward_item(all_item_indices[i:i+4096], all_genres_t[i:i+4096], all_stats_t[i:i+4096])
            item_embs_list.append(embs.cpu().numpy().astype('float32'))
        item_pool_np = np.vstack(item_embs_list)

    index = faiss.IndexFlatIP(item_pool_np.shape[1])
    index.add(item_pool_np)

    # 2. 准备统计变量
    user_sequences = meta['user_sequences']
    user_feat_map = meta['user_feat_map']
    movie_genres = meta['movie_genres']
    
    total_users = len(test_df)
    overlap_ratios = [] # 记录每个用户的候选集重合度
    
    hits_tower = set() # 双塔命中的用户 ID
    hits_itemcf = set() # ItemCF 命中的用户 ID

    print(f"正在分析 {total_users} 个用户的召回重合度...")
    u_batch_size = 512
    with torch.no_grad():
        for i in tqdm(range(0, total_users, u_batch_size)):
            batch_df = test_df.iloc[i : i + u_batch_size]
            u_ids = batch_df['user_idx'].values
            targets = batch_df['item_idx'].values
            
            # --- A. 获取双塔召回结果 ---
            batch_u_vecs = []
            for idx_in_batch in range(len(u_ids)):
                u_idx = u_ids[idx_in_batch]
                pos = batch_df.iloc[idx_in_batch]['pos']
                random.seed(u_idx) # 保证可复现
                u_hist = get_mixed_history(user_sequences.get(u_idx, []), pos)
                u_hgenre = get_hist_genres(u_hist, movie_genres)
                f = user_feat_map.get(u_idx, {})
                u_stat = [f.get('rating_bucket', 0), f.get('count_bucket', 0)]
                u_vec = model.forward_user(torch.tensor([u_idx]), torch.tensor([u_hist]), 
                                         torch.tensor([u_hgenre]), torch.tensor([u_stat]))
                batch_u_vecs.append(u_vec.cpu().numpy().astype('float32'))
            
            _, tower_topk_all = index.search(np.vstack(batch_u_vecs), k)
            
            # --- B. 逐用户计算 ItemCF 并对比 ---
            for j in range(len(targets)):
                u_idx = u_ids[j]
                target = targets[j]
                random.seed(u_idx)
                
                # 获取历史用于过滤
                current_pos = int(batch_df.iloc[j]['pos'])
                history_list = user_sequences.get(u_idx, [])[:current_pos]
                history_set = set(history_list)
                
                # 1. 计算 ItemCF 召回
                item_cf_rank = defaultdict(float)
                for h_item in history_list:
                    if h_item in item_sim_matrix:
                        for neighbor, score in item_sim_matrix[h_item]:
                            if neighbor not in history_set:
                                item_cf_rank[neighbor] += score
                item_cf_topk = [x[0] for x in sorted(item_cf_rank.items(), key=lambda x: (-x[1], x[0]))[:k]]
                
                # 2. 获取双塔召回
                tower_topk = tower_topk_all[j].tolist()
                
                # 3. 计算候选集重合度
                set_tower = set(tower_topk)
                set_itemcf = set(item_cf_topk)
                intersection = set_tower.intersection(set_itemcf)
                overlap_ratios.append(len(intersection) / k)
                
                # 4. 记录命中情况
                if target in set_tower: hits_tower.add(u_idx)
                if target in set_itemcf: hits_itemcf.add(u_idx)

    # 3. 输出分析报告
    print("" + "="*50)
    print(f"两路召回互补性深度分析报告 (K={k})")
    print("="*50)
    
    # A. 候选集重合度
    avg_overlap = np.mean(overlap_ratios)
    print(f"平均候选集重合度: {avg_overlap:.2%} (即平均有 {avg_overlap*k:.1f} 个物品相同)")
    
    # B. 命中用户分析 (维恩图逻辑)
    both_hit = hits_tower.intersection(hits_itemcf)
    only_tower = hits_tower - hits_itemcf
    only_itemcf = hits_itemcf - hits_tower
    either_hit = hits_tower.union(hits_itemcf)
    
    print(f"双塔路 Hit 用户数: {len(hits_tower)} (HR: {len(hits_tower)/total_users:.4f})")
    print(f"ItemCF路 Hit 用户数: {len(hits_itemcf)} (HR: {len(hits_itemcf)/total_users:.4f})")
    
    print(f"--- 命中用户分布 (Venn Stats) ---")
    print(f"两路同时命中用户数: {len(both_hit)} (占总命中 {len(both_hit)/len(either_hit):.2%})")
    print(f"仅被双塔命中的用户: {len(only_tower)} (占总命中 {len(only_tower)/len(either_hit):.2%})")
    print(f"仅被ItemCF命中的用户: {len(only_itemcf)} (占总命中 {len(only_itemcf)/len(either_hit):.2%})")
    print(f"两路合并后总命中用户: {len(either_hit)} (融合后 HR: {len(either_hit)/total_users:.4f})")
    
    # C. 增量贡献
    print(f"ItemCF 对双塔的纯增量贡献: +{len(only_itemcf)} 人 (相对提升: {len(only_itemcf)/len(hits_tower):.2%})")
    print("="*50)

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else "best_tower_model.pth"
    analyze_overlap(model_file)
