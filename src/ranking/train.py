import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict

from .model import DeepFM
from .dataset import RankingDataset, ranking_collate_fn

# --- 设备自动检测与路由逻辑 ---
def get_device():
    """
    智能选择计算设备。本地 MacBook Air 优先使用 mps。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# --- 训练全局配置项 (本地优化版) ---
# 本地 Mac 建议维持在 4096，确保系统响应流畅
BATCH_SIZE = 4096
EPOCHS = 10
LEARNING_RATE = 0.001

def evaluate_gauc(model, meta, device, exam_path='data/ranking_processed/val_exams_10k.parquet'):
    """
    高性能批量 GAUC 计算：通过将多个用户候选集拼接为大批次进行推理，极大减少调用开销。
    """
    if not os.path.exists(exam_path):
        return 0.0
    
    model.eval()
    import pandas as pd
    val_df = pd.read_parquet(exam_path)
    
    # 提取索引
    user_sequences = meta['user_sequences']
    movie_genres = meta['movie_genres']
    item_feat_map = meta['item_feat_map']
    user_feat_map = meta['user_feat_map']
    user_ratings_seq = meta.get('user_ratings_seq', {}) 
    
    gauc_list = []
    
    # --- 批量化配置 ---
    # 每批处理 100 个用户，即单次推理 35,000 个样本
    USER_BATCH_SIZE = 100 
    
    for i in range(0, len(val_df), USER_BATCH_SIZE):
        batch_chunk = val_df.iloc[i : i + USER_BATCH_SIZE]
        
        # 存储当前 Batch 所有用户的所有候选特征
        full_cand_cat = defaultdict(list)
        full_cand_seq = {'hist_movie_ids': [], 'hist_genre_ids': []}
        full_cand_num = {'genre_density': [], 'genre_recent_match': [], 'genre_rating_bias': []}
        
        user_targets = [] # 记录该 batch 内各用户的 GT 索引
        
        for _, row in batch_chunk.iterrows():
            u_idx, target_item, candidates, pos = int(row['user_idx']), int(row['item_idx']), row['candidates'], int(row['pos'])
            
            # 1. 构造快照 (与训练逻辑对齐)
            u_full_seq = user_sequences.get(u_idx, [])
            u_full_ratings = user_ratings_seq.get(u_idx, [4.0]*len(u_full_seq))
            snap_items = u_full_seq[:pos]; snap_ratings = u_full_ratings[:pos]
            
            hist_ids = (snap_items[-50:] + [0]*50)[:50]
            hist_gs = []
            for h in hist_ids:
                if h != 0: hist_gs.extend(movie_genres.get(h, []))
            hist_gs_truncated = (hist_gs[-100:] + [0]*100)[:100]
            u_stats = user_feat_map.get(u_idx, {})
            
            # 2. 统计当前用户题材分布
            recent_200 = snap_items[-200:]
            g_counts = defaultdict(int); g_total_score = defaultdict(float)
            for it, rt in zip(recent_200, snap_ratings[-200:]):
                for g in movie_genres.get(it, []): g_counts[g] += 1; g_total_score[g] += rt
            total_g = sum(g_counts.values())
            
            r5_g = set()
            for it in snap_items[-5:]: r5_g.update(movie_genres.get(it, []))

            # 3. 展开 350 个候选人
            for c_id in candidates:
                i_f = item_feat_map.get(c_id, {})
                full_cand_cat['user_idx'].append(u_idx); full_cand_cat['item_idx'].append(c_id)
                full_cand_cat['year_bucket'].append(i_f.get('year_bucket', 0))
                full_cand_cat['item_rating_bucket'].append(i_f.get('rating_bucket', 0))
                full_cand_cat['item_count_bucket'].append(i_f.get('count_bucket', 0))
                full_cand_cat['user_rating_bucket'].append(u_stats.get('rating_bucket', 0))
                full_cand_cat['user_count_bucket'].append(u_stats.get('count_bucket', 0))
                full_cand_seq['hist_movie_ids'].append(hist_ids); full_cand_seq['hist_genre_ids'].append(hist_gs_truncated)
                
                # 数值组合拳
                c_gs = movie_genres.get(c_id, [])
                full_cand_num['genre_density'].append(sum(g_counts[g] for g in c_gs)/(total_g+1e-8))
                full_cand_num['genre_recent_match'].append(1.0 if (set(c_gs) & r5_g) else 0.0)
                scores = [g_total_score[g]/(g_counts[g]+1e-8) for g in c_gs if g in g_counts]
                full_cand_num['genre_rating_bias'].append(np.mean(scores)/5.0 if scores else 0.6)
            
            # 记录 GT 索引以供后续分段计算 AUC
            try:
                gt_idx = candidates.tolist().index(target_item) if isinstance(candidates, np.ndarray) else candidates.index(target_item)
                user_targets.append(gt_idx)
            except:
                user_targets.append(-1) # 异常标记

        # --- 批量推理 ---
        with torch.no_grad():
            t_cat = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in full_cand_cat.items()}
            t_seq = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in full_cand_seq.items()}
            t_num = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in full_cand_num.items()}
            
            # 暴力推理 [USER_BATCH_SIZE * 350]
            batch_scores = torch.sigmoid(model(t_cat, t_seq, t_num)).cpu().numpy()
            
        # --- 分段还原计算每个用户的 AUC ---
        for u_batch_idx in range(len(batch_chunk)):
            gt_i = user_targets[u_batch_idx]
            if gt_i == -1: continue
            
            # 取出当前用户的 350 个预测分
            start, end = u_batch_idx * 350, (u_batch_idx + 1) * 350
            u_scores = batch_scores[start:end]
            
            labels = np.zeros(350)
            labels[gt_i] = 1
            gauc_list.append(roc_auc_score(labels, u_scores))
                
    return np.mean(gauc_list) if gauc_list else 0.5

def train_ranking():
    """
    排序模型训练主循环。
    """
    print(f"--- 排序阶段训练启动 (本地 M系列芯片优化版) ---")
    print(f"当前运行设备: {DEVICE}")
    
    with open('data/processed/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    feature_dims = {
        'user_idx': meta['user_count'], 'item_idx': meta['item_count'],
        'year_bucket': 12, 'item_rating_bucket': 12, 'item_count_bucket': 12,
        'user_rating_bucket': 12, 'user_count_bucket': 12,
        'hist_movie_ids': meta['item_count'], 'hist_genre_ids': meta['genre_count']
    }

    print("加载 Parquet 训练集...")
    train_dataset = RankingDataset('data/ranking_processed/train.parquet')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=ranking_collate_fn, num_workers=0)

    model = DeepFM(feature_dims, embed_dim=16).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_gauc = 0.0; patience = 3; wait = 0

    print(f"开始训练 (BatchSize={BATCH_SIZE}, Epochs={EPOCHS})...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for i, (cat, seq, num, labels) in pbar:
            cat = {k: v.to(DEVICE) for k, v in cat.items()}
            seq = {k: v.to(DEVICE) for k, v in seq.items()}
            num = {k: v.to(DEVICE) for k, v in num.items()}
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(cat, seq, num)
            loss = criterion(preds, labels)
            loss.backward(); optimizer.step()
            
            total_loss += loss.item()
            if i % 100 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- 验证阶段：高性能批量评估 ---
        print(f"正在进行轮次结算：计算 1万用户批量 GAUC...")
        current_gauc = evaluate_gauc(model, meta, DEVICE)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 结束 | 平均 Loss: {avg_loss:.4f} | 验证集 GAUC: {current_gauc:.4f}")

        if current_gauc > best_gauc:
            best_gauc = current_gauc; wait = 0
            torch.save(model.state_dict(), "best_deepfm_model.pth")
            print(">>> 排序性能突破，已保存权重。")
        else:
            wait += 1
            if wait >= patience:
                print(f"GAUC 连续 {patience} 代不再增长，自动触发早停。"); break

if __name__ == "__main__":
    import pandas as pd
    train_ranking()
