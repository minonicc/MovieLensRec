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

# 加载 DIN 增强版模型
from .model_din import DIN_DCN_PPNet
from .dataset import RankingDataset, ranking_collate_fn

# --- 设备检测 ---
def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# --- 训练全局配置 (针对长序列优化) ---
# 由于 DIN 涉及 100 维的局部注意力矩阵运算，BatchSize 适当回调以防止显存溢出
BATCH_SIZE = 8192 if DEVICE.type == 'cuda' else 2048
EPOCHS = 10
LEARNING_RATE = 0.001
FEAT_SNAPSHOT_MAX = 200
DIN_HIST_LEN = 100 # 必须与 data_prep_din 保持一致

def evaluate_gauc_din(model, meta, device, exam_path='data/ranking_processed/val_exams_10k.parquet'):
    """
    针对 DIN 模型的 100 维验证集评估。
    """
    if not os.path.exists(exam_path): return 0.0
    model.eval()
    import pandas as pd
    val_df = pd.read_parquet(exam_path)
    
    user_sequences = meta['user_sequences']
    user_ratings_seq = meta.get('user_ratings_seq', {})
    movie_genres = meta['movie_genres']
    item_feat_map = meta['item_feat_map']
    user_feat_map = meta['user_feat_map']
    
    gauc_list = []
    # 每次打分 50 个用户以平衡速度与显存
    USER_BATCH = 50 

    with torch.no_grad():
        for i in range(0, len(val_df), USER_BATCH):
            batch_chunk = val_df.iloc[i : i + USER_BATCH]
            full_cat, full_num = defaultdict(list), defaultdict(list)
            full_seq = {'hist_movie_ids': [], 'hist_genre_ids': []}
            user_targets = []

            for _, row in batch_chunk.iterrows():
                u_idx, target, candidates, pos = int(row['user_idx']), int(row['item_idx']), row['candidates'], int(row['pos'])
                u_seq = user_sequences.get(u_idx, [])
                u_rts = user_ratings_seq.get(u_idx, [4.0]*len(u_seq))
                snap_it, snap_rt = u_seq[:pos], u_rts[:pos]
                
                # --- 核心：验证集也采用 100 长度的历史 ---
                hist_ids = (snap_it[-DIN_HIST_LEN:] + [0]*DIN_HIST_LEN)[:DIN_HIST_LEN]
                h_gs = []
                for h in hist_ids:
                    if h != 0: h_gs.extend(movie_genres.get(h, []))
                h_gs_trun = (h_gs[-100:] + [0]*100)[:100]
                u_stats = user_feat_map.get(u_idx, {})

                # 预计算组合拳
                h_200, r_200 = snap_it[-FEAT_SNAPSHOT_MAX:], snap_rt[-FEAT_SNAPSHOT_MAX:]
                g_counts, g_scores = defaultdict(int), defaultdict(float)
                for it, rt in zip(h_200, r_200):
                    for g in movie_genres.get(it, []): g_counts[g] += 1; g_scores[g] += rt
                total_g = sum(g_counts.values())
                r5_g = set()
                for it in snap_it[-5:]: r5_g.update(movie_genres.get(it, []))

                for c_id in candidates:
                    i_f = item_feat_map.get(c_id, {})
                    full_cat['user_idx'].append(u_idx); full_cat['item_idx'].append(c_id)
                    full_cat['year_bucket'].append(i_f.get('year_bucket', 0))
                    full_cat['item_rating_bucket'].append(i_f.get('rating_bucket', 0))
                    full_cat['item_count_bucket'].append(i_f.get('count_bucket', 0))
                    full_cat['user_rating_bucket'].append(u_stats.get('rating_bucket', 0))
                    full_cat['user_count_bucket'].append(u_stats.get('count_bucket', 0))
                    full_seq['hist_movie_ids'].append(hist_ids); full_seq['hist_genre_ids'].append(h_gs_trun)
                    
                    c_gs = movie_genres.get(c_id, [])
                    full_num['genre_density'].append(sum(g_counts[g] for g in c_gs)/(total_g+1e-8))
                    full_num['genre_recent_match'].append(1.0 if (set(c_gs) & r5_g) else 0.0)
                    bias_s = [g_scores[g]/(g_counts[g]+1e-8) for g in c_gs if g in g_counts]
                    full_num['genre_rating_bias'].append(np.mean(bias_s)/5.0 if bias_s else 0.6)
                
                try: user_targets.append(candidates.tolist().index(target))
                except: user_targets.append(-1)

            t_cat = {k: torch.tensor(v).to(device) for k, v in full_cat.items()}
            t_seq = {k: torch.tensor(v).to(device) for k, v in full_seq.items()}
            t_num = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in full_num.items()}
            batch_scores = torch.sigmoid(model(t_cat, t_seq, t_num)).cpu().numpy()

            for idx in range(len(batch_chunk)):
                gt_i = user_targets[idx]
                if gt_i == -1: continue
                u_scores = batch_scores[idx*350 : (idx+1)*350]
                labels = np.zeros(350); labels[gt_i] = 1
                gauc_list.append(roc_auc_score(labels, u_scores))
                
    return np.mean(gauc_list) if gauc_list else 0.5

def train_din():
    print(f"--- DIN+DCN+PPNet 综合实验启动 ---")
    print(f"训练设备: {DEVICE}")
    
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    with open('data/ranking_processed/ranking_user_pack.pkl', 'rb') as f:
        user_pack = pickle.load(f)
        meta['user_ratings_seq'] = user_pack['user_ratings_seq']

    feature_dims = {
        'user_idx': meta['user_count'], 'item_idx': meta['item_count'],
        'year_bucket': 12, 'item_rating_bucket': 12, 'item_count_bucket': 12,
        'user_rating_bucket': 12, 'user_count_bucket': 12,
        'hist_movie_ids': meta['item_count'], 'hist_genre_ids': meta['genre_count']
    }

    # 训练集加载 (使用 100 维长序列数据)
    train_dataset = RankingDataset('data/ranking_processed/train_din.parquet')
    num_workers = 8 if DEVICE.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=ranking_collate_fn, num_workers=num_workers, pin_memory=True)

    # 初始化最强战力模型
    model = DIN_DCN_PPNet(feature_dims, embed_dim=16).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    
    best_gauc = 0.0; patience = 3; wait = 0

    print(f"开始长序列迭代 (Batch={BATCH_SIZE})...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for i, (cat, seq, num, labels) in pbar:
            cat = {k: v.to(DEVICE, non_blocking=True) for k, v in cat.items()}
            seq = {k: v.to(DEVICE, non_blocking=True) for k, v in seq.items()}
            num = {k: v.to(DEVICE, non_blocking=True) for k, v in num.items()}
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            preds = model(cat, seq, num)
            loss = criterion(preds, labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
            if i % 20 == 0: pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"正在结算长序列 GAUC...")
        current_gauc = evaluate_gauc_din(model, meta, DEVICE)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val GAUC: {current_gauc:.4f}")

        if current_gauc > best_gauc:
            best_gauc = current_gauc; wait = 0
            torch.save(model.state_dict(), "best_din_model.pth")
            print(">>> DIN 个性化能力更优，模型已保存。")
        else:
            wait += 1
            if wait >= patience: break

if __name__ == "__main__":
    import pandas as pd
    train_din()
