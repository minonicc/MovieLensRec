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
    智能选择当前计算能力最强的硬件设备。
    逻辑：NVIDIA GPU (CUDA) > Mac GPU (MPS) > CPU。
    确保代码在服务器与本地环境切换时无需手动改动。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# 全局训练设备变量
DEVICE = get_device()

# --- 训练全局配置项 ---
# Batch Size 设为 4096 以充分平衡 GPU 利用率与训练稳定性
BATCH_SIZE = 4096
# 最大训练轮数
EPOCHS = 10
# 优化器初始学习率
LEARNING_RATE = 0.001
# 特征统计时的快照截断长度，需与数据预处理阶段保持严格一致
FEAT_SNAPSHOT_MAX = 200

def evaluate_gauc(model, meta, device, exam_path='data/ranking_processed/val_exams_10k.parquet'):
    """
    在验证集考卷上计算 GAUC (Group AUC)。
    GAUC 衡量模型在单个用户内部进行正确排序的能力，是精排模型最重要的离线性能预测指标。
    本实现采用了『批量用户合并推理』优化，极大减少了 MPS/CUDA 内核调度开销。
    """
    if not os.path.exists(exam_path):
        print(f"警告：考卷文件 {exam_path} 不存在，跳过验证环节。")
        return 0.0
    
    model.eval()
    import pandas as pd
    val_df = pd.read_parquet(exam_path)
    
    # 提取评估所需的元数据索引
    user_sequences = meta['user_sequences']
    user_ratings_seq = meta.get('user_ratings_seq', {})
    movie_genres = meta['movie_genres']
    item_feat_map = meta['item_feat_map']
    user_feat_map = meta['user_feat_map']
    
    gauc_list = []
    # 批量化配置：每次合并 100 个用户进行推理，即单次打分 35,000 个物品
    USER_BATCH = 100 

    with torch.no_grad():
        for i in range(0, len(val_df), USER_BATCH):
            batch_chunk = val_df.iloc[i : i + USER_BATCH]
            
            # 存储当前 Batch 所有用户的所有候选特征矩阵
            full_cat = defaultdict(list)
            full_seq = {'hist_movie_ids': [], 'hist_genre_ids': []}
            full_num = {'genre_density': [], 'genre_recent_match': [], 'genre_rating_bias': []}
            user_targets = [] # 记录该 batch 内各用户的 GT 在其 350 个候选人中的位置

            for _, row in batch_chunk.iterrows():
                u_idx, target, candidates, pos = int(row['user_idx']), int(row['item_idx']), row['candidates'], int(row['pos'])
                
                # --- 1. 恢复行为快照 (严禁穿越) ---
                u_seq = user_sequences.get(u_idx, [])
                u_rts = user_ratings_seq.get(u_idx, [4.0]*len(u_seq))
                # 截取当前时刻之前的历史流水
                snap_items = u_seq[:pos]
                snap_rts = u_rts[:pos]
                
                # 构造序列特征
                hist_ids = (snap_items[-50:] + [0]*50)[:50]
                h_gs = []
                for h in hist_ids:
                    if h != 0: h_gs.extend(movie_genres.get(h, []))
                h_gs_trun = (h_gs[-100:] + [0]*100)[:100]
                u_stats = user_feat_map.get(u_idx, {})

                # 预计算当前用户题材分布 (用于组合拳特征)
                h_200 = snap_items[-FEAT_SNAPSHOT_MAX:]
                r_200 = snap_rts[-FEAT_SNAPSHOT_MAX:]
                g_counts, g_scores = defaultdict(int), defaultdict(float)
                for it, rt in zip(h_200, r_200):
                    for g in movie_genres.get(it, []): 
                        g_counts[g] += 1; g_scores[g] += rt
                total_g = sum(g_counts.values())
                
                r5_g = set()
                for it in snap_items[-5:]: r5_g.update(movie_genres.get(it, []))

                # 展开 350 个候选人进行特征封装
                for c_id in candidates:
                    i_f = item_feat_map.get(c_id, {})
                    full_cat['user_idx'].append(u_idx); full_cat['item_idx'].append(c_id)
                    full_cat['year_bucket'].append(i_f.get('year_bucket', 0))
                    full_cat['item_rating_bucket'].append(i_f.get('rating_bucket', 0))
                    full_cat['item_count_bucket'].append(i_f.get('count_bucket', 0))
                    full_cat['user_rating_bucket'].append(u_stats.get('rating_bucket', 0))
                    full_cat['user_count_bucket'].append(u_stats.get('count_bucket', 0))
                    full_seq['hist_movie_ids'].append(hist_ids)
                    full_seq['hist_genre_ids'].append(h_gs_trun)
                    
                    # 组合拳计算：密度、匹配、偏好
                    c_gs = movie_genres.get(c_id, [])
                    full_num['genre_density'].append(sum(g_counts[g] for g in c_gs)/(total_g+1e-8))
                    full_num['genre_recent_match'].append(1.0 if (set(c_gs) & r5_g) else 0.0)
                    bias_s = [g_scores[g]/(g_counts[g]+1e-8) for g in c_gs if g in g_counts]
                    full_num['genre_rating_bias'].append(np.mean(bias_s)/5.0 if bias_s else 0.6)
                
                # 记录 Ground Truth 索引
                try:
                    gt_idx = candidates.tolist().index(target) if isinstance(candidates, np.ndarray) else candidates.index(target)
                    user_targets.append(gt_idx)
                except: user_targets.append(-1)

            # --- 2. 批量执行模型推理 ---
            t_cat = {k: torch.tensor(v).to(device) for k, v in full_cat.items()}
            t_seq = {k: torch.tensor(v).to(device) for k, v in full_seq.items()}
            t_num = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in full_num.items()}
            # 推理获得概率分
            batch_scores = torch.sigmoid(model(t_cat, t_seq, t_num)).cpu().numpy()

            # --- 3. 结果还原并计算分用户 AUC ---
            for idx_in_batch in range(len(batch_chunk)):
                gt_i = user_targets[idx_in_batch]
                if gt_i == -1: continue
                # 截取属于当前用户的 350 个预测结果
                u_scores = batch_scores[idx_in_batch*350 : (idx_in_batch+1)*350]
                labels = np.zeros(350); labels[gt_i] = 1
                gauc_list.append(roc_auc_score(labels, u_scores))
                
    return np.mean(gauc_list) if gauc_list else 0.5

def train_ranking():
    """
    排序模型训练主循环：实现 DeepFM 的高性能训练与实时 GAUC 监控。
    """
    print(f"--- 排序阶段训练启动 ---")
    print(f"当前活跃设备: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    
    # 1. 加载元数据并注入真实评分序列
    print("加载元数据并对齐特征空间...")
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    # 重要：从黄金用户包中获取评分，激活评分偏好特征
    with open('data/ranking_processed/ranking_user_pack.pkl', 'rb') as f:
        user_pack = pickle.load(f)
        meta['user_ratings_seq'] = user_pack['user_ratings_seq']

    # 配置离散特征的 Embedding 空间维度
    feature_dims = {
        'user_idx': meta['user_count'], 'item_idx': meta['item_count'],
        'year_bucket': 12, 'item_rating_bucket': 12, 'item_count_bucket': 12,
        'user_rating_bucket': 12, 'user_count_bucket': 12,
        'hist_movie_ids': meta['item_count'], 'hist_genre_ids': meta['genre_count']
    }

    # 2. 挂载数据加载器
    print("挂载 Parquet 训练集加载器...")
    train_dataset = RankingDataset('data/ranking_processed/train.parquet')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ranking_collate_fn)

    # 3. 初始化模型组件
    model = DeepFM(feature_dims, embed_dim=16).to(DEVICE)
    # 使用 BCEWithLogitsLoss 提高数值计算的稳定性
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_gauc = 0.0
    patience = 3
    wait = 0

    print(f"进入迭代训练过程 (BatchSize={BATCH_SIZE})...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for i, (cat, seq, num, labels) in pbar:
            # 数据设备转移
            cat = {k: v.to(DEVICE) for k, v in cat.items()}
            seq = {k: v.to(DEVICE) for k, v in seq.items()}
            num = {k: v.to(DEVICE) for k, v in num.items()}
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            # A. 前向传播计算原始得分
            preds = model(cat, seq, num)
            # B. 计算交叉熵损失
            loss = criterion(preds, labels)
            # C. 梯度回传
            loss.backward(); optimizer.step()
            
            total_loss += loss.item()
            if i % 100 == 0: pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- 轮次结算阶段 ---
        print(f"正在进行轮次结算：计算 1万用户批量 GAUC...")
        current_gauc = evaluate_gauc(model, meta, DEVICE)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 总结 | 平均 Loss: {avg_loss:.4f} | 验证集 GAUC: {current_gauc:.4f}")

        # 早停与最佳权重持久化
        if current_gauc > best_gauc:
            best_gauc = current_gauc
            wait = 0
            torch.save(model.state_dict(), "best_deepfm_model.pth")
            print(">>> 发现更优的排序能力，模型权重已更新。")
        else:
            wait += 1
            if wait >= patience:
                print(f"GAUC 连续 {patience} 代不再增长，自动触发早停。"); break

if __name__ == "__main__":
    import pandas as pd
    train_ranking()
