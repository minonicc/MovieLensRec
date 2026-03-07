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
    这保证了代码在 V100 服务器与本地 MacBook 之间迁移时无需修改代码。
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
# 批量大小设定，平衡显存占用与梯度估算准确度
BATCH_SIZE = 4096
# 最大训练迭代轮数
EPOCHS = 10
# 优化器学习率
LEARNING_RATE = 0.001

def evaluate_gauc(model, meta, device, exam_path='data/ranking_processed/val_exams_10k.parquet'):
    """
    在验证集考卷上执行精细化 GAUC (Group AUC) 评估。
    GAUC 衡量模型在单个用户内将目标物品排在干扰项之前的能力，是排序阶段最核心的离线预测指标。
    """
    if not os.path.exists(exam_path):
        print(f"警告：未找到考卷文件 {exam_path}，无法进行验证。")
        return 0.0
    
    model.eval()
    import pandas as pd
    val_df = pd.read_parquet(exam_path)
    
    # 获取评估索引字典，对齐特征提取逻辑
    user_sequences = meta['user_sequences']
    movie_genres = meta['movie_genres']
    item_feat_map = meta['item_feat_map']
    user_feat_map = meta['user_feat_map']
    # 注意：为了特征组合拳计算，我们需要评分序列
    user_ratings_seq = meta.get('user_ratings_seq', {}) 
    
    gauc_list = []
    
    # 模拟真实推荐场景进行打分
    with torch.no_grad():
        for _, row in val_df.iterrows():
            u_idx, target_item, candidates, pos = int(row['user_idx']), int(row['item_idx']), row['candidates'], int(row['pos'])
            
            # --- 步骤 1: 构造该时刻的用户历史快照特征 (严禁穿越) ---
            full_seq = user_sequences.get(u_idx, [])
            full_ratings = user_ratings_seq.get(u_idx, [4.0]*len(full_seq))
            
            snap_items = full_seq[:pos]
            snap_ratings = full_ratings[:pos]
            
            # 基础 ID 序列 (最近 50)
            hist_ids = (snap_items[-50:] + [0]*50)[:50]
            # 题材偏好序列 (最近 100)
            hist_gs = []
            for h in hist_ids:
                if h != 0: hist_gs.extend(movie_genres.get(h, []))
            hist_gs_truncated = (hist_gs[-100:] + [0]*100)[:100]
            hist_gs_set = set(hist_gs)
            
            u_stats = user_feat_map.get(u_idx, {})
            
            # --- 步骤 2: 批量封装候选物品特征输入 ---
            cand_cat = defaultdict(list)
            cand_seq = {'hist_movie_ids': [], 'hist_genre_ids': []}
            cand_num = {'genre_density': [], 'genre_recent_match': [], 'genre_rating_bias': []}
            
            for c_id in candidates:
                i_f = item_feat_map.get(c_id, {})
                # 基础分桶与 ID
                cand_cat['user_idx'].append(u_idx); cand_cat['item_idx'].append(c_id)
                cand_cat['year_bucket'].append(i_f.get('year_bucket', 0))
                cand_cat['item_rating_bucket'].append(i_f.get('rating_bucket', 0))
                cand_cat['item_count_bucket'].append(i_f.get('count_bucket', 0))
                cand_cat['user_rating_bucket'].append(u_stats.get('rating_bucket', 0))
                cand_cat['user_count_bucket'].append(u_stats.get('count_bucket', 0))
                # 共享行为序列
                cand_seq['hist_movie_ids'].append(hist_ids); cand_seq['hist_genre_ids'].append(hist_gs_truncated)
                
                # --- 计算题材组合拳实时数值特征 ---
                # 1. 密度 (Density)
                c_gs_list = movie_genres.get(c_id, [])
                # 统计最近 200 个点击中当前题材的占比
                recent_pool = snap_items[-200:]
                g_counts = defaultdict(int)
                g_total_score = defaultdict(float)
                for it, rt in zip(recent_pool, snap_ratings[-200:]):
                    for g in movie_genres.get(it, []): g_counts[g] += 1; g_total_score[g] += rt
                total_g = sum(g_counts.values())
                cand_num['genre_density'].append(sum(g_counts[g] for g in c_gs_list)/(total_g+1e-8))
                # 2. 最近匹配 (Recent Match)
                r5_g = set()
                for it in snap_items[-5:]: r5_g.update(movie_genres.get(it, []))
                cand_num['genre_recent_match'].append(1.0 if (set(c_gs_list) & r5_g) else 0.0)
                # 3. 评分偏好 (Rating Bias)
                scores = [g_total_score[g]/(g_counts[g]+1e-8) for g in c_gs_list if g in g_counts]
                cand_num['genre_rating_bias'].append(np.mean(scores)/5.0 if scores else 0.6)
            
            # --- 步骤 3: 转化为 Tensor 并执行设备加速推理 ---
            t_cat = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in cand_cat.items()}
            t_seq = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in cand_seq.items()}
            t_num = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in cand_num.items()}
            
            # 计算 Sigmoid 概率
            scores = torch.sigmoid(model(t_cat, t_seq, t_num)).cpu().numpy()
            
            # --- 步骤 4: 统计 AUC 分量 ---
            labels = np.zeros(len(candidates))
            try:
                # 定位 Ground Truth 在考卷中的原始索引
                gt_idx = candidates.tolist().index(target_item) if isinstance(candidates, np.ndarray) else candidates.index(target_item)
                labels[gt_idx] = 1
                gauc_list.append(roc_auc_score(labels, scores))
            except:
                continue
                
    return np.mean(gauc_list) if gauc_list else 0.5

def train_ranking():
    """
    精排模型训练主入口：负责环境检查、数据加载、训练循环与早停管理。
    """
    print(f"--- 排序阶段训练启动 ---")
    print(f"当前训练算力核心: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU 信息: {torch.cuda.get_device_name(0)}")
    
    # 1. 准备模型特征维度元数据
    print("加载元数据并对齐特征空间...")
    with open('data/processed/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    feature_dims = {
        'user_idx': meta['user_count'], 'item_idx': meta['item_count'],
        'year_bucket': 12, 'item_rating_bucket': 12, 'item_count_bucket': 12,
        'user_rating_bucket': 12, 'user_count_bucket': 12,
        'hist_movie_ids': meta['item_count'], 'hist_genre_ids': meta['genre_count']
    }

    # 2. 加载经过 data_prep 预处理的高性能 Parquet 训练集
    print("挂载 Parquet 训练数据加载器...")
    train_dataset = RankingDataset('data/ranking_processed/train.parquet')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=ranking_collate_fn, num_workers=0)

    # 3. 构建 DeepFM 模型骨架与优化器
    model = DeepFM(feature_dims, embed_dim=16).to(DEVICE)
    # 使用 BCEWithLogitsLoss 以确保数值计算的鲁棒性，有效防止梯度消失或爆炸
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 监控变量初始化
    best_gauc = 0.0
    patience = 3 # 容忍 GAUC 连续 3 代不涨则早停
    wait = 0

    print(f"开始训练流程 (预设轮数: {EPOCHS})...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for i, (cat, seq, num, labels) in pbar:
            # 搬运数据至设备
            cat = {k: v.to(DEVICE) for k, v in cat.items()}
            seq = {k: v.to(DEVICE) for k, v in seq.items()}
            num = {k: v.to(DEVICE) for k, v in num.items()}
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            # 步骤 A: 模型前向传播，产出原始 Logits
            preds = model(cat, seq, num)
            # 步骤 B: 计算二分类对比损失
            loss = criterion(preds, labels)
            
            # 步骤 C: 反向传播梯度并更新权重
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 100 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- 轮次结束后的实战评估 ---
        print(f"正在进行轮次结算：计算 1万用户 GAUC...")
        current_gauc = evaluate_gauc(model, meta, DEVICE)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 总结 | 平均 Loss: {avg_loss:.4f} | 验证集 GAUC: {current_gauc:.4f}")

        # 早停与模型持久化逻辑
        if current_gauc > best_gauc:
            best_gauc = current_gauc
            wait = 0
            torch.save(model.state_dict(), "best_deepfm_model.pth")
            print(">>> 排序性能突破，已保存当前最佳权重。")
        else:
            wait += 1
            if wait >= patience:
                print(f"GAUC 连续 {patience} 代不再增长，自动触发早停保护。")
                break

if __name__ == "__main__":
    import pandas as pd
    train_ranking()
