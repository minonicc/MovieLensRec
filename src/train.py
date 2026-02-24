import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pickle
import os
import faiss
import numpy as np
import pandas as pd
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from src.dataset import MovieLensDataset, collate_fn
from src.model import DualTowerModel


def evaluate_fast(model, meta, device, k=50):
    """
    在训练过程中快速计算验证集的 Hit Rate，用于早停判断。
    """
    model.eval()
    # 为了稳定性，评估过程在 CPU 上进行
    eval_device = torch.device("cpu")
    model.to(eval_device)
    faiss.omp_set_num_threads(1)

    # 1. 导出全量电影向量池
    item_count = meta["item_count"]
    all_item_indices = torch.arange(item_count).to(eval_device)
    all_genres = []
    all_stats = []
    for i in range(item_count):
        g = meta["movie_genres"].get(i, [])
        all_genres.append((g[:6] + [0]*6)[:6])
        f = meta["item_feat_map"].get(i, {})
        all_stats.append([f.get("year_bucket", 0), f.get("rating_bucket", 0), f.get("count_bucket", 0)])
    
    all_genres_t = torch.tensor(all_genres).to(eval_device)
    all_stats_t = torch.tensor(all_stats).to(eval_device)

    with torch.no_grad():
        item_embs = []
        for i in range(0, item_count, 4096):
            embs = model.forward_item(all_item_indices[i:i+4096], all_genres_t[i:i+4096], all_stats_t[i:i+4096])
            item_embs.append(embs.numpy().astype("float32"))
        item_pool = np.vstack(item_embs)

    index = faiss.IndexFlatIP(item_pool.shape[1])
    index.add(item_pool)

    # 2. 加载验证集数据 (这里我们简单处理，不使用 DataLoader 以提速)
    val_df = pd.read_csv("data/processed/val.csv")
    user_sequences = meta["user_sequences"]
    user_feat_map = meta["user_feat_map"]
    movie_genres = meta["movie_genres"]
    
    hit_count = 0
    total = len(val_df)
    
    # 分批检索
    u_batch_size = 2048
    with torch.no_grad():
        for i in range(0, total, u_batch_size):
            batch_df = val_df.iloc[i : i + u_batch_size]
            u_ids = torch.tensor(batch_df["user_idx"].values).to(eval_device)
            
            # 构造混合历史特征
            u_hists = []
            u_hgenres = []
            u_stats = []
            for u, pos in zip(batch_df["user_idx"], batch_df["pos"]):
                # 50最近 + 10随机
                full_seq = user_sequences.get(u, [])[:pos]
                recent = full_seq[-50:]
                rem = full_seq[:-50]
                rand = random.sample(rem, 10) if len(rem) > 10 else rem
                h = (recent + rand + [0]*60)[:60]
                u_hists.append(h)
                
                # 题材展平
                hg = []
                for item_id in h: 
                    if item_id != 0: hg.extend(movie_genres.get(item_id, []))
                u_hgenres.append((hg[:100] + [0]*100)[:100])

                # 用户侧统计特征
                f = user_feat_map.get(u, {})
                u_stats.append([f.get("rating_bucket", 0), f.get("count_bucket", 0)])

            u_vecs = model.forward_user(u_ids, torch.tensor(u_hists), torch.tensor(u_hgenres), torch.tensor(u_stats)).numpy().astype("float32")
            _, topk_indices = index.search(u_vecs, k)
            
            targets = batch_df["item_idx"].values
            for j in range(len(targets)):
                if targets[j] in topk_indices[j]: hit_count += 1

    # 评估完后切回训练模式并移回原设备
    model.train()
    model.to(device)
    return hit_count / total
def train():
    # --- 设置全局随机种子以保证实验可复现性 ---
    import random
    import numpy as np
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    print(f"正在使用的训练设备: {device}")

    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)

    train_dataset = MovieLensDataset('data/processed/train.csv', meta, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)

    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    # --- 早停机制初始化 ---
    best_hr = 0.0
    patience = 3
    wait = 0

    print("开始增强版模型训练...")
    for epoch in range(20): # 增加最大轮数以配合早停
        start_time = time.time()
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for batch_idx, batch in pbar:
            # 移动所有张量到设备 (共有 8 个张量：4个User侧, 3个Item侧, 1个Label)
            batch = [t.to(device) for r in batch for t in (r if isinstance(r, list) else [r])]
            # 解包 (根据 dataset.py 的返回顺序)
            u_ids, u_hists, u_hgenres, u_stats, i_ids, i_genres, i_stats, labels = batch

            optimizer.zero_grad()
            outputs = model(u_ids, u_hists, u_hgenres, u_stats, i_ids, i_genres, i_stats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} 结束。平均 Loss: {total_loss/len(train_loader):.4f} | 耗时: {epoch_time:.2f}s")
        # 执行快速验证
        current_hr = evaluate_fast(model, meta, device, k=50)
        print(f"Epoch {epoch+1} 验证集 HitRate@50: {current_hr:.4f}")

        # 早停逻辑判断
        if current_hr > best_hr:
            best_hr = current_hr
            wait = 0
            torch.save(model.state_dict(), "best_tower_model.pth")
            print("发现更好的模型，已保存权重至 best_tower_model.pth")
        else:
            wait += 1
            print(f"验证集表现未提升，Patience: {wait}/{patience}")
            if wait >= patience:
                print("满足早停条件，停止训练。")
                break

if __name__ == "__main__":
    train()
