import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pickle
import os
from src.dataset import MovieLensDataset, collate_fn
from src.model import DualTowerModel

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

    print("开始增强版模型训练...")
    for epoch in range(5):
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
        torch.save(model.state_dict(), f"tower_model_v2_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
