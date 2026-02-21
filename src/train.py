import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import os
from src.dataset import MovieLensDataset, collate_fn
from src.model import DualTowerModel

def train():
    """
    模型训练主循环。
    """
    # 1. 设备选择：优先使用 Mac 的 MPS 加速，其次使用 Nvidia 的 CUDA，最后回退到 CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"正在使用的训练设备: {device}")

    # 2. 加载预处理好的元数据
    with open('data/processed/meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    # 3. 数据加载
    # 在 MacBook Air 上建议将 batch_size 设为 512，全量训练时可调大
    train_dataset = MovieLensDataset('data/processed/train.csv', meta, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)

    # 4. 模型初始化
    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Pointwise 召回本质是二分类任务
    criterion = nn.BCELoss()

    print("开始模型训练...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch_idx, (users, u_hists, items, i_genres, labels) in enumerate(train_loader):
            # 将数据移动到计算设备
            users, u_hists = users.to(device), u_hists.to(device)
            items, i_genres, labels = items.to(device), i_genres.to(device), labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(users, u_hists, items, i_genres)
            
            # 计算 Loss (正样本得分接近1，负样本得分接近0)
            loss = criterion(outputs, labels)
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 结束。平均 Loss: {avg_loss:.4f}")
        
        # 每个 Epoch 结束后保存一次权重
        torch.save(model.state_dict(), f"tower_model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
