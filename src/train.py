import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import os
from src.dataset import MovieLensDataset, collate_fn
from src.model import DualTowerModel

def train():
    # 自动选择设备 (MPS/CUDA/CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    # 加载元数据
    with open('data/processed/meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    # 数据加载 (batch_size 调小一点以适应 MacBook 内存)
    train_dataset = MovieLensDataset('data/processed/train.csv', meta, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)

    # 模型初始化
    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print("Starting training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch_idx, (users, u_hists, items, i_genres, labels) in enumerate(train_loader):
            # 移动到设备
            users = users.to(device)
            u_hists = u_hists.to(device)
            items = items.to(device)
            i_genres = i_genres.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(users, u_hists, items, i_genres)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        # 保存模型
        torch.save(model.state_dict(), f"tower_model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
