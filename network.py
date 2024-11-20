import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

dataset_device = "cuda"
# 自定义数据集
class PoswDirectionColorDataset(Dataset):
    def __init__(self, posw, direction, color):
        # Flatten 数据并组合输入特征
        self.inputs = torch.tensor(
            np.concatenate([posw.reshape(-1, 3), direction.reshape(-1, 2)], axis=1), dtype=torch.float32
        ).to(dataset_device)
        # Flatten 输出颜色值
        self.targets = torch.tensor(color.reshape(-1, 3), dtype=torch.float32).to(dataset_device)
        print(self.inputs.device)
        print(self.targets.device)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Fully Fused MLP 定义
class FullyFusedMLP(nn.Module):
    def __init__(self):
        super(FullyFusedMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 128),  # 输入 5 个特征 (x, y, z, φ, θ)
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 输出 3 个颜色值 (r, g, b)
        )

    def forward(self, x):
        return self.model(x)

# 数据加载与训练
def train_model(all_posws, all_directions, all_colors, epochs=10, batch_size=32, lr=1e-3, device='cuda'):
    """
    训练神经网络模型。
    """
    
    # tensorboard writer
    writer = SummaryWriter(log_dir='./logs')  # 设置日志路径

    dataset = PoswDirectionColorDataset(all_posws, all_directions, all_colors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化网络、损失函数和优化器
    model = FullyFusedMLP().to(device)
    criterion = nn.MSELoss()  # 均方误差
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # 将数据移动到 GPU
            # inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)

            # 计算损失并反向传播
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 更新进度条和损失值
            epoch_loss += loss.item()
            # 每个 batch 都记录损失
            writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + batch_idx)

        # 计算并记录平均损失
        average_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.6f}")
        writer.add_scalar("Loss/epoch", average_loss, epoch)

    return model

# 模型推理
# posws: (x, 3) 的 numpy 数组，表示合法的像素点的实际坐标列表。
# direction_maps: (x, 2) 的 numpy 数组，表示每个点到相机的方向 (φ, θ)。
# return: (x, 3) 的 numpy 数组，表示预测的颜色值。
def predict(model, posws, direction_maps, device='cuda'):
    """
    使用训练好的模型进行推理。
    """
    model.eval()

    # 执行推理
    with torch.no_grad():
        inputs = torch.tensor(
            np.concatenate([posws, direction_maps], axis=1), dtype=torch.float32
        ).to(device)
        outputs = model(inputs)

    return outputs.cpu().numpy()
