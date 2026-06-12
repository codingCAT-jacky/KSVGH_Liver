import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFromDiagram(nn.Module):
    """
    對應圖中的結構：
    1×310×310 → Conv(8) → BN → Pool
              → Conv(16) → Pool
              → Conv(32) → Pool
              → Conv(64) → Pool
              → Conv(128)→ Pool
              → Flatten() → Dense(10) → Act → Dense(4)
    """
    def __init__(self, act='relu', out_dim=1, p_drop_conv=0.15, p_drop_fc=0.5):
        super().__init__()
        # ---- Stem ----
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # SAME
        self.bn1   = nn.BatchNorm2d(8)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.doc   = nn.Dropout2d(p_drop_conv)
        self.dof   = nn.Dropout(p_drop_fc)

        # ---- 后續 conv-blocks（無 BN，照圖）----
        self.conv2 = nn.Conv2d(8,  16, kernel_size=3, stride=1, padding=1)  
        self.bn2   = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  
        self.bn3   = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
        self.bn4   = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128,kernel_size=3, stride=1, padding=1) 
        self.bn5   = nn.BatchNorm2d(128)

        # ---- 全連接 ----
        self.flatten_dim = 12 * 12 * 128     # 圖中對應 
        self.fc1   = nn.Linear(self.flatten_dim, 10)
        self.fc2   = nn.Linear(10, out_dim)

        # 激活函數（Dense(10) 之後）
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported act: {act}")

    def forward(self, x):
        # x: [B,1,360,360]
        x = self.conv1(x); x = self.bn1(x); x = F.relu(x, inplace=True); x = self.pool(x);  x = self.doc(x);  
        x = self.conv2(x); x = self.bn2(x); x = F.relu(x, inplace=True); x = self.pool(x);  x = self.doc(x);                 
        x = self.conv3(x); x = self.bn3(x); x = F.relu(x, inplace=True); x = self.pool(x);  x = self.doc(x);                 
        x = self.conv4(x); x = self.bn4(x); x = F.relu(x, inplace=True); x = self.pool(x);  x = self.doc(x);              
        x = self.conv5(x); x = self.bn5(x); x = F.relu(x, inplace=True); x = self.pool(x);  x = self.doc(x);              

        x = torch.flatten(x, 1)                                                          
        x = self.dof(x)
        x = self.fc1(x); x = self.act(x)                                                   # -> [B,10]
        x = self.dof(x)
        x = self.fc2(x)                                                                    # -> [B,out_dim]
        return x

# －－－ 快速測試 －－－
if __name__ == "__main__":
    model = CNNFromDiagram(act='relu', out_dim=1)  # 回歸：out_dim=1；分類：out_dim=類別數
    dummy = torch.randn(2, 1, 360, 360)
    y = model(dummy)
    print(y.shape)  # 回歸時 [2,1]；分類時 [2,num_classes]
