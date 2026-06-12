import torch
import torch.nn as nn


# 768 + 10 -> 64 + 1
class MultiModalConv(nn.Module):
    def __init__(self, pretrained_conv, num_scalars):
        super(MultiModalConv, self).__init__()
        
        # ==========================================
        # 1. 影像特徵提取器 (無損輸出 768 維)
        # ==========================================
        # 1.1 修改第一層為單 通道 (如果你仍使用 Raw 或單通道灰階)
        original_stem = pretrained_conv.features[0][0]
        # 1.2 建立新的單通道卷積層
        new_stem = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=0, bias=True)
        # 1.3 進行權重平均轉移 (無損繼承 ImageNet 視覺經驗)
        with torch.no_grad():
            new_stem.weight = nn.Parameter(original_stem.weight.mean(dim=1, keepdim=True))
            new_stem.bias = original_stem.bias
        # 1.4 替換回模型中
        pretrained_conv.features[0][0] = new_stem
        # 1.5 取得特徵維度並移除原本的 classifier
        in_features = pretrained_conv.classifier[2].in_features
        pretrained_conv.classifier[2] = nn.Identity()
        self.image_extractor = pretrained_conv
        

        # ==========================================
        # 2. 數值特徵防護罩 (10 維)
        # ==========================================
        self.scalar_norm = nn.BatchNorm1d(num_scalars)
        
        # ==========================================
        # 3. 跨模態深層特徵萃取 (不含公式: 768 + 10 = 778 維)
        # ==========================================
        self.fc_features = nn.Sequential(
            nn.Linear(in_features + num_scalars, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
             
            nn.Linear(384, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # ==========================================
        # 4. 最終決策層 (晚期注入公式: 64 + 1 = 65 維)
        # ==========================================
        # 這層只有 65 個權重，它能清晰地決定「要聽深層特徵的，還是聽大師公式的」
        self.fc_decision = nn.Linear(65, 1)

    def forward(self, img, scalars):
        # 取得影像與數值原料
        tai = scalars[:, :5]  # 前5個是 TAI
        tsi = scalars[:, 5:]  # 後5個是 TSI
        tai_mean = tai.mean(dim=1)  # [Batch]
        tsi_mean = tsi.mean(dim=1)  # [Batch]
        expert_usff = (-44.3 + 41.9 * tai_mean + 0.23 * tsi_mean) / 100.0
        

        img_embed = self.image_extractor(img)          # (Batch, 768)
        scalar_embed = self.scalar_norm(scalars)       # (Batch, 10)
        
        # 第一階段拼接：讓神經網路去挖掘影像與數值的隱藏關聯
        fused_base = torch.cat((img_embed, scalar_embed), dim=1)
        
        # 提煉出 64 維的深層跨模態精華
        deep_features = self.fc_features(fused_base)  # (Batch, 64)
        
        combined = torch.cat((deep_features, expert_usff.unsqueeze(1)), dim=1)
        
        # 做出最終預測
        output = self.fc_decision(combined)        # (Batch, 1)
        
        return output

class BaseConv(nn.Module):
    def __init__(self, pretrained_conv):
        super(BaseConv, self).__init__()

        # 1.1 修改第一層為單 通道 (如果你仍使用 Raw 或單通道灰階)
        original_stem = pretrained_conv.features[0][0]
        # 1.2 建立新的單通道卷積層
        new_stem = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=0, bias=True)
        # 1.3 進行權重平均轉移 (無損繼承 ImageNet 視覺經驗)
        with torch.no_grad():
            new_stem.weight = nn.Parameter(original_stem.weight.mean(dim=1, keepdim=True))
            new_stem.bias = original_stem.bias
        # 1.4 替換回模型中
        pretrained_conv.features[0][0] = new_stem
        # 1.5 取得特徵維度並移除原本的 classifier
        in_features = pretrained_conv.classifier[2].in_features
        pretrained_conv.classifier[2] = nn.Linear(in_features, 1)
        self.regressor = pretrained_conv
        

    def forward(self, img):

        output = self.regressor(img)           
        return output



# 4096 + 10 -> 64 + 1
class MultiModalVGG(nn.Module):
    def __init__(self, pretrained_vgg, num_scalars=10):
        super(MultiModalVGG, self).__init__()

        # ==========================================
        # 1. 影像特徵提取器 (無損輸出 4096 維)
        # ==========================================        
        # 修改第一層卷積以接受單通道輸入
        original_conv1 = pretrained_vgg.features[0]
        pretrained_vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        # 將3通道權重平均為單通道
        with torch.no_grad():
            pretrained_vgg.features[0].weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))
            pretrained_vgg.features[0].bias = original_conv1.bias

        in_features = pretrained_vgg.classifier[6].in_features
        pretrained_vgg.classifier[6] = nn.Identity()
        self.image_extractor = pretrained_vgg


        # ==========================================
        # 2. 數值特徵防護罩 (10 維)
        # ==========================================
        self.scalar_norm = nn.BatchNorm1d(num_scalars)

        # ==========================================
        # 3. 跨模態深層特徵萃取 (不含公式: 4096 + 10 = 4106 維)
        # ==========================================
        self.fc_features = nn.Sequential(
            nn.Linear(in_features + num_scalars, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # ==========================================
        # 4. 最終決策層 (晚期注入公式: 64 + 1 = 65 維)
        # ==========================================
        # 這層只有 65 個權重，它能清晰地決定「要聽深層特徵的，還是聽大師公式的」
        self.fc_decision = nn.Linear(65, 1)

    def forward(self, img, scalars):
        # 取得影像與數值原料
        tai = scalars[:, :5]  # 前5個是 TAI
        tsi = scalars[:, 5:]  # 後5個是 TSI
        tai_mean = tai.mean(dim=1)  # [Batch]
        tsi_mean = tsi.mean(dim=1)  # [Batch]
        expert_usff = (-44.3 + 41.9 * tai_mean + 0.23 * tsi_mean) / 100.0

        img_embed = self.image_extractor(img)          # (Batch, 4096)
        scalar_embed = self.scalar_norm(scalars)       # (Batch, 10)

        # 第一階段拼接：讓神經網路去挖掘影像與數值的隱藏關聯
        fused_base = torch.cat((img_embed, scalar_embed), dim=1)

        # 提煉出 64 維的深層跨模態精華
        deep_features = self.fc_features(fused_base)  # (Batch, 64)

        combined = torch.cat((deep_features, expert_usff.unsqueeze(1)), dim=1)

        # 做出最終預測
        output = self.fc_decision(combined)        # (Batch, 1)

        return output


class BaseVGG(nn.Module):
    def __init__(self, pretrained_vgg):
        super(BaseVGG, self).__init__()

        # 修改第一層卷積以接受單通道輸入
        original_conv1 = pretrained_vgg.features[0]
        pretrained_vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        # 將3通道權重平均為單通道
        with torch.no_grad():
            pretrained_vgg.features[0].weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))
            pretrained_vgg.features[0].bias = original_conv1.bias
        pretrained_vgg.classifier[6] = nn.Linear(4096, 1)
        self.regressor = pretrained_vgg

    def forward(self, img):
        return self.regressor(img)











