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
        tsi = scalars[:, 5:10] 
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


class CrossAttentionBlock(nn.Module):
    """
    標準 cross-attention + 殘差連接：
        out = x + MultiheadAttention(Q=x, K=context, V=context)
    輸入/輸出皆為 [B, N, dim]
    """
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, return_attn=False):
        """
        x       : [B, Nx, dim]  -> Query
        context : [B, Nc, dim]  -> Key, Value
        return_attn : 若為 True，額外回傳 attention weights [B, Nx, Nc]
                      (各 head 平均後的結果，由 nn.MultiheadAttention 預設行為提供)
        """
        q = self.norm_q(x)
        kv = self.norm_kv(context)
        attn_out, attn_weights = self.attn(
            query=q, key=kv, value=kv,
            need_weights=True, average_attn_weights=True
        )
        out = x + self.dropout(attn_out)  # 殘差連接
        if return_attn:
            return out, attn_weights   # attn_weights: [B, Nx, Nc]
        return out


# 768 + 768 -> 64 + 1 (with cross-attention fusion)
class MultiModalAttnConv(nn.Module):
    def __init__(self, pretrained_conv, num_scalars, num_qus_types=3, num_heads=8, dropout=0.1):
        """
        num_scalars   : QUS 數值總數 (例如 15 = 3 種 QUS * 5 個數字)
        num_qus_types : QUS 參數種類數 (例如 3: TAI, TSI, SWE...)
        """
        super(MultiModalAttnConv, self).__init__()

        self.num_qus_types = num_qus_types

        # ==========================================
        # 1. 影像特徵提取器 (取出 GAP 前的特徵圖 [B, 768, 7, 7])
        # ==========================================
        # 1.1 修改第一層為單通道 (如果你仍使用 Raw 或單通道灰階)
        original_stem = pretrained_conv.features[0][0]
        # 1.2 建立新的單通道卷積層
        new_stem = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=0, bias=True)
        # 1.3 進行權重平均轉移 (無損繼承 ImageNet 視覺經驗)
        with torch.no_grad():
            new_stem.weight = nn.Parameter(original_stem.weight.mean(dim=1, keepdim=True))
            new_stem.bias = original_stem.bias
        # 1.4 替換回模型中
        pretrained_conv.features[0][0] = new_stem
        # 1.5 取得特徵維度 (768)；只使用 features，不使用 classifier (GAP 移到後面手動做)
        in_features = pretrained_conv.classifier[2].in_features  # = 768
        self.feature_dim = in_features
        self.image_extractor = pretrained_conv.features  # 輸出 [B, 768, 7, 7]

        # ==========================================
        # 2. QUS 異質維度對齊 (Parameter-Specific Encoders)
        # ==========================================
        self.scalar_norm = nn.BatchNorm1d(num_scalars)
        
        # 🌟 核心創新：利用 ModuleDict 動態生成專屬編碼器
        self.qus_encoders = nn.ModuleDict({
            'tai': nn.Sequential(nn.Linear(5, self.feature_dim//2), nn.GELU()),
            'tsi': nn.Sequential(nn.Linear(5, self.feature_dim//2), nn.GELU())
        })
        if self.num_qus_types >= 3:
            self.qus_encoders['swe'] = nn.Sequential(nn.Linear(5, self.feature_dim//2), nn.GELU())
        if self.num_qus_types >= 4:
            # EZHRI 只有 1 維，但我們一樣能把它映射成 1024 維的 Token！
            self.qus_encoders['ezhri'] = nn.Sequential(nn.Linear(1, self.feature_dim//2), nn.GELU())

        self.shared_qus_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim//2),
            nn.Linear(self.feature_dim//2, self.feature_dim)
        )

        # ==========================================
        # 3. 雙向 Cross-Attention Fusion
        #    Fa = [B, 49, 768] 當 Q, Fb = [B, num_qus_types, 768] 當 K,V
        #    Fb 當 Q, Fa 當 K,V
        # ==========================================
        self.cross_attn_img2qus = CrossAttentionBlock(
            dim=self.feature_dim, num_heads=num_heads, dropout=dropout
        )
        self.cross_attn_qus2img = CrossAttentionBlock(
            dim=self.feature_dim, num_heads=num_heads, dropout=dropout
        )

        # ==========================================
        # 4. 跨模態深層特徵萃取
        #    GAP(Fa) + GAP(Fb) -> [B, 768] + [B, 768] -> concat -> [B, 1536]
        # ==========================================
        self.fc_features = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(384, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # ==========================================
        # 5. 最終決策層 (晚期注入公式: 64 + 1 = 65 維)
        # ==========================================
        # 這層只有 65 個權重，它能清晰地決定「要聽深層特徵的，還是聽大師公式的」
        self.fc_decision = nn.Linear(65, 1)

    def forward(self, img, scalars, return_attn=False):
        B = img.shape[0]

        # --- 取得影像與數值原料 (專家公式，仍使用前 10 個數字: TAI + TSI) ---
        tai = scalars[:, :5]   # 前5個是 TAI
        tsi = scalars[:, 5:10] # 接著5個是 TSI
        tai_mean = tai.mean(dim=1)  # [B]
        tsi_mean = tsi.mean(dim=1)  # [B]
        expert_usff = (-44.3 + 41.9 * tai_mean + 0.23 * tsi_mean) / 100.0

        # --- 影像分支: 取得 GAP 前特徵圖 [B, 768, 7, 7] ---
        feat_map = self.image_extractor(img)            # [B, 768, 7, 7]
        Fa = feat_map.flatten(2).transpose(1, 2)        # [B, 49, 768]

        # --- QUS 分支: [B, num_scalars] 
        scalar_normed = self.scalar_norm(scalars)         # [B, num_scalars]
        tai_feats = scalar_normed[:, 0:5]
        tsi_feats = scalar_normed[:, 5:10]
        
        # 動態收集 Tokens
        qus_list = [
            self.qus_encoders['tai'](tai_feats),
            self.qus_encoders['tsi'](tsi_feats)
        ]
        if self.num_qus_types >= 3:
            swe_feats = scalar_normed[:, 10:15]
            qus_list.append(self.qus_encoders['swe'](swe_feats))
            
        if self.num_qus_types >= 4:
            ezhri_feats = scalar_normed[:, 15:16]
            qus_list.append(self.qus_encoders['ezhri'](ezhri_feats))

        # 將列表堆疊成 Token 序列: [B, num_qus_types, 768]
        stacked_qus = torch.stack(qus_list, dim=1)
        Fb = self.shared_qus_proj(stacked_qus)

        # --- 雙向 Cross-Attention Fusion (各自殘差) ---
        Fa_fused = self.cross_attn_img2qus(x=Fa, context=Fb)  # [B, 49, 768]
        Fb_fused, qus2img_attn = self.cross_attn_qus2img(
            x=Fb, context=Fa, return_attn=True
        )  # Fb_fused: [B, num_qus_types, 768], qus2img_attn: [B, num_qus_types, 49]

        # --- GAP 後 concat ---
        Fa_pooled = Fa_fused.mean(dim=1)   # [B, 768]
        Fb_pooled = Fb_fused.mean(dim=1)   # [B, 768]
        fused_base = torch.cat((Fa_pooled, Fb_pooled), dim=1)  # [B, 1536]

        # --- 提煉出 64 維的深層跨模態精華 ---
        deep_features = self.fc_features(fused_base)   # [B, 64]

        combined = torch.cat((deep_features, expert_usff.unsqueeze(1)), dim=1)  # [B, 65]

        # --- 做出最終預測 ---
        output = self.fc_decision(combined)             # [B, 1]

        if return_attn:
            # qus2img_attn: [B, num_qus_types, 49] -> 每個 QUS token 對 49 個空間位置的關注度
            return output, qus2img_attn
        return output

class BaseConv(nn.Module):
    def __init__(self, pretrained_conv):
        super(BaseConv, self).__init__()


        original_stem = pretrained_conv.features[0][0]
        new_stem = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=0, bias=True)
        with torch.no_grad():
            new_stem.weight = nn.Parameter(original_stem.weight.mean(dim=1, keepdim=True))
            new_stem.bias = original_stem.bias
        pretrained_conv.features[0][0] = new_stem
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

        original_conv1 = pretrained_vgg.features[0]
        pretrained_vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        with torch.no_grad():
            pretrained_vgg.features[0].weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))
            pretrained_vgg.features[0].bias = original_conv1.bias

        in_features = pretrained_vgg.classifier[6].in_features
        pretrained_vgg.classifier[6] = nn.Identity()
        self.image_extractor = pretrained_vgg

        self.scalar_norm = nn.BatchNorm1d(num_scalars)
        
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

        self.fc_decision = nn.Linear(65, 1)

    def forward(self, img, scalars):
        # 取得影像與數值原料
        tai = scalars[:, :5]  # 前5個是 TAI
        tsi = scalars[:, 5:10]  
        tai_mean = tai.mean(dim=1)  # [Batch]
        tsi_mean = tsi.mean(dim=1)  # [Batch]
        expert_usff = (-44.3 + 41.9 * tai_mean + 0.23 * tsi_mean) / 100.0

        img_embed = self.image_extractor(img)          # (Batch, 4096)
        scalar_embed = self.scalar_norm(scalars)       # (Batch, 10)

        fused_base = torch.cat((img_embed, scalar_embed), dim=1)
        deep_features = self.fc_features(fused_base)  # (Batch, 64)
        combined = torch.cat((deep_features, expert_usff.unsqueeze(1)), dim=1)
        output = self.fc_decision(combined)        # (Batch, 1)

        return output


# 🌟 新增：VGG 版本的 Cross-Attention 模型
class MultiModalAttnVGG(nn.Module):
    def __init__(self, pretrained_vgg, num_scalars, num_qus_types=3, num_heads=8, dropout=0.1):
        super(MultiModalAttnVGG, self).__init__()

        self.num_qus_types = num_qus_types

        # 1. 影像特徵提取器
        original_conv1 = pretrained_vgg.features[0]
        pretrained_vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        with torch.no_grad():
            pretrained_vgg.features[0].weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))
            pretrained_vgg.features[0].bias = original_conv1.bias

        # VGG 提取器 (最後輸出維度是 [B, 512, 7, 7])
        self.feature_dim = 512
        self.image_extractor = pretrained_vgg.features

        # 2. QUS 動態編碼器
        self.scalar_norm = nn.BatchNorm1d(num_scalars)
        self.qus_encoders = nn.ModuleDict({
            'tai': nn.Sequential(nn.Linear(5, self.feature_dim//2), nn.GELU()),
            'tsi': nn.Sequential(nn.Linear(5, self.feature_dim//2), nn.GELU())
        })
        if self.num_qus_types >= 3:
            self.qus_encoders['swe'] = nn.Sequential(nn.Linear(5, self.feature_dim//2), nn.GELU())
        if self.num_qus_types >= 4:
            self.qus_encoders['ezhri'] = nn.Sequential(nn.Linear(1, self.feature_dim//2), nn.GELU())

        self.shared_qus_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim//2),
            nn.Linear(self.feature_dim//2, self.feature_dim)
        )

        # 3. 雙向 Cross-Attention
        self.cross_attn_img2qus = CrossAttentionBlock(dim=self.feature_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn_qus2img = CrossAttentionBlock(dim=self.feature_dim, num_heads=num_heads, dropout=dropout)

        # 4. 跨模態深層特徵萃取 (512+512 -> 1024 -> 384 -> 64)
        self.fc_features = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.fc_decision = nn.Linear(65, 1)

    def forward(self, img, scalars, return_attn=False):
        # --- 專家公式 ---
        tai = scalars[:, :5]
        tsi = scalars[:, 5:10]
        expert_usff = (-44.3 + 41.9 * tai.mean(dim=1) + 0.23 * tsi.mean(dim=1)) / 100.0

        # --- 影像分支 ---
        feat_map = self.image_extractor(img)            # [B, 512, 7, 7]
        Fa = feat_map.flatten(2).transpose(1, 2)        # [B, 49, 512]

        # --- QUS 分支 ---
        scalar_normed = self.scalar_norm(scalars)
        qus_list = [
            self.qus_encoders['tai'](scalar_normed[:, 0:5]),
            self.qus_encoders['tsi'](scalar_normed[:, 5:10])
        ]
        if self.num_qus_types >= 3:
            qus_list.append(self.qus_encoders['swe'](scalar_normed[:, 10:15]))
        if self.num_qus_types >= 4:
            qus_list.append(self.qus_encoders['ezhri'](scalar_normed[:, 15:16]))

        stacked_qus = torch.stack(qus_list, dim=1)
        Fb = self.shared_qus_proj(stacked_qus)                     

        # --- 雙向 Cross-Attention ---
        Fa_fused = self.cross_attn_img2qus(x=Fa, context=Fb)
        Fb_fused, qus2img_attn = self.cross_attn_qus2img(
            x=Fb, context=Fa, return_attn=True
        )  # qus2img_attn: [B, num_qus_types, 49]

        # --- GAP 後 Concat ---
        Fa_pooled = Fa_fused.mean(dim=1)                         
        Fb_pooled = Fb_fused.mean(dim=1)                         
        fused_base = torch.cat((Fa_pooled, Fb_pooled), dim=1)   # [B, 1024]

        # --- 深層特徵萃取 ---
        deep_features = self.fc_features(fused_base)             # [B, 64]
        combined = torch.cat((deep_features, expert_usff.unsqueeze(1)), dim=1)  # [B, 65]
        output = self.fc_decision(combined)                      # [B, 1]
        
        if return_attn:
            return output, qus2img_attn
        return output

class BaseVGG(nn.Module):
    def __init__(self, pretrained_vgg):
        super(BaseVGG, self).__init__()

        original_conv1 = pretrained_vgg.features[0]
        pretrained_vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        with torch.no_grad():
            pretrained_vgg.features[0].weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))
            pretrained_vgg.features[0].bias = original_conv1.bias
        pretrained_vgg.classifier[6] = nn.Linear(4096, 1)
        self.regressor = pretrained_vgg

    def forward(self, img):
        return self.regressor(img)



# ==============================================================================
# MedViT-based Models
#
# ★ MedViT_small 真實架構（讀完 MedViT.py 後的正確理解）:
#
#   stem     : 4× ConvBNReLU
#
#   features : 20 個 Block (ECB / LTB 混合)
#              全程維持空間特徵圖形狀 [B, C, H, W]，最終輸出 [B, 1024, 7, 7]
#              (MedViT_small depths=[3,4,10,3], 最後一層輸出 channel = 1024)
#
#   norm     : BatchNorm2d(1024)  -> [B, 1024, 7, 7]
#   avgpool  : AdaptiveAvgPool2d(1,1) -> [B, 1024, 1, 1]
#   flatten  : -> [B, 1024]
#   proj_head: Linear(1024, num_classes)
#
# ★ 關鍵結論:
#   2. 全程是空間特徵圖 [B, C, H, W]，沒有 token 序列 [B, N, C]
#   3. norm 之後、avgpool 之前的特徵圖 [B, 1024, 7, 7] 就是「空間特徵」
#      flatten(2).transpose(1,2) -> [B, 49, 1024] 即可當 patch token 用
#   4. embed_dim = 1024 (MedViT_small 最後一個 stage 的輸出 channel)
# ==============================================================================


# ──────────────────────────────────────────────────────────────
# 1. BaseMed：只輸入 B-mode 影像
#    MedViT 特徵 [B, 1024] -> Linear(1024, 1)
# ──────────────────────────────────────────────────────────────
class BaseMed(nn.Module):
    """
    最簡版 MedViT 回歸模型。
    僅接受單通道 B-mode 影像，直接輸出 PDFF 預測值。

    架構:
        影像 [B,1,224,224]
        -> stem [B,64,56,56]
        -> features (ECB/LTB blocks) [B,1024,7,7]
        -> norm -> avgpool -> flatten [B,1024]
        -> proj_head: Linear(1024, 1)
    """
    def __init__(self, pretrained_med):
        super(BaseMed, self).__init__()

        
        # 載入後再做通道平均轉換
        old_conv = pretrained_med.stem[0].conv          # [64, 3, 3, 3]
        new_conv = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                old_conv.weight.mean(dim=1, keepdim=True)  # [64, 3, 3, 3] -> [64, 1, 3, 3]
            )
        pretrained_med.stem[0].conv = new_conv
        # 取出 embed_dim，並換掉分類頭 -> 回歸頭
        embed_dim = pretrained_med.proj_head[0].in_features  # 1024
        self.backbone = pretrained_med  # stem 已是單通道，不需修改
        self.backbone.proj_head = nn.Sequential(nn.Linear(embed_dim, 1))

    def forward(self, img):
        return self.backbone(img)   # [B, 1]


# ──────────────────────────────────────────────────────────────
# 2. MultiModalMed：B-mode 影像 + QUS 數值，Concat 融合
#    1024 + 10 -> 384 -> 64 + 1
#
#    流程與 MultiModalConv 完全對應，backbone 換成 MedViT。
# ──────────────────────────────────────────────────────────────
class MultiModalMed(nn.Module):
    """
    MedViT + QUS 數值 Concat 融合版本。

    架構:
        影像 -> MedViT (avgpool 前截斷) -> [B, 1024]
        QUS  -> BatchNorm1d             -> [B, num_scalars]
        Concat [B, 1024 + num_scalars]
        -> MLP(1034 -> 384 -> 64) -> concat expert_usff -> Linear(65, 1)
    """
    def __init__(self, pretrained_med, num_scalars=10):
        super(MultiModalMed, self).__init__()

        # ==========================================
        # 1. 影像特徵提取器
        #    用 proj_head = Identity 讓 backbone(img) 輸出 flatten 後的 [B, 1024]
        # ==========================================
        # 載入後再做通道平均轉換
        old_conv = pretrained_med.stem[0].conv          # [64, 3, 3, 3]
        new_conv = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                old_conv.weight.mean(dim=1, keepdim=True)  # [64, 3, 3, 3] -> [64, 1, 3, 3]
            )
        # 取出 embed_dim，並換掉分類頭 -> 回歸頭
        pretrained_med.stem[0].conv = new_conv
        embed_dim = pretrained_med.proj_head[0].in_features  # 1024
        self.backbone = pretrained_med
        self.backbone.proj_head = nn.Identity()
        self.embed_dim = embed_dim

        # ==========================================
        # 2. 數值特徵防護罩
        # ==========================================
        self.scalar_norm = nn.BatchNorm1d(num_scalars)

        # ==========================================
        # 3. 跨模態深層特徵萃取 (1024 + num_scalars -> 384 -> 64)
        # ==========================================
        self.fc_features = nn.Sequential(
            nn.Linear(embed_dim + num_scalars, 384),
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
        self.fc_decision = nn.Linear(65, 1)

    def forward(self, img, scalars):
        # --- 專家公式 ---
        tai = scalars[:, :5]
        tsi = scalars[:, 5:10]
        tai_mean = tai.mean(dim=1)
        tsi_mean = tsi.mean(dim=1)
        expert_usff = (-44.3 + 41.9 * tai_mean + 0.23 * tsi_mean) / 100.0

        # --- 影像分支: backbone(img) 走完 stem+features+norm+avgpool+flatten
        #              proj_head=Identity -> [B, 1024]
        img_embed = self.backbone(img)            # [B, 1024]

        # --- QUS 分支 ---
        scalar_embed = self.scalar_norm(scalars)  # [B, num_scalars]

        # --- Concat 融合 ---
        fused_base = torch.cat((img_embed, scalar_embed), dim=1)  # [B, 1034]

        # --- 深層特徵萃取 ---
        deep_features = self.fc_features(fused_base)  # [B, 64]

        # --- 晚期注入專家公式 ---
        combined = torch.cat((deep_features, expert_usff.unsqueeze(1)), dim=1)  # [B, 65]

        output = self.fc_decision(combined)   # [B, 1]
        return output


# ──────────────────────────────────────────────────────────────
# 3. MultiModalAttnMed：B-mode 影像 + QUS 數值，雙向 Cross-Attention 融合
#
#    MedViT 的特徵圖在 norm 之後、avgpool 之前是 [B, 1024, 7, 7]
#    → flatten(2).transpose(1,2) → [B, 49, 1024]  這 49 個就是「空間 token」
#
#    流程與 MultiModalAttnConv 完全對應，backbone 換成 MedViT。
# ──────────────────────────────────────────────────────────────
class MultiModalAttnMed(nn.Module):
    """
    MedViT + QUS 數值 雙向 Cross-Attention 融合版本。

    架構:
        影像 -> stem + features + norm [B, 1024, 7, 7]
             -> flatten to tokens Fa:  [B, 49, 1024]
        QUS  -> BN -> reshape
             -> MLP 投影 Fb: [B, num_qus_types, 1024]
        Fa, Fb -> 雙向 CrossAttentionBlock
        GAP(Fa_fused) + GAP(Fb_fused) -> Concat [B, 2048]
        -> MLP(2048 -> 384 -> 64) -> concat expert_usff -> Linear(65, 1)
    """
    def __init__(self, pretrained_med, num_scalars=10, num_qus_types=2,
                 num_heads=8, dropout=0.1):
        """
        num_scalars   : QUS 數值總數 (預設 10 = TAI×5 + TSI×5)
        num_qus_types : QUS 參數種類數 (預設 2: TAI, TSI)
                        num_scalars 必須能被 num_qus_types 整除
        """
        super(MultiModalAttnMed, self).__init__()

        self.num_qus_types = num_qus_types

        # ==========================================
        # 1. 影像特徵提取器
        #    只保留 stem + features + norm；avgpool/flatten/proj_head 改為手動執行
        # ==========================================
        # 載入後再做通道平均轉換
        old_conv = pretrained_med.stem[0].conv          # [64, 3, 3, 3]
        new_conv = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                old_conv.weight.mean(dim=1, keepdim=True)  # [64, 3, 3, 3] -> [64, 1, 3, 3]
            )
        # 取出 embed_dim，並換掉分類頭 -> 回歸頭
        pretrained_med.stem[0].conv = new_conv
        embed_dim = pretrained_med.proj_head[0].in_features  # 1024
        self.feature_dim = embed_dim
        self.backbone = pretrained_med
        self.backbone.proj_head = nn.Identity()

        # ==========================================
        # 2. QUS 異質維度對齊 (Parameter-Specific Encoders)
        # ==========================================
        self.scalar_norm = nn.BatchNorm1d(num_scalars)
        
        # 🌟 核心創新：利用 ModuleDict 動態生成專屬編碼器
        self.qus_encoders = nn.ModuleDict({
            'tai': nn.Sequential(nn.Linear(5, self.feature_dim//2), nn.GELU()),
            'tsi': nn.Sequential(nn.Linear(5, self.feature_dim//2), nn.GELU())
        })
        if self.num_qus_types >= 3:
            self.qus_encoders['swe'] = nn.Sequential(nn.Linear(5, self.feature_dim//2), nn.GELU())
        if self.num_qus_types >= 4:
            # EZHRI 只有 1 維，但我們一樣能把它映射成 1024 維的 Token！
            self.qus_encoders['ezhri'] = nn.Sequential(nn.Linear(1, self.feature_dim//2), nn.GELU())

        self.shared_qus_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim//2),
            nn.Linear(self.feature_dim//2, self.feature_dim)
        )

        # ==========================================
        # 3. 雙向 Cross-Attention Fusion
        #    Fa: [B, 49, 1024]  (影像空間 token，來自 7×7 特徵圖)
        #    Fb: [B, num_qus_types, 1024]  (QUS token)
        # ==========================================
        self.cross_attn_img2qus = CrossAttentionBlock(
            dim=self.feature_dim, num_heads=num_heads, dropout=dropout
        )
        self.cross_attn_qus2img = CrossAttentionBlock(
            dim=self.feature_dim, num_heads=num_heads, dropout=dropout
        )

        # ==========================================
        # 4. 跨模態深層特徵萃取
        #    GAP(Fa) + GAP(Fb) -> [B, 2048] -> MLP -> [B, 64]
        # ==========================================
        self.fc_features = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(384, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # ==========================================
        # 5. 最終決策層 (晚期注入公式: 64 + 1 = 65 維)
        # ==========================================
        self.fc_decision = nn.Linear(65, 1)

    def _get_spatial_tokens(self, img):
        """
        取得 MedViT norm 之後、avgpool 之前的空間特徵，轉成 token 序列。

        MedViT.forward() 流程:
            x = self.stem(img)          # [B, 64,  56, 56]
            x = self.features(x)        # [B, 1024, 7,  7]
            x = self.norm(x)            # [B, 1024, 7,  7]  ← 我們要這裡的輸出
            x = self.avgpool(x)         # [B, 1024, 1,  1]
            x = torch.flatten(x, 1)     # [B, 1024]
            x = self.proj_head(x)       # [B, num_classes]  (已換成 Identity)

        作法：直接呼叫 backbone.stem / backbone.features / backbone.norm，
        繞過 avgpool，然後把 [B, 1024, 7, 7] 攤平成 [B, 49, 1024]。
        """
        x = self.backbone.stem(img)       # [B, 64,  56, 56]
        x = self.backbone.features(x)     # [B, 1024, 7,  7]
        x = self.backbone.norm(x)         # [B, 1024, 7,  7]
        # [B, C, H, W] -> [B, H*W, C] = [B, 49, 1024]
        tokens = x.flatten(2).transpose(1, 2)
        return tokens

    def forward(self, img, scalars, return_attn=False):
        B = img.shape[0]

        # --- 專家公式 ---
        tai = scalars[:, :5]
        tsi = scalars[:, 5:10]
        tai_mean = tai.mean(dim=1)
        tsi_mean = tsi.mean(dim=1)
        expert_usff = (-44.3 + 41.9 * tai_mean + 0.23 * tsi_mean) / 100.0

        # --- 影像分支: 空間 token [B, 49, 1024] ---
        Fa = self._get_spatial_tokens(img)

        # --- QUS 分支: [B, 10] 
        scalar_normed = self.scalar_norm(scalars)
        tai_feats = scalar_normed[:, 0:5]
        tsi_feats = scalar_normed[:, 5:10]
        
        # 動態收集 Tokens
        qus_list = [
            self.qus_encoders['tai'](tai_feats),
            self.qus_encoders['tsi'](tsi_feats)
        ]
        if self.num_qus_types >= 3:
            swe_feats = scalar_normed[:, 10:15]
            qus_list.append(self.qus_encoders['swe'](swe_feats))
            
        if self.num_qus_types >= 4:
            ezhri_feats = scalar_normed[:, 15:16]
            qus_list.append(self.qus_encoders['ezhri'](ezhri_feats))

        # 將列表堆疊成 Token 序列: [B, num_qus_types, 1024]
        stacked_qus = torch.stack(qus_list, dim=1)
        Fb = self.shared_qus_proj(stacked_qus)                     

        # --- 雙向 Cross-Attention ---
        Fa_fused = self.cross_attn_img2qus(x=Fa, context=Fb)    # [B, 49, 1024]
        Fb_fused, qus2img_attn = self.cross_attn_qus2img(
            x=Fb, context=Fa, return_attn=True
        )  # Fb_fused: [B, num_qus_types, 1024], qus2img_attn: [B, num_qus_types, 49]

        # --- GAP 後 Concat ---
        Fa_pooled = Fa_fused.mean(dim=1)                         # [B, 1024]
        Fb_pooled = Fb_fused.mean(dim=1)                         # [B, 1024]
        fused_base = torch.cat((Fa_pooled, Fb_pooled), dim=1)   # [B, 2048]

        # --- 深層特徵萃取 ---
        deep_features = self.fc_features(fused_base)             # [B, 64]

        # --- 晚期注入專家公式 ---
        combined = torch.cat((deep_features, expert_usff.unsqueeze(1)), dim=1)  # [B, 65]

        output = self.fc_decision(combined)                      # [B, 1]

        if return_attn:
            return output, qus2img_attn
        return output