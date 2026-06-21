"""
convGradCAM.py  ─  支援最新架構的 Grad-CAM (自動適應 VGG/ConvNeXt/MedViT 與多種 QUS 維度)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.models as models
from MedViT import MedViT_small
import convModel
import convDataset
from convUtils import *

# ─────────────────────────────────────────────
# 1. Regression Target & Wrapper
# ─────────────────────────────────────────────
class PDFFRegressionTarget:
    def __call__(self, output):
        return output[0]

class MultiModalWrapper(nn.Module):
    """
    用於 GradCAM：固定 QUS 輸入，只暴露 img 給 pytorch-grad-cam。
    強制 return_attn=False（即使是 MODE_MULTI_ATTN 模型），確保 forward 只回傳
    [B,1] 的純量 tensor，符合 GradCAM 對 model 輸出格式的要求。
    """
    def __init__(self, model, fixed_qus: torch.Tensor, is_attn_model: bool = False):
        super().__init__()
        self.model = model
        self.fixed_qus = fixed_qus
        self.is_attn_model = is_attn_model

    def forward(self, img):
        if self.is_attn_model:
            return self.model(img, self.fixed_qus, return_attn=False)
        return self.model(img, self.fixed_qus)

# ─────────────────────────────────────────────
# 2. 智慧尋找 Target Layer
# ─────────────────────────────────────────────
def get_target_layer(model):
    """
    根據不同 Backbone 結構動態尋找最後一個卷積層作為 CAM 的 Target。
    MODE_MULTI 與 MODE_MULTI_ATTN 的影像分支屬性名稱一致 (image_extractor / backbone)，
    故不需特別分支，仍依 CURRENT_MODEL 判斷即可。
    """
    if CURRENT_MODEL == MODEL_CONVNEXT:
        # ConvNeXt 的 features 模組最後一塊 (MODE_MULTI / MODE_MULTI_ATTN 皆同名)
        return model.image_extractor[-1] if CURRENT_MODE == MODE_MULTI_ATTN else model.image_extractor.features[-1]
    elif CURRENT_MODEL == MODEL_VGG:
        # VGG features 的最後一層通常是 MaxPool，我們取倒數第二層 Conv2d (index 28)
        return model.image_extractor[28]
    elif CURRENT_MODEL == MODEL_MEDVIT:
        # MedViT 的空間特徵是從 norm 輸出的 (MODE_MULTI / MODE_MULTI_ATTN 皆同名)
        return model.backbone.norm
    else:
        raise ValueError("Unknown Target Layer for Current Model")

# ─────────────────────────────────────────────
# 3. 影像前處理
# ─────────────────────────────────────────────
def preprocess_image(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    img = img / 255.0

    H, W, _ = img.shape
    cx, cy   = W // 2, H // 2
    hw, hh   = 270, 270
    x1, y1   = max(0, cx - hw), max(0, cy - hh)
    x2, y2   = min(W, cx + hw), min(H, cy + hh)
    img      = img[y1:y2, x1:x2, :].astype(np.float32)
    img01    = img[..., 0]          

    transform = A.Compose([A.Resize(224, 224), ToTensorV2()])
    img_tensor = transform(image=img01)["image"].unsqueeze(0).float()  

    img_vis = cv2.resize(img01, (224, 224))
    img_vis = np.stack([img_vis] * 3, axis=-1).astype(np.float32)
    img_vis = np.clip(img_vis, 0.0, 1.0)

    return img_tensor, img_vis

# ─────────────────────────────────────────────
# 4. 運算與繪圖邏輯
# ─────────────────────────────────────────────
def compute_gradcam(model, img_tensor, qus_tensor, device):
    """
    一般 Grad-CAM。同時支援 MODE_MULTI 與 MODE_MULTI_ATTN 模型：
    對於 Attn 模型，forward 時強制 return_attn=False，確保輸出為單一 [B,1] tensor，
    這樣 Grad-CAM 才能對它做反向傳播。
    """
    model.eval()
    img_tensor = img_tensor.to(device)

    if qus_tensor.dim() == 1:
        qus_tensor = qus_tensor.unsqueeze(0)

    is_attn_model = (CURRENT_MODE == MODE_MULTI_ATTN)
    cam_model = MultiModalWrapper(model, qus_tensor.to(device), is_attn_model=is_attn_model).to(device)
    target_layer = get_target_layer(model)

    cam = GradCAM(model=cam_model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor, targets=[PDFFRegressionTarget()])
    return grayscale_cam[0]   

def get_prediction(model, img_tensor, qus_tensor, device):
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        if qus_tensor.dim() == 1:
            qus_tensor = qus_tensor.unsqueeze(0)
        pred = model(img_tensor, qus_tensor.to(device)).item()
    return pred

# ─────────────────────────────────────────────
# 4.5 Cross-Attention CAM (僅適用於 MODE_MULTI_ATTN)
#
#     利用 QUS→Image 的 cross-attention weights 作為熱力圖，
#     不需要梯度，直接反映「每個 QUS token 關注 B-mode 影像的哪個空間位置」。
#     49 個空間位置 -> reshape 成 7x7 -> upsample -> 224x224
# ─────────────────────────────────────────────
def get_qus_token_names(num_qus_types):
    names = ["TAI", "TSI"]
    if num_qus_types >= 3:
        names.append("SWE")
    if num_qus_types >= 4:
        names.append("EzHRI")
    return names[:num_qus_types]


def compute_crossattn_cam(model, img_tensor, qus_tensor, device):
    """
    回傳每個 QUS token 對應的空間注意力熱力圖 (已 upsample 到 224x224，正規化到 [0,1])

    Returns
    -------
    cam_list : list of np.ndarray，每個 shape [224, 224]，順序對應 get_qus_token_names()
    """
    model.eval()
    img_tensor = img_tensor.to(device)
    if qus_tensor.dim() == 1:
        qus_tensor = qus_tensor.unsqueeze(0)
    qus_tensor = qus_tensor.to(device)

    with torch.no_grad():
        _, qus2img_attn = model(img_tensor, qus_tensor, return_attn=True)
        # qus2img_attn: [1, num_qus_types, 49]

    attn = qus2img_attn[0].detach().cpu().numpy()  # [num_qus_types, 49]
    num_qus_types = attn.shape[0]

    cam_list = []
    for i in range(num_qus_types):
        spatial = attn[i].reshape(7, 7)  # [7, 7]
        # 正規化到 [0, 1]（每個 token 各自正規化，方便比較相對熱區）
        spatial = spatial - spatial.min()
        denom = spatial.max()
        if denom > 1e-8:
            spatial = spatial / denom
        heatmap = cv2.resize(spatial, (224, 224), interpolation=cv2.INTER_LINEAR)
        cam_list.append(heatmap.astype(np.float32))

    return cam_list


def show_crossattn_cam(img_vis, cam_list, qus_token_names, pred=None, true_val=None, save_path=None):
    """
    並排顯示：原圖 + 每個 QUS token 的 cross-attention 熱力圖
    """
    n = len(cam_list)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

    axes[0].imshow(img_vis[..., 0], cmap="gray")
    title0 = "Original"
    if true_val is not None:
        title0 += f"\nTrue: {true_val*100:.1f}%"
    if pred is not None:
        title0 += f"\nPred: {pred*100:.1f}%"
    axes[0].set_title(title0)
    axes[0].axis("off")

    for i, (cam, name) in enumerate(zip(cam_list, qus_token_names)):
        overlay = show_cam_on_image(img_vis, cam, use_rgb=True)
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f"{name} → Image\n(Cross-Attn)")
        axes[i + 1].axis("off")

    plt.tight_layout()
    # if save_path:
    #     os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    #     plt.savefig(save_path, dpi=150, bbox_inches="tight")
    #     print(f"Saved → {save_path}")
    plt.show()


def compare_crossattn_cam(img_vis, cam_list_a, cam_list_b, qus_token_names,
                          pred_a=None, pred_b=None,
                          label_a="Normal QUS", label_b="Extreme QUS",
                          save_path=None):
    """
    對比兩種 QUS 輸入下，每個 QUS token 的 cross-attention 熱力圖差異
    每個 QUS token 一列：[原圖 | Normal heatmap | Extreme heatmap | Diff]
    """
    n = len(cam_list_a)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]  # 保持 2D 索引一致

    for i, name in enumerate(qus_token_names):
        cam_a = cam_list_a[i]
        cam_b = cam_list_b[i]
        overlay_a = show_cam_on_image(img_vis, cam_a, use_rgb=True)
        overlay_b = show_cam_on_image(img_vis, cam_b, use_rgb=True)
        diff = cam_b.astype(np.float32) - cam_a.astype(np.float32)

        axes[i, 0].imshow(img_vis[..., 0], cmap="gray")
        axes[i, 0].set_title(f"{name}\nOriginal")
        axes[i, 0].axis("off")

        title_a = label_a if pred_a is None else f"{label_a}\nPred: {pred_a*100:.1f}%"
        axes[i, 1].imshow(overlay_a)
        axes[i, 1].set_title(title_a)
        axes[i, 1].axis("off")

        title_b = label_b if pred_b is None else f"{label_b}\nPred: {pred_b*100:.1f}%"
        axes[i, 2].imshow(overlay_b)
        axes[i, 2].set_title(title_b)
        axes[i, 2].axis("off")

        vmax = max(abs(diff.min()), abs(diff.max())) + 1e-8
        im = axes[i, 3].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i, 3].set_title("Diff (Extreme − Normal)")
        axes[i, 3].axis("off")
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()

def compare_heatmaps(img_vis, cam_a, cam_b, pred_a=None, pred_b=None, label_a="Normal QUS", label_b="Extreme QUS", save_path=None):
    overlay_a = show_cam_on_image(img_vis, cam_a, use_rgb=True)
    overlay_b = show_cam_on_image(img_vis, cam_b, use_rgb=True)
    diff = cam_b.astype(np.float32) - cam_a.astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(img_vis[..., 0], cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    title_a = label_a if pred_a is None else f"{label_a}\nPred: {pred_a*100:.1f}%"
    axes[1].imshow(overlay_a)
    axes[1].set_title(title_a)
    axes[1].axis("off")

    title_b = label_b if pred_b is None else f"{label_b}\nPred: {pred_b*100:.1f}%"
    axes[2].imshow(overlay_b)
    axes[2].set_title(title_b)
    axes[2].axis("off")

    vmax = max(abs(diff.min()), abs(diff.max())) + 1e-8
    im = axes[3].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[3].set_title("Diff (Extreme − Normal)")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()

# ─────────────────────────────────────────────
# 5. 主程式
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # --- 動態載入模型與權重 (與 convVal 完全一致的邏輯) ---
    if CURRENT_MODEL == MODEL_CONVNEXT:
        pretrained_convnext = models.convnext_tiny(weights='DEFAULT')
        if CURRENT_MODE == MODE_MULTI:
            model = convModel.MultiModalConv(pretrained_convnext, num_scalars=NUM_SCALARS)
        elif CURRENT_MODE == MODE_MULTI_ATTN:
            model = convModel.MultiModalAttnConv(pretrained_convnext, num_scalars=NUM_SCALARS, num_qus_types=NUM_QUS_TYPES)
            
    elif CURRENT_MODEL == MODEL_MEDVIT:
        pretrained_med = MedViT_small()  
        pretrained_med.load_state_dict(torch.load(MEDVIT_LOAD_PRETEAINMODEL_PATH), strict=False)
        if CURRENT_MODE == MODE_MULTI:
            model = convModel.MultiModalMed(pretrained_med, num_scalars=NUM_SCALARS)
        elif CURRENT_MODE == MODE_MULTI_ATTN:
            model = convModel.MultiModalAttnMed(pretrained_med, num_scalars=NUM_SCALARS, num_qus_types=NUM_QUS_TYPES)
            
    elif CURRENT_MODEL == MODEL_VGG:
        pretrained_vgg = models.vgg16(weights='DEFAULT')  
        if CURRENT_MODE == MODE_MULTI:
            model = convModel.MultiModalVGG(pretrained_vgg, num_scalars=NUM_SCALARS)
        elif CURRENT_MODE == MODE_MULTI_ATTN:
            model = convModel.MultiModalAttnVGG(pretrained_vgg, num_scalars=NUM_SCALARS, num_qus_types=NUM_QUS_TYPES)
            
    model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model Loaded: {type(model).__name__} from {LOAD_MODEL_PATH}")

    # --- 讀取圖片 ---
    train_idx, val_idx = convDataset.splits[0]
    val_list = convDataset.build_imagelist(val_idx)
    item = val_list[280]   # 可自行更換圖片 idx
    print(f"patient is {item.patient_id}")
    img_tensor, img_vis = preprocess_image(item.imgpath)

    # --- 組裝 Normal QUS (動態支援 2, 3, 4 種特徵) ---
    qus_list = list(item.tai) + list(item.tsi)
    if NUM_QUS_TYPES >= 3 and item.swe is not None:
        qus_list += list(item.swe)
    if NUM_QUS_TYPES >= 4 and item.ezhri is not None:
        qus_list += list(item.ezhri)
        
    normal_qus = torch.tensor(qus_list, dtype=torch.float32)
    
    # --- 極端 QUS (全部填 0 測試) ---
    extreme_4qus = torch.tensor([1.04, 1.04, 1.04, 1.04, 1.04, 99.63, 99.63, 99.63, 99.63, 99.63, 5.97, 5.97, 5.97, 5.97, 5.97, 1.93], dtype=torch.float32)
    extreme_3qus = torch.tensor([1.04, 1.04, 1.04, 1.04, 1.04, 99.63, 99.63, 99.63, 99.63, 99.63, 5.97, 5.97, 5.97, 5.97, 5.97], dtype=torch.float32)
    extreme_2qus = torch.tensor([1.04, 1.04, 1.04, 1.04, 1.04, 99.63, 99.63, 99.63, 99.63, 99.63], dtype=torch.float32)
    extreme_qus = extreme_2qus

    
    # --- 計算並畫圖 ---
    # ★ 一般 Grad-CAM：MODE_MULTI 與 MODE_MULTI_ATTN 都支援
    print("Computing Normal QUS Grad-CAM ...")
    cam_normal  = compute_gradcam(model, img_tensor, normal_qus,  device)
    pred_normal = get_prediction(model, img_tensor, normal_qus,  device)

    print("Computing Extreme QUS Grad-CAM ...")
    cam_extreme  = compute_gradcam(model, img_tensor, extreme_qus, device)
    pred_extreme = get_prediction(model, img_tensor, extreme_qus, device)

    print(f"True PDFF   : {item.value*100:.1f}%")
    print(f"Pred Normal : {pred_normal*100:.1f}%")
    print(f"Pred Extreme: {pred_extreme*100:.1f}%")

    compare_heatmaps(
        img_vis, cam_normal, cam_extreme,
        pred_a=pred_normal, pred_b=pred_extreme,
        save_path=f"./picture/gradcam_{CURRENT_MODEL}_{CURRENT_MODE}.png",
    )

    # ★ Cross-Attention CAM：僅 MODE_MULTI_ATTN 額外提供（細粒度、每個 QUS token 各一張）
    if CURRENT_MODE == MODE_MULTI_ATTN:
        print("\nComputing Cross-Attention CAM (Normal QUS) ...")
        cam_list_normal = compute_crossattn_cam(model, img_tensor, normal_qus, device)

        print("Computing Cross-Attention CAM (Extreme QUS) ...")
        cam_list_extreme = compute_crossattn_cam(model, img_tensor, extreme_qus, device)

        qus_token_names = get_qus_token_names(NUM_QUS_TYPES)

        # 單獨顯示 Normal QUS 的各 token 熱力圖
        show_crossattn_cam(
            img_vis, cam_list_normal, qus_token_names,
            pred=pred_normal, true_val=item.value,
            save_path=f"./picture/crossattn_cam_{CURRENT_MODEL}_normal.png",
        )

        # Normal vs Extreme 對比
        compare_crossattn_cam(
            img_vis, cam_list_normal, cam_list_extreme, qus_token_names,
            pred_a=pred_normal, pred_b=pred_extreme,
            save_path=f"./picture/crossattn_cam_{CURRENT_MODEL}_compare.png",
        )