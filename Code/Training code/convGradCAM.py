"""
convGradCAM.py  ─  非 Attention 版 Grad-CAM
支援 MultiModalConv（10 或 20 個 QUS 數字）

典型用法：
    img_t, img_vis = preprocess_image("path/to/image.png")

    normal_qus  = torch.tensor([...], dtype=torch.float32)   # 10 或 20 個數字
    extreme_qus = torch.tensor([...], dtype=torch.float32)   # 自行替換

    cam_normal  = compute_gradcam(model, img_t, normal_qus,  device)
    cam_extreme = compute_gradcam(model, img_t, extreme_qus, device)

    compare_heatmaps(img_vis, cam_normal, cam_extreme,
                     label_a="Normal QUS", label_b="Extreme QUS")
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
import convModel
from convModel import MultiModalVGG
import convDataset
from convUtils import *


# ─────────────────────────────────────────────
# 1. Regression Target
# ─────────────────────────────────────────────
class PDFFRegressionTarget:
    """output: [B, 1]  →  scalar per sample，供 Grad-CAM 反向傳播"""
    def __call__(self, output):
        return output[0]


# ─────────────────────────────────────────────
# 2. MultiModal Wrapper
#    固定 QUS，只暴露 img 給 pytorch-grad-cam
# ─────────────────────────────────────────────
class MultiModalWrapper(nn.Module):
    def __init__(self, model, fixed_qus: torch.Tensor):
        """
        fixed_qus : [1, num_scalars]，已在正確 device 上
        """
        super().__init__()
        self.model = model
        self.fixed_qus = fixed_qus

    def forward(self, img):
        return self.model(img, self.fixed_qus)


# ─────────────────────────────────────────────
# 3. 取得 Target Layer
#    ConvNeXt-Tiny features[7] → [B, 768, 7, 7]
# ─────────────────────────────────────────────
def get_target_layer(model):
    return model.image_extractor.features[7]



# ─────────────────────────────────────────────
# 4. 影像前處理（與 convDataset.py 完全一致）
# ─────────────────────────────────────────────
def preprocess_image(img_path: str):
    """
    Returns
    -------
    img_tensor : [1, 1, 224, 224]  float32 Tensor（模型輸入）
    img_vis    : [224, 224, 3]     float32 numpy，值域 [0,1]（疊加熱力圖用）
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    img = img / 255.0

    # Center crop
    H, W, _ = img.shape
    cx, cy   = W // 2, H // 2
    hw, hh   = 270, 270
    x1, y1   = max(0, cx - hw), max(0, cy - hh)
    x2, y2   = min(W, cx + hw), min(H, cy + hh)
    img      = img[y1:y2, x1:x2, :].astype(np.float32)
    img01    = img[..., 0]          # 單通道

    transform = A.Compose([A.Resize(224, 224), ToTensorV2()])
    img_tensor = transform(image=img01)["image"].unsqueeze(0).float()  # [1,1,224,224]

    img_vis = cv2.resize(img01, (224, 224))
    img_vis = np.stack([img_vis] * 3, axis=-1).astype(np.float32)
    img_vis = np.clip(img_vis, 0.0, 1.0)

    return img_tensor, img_vis


# ─────────────────────────────────────────────
# 5. 計算 Grad-CAM
# ─────────────────────────────────────────────
def compute_gradcam(
    model,
    img_tensor: torch.Tensor,
    qus_tensor: torch.Tensor,
    device: str,
) -> np.ndarray:
    """
    Parameters
    ----------
    model      :  MultiModalConv
    img_tensor : [1, 1, 224, 224]
    qus_tensor : [num_scalars] 或 [1, num_scalars]
    device     : 'cuda' 或 'cpu'

    Returns
    -------
    cam_map : [224, 224]，float32，值域 [0, 1]
    """
    model.eval()
    img_tensor = img_tensor.to(device)


    if qus_tensor.dim() == 1:
        qus_tensor = qus_tensor.unsqueeze(0)
    cam_model    = MultiModalWrapper(model, qus_tensor.to(device))
    cam_model = cam_model.to(device)
    target_layer = model.image_extractor.features[7]


    cam = GradCAM(model=cam_model, target_layers=[target_layer])
    targets = [PDFFRegressionTarget()]
    grayscale_cam = cam(
        input_tensor=img_tensor,
        targets=targets
    )
    return grayscale_cam[0]   # [224, 224]


# ─────────────────────────────────────────────
# 6. 單張視覺化
# ─────────────────────────────────────────────
def show_heatmap(
    img_vis: np.ndarray,
    cam_map: np.ndarray,
    pred_pdff: float,
    true_pdff: float,
    title: str = "",
    save_path: str | None = None,
):
    overlay = show_cam_on_image(img_vis, cam_map, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_vis[..., 0], cmap="gray")
    axes[0].set_title(f"Original  True: {true_pdff*100:.1f}%")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM  Pred: {pred_pdff*100:.1f}%  {title}")
    axes[1].axis("off")

    plt.tight_layout()
    # if save_path:
    #     os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    #     plt.savefig(save_path, dpi=150, bbox_inches="tight")
        # print(f"Saved → {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# 7. 對比兩張熱力圖（Normal QUS vs Extreme QUS）
# ─────────────────────────────────────────────
def compare_heatmaps(
    img_vis: np.ndarray,
    cam_a: np.ndarray,
    cam_b: np.ndarray,
    pred_a: float | None = None,
    pred_b: float | None = None,
    label_a: str = "Normal QUS",
    label_b: str = "Extreme QUS",
    save_path: str | None = None,
):
    """
    並排顯示兩張熱力圖及差異圖（cam_b - cam_a）

    Parameters
    ----------
    cam_a, cam_b : [224, 224]，值域 [0, 1]
    pred_a/b     : 對應的模型預測值（小數，如 0.15）；可為 None
    """
    overlay_a = show_cam_on_image(img_vis, cam_a, use_rgb=True)
    overlay_b = show_cam_on_image(img_vis, cam_b, use_rgb=True)

    # 差異圖：正值表示 extreme QUS 使該區域更受關注
    diff = cam_b.astype(np.float32) - cam_a.astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    # cv2.imshow("img_vis", img_vis)  # Debug: 查看 img_vis 的內容和範圍

    # 原圖
    axes[0].imshow(img_vis[..., 0], cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Normal QUS 熱力圖
    title_a = label_a if pred_a is None else f"{label_a}\nPred: {pred_a*100:.1f}%"
    axes[1].imshow(overlay_a)
    axes[1].set_title(title_a)
    axes[1].axis("off")

    # Extreme QUS 熱力圖
    title_b = label_b if pred_b is None else f"{label_b}\nPred: {pred_b*100:.1f}%"
    axes[2].imshow(overlay_b)
    axes[2].set_title(title_b)
    axes[2].axis("off")

    # 差異圖（diverging colormap）
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
    plt.close()


# ─────────────────────────────────────────────
# 8. 取得模型預測值（不含梯度）
# ─────────────────────────────────────────────
def get_prediction(model, img_tensor, qus_tensor, device):
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        if qus_tensor.dim() == 1:
            qus_tensor = qus_tensor.unsqueeze(0)
        pred = model(img_tensor, qus_tensor.to(device)).item()

    return pred


# ─────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    SAVE_DIR   = "./picture"
    MODEL_PATH = CONV_MULTI_MODEL_PATH   # 改成你的模型路徑

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if MODEL_PATH == VGG_MULTI_MODEL_PATH:
        # pretrained_vgg = models.vgg16(weights='DEFAULT')
        # model = convModel.MultiModalVGG(pretrained_vgg, num_scalars=NUM_SCALARS)
        # model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    elif MODEL_PATH == CONV_MULTI_MODEL_PATH:
        pretrained_convnext = models.convnext_tiny(weights='DEFAULT')
        model = convModel.MultiModalConv(pretrained_convnext, num_scalars=NUM_SCALARS)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model: {type(model).__name__}")

    # ── 選一張驗證集圖片 ──────────────────────────────
    train_idx, val_idx = convDataset.splits[0]
    val_list   = convDataset.build_imagelist(
        val_idx, IMG_FOLDER, MASK_FOLDER,
        convDataset.labels_reg,
        convDataset.tai_values,
        convDataset.tsi_values,
    )
    item = val_list[280]   # 取第 0 張；可自行更換
    # print(f"patient id: {item.patient_id}, true PDFF: {item.value*100:.1f}%")
    img_tensor, img_vis = preprocess_image(item.imgpath)

    # ── 定義 QUS（支援 10 或 20 個數字）──────────────
    # 10 個數字（TAI×5 + TSI×5）
    normal_qus = torch.tensor(
        list(item.tai) + list(item.tsi),
        dtype=torch.float32,
    )

    # ★ 極端 QUS 由使用者自行修改下面這行 ★
    extreme_qus = torch.tensor(
        [0.0] * 5 + [0.0] * 5,   # 範例：全部設為 0
        dtype=torch.float32,
    )

    # ── 計算熱力圖 ────────────────────────────────────
    print("Computing Normal QUS Grad-CAM ...")
    cam_normal  = compute_gradcam(model, img_tensor, normal_qus,  device)
    pred_normal = get_prediction (model, img_tensor, normal_qus,  device)

    print("Computing Extreme QUS Grad-CAM ...")
    cam_extreme  = compute_gradcam(model, img_tensor, extreme_qus, device)
    pred_extreme = get_prediction (model, img_tensor, extreme_qus, device)

    print(f"True PDFF   : {item.value*100:.1f}%")
    print(f"Pred Normal : {pred_normal*100:.1f}%")
    print(f"Pred Extreme: {pred_extreme*100:.1f}%")

    # ── 對比顯示 ─────────────────────────────────────
    compare_heatmaps(
        img_vis,
        cam_normal, 
        cam_extreme,
        pred_a=pred_normal,
        pred_b=pred_extreme,
        label_a="Normal QUS",
        label_b="Extreme QUS",
        save_path=os.path.join(SAVE_DIR, "convGradcam.png"),
    )
