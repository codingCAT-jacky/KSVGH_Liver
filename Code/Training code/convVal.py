from convUtils import *  
import convModel 
from collections import defaultdict
import torch
import torch.nn as nn, torch.optim as optim
import convDataset  
import numpy as np
from torch.utils.data import  DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, r2_score
import cv2
# import matlab.engine
# import usFilter
from torch.utils.tensorboard import SummaryWriter
import time
import torch.optim.lr_scheduler as lr_scheduler
import os
import torchvision.models as models
from scipy.stats import pearsonr
from MedViT import MedViT_small 

def create_optimizer_vgg(model, requires_grad, T_max):
    for param in model.image_extractor.parameters():
        param.requires_grad = requires_grad
    optimizer = optim.AdamW([
        # 老手：步伐極小，只做微調配合
        {'params': model.image_extractor.parameters(), 'lr': BASE_LR * LR_FACTOR},
        # 其他融合層：步伐調降，進行精細雕琢
        {'params': model.scalar_norm.parameters(), 'lr': BASE_LR},
        {'params': model.fc_features.parameters(), 'lr': BASE_LR},
        {'params': model.fc_decision.parameters(), 'lr': BASE_LR}
    ], weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=BASE_LR * 1e-2)
    return optimizer, scheduler

def create_optimizer_vgg_attn(model, requires_grad, T_max):
    for param in model.image_extractor.parameters():
        param.requires_grad = requires_grad
    optimizer = optim.AdamW([
        {'params': model.image_extractor.parameters(), 'lr': BASE_LR * LR_FACTOR},
        {'params': model.scalar_norm.parameters(), 'lr': BASE_LR},
        {'params': model.qus_encoders.parameters(),         'lr': BASE_LR},
        {'params': model.shared_qus_proj.parameters(),      'lr': BASE_LR},
        {'params': model.cross_attn_img2qus.parameters(), 'lr': BASE_LR},
        {'params': model.cross_attn_qus2img.parameters(), 'lr': BASE_LR},
        {'params': model.fc_features.parameters(), 'lr': BASE_LR},
        {'params': model.fc_decision.parameters(), 'lr': BASE_LR}
    ], weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=BASE_LR * 1e-2)
    return optimizer, scheduler

def create_optimizer_conv(model, requires_grad, T_max):
    for param in model.image_extractor.parameters():
        param.requires_grad = requires_grad
    optimizer = optim.AdamW([
        # 老手：步伐極小，只做微調配合
        {'params': model.image_extractor.parameters(), 'lr': BASE_LR * LR_FACTOR},
        # 其他融合層：步伐調降，進行精細雕琢
        {'params': model.scalar_norm.parameters(), 'lr': BASE_LR},
        {'params': model.fc_features.parameters(), 'lr': BASE_LR},
        {'params': model.fc_decision.parameters(), 'lr': BASE_LR}
    ], weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=BASE_LR * 1e-2)
    return optimizer, scheduler

def create_optimizer_conv_Attn(model, requires_grad, T_max):
    for param in model.image_extractor.parameters():
        param.requires_grad = requires_grad
    optimizer = optim.AdamW([
        # 老手：步伐極小，只做微調配合 (ConvNeXt backbone)
        {'params': model.image_extractor.parameters(), 'lr': BASE_LR * LR_FACTOR},
        # 其他融合層：步伐調降，進行精細雕琢
        {'params': model.scalar_norm.parameters(), 'lr': BASE_LR},
        {'params': model.qus_encoders.parameters(),         'lr': BASE_LR},
        {'params': model.shared_qus_proj.parameters(),      'lr': BASE_LR},
        {'params': model.cross_attn_img2qus.parameters(), 'lr': BASE_LR},
        {'params': model.cross_attn_qus2img.parameters(), 'lr': BASE_LR},
        {'params': model.fc_features.parameters(), 'lr': BASE_LR},
        {'params': model.fc_decision.parameters(), 'lr': BASE_LR}
    ], weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=BASE_LR * 1e-2)
    return optimizer, scheduler

def create_optimizer_med(model, requires_grad, T_max):
    """
    MultiModalMed 用的 optimizer。
    backbone.stem / features / norm 凍結或解凍（預訓練部分）。
    proj_head 已被換成 Identity，不含可訓練參數，不需列入。
    """
    for param in model.backbone.stem.parameters():
        param.requires_grad = requires_grad
    for param in model.backbone.features.parameters():
        param.requires_grad = requires_grad
    for param in model.backbone.norm.parameters():
        param.requires_grad = requires_grad
    backbone_params = (
        list(model.backbone.stem.parameters()) +
        list(model.backbone.features.parameters()) +
        list(model.backbone.norm.parameters())
    )
    optimizer = optim.AdamW([
        {'params': backbone_params,                'lr': BASE_LR * LR_FACTOR},
        {'params': model.scalar_norm.parameters(), 'lr': BASE_LR},
        {'params': model.fc_features.parameters(), 'lr': BASE_LR},
        {'params': model.fc_decision.parameters(), 'lr': BASE_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=BASE_LR * 1e-2)
    return optimizer, scheduler


def create_optimizer_med_attn(model, requires_grad, T_max):
    """
    MultiModalAttnMed 用的 optimizer。
    backbone.stem / features / norm 凍結或解凍（預訓練部分）。
    """
    for param in model.backbone.stem.parameters():
        param.requires_grad = requires_grad
    for param in model.backbone.features.parameters():
        param.requires_grad = requires_grad
    for param in model.backbone.norm.parameters():
        param.requires_grad = requires_grad
    backbone_params = (
        list(model.backbone.stem.parameters()) +
        list(model.backbone.features.parameters()) +
        list(model.backbone.norm.parameters())
    )
    optimizer = optim.AdamW([
        {'params': backbone_params,                         'lr': BASE_LR * LR_FACTOR},
        {'params': model.scalar_norm.parameters(),          'lr': BASE_LR},
        {'params': model.qus_encoders.parameters(),         'lr': BASE_LR},
        {'params': model.shared_qus_proj.parameters(),      'lr': BASE_LR},
        {'params': model.cross_attn_img2qus.parameters(),   'lr': BASE_LR},
        {'params': model.cross_attn_qus2img.parameters(),   'lr': BASE_LR},
        {'params': model.fc_features.parameters(),          'lr': BASE_LR},
        {'params': model.fc_decision.parameters(),          'lr': BASE_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=BASE_LR * 1e-2)
    return optimizer, scheduler




def evaluate_patient_level(model, val_loader, criterion, device):
    model.eval()
    
    patient_preds_dict = defaultdict(list)
    patient_targets_dict = {}
    all_img_preds = []
    all_img_targets = []

    with torch.no_grad():
        for img, targets, patient_ids, taitsi, swe, ezhri in val_loader:
            img = img.to(device)
            targets = targets.to(device)
            taitsi = taitsi.to(device)
            
            with torch.amp.autocast('cuda'):
                # [關鍵] 傳入雙模態資料 (影像, 數值)
                if CURRENT_MODE == MODE_BASE:              
                    outputs = model(img)      # (B,1)
                elif CURRENT_MODE in [MODE_MULTI, MODE_MULTI_ATTN]:
                    if NUM_QUS_TYPES == 2:
                        outputs = model(img, taitsi)      
                    elif NUM_QUS_TYPES == 3:
                        swe = swe.to(device)
                        three_qus = torch.cat([taitsi, swe], dim=1)
                        outputs = model(img, three_qus)   
                    elif NUM_QUS_TYPES == 4:
                        swe = swe.to(device)      
                        ezhri = ezhri.to(device)      
                        four_qus = torch.cat([taitsi, swe, ezhri], dim=1)
                        outputs = model(img, four_qus) 
            
            all_img_preds.append(outputs.float())
            all_img_targets.append(targets.unsqueeze(1).float())
            
            pids = patient_ids.numpy() if isinstance(patient_ids, torch.Tensor) else patient_ids
            
            for i, pid in enumerate(pids):
                patient_preds_dict[pid].append(outputs[i].float())
                patient_targets_dict[pid] = targets[i].float()

    # --- 1. 計算 Image-level Metrics ---
    full_img_preds = torch.cat(all_img_preds, dim=0)
    full_img_targets = torch.cat(all_img_targets, dim=0)
    image_val_loss = criterion(full_img_preds, full_img_targets).item()
    # print(f"len full_img_preds {len(full_img_preds)}, len full_img_targets {len(full_img_targets)} image_val_loss {image_val_loss}")

    img_preds_np = full_img_preds.detach().cpu().numpy().flatten()
    img_targets_np = full_img_targets.detach().cpu().numpy().flatten()
    img_pred_cls = pdff_to_class(img_preds_np)
    img_true_cls = pdff_to_class(img_targets_np)
    
    image_acc = np.mean(img_pred_cls == img_true_cls) 
    image_mae = np.mean(np.abs(img_preds_np - img_targets_np))
    image_cm = confusion_matrix(img_true_cls, img_pred_cls, labels=[0,1,2,3])

    # --- 2. 計算 Patient-level Metrics ---
    final_preds_list = []
    final_targets_list = []
    sorted_pids = sorted(patient_preds_dict.keys())
    
    for pid in sorted_pids:
        preds = patient_preds_dict[pid] 
        avg_pred = torch.stack(preds).mean() 
        final_preds_list.append(avg_pred)
        final_targets_list.append(patient_targets_dict[pid])

    pat_preds_t = torch.stack(final_preds_list).unsqueeze(1)   
    pat_targets_t = torch.stack(final_targets_list).unsqueeze(1) 
    patient_val_loss = criterion(pat_preds_t, pat_targets_t).item()
    # print(f"len pat_preds_t {len(pat_preds_t)}, len pat_targets_t {len(pat_targets_t)} patient_val_loss {patient_val_loss}")

    final_preds_np = pat_preds_t.detach().cpu().numpy().flatten()
    final_targets_np = pat_targets_t.detach().cpu().numpy().flatten()

    mae = np.mean(np.abs(final_preds_np - final_targets_np))
    y_pred_cls = pdff_to_class(final_preds_np)
    y_true_cls = pdff_to_class(final_targets_np)
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0,1,2,3])
    overall_acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0

    return cm, mae, overall_acc, final_preds_np, final_targets_np, image_val_loss, patient_val_loss, image_acc, image_mae, image_cm



if __name__ == "__main__":
    # 取第一組分割；可自行更換
    train_idx, val_idx = convDataset.splits[0]  

    # 1. 讀取資料、建立 Dataset 和 DataLoader
    train_list = convDataset.build_imagelist(train_idx)
    val_list   = convDataset.build_imagelist(val_idx)
    train_ds = convDataset.PDFFDataset(train_list, isTrain=True)  # 你的 Dataset
    val_ds   = convDataset.PDFFDataset(val_list, isTrain=False)
    train_image_targets = [item.pdffClass for item in train_ds.dataList] 
    train_image_targets = np.array(train_image_targets)
    unique_classes, counts = np.unique(train_image_targets, return_counts=True)
    class_weight_dict = {}
    for cls, count in zip(unique_classes, counts):
        class_weight_dict[cls] = 1.0 / count
    samples_weight = np.array([class_weight_dict[t] for t in train_image_targets])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, shuffle=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
    print(f"Train imgs: {len(train_ds)} | Val imgs: {len(val_ds)}")
    print("Train dist:", count_by_class(train_idx, convDataset.labels_cls))
    print("Val   dist:", count_by_class(val_idx,   convDataset.labels_cls))


    # 2. 設定model、訓練參數等
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if CURRENT_MODEL == MODEL_CONVNEXT:
        pretrained_convnext = models.convnext_tiny(weights='DEFAULT')
        if CURRENT_MODE == MODE_BASE:
            multi_model = convModel.BaseConv(pretrained_convnext)
        elif CURRENT_MODE == MODE_MULTI:
            multi_model = convModel.MultiModalConv(pretrained_convnext, num_scalars=NUM_SCALARS)
        elif CURRENT_MODE == MODE_MULTI_ATTN:
            multi_model = convModel.MultiModalAttnConv(pretrained_convnext, num_scalars=NUM_SCALARS, num_qus_types=NUM_QUS_TYPES)
    elif CURRENT_MODEL == MODEL_MEDVIT:
        pretrained_med = MedViT_small()  # MedViT_small，num_classes=4（原始碼預設）
        pretrained_med.load_state_dict(torch.load(MEDVIT_LOAD_PRETEAINMODEL_PATH), strict=False)
        if CURRENT_MODE == MODE_BASE:
            multi_model = convModel.BaseMed(pretrained_med)
        elif CURRENT_MODE == MODE_MULTI:
            multi_model = convModel.MultiModalMed(pretrained_med, num_scalars=NUM_SCALARS)
        elif CURRENT_MODE == MODE_MULTI_ATTN:
            multi_model = convModel.MultiModalAttnMed(pretrained_med, num_scalars=NUM_SCALARS, num_qus_types=NUM_QUS_TYPES)
    elif CURRENT_MODEL == MODEL_VGG:
        pretrained_vgg = models.vgg16(weights='DEFAULT')  # 有使用預訓練權重
        if CURRENT_MODE == MODE_BASE:
            multi_model = convModel.BaseVGG(pretrained_vgg)
        elif CURRENT_MODE == MODE_MULTI:
            multi_model = convModel.MultiModalVGG(pretrained_vgg, num_scalars=NUM_SCALARS)
        elif CURRENT_MODE == MODE_MULTI_ATTN:
            multi_model = convModel.MultiModalAttnVGG(pretrained_vgg, num_scalars=NUM_SCALARS, num_qus_types=NUM_QUS_TYPES)
    # 👇 只需要在這裡寫一次 load，它會自動去 utils 抓對應的字典路徑！
    multi_model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device))
    multi_model = multi_model.to(device)
    # elif CURRENT_MODEL == MODEL_MEDVIT:
    criterion = nn.MSELoss()



    # --- 執行 Training Set 評估 ---
    # print("\n[*] Running evaluation on Training Set...")
    # train_cm, train_mae, train_acc, train_preds, train_targets, train_img_loss, train_pat_loss, train_img_acc, train_img_mae, train_img_cm = evaluate_patient_level(
    #     multi_model, train_loader, criterion, device
    # )
    # --- 執行 Validation Set 評估 ---
    print("[*] Running evaluation on Validation Set...")
    val_cm, val_mae, val_acc, val_preds, val_targets, val_img_loss, val_pat_loss, val_img_acc, val_img_mae, val_img_cm = evaluate_patient_level(
        multi_model, val_loader, criterion, device
    )
    # 計算 Pearson 相關係數
    # train_r, train_p = pearsonr(train_targets, train_preds)
    val_r, val_p = pearsonr(val_targets, val_preds)

    # ==========================================
    # 輸出結果
    # ==========================================
    # print("\n" + "="*50)
    # print(" 🎯 training RESULTS 🎯")
    # print("="*50)
    # print(f" - Image-level")
    # print(f"MSE Loss     : {train_img_loss:.6f}")
    # print(f"分類準確率    : {train_img_acc*100:.2f}%")
    # print(f"預測脂肪肝百分比 平均誤差: {train_img_mae*100:.2f}%")
    # print("-" * 50)
    # print(f" - Patient-level")
    # print(f"分類準確率    : {train_acc*100:.2f}%")
    # print("\n" + "="*50)
    print(" 🏆 VALIDATION RESULTS 🏆")
    print("="*50)
    print(f" - Image-level")
    print(f"MSE Loss     : {val_img_loss:.6f}")
    print(f"分類準確率    : {val_img_acc*100:.2f}%")
    print(f"預測脂肪肝百分比 平均誤差: {val_img_mae*100:.2f}%")
    print("-" * 50)
    print(f" - Patient-level")
    print(f"MSE Loss     : {val_pat_loss:.6f}")
    print(f"分類準確率    : {val_acc*100:.2f}%")
    print(f"預測脂肪肝百分比 平均誤差: {val_mae*100:.2f}%")
    print(f"與 MRI-PDFF 相關性 (Pearson r): {val_r:.4f} (p-value: {val_p:.2e})")
    print("="*50)
    # ==========================================
    # 計算 AUC 與各項診斷指標 (Validation Set)
    # ==========================================
    thresholds_dict = {
        "S1_Mild (>=0.064)": 0.064,      
        "S2_Moderate (>=0.174)": 0.174,  
        "S3_Severe (>=0.221)": 0.221     
        # "S1_Mild (>=0.05)": 0.05,      
        # "S2_Moderate (>=0.15)": 0.15,  
        # "S3_Severe (>=0.25)": 0.25    
    }

    print("\n" + "="*100)
    print(" 📊 Diagnostic Performance (Validation Patient-level)")
    print("="*100)
    print(f"{'Task':<22} | {'AUC':<6} | {'Cutoff':<6} | {'Sens(%)':<14} | {'Spec(%)':<14} | {'PPV(%)':<14} | {'NPV(%)':<14}")
    print("-" * 100)
    for name, thresh in thresholds_dict.items():
        binary_true = (val_targets >= thresh).astype(int)
        
        if len(np.unique(binary_true)) > 1:
            auc_score = roc_auc_score(binary_true, val_preds)
            fpr, tpr, roc_thresholds = roc_curve(binary_true, val_preds)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            optimal_cutoff = roc_thresholds[optimal_idx]
            
            binary_pred = (val_preds >= optimal_cutoff).astype(int)
            tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred).ravel()
            
            sens_num, sens_den = tp, tp + fn
            spec_num, spec_den = tn, tn + fp
            ppv_num, ppv_den = tp, tp + fp
            npv_num, npv_den = tn, tn + fn
            
            sens_pct = (sens_num / sens_den * 100) if sens_den > 0 else 0.0
            spec_pct = (spec_num / spec_den * 100) if spec_den > 0 else 0.0
            ppv_pct  = (ppv_num / ppv_den * 100) if ppv_den > 0 else 0.0
            npv_pct  = (npv_num / npv_den * 100) if npv_den > 0 else 0.0
            
            sens_str = f"{sens_pct:.0f} ({sens_num}/{sens_den})"
            spec_str = f"{spec_pct:.0f} ({spec_num}/{spec_den})"
            ppv_str  = f"{ppv_pct:.0f} ({ppv_num}/{ppv_den})"
            npv_str  = f"{npv_pct:.0f} ({npv_num}/{npv_den})"
            
            print(f"{name:<22} | {auc_score:.4f} | {optimal_cutoff:.4f} | {sens_str:<14} | {spec_str:<14} | {ppv_str:<14} | {npv_str:<14}")
        else:
            print(f"{name:<22} | N/A (Only one class present in Val Set)")
    print("="*100)

    # 儲存圖片時順便幫你確認路徑存在 (os.makedirs 已補上在 plot 內)
    plot_confusion_matrix(val_img_cm, ["S0", "S1", "S2", "S3"], title="Val Image-level CM", filename="./picture/convVal_image_cm.png", save=True)
    plot_confusion_matrix(val_cm, ["S0", "S1", "S2", "S3"], title="Val Patient-level CM", filename="./picture/convVal_patient_cm.png", save=True)
    plot_bland_altman(val_preds, val_targets, title="Validation Bland-Altman Plot", filename="./picture/convVal_bland_altman.png")
    plot_correlation_scatter(val_preds, val_targets, title="Correlation between Prediction and MRI-PDFF", filename="./picture/convVal_correlation.png")
    print("\n✅ 所有結果輸出完畢，並已存下 Validation 的 Confusion Matrix 圖片！")