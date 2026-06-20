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
import sys
import torchvision.models as models
import convVal
from MedViT import MedViT_small 


if __name__ == "__main__":
    # 5 fold training
    for fold, (train_idx, val_idx) in enumerate(convDataset.splits, start=1):
        print(f"\n=== Fold {fold}/5 ===")
        if fold>1:
            continue
        current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        log_dir = os.path.join("./outcome", current_time)
        tb_writer = SummaryWriter(log_dir=log_dir)

        # 1. 讀取資料、建立 Dataset 和 DataLoader
        train_list = convDataset.build_imagelist(train_idx)
        val_list   = convDataset.build_imagelist(val_idx)

        train_ds = convDataset.PDFFDataset(train_list, isTrain=True)  
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

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)


        print(f"Train imgs: {len(train_ds)} | Val imgs: {len(val_ds)}")
        print("Train dist:", count_by_class(train_idx, convDataset.labels_cls))
        print("Val   dist:", count_by_class(val_idx,   convDataset.labels_cls))
        # 2. 設定model、訓練參數等
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if CURRENT_MODEL == MODEL_CONVNEXT:
            pretrained_convnext = models.convnext_tiny(weights='DEFAULT')
            if CURRENT_MODE == MODE_BASE:
                multi_model = convModel.BaseConv(pretrained_convnext)
                optimizer = optim.AdamW(multi_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=BASE_LR * 1e-2)
            elif CURRENT_MODE == MODE_MULTI:
                multi_model = convModel.MultiModalConv(pretrained_convnext, num_scalars=NUM_SCALARS)
                optimizer, scheduler = convVal.create_optimizer_conv(multi_model, requires_grad=False, T_max=NUM_EPOCHS)
            elif CURRENT_MODE == MODE_MULTI_ATTN:
                multi_model = convModel.MultiModalAttnConv(pretrained_convnext, num_scalars=NUM_SCALARS, num_qus_types=NUM_QUS_TYPES)
                optimizer, scheduler = convVal.create_optimizer_conv_Attn(multi_model, requires_grad=False, T_max=NUM_EPOCHS)
        elif CURRENT_MODEL == MODEL_MEDVIT:
            pretrained_med = MedViT_small()  # MedViT_small，num_classes=4（原始碼預設）
            pretrained_med.load_state_dict(torch.load(MEDVIT_LOAD_PRETEAINMODEL_PATH), strict=False)
            if CURRENT_MODE == MODE_BASE:
                multi_model = convModel.BaseMed(pretrained_med)
                optimizer = optim.AdamW(multi_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=BASE_LR * 1e-2)
            elif CURRENT_MODE == MODE_MULTI:
                multi_model = convModel.MultiModalMed(pretrained_med, num_scalars=NUM_SCALARS)
                optimizer, scheduler = convVal.create_optimizer_med(multi_model, requires_grad=False, T_max=NUM_EPOCHS)
            elif CURRENT_MODE == MODE_MULTI_ATTN:
                multi_model = convModel.MultiModalAttnMed(pretrained_med, num_scalars=NUM_SCALARS, num_qus_types=NUM_QUS_TYPES)
                optimizer, scheduler = convVal.create_optimizer_med_attn(multi_model, requires_grad=False, T_max=NUM_EPOCHS)
        elif CURRENT_MODEL == MODEL_VGG:
            pretrained_vgg = models.vgg16(weights='DEFAULT')  # 有使用預訓練權重
            if CURRENT_MODE == MODE_BASE:
                multi_model = convModel.BaseVGG(pretrained_vgg)
                optimizer = optim.AdamW(multi_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=BASE_LR * 1e-2)
            elif CURRENT_MODE == MODE_MULTI:
                multi_model = convModel.MultiModalVGG(pretrained_vgg, num_scalars=NUM_SCALARS)
                optimizer, scheduler = convVal.create_optimizer_vgg(multi_model, requires_grad=False, T_max=NUM_EPOCHS)
            elif CURRENT_MODE == MODE_MULTI_ATTN:
                multi_model = convModel.MultiModalAttnVGG(pretrained_vgg, num_scalars=NUM_SCALARS, num_qus_types=NUM_QUS_TYPES)
                optimizer, scheduler = convVal.create_optimizer_vgg_attn(multi_model, requires_grad=False, T_max=NUM_EPOCHS)


        multi_model = multi_model.to(device)
        criterion = nn.MSELoss()



        print("start training...")
        # 3. 訓練
        best_mae = float('inf')  # 初始設為無限大
        counter = 0                   # 計數器
        for epoch in range(NUM_EPOCHS):
            # ===== 訓練狀態 =====
            multi_model.train()
            sum_errors = 0.0      # sum of squared errors
            n_train   = 0        # 總樣本數（逐元素）
            # === 【狀態機：第 x 個 Epoch 觸發全面解凍】 ===
            if epoch == UN_FREEZE_EPOCH:
                if CURRENT_MODEL == MODEL_CONVNEXT:
                    if CURRENT_MODE == MODE_MULTI:
                        optimizer, scheduler = convVal.create_optimizer_conv(multi_model, requires_grad=True, T_max=NUM_EPOCHS-UN_FREEZE_EPOCH)
                    elif CURRENT_MODE == MODE_MULTI_ATTN:
                        optimizer, scheduler = convVal.create_optimizer_conv_Attn(multi_model, requires_grad=True, T_max=NUM_EPOCHS-UN_FREEZE_EPOCH)
                elif CURRENT_MODEL == MODEL_MEDVIT:
                    if CURRENT_MODE == MODE_MULTI:
                        optimizer, scheduler = convVal.create_optimizer_med(multi_model, requires_grad=True, T_max=NUM_EPOCHS-UN_FREEZE_EPOCH)
                    elif CURRENT_MODE == MODE_MULTI_ATTN:
                        optimizer, scheduler = convVal.create_optimizer_med_attn(multi_model, requires_grad=True, T_max=NUM_EPOCHS-UN_FREEZE_EPOCH)
                elif CURRENT_MODEL == MODEL_VGG:
                    if CURRENT_MODE == MODE_MULTI:
                        optimizer, scheduler = convVal.create_optimizer_vgg(multi_model, requires_grad=True, T_max=NUM_EPOCHS-UN_FREEZE_EPOCH)
                    elif CURRENT_MODE == MODE_MULTI_ATTN:
                        optimizer, scheduler = convVal.create_optimizer_vgg_attn(multi_model, requires_grad=True, T_max=NUM_EPOCHS-UN_FREEZE_EPOCH)
                        
            # 資料輸入model
            for img, targets, patient_ids, taitsi, swe, ezhri in train_loader:
                img = img.to(device)
                taitsi = taitsi.to(device)      # shape: (B,10)
                targets = targets.to(device)  # shape: (B,1)
                if CURRENT_MODE == MODE_BASE:              
                    outputs = multi_model(img)      # (B,1)
                elif CURRENT_MODE in [MODE_MULTI, MODE_MULTI_ATTN]:
                    if NUM_QUS_TYPES == 2:
                        outputs = multi_model(img, taitsi)      
                    elif NUM_QUS_TYPES == 3:
                        swe = swe.to(device)
                        three_qus = torch.cat([taitsi, swe], dim=1)
                        outputs = multi_model(img, three_qus)   
                    elif NUM_QUS_TYPES == 4:
                        swe = swe.to(device)      
                        ezhri = ezhri.to(device)      
                        four_qus = torch.cat([taitsi, swe, ezhri], dim=1)
                        outputs = multi_model(img, four_qus)
                loss = criterion(outputs, targets.unsqueeze(1))         # MSE (mean over elements)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                # 累積 sum_errors 與樣本數以計算「真正的」epoch MSE
                batch_elems = targets.numel()
                n_train   += batch_elems
                sum_errors += loss.item() * batch_elems
            epoch_train_loss  = sum_errors / n_train
            if LR_DECAY:
                scheduler.step() 


            # ===== 驗證 =====
            cm, mae, overall_acc, final_preds, final_targets, image_val_loss, patient_val_loss, img_acc, img_mae, img_cm = convVal.evaluate_patient_level(
                multi_model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, patient-level MAE: {mae:.6f}, image_val_loss: {image_val_loss:.6f}, learning rate: {optimizer.param_groups[0]['lr']:.6e}")
            # --- Early Stopping 邏輯 ---
            if epoch>=UN_FREEZE_EPOCH:  # 給模型一些時間適應新的權重，前面的epoch不早停
                if mae < best_mae:
                    best_mae = mae
                    counter = 0  # 重置計數器
                    # [關鍵] 只在變好時存檔，這樣留下來的一定是最好的
                    torch.save(multi_model.state_dict(), SAVE_MODEL_PATH)
                else:
                    counter += 1
                    if counter >= EARLY_STOPPING_PATIENCE:
                        print("Early stopping triggered! Training finished.")
                        break  # 跳出 epoch 迴圈


            # 4. 記錄原本的Loss  畫圖並寫入 TensorBoard
            # fig_cm = plot_confusion_matrix(cm, class_names=["S0", "S1", "S2", "S3"])
            # tb_writer.add_figure("Confusion_Matrix/val", fig_cm, epoch)
            # 計算每一類的 AUC (Threshold-based)
            thresholds_dict = {
                "S1_Mild": 0.064,      # >= S1
                "S2_Moderate": 0.174,  # >= S2
                "S3_Severe": 0.221     # >= S3
            }
            for name, thresh in thresholds_dict.items():
                # 建立二元標籤：真實值是否大於等於該閾值
                binary_true = (final_targets >= thresh).astype(int)
                
                # 檢查是否只有單一類別 (例如驗證集剛好沒有嚴重脂肪肝)，避免 sklearn 報錯
                if len(np.unique(binary_true)) > 1:
                    # 計算 AUC，使用原始回歸預測值 (final_preds) 作為分數
                    score = roc_auc_score(binary_true, final_preds)
                    tb_writer.add_scalar(f'AUC/{name}', score, epoch)
                else:
                    print(f"Skipping AUC for {name} (only one class present in val set)")
            
            tb_writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            tb_writer.add_scalar('Loss/val_image', image_val_loss, epoch)
            tb_writer.add_scalar('Loss/val_patient', patient_val_loss, epoch)
            tb_writer.add_scalar('Loss/val_patient_mae', mae, epoch)
        
        tb_writer.close()
