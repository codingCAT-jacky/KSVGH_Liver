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
import convVal



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
        train_list = convDataset.build_imagelist(train_idx, IMG_FOLDER, MASK_FOLDER, convDataset.labels_reg, convDataset.tai_values, convDataset.tsi_values)
        val_list   = convDataset.build_imagelist(val_idx, IMG_FOLDER, MASK_FOLDER, convDataset.labels_reg, convDataset.tai_values, convDataset.tsi_values)

        train_ds = convDataset.PDFFDataset(train_list, isTrain=True)  # 你的 Dataset
        val_ds   = convDataset.PDFFDataset(val_list, isTrain=False)

        train_image_targets = [item.pdffClass for item in train_list] 
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
        pretrained_convnext = models.convnext_tiny(weights='DEFAULT')
        if MODE == MODE_BASE:
            multi_model = convModel.BaseConv(pretrained_convnext)
            if PRE_TRAINED:
                multi_model = torch.load(CONV_BASE_MODEL_PATH, map_location=device, weights_only=False)
            optimizer = optim.AdamW(multi_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        else:
            multi_model = convModel.MultiModalConv(pretrained_convnext, num_scalars=NUM_SCALARS)
            if PRE_TRAINED:
                multi_model.load_state_dict(torch.load(CONV_MULTI_MODEL_PATH, map_location=device))
            optimizer, scheduler = convVal.create_optimizer(multi_model, requires_grad=False, T_max=NUM_EPOCHS)
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
            if epoch == UN_FREEZE_EPOCH and MODE == MODE_MULTI:
                optimizer, scheduler = convVal.create_optimizer(multi_model, requires_grad=True, T_max=NUM_EPOCHS-UN_FREEZE_EPOCH)

            # 資料輸入model
            for inputs, targets, patient_ids, taitsi in train_loader:
                inputs = inputs.to(device)
                taitsi = taitsi.to(device)      # shape: (B,10)
                targets = targets.to(device)  # shape: (B,1)
                if MODE == MODE_BASE:              
                    outputs = multi_model(inputs)      # (B,1)
                elif MODE == MODE_MULTI:
                    outputs = multi_model(inputs, taitsi)                   
                loss = criterion(outputs, targets.unsqueeze(1))         # MSE (mean over elements)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                

                # 累積 sum_errors 與樣本數以計算「真正的」epoch MSE
                batch_elems = targets.numel()
                n_train   += batch_elems
                sum_errors += loss.item() * batch_elems
            epoch_train_loss  = sum_errors / n_train
            if LR_DECAY and MODE == MODE_MULTI:
                scheduler.step() 


            # ===== 驗證 =====
            cm, mae, overall_acc, final_preds, final_targets, image_val_loss, patient_val_loss, img_acc, img_mae, img_cm = convVal.evaluate_patient_level(
                multi_model, val_loader, criterion, device, MODE)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, patient-level MAE: {mae:.6f}, image_val_loss: {image_val_loss:.6f}, learning rate: {optimizer.param_groups[0]['lr']:.6e}")
            # --- Early Stopping 邏輯 ---
            if epoch>=UN_FREEZE_EPOCH:  # 給模型一些時間適應新的權重，前面的epoch不早停
                if mae < best_mae:
                    best_mae = mae
                    counter = 0  # 重置計數器
                    # [關鍵] 只在變好時存檔，這樣留下來的一定是最好的
                    torch.save(multi_model.state_dict(), CONV_SAVE_MODEL_PATH)
                else:
                    counter += 1
                    if counter >= EARLY_STOPPING_PATIENCE:
                        print("Early stopping triggered! Training finished.")
                        break  # 跳出 epoch 迴圈


            # 4. 記錄原本的Loss  畫圖並寫入 TensorBoard
            fig_cm = plot_confusion_matrix(cm, class_names=["S0", "S1", "S2", "S3"], title='Validation Confusion Matrix', filename='validation_confusion_matrix.png')
            tb_writer.add_figure("Confusion_Matrix/val", fig_cm, epoch)
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
