import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 1. 在 convUtils.py 頂部（import os 等地方）加入這行：
from scipy.stats import pearsonr



PRE_TRAINED = False     #
NUM_EPOCHS = 150         #
NUM_SCALARS = 10        #
EARLY_STOPPING_PATIENCE = 30
UN_FREEZE_EPOCH = 10 
BASE_LR = 4.4e-3        #
CONV_LR_FACTOR = 0.1
WEIGHT_DECAY = 3e-2
LR_DECAY = False         #
VGG_BASE_MODEL_PATH = "./outcome/vgg_save_model/vggBase.pth"
VGG_MULTI_MODEL_PATH = "./outcome/vgg_save_model/base_aug_32.pth"
CONV_BASE_MODEL_PATH = "./outcome/conv_save_model/convBase.pth"
CONV_MULTI_MODEL_PATH = "./outcome/conv_save_model/convMulti2QUS.pth"
MEDVIT_BASE_MODEL_PATH = "./outcome/medvit_save_model/convBase.pth"
CONV_SAVE_MODEL_PATH = "./outcome/conv_save_model/convMulti2QUS.pth"
MODE_MULTI = "multi" 
MODE_BASE = "base"
MODE = MODE_MULTI #
IMG_FOLDER = "./nckuPng/CPng"
MASK_FOLDER = "./nckuPng/CMask"
PDFF_FILE  = "./numeric/nckuPdff.txt"
TAITSI_FILE = "./numeric/nckuTAITSI.txt"


def pdff_to_class(x: np.ndarray) -> np.ndarray:
    """
    依照門檻 (單位：小數而非百分比)：
      s0: x < 0.064
      s1: 0.064 <= x < 0.174
      s2: 0.174 <= x < 0.221
      s3: x >= 0.221
    """
    bins = [0.064, 0.174, 0.221]
    # bins = [0.05, 0.15, 0.25]
    return np.digitize(x, bins, right=False)  # 產生 0,1,2,3


def count_by_class(indices, y):
    vals, cnts = np.unique(y[indices], return_counts=True)
    # 將回傳格式統一為 numpy array，索引對應類別 0..(C-1)
    num_classes = int(np.max(y)) + 1
    counts = np.zeros(num_classes, dtype=int)
    counts[vals.astype(int)] = cnts
    return counts


def plot_confusion_matrix(cm, class_names, title='Validation Confusion Matrix', filename='validation_confusion_matrix.png'):
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    # plt.savefig(filename)
    # print(f"Confusion matrix saved as '{filename}'")
    # plt.close() 
    return figure

def plot_bland_altman(preds, targets, title='Validation Bland-Altman Plot', filename='./picture/val_bland_altman.png'):
    """
    繪製 Bland-Altman Plot 來評估模型預測 (USFF) 與真實值 (MRI-PDFF) 的一致性。
    注意：此處依照臨床常見做法，X軸放置 Reference Standard (MRI-PDFF)。
    """
    # 1. 將預測值與真實值從數值 (0~1) 轉換為百分比 (%)
    preds_pct = preds * 100.0
    targets_pct = targets * 100.0
    
    # 2. 計算差異 (Predicted - Ground Truth)
    diff = preds_pct - targets_pct
    
    # 3. 計算統計量 (平均誤差與標準差)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1) # ddof=1 代表樣本標準差
    
    # 4. 計算 95% 一致性界限 (Limits of Agreement, LoA)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    # --- 開始繪圖 ---
    plt.figure(figsize=(8, 6))
    
    # 繪製散點 (標記設為菱形 'D'，顏色為黑色，最接近你提供的參考圖)
    plt.scatter(targets_pct, diff, color='black', marker='D', s=15, alpha=0.8)
    
    # 繪製平均差異線 (藍色實線)
    plt.axhline(mean_diff, color='royalblue', linestyle='-', linewidth=1.5)
    
    # 繪製上下 95% 界限線 (紅色虛線)
    plt.axhline(upper_loa, color='firebrick', linestyle='--', linewidth=1.2)
    plt.axhline(lower_loa, color='firebrick', linestyle='--', linewidth=1.2)
    
    # 加上文字標籤 (放置在圖表右側)
    x_pos = np.max(targets_pct) 
    plt.text(x_pos, mean_diff + 0.5, f'Mean\n{mean_diff:.1f}', ha='right', va='bottom', color='black', fontsize=10)
    plt.text(x_pos, upper_loa + 0.5, f'+1.96 SD\n{upper_loa:.1f}', ha='right', va='bottom', color='black', fontsize=10)
    plt.text(x_pos, lower_loa - 0.5, f'-1.96 SD\n{lower_loa:.1f}', ha='right', va='top', color='black', fontsize=10)
    
    # 設定軸標籤與範圍
    plt.xlabel('MRI-PDFF (%)', fontsize=12)
    plt.ylabel('USFF - MRI-PDFF (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlim(left=0) # 確保 X 軸從 0 開始
    
    # [關鍵修改] 設定 X 軸與 Y 軸的絕對範圍
    plt.xlim(0, 60)      # X軸範圍設為 0~60
    plt.ylim(-30, 10)    # Y軸範圍設為 -30~10
    
    # 儲存圖片
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300) # 使用高解析度儲存
    print(f"Bland-Altman plot saved as '{filename}'")
    plt.close()


# 2. 在 convUtils.py 的最下方加入這個 function：
def plot_correlation_scatter(preds, targets, title='Correlation between USFF and MRI-PDFF', filename='./picture/val_correlation.png'):
    """
    繪製預測值與真實值的相關性散點圖 (Scatter Plot) 並計算 Pearson r
    """
    # 轉換為百分比 (%)
    preds_pct = preds * 100.0
    targets_pct = targets * 100.0
    
    # 計算 Pearson 相關係數與 p-value
    r_val, p_val = pearsonr(targets_pct, preds_pct)
    
    plt.figure(figsize=(8, 8))
    
    # 繪製散點
    plt.scatter(targets_pct, preds_pct, color='royalblue', marker='o', s=30, alpha=0.7)
    
    # 計算並繪製線性迴歸擬合線 (y = mx + b)
    m, b = np.polyfit(targets_pct, preds_pct, 1)
    plt.plot(targets_pct, m * targets_pct + b, color='firebrick', linestyle='-', linewidth=2, 
             label=f'Linear Fit (y = {m:.2f}x + {b:.2f})')
    
    # 繪製理想的對角線 (y = x)
    max_val = max(np.max(targets_pct), np.max(preds_pct)) + 5
    plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--', linewidth=1.5, label='Ideal (y = x)')
    
    # 將 R 值與 p-value 標註在圖表左上角
    plt.text(0.05, 0.95, f'Pearson $r$ = {r_val:.4f}\n$p$-value = {p_val:.2e}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.xlabel('MRI-PDFF (%)', fontsize=12)
    plt.ylabel('Predicted USFF (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlim(0, 50)  # 根據你的資料可以微調範圍 (例如 0~60)
    plt.ylim(0, 50)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 儲存圖片
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    print(f"Correlation plot saved as '{filename}'")
    plt.close()
    
    return r_val, p_val

