import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp

# ==========================================
# 1. 基礎訓練參數設定 
# "needs to change for execute"
# ==========================================
BATCH_SIZE = 16        
NUM_EPOCHS = 200         
NUM_QUS_TYPES = 2       
NUM_SCALARS = {2: 10, 3: 15, 4: 16}.get(NUM_QUS_TYPES, 10)
EARLY_STOPPING_PATIENCE = 30
UN_FREEZE_EPOCH = 10 
BASE_LR = 3e-04    
LR_FACTOR = 0.1     
WEIGHT_DECAY = 3e-2
LR_DECAY = False         

# ==========================================
# 2. 模式與模型常數定義
# ==========================================
MODE_BASE = "base"
MODE_MULTI = "multi"
MODE_MULTI_ATTN = "multiAttn"

MODEL_VGG = "model_vgg"
MODEL_CONVNEXT = "model_convnext"
MODEL_MEDVIT = "model_medvit"

# "needs to change for execute" 
CURRENT_MODEL = MODEL_CONVNEXT
CURRENT_MODE = MODE_MULTI_ATTN


# ==========================================
# 3. 路徑動態路由 (Router)
# ==========================================
IMG_FOLDER = "./nckuPng/CPng"
MASK_FOLDER = "./nckuPng/CMask"
PDFF_FILE  = "./numeric/nckuPdff.txt"
TAITSI_FILE = "./numeric/nckuTAITSI.txt"
SWE_FILE = "./numeric/nckuSWE.txt"
EZHRI_FILE = "./numeric/nckuEzHRI.txt"
MEDVIT_LOAD_PRETEAINMODEL_PATH = "./MedViT/MedViT_small_im1k.pth"

# 🌟 核心簡化：使用巢狀字典來管理所有可能組合的存檔路徑
_MODEL_PATHS = {
    MODEL_VGG: {
        MODE_BASE:       "./outcome/vgg_save_model/vggBase.pth",
        MODE_MULTI:      f"./outcome/vgg_save_model/vgg{NUM_QUS_TYPES}QUS.pth",
        MODE_MULTI_ATTN: f"./outcome/vgg_save_model/vggAttn{NUM_QUS_TYPES}QUS.pth",
    },
    MODEL_CONVNEXT: {
        MODE_BASE:       "./outcome/conv_save_model/convBase.pth",
        MODE_MULTI:      f"./outcome/conv_save_model/conv{NUM_QUS_TYPES}QUS.pth",
        MODE_MULTI_ATTN: f"./outcome/conv_save_model/convAttn{NUM_QUS_TYPES}QUS.pth",
    },
    MODEL_MEDVIT: {
        MODE_BASE:       "./outcome/medvit_save_model/medvitBase.pth",
        MODE_MULTI:      f"./outcome/medvit_save_model/med{NUM_QUS_TYPES}QUS.pth",
        MODE_MULTI_ATTN: f"./outcome/medvit_save_model/medAttn{NUM_QUS_TYPES}QUS.pth",
    }
}

# 動態分發：直接根據當前設定抓出正確的存檔與讀檔路徑！
SAVE_MODEL_PATH = _MODEL_PATHS[CURRENT_MODEL][CURRENT_MODE]
LOAD_MODEL_PATH = _MODEL_PATHS[CURRENT_MODEL][CURRENT_MODE]

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


def plot_confusion_matrix(cm, class_names, title='Validation Confusion Matrix', filename='./picture/validation_confusion_matrix.png', save=False):
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if save is True:
        plt.savefig(filename) 
    print(f"Confusion matrix saved as '{filename}'")
    return figure

def plot_bland_altman(preds, targets, title='Validation Bland-Altman Plot', filename='./picture/val_bland_altman.png'):
    # 1. 轉換為百分比 (%)
    preds_pct = preds * 100.0
    targets_pct = targets * 100.0
    
    # 2. 計算差異 (Predicted - Ground Truth)
    diff = preds_pct - targets_pct
    # 3. 計算統計量 (平均誤差與標準差)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    # 🌟 核心新增：計算 One-Sample t-test 的 p-value
    t_stat, p_value = ttest_1samp(diff, 0.0)
    
    # 格式化 P-value (符合醫學期刊規範)
    if p_value < 0.001:
        p_str = "P < .001"
    else:
        p_str = f"P = {p_value:.3f}"
    
    # 4. 計算 95% 一致性界限
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    # --- 開始繪圖 ---
    plt.figure(figsize=(8, 6))
    plt.scatter(targets_pct, diff, color='black', marker='D', s=15, alpha=0.8)
    
    plt.axhline(mean_diff, color='royalblue', linestyle='-', linewidth=1.5)
    plt.axhline(upper_loa, color='firebrick', linestyle='--', linewidth=1.2)
    plt.axhline(lower_loa, color='firebrick', linestyle='--', linewidth=1.2)
    
    # 🌟 核心修改：將 P-value 加入 Mean Bias 的文字標籤中
    x_pos = np.max(targets_pct) 
    plt.text(x_pos, mean_diff + 0.5, f'Mean Bias\n{mean_diff:.1f} ({p_str})', ha='right', va='bottom', color='black', fontsize=10)
    plt.text(x_pos, upper_loa + 0.5, f'+1.96 SD\n{upper_loa:.1f}', ha='right', va='bottom', color='black', fontsize=10)
    plt.text(x_pos, lower_loa - 0.5, f'-1.96 SD\n{lower_loa:.1f}', ha='right', va='top', color='black', fontsize=10)
    
    plt.xlabel('MRI-PDFF (%)', fontsize=12)
    plt.ylabel('USFF - MRI-PDFF (%)', fontsize=12)
    plt.title(title, fontsize=14)
    
    plt.xlim(0, 60)      
    plt.ylim(-30, 10)    
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300) 
    print(f"Bland-Altman plot saved as '{filename}'")
    plt.close()


def plot_correlation_scatter(preds, targets, title='Correlation between USFF and MRI-PDFF', filename='./picture/val_correlation.png'):
    preds_pct = preds * 100.0
    targets_pct = targets * 100.0
    
    # 計算 Pearson r 與 p-value
    r_val, p_val = pearsonr(targets_pct, preds_pct)
    
    # 🌟 核心新增：利用 Fisher's z-transformation 計算 r 的 95% CI
    n = len(targets_pct)
    z = np.arctanh(r_val)             # 將 r 轉為 z 分數
    se = 1.0 / np.sqrt(n - 3)         # 計算標準誤
    z_crit = 1.96                     # 95% 信心水準的 Z 值
    
    ci_lower = np.tanh(z - z_crit * se) # 轉回 r 區間 (下界)
    ci_upper = np.tanh(z + z_crit * se) # 轉回 r 區間 (上界)
    
    # 格式化 P-value
    if p_val < 0.001:
        p_str = "P < .001"
    else:
        p_str = f"P = {p_val:.3f}"
    
    # --- 開始繪圖 ---
    plt.figure(figsize=(8, 8))
    plt.scatter(targets_pct, preds_pct, color='royalblue', marker='o', s=30, alpha=0.7)
    
    m, b = np.polyfit(targets_pct, preds_pct, 1)
    plt.plot(targets_pct, m * targets_pct + b, color='firebrick', linestyle='-', linewidth=2, 
             label=f'Linear Fit (y = {m:.2f}x + {b:.2f})')
    
    max_val = max(np.max(targets_pct), np.max(preds_pct)) + 5
    plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--', linewidth=1.5, label='Ideal (y = x)')
    
    # 🌟 核心修改：將 95% CI 寫入資訊方塊中 (符合論文規範格式)
    info_text = f'Pearson $r$ = {r_val:.2f}\n95% CI: {ci_lower:.2f}, {ci_upper:.2f}\n{p_str}'
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.xlabel('MRI-PDFF (%)', fontsize=12)
    plt.ylabel('Predicted USFF (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    print(f"Correlation plot saved as '{filename}'")
    plt.close()
    
    return r_val, p_val

