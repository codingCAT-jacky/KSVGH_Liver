import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch_lr_finder import LRFinder, TrainDataLoaderIter
import torchvision.models as models

# 导入项目模块
import convDataset
import convModel
import convVal
from convUtils import *
from MedViT import MedViT_small


# 1. DataLoader 封裝器：負責把 6 個變數打包成套件規定的 2 個 (Inputs, Targets)
class LRFinderDataWrapper(TrainDataLoaderIter):
    def __init__(self, dataloader):
        super().__init__(dataloader)
        
    def inputs_labels_from_batch(self, batch_data):
        # 接收最新版的 6 個輸出
        img, targets, patient_ids, taitsi, swe, ezhri = batch_data
        targets = targets.view(-1, 1)
        
        # 根據全局設定動態 Concat 數值特徵
        if NUM_QUS_TYPES == 2:
            qus = taitsi
        elif NUM_QUS_TYPES == 3:
            qus = torch.cat([taitsi, swe], dim=1)
        elif NUM_QUS_TYPES == 4:
            qus = torch.cat([taitsi, swe, ezhri], dim=1)
            
        return (img, qus), targets

# 2. Model 封裝器 
class LRFinderModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, inputs):
        image, qus = inputs
        # 🌟 修正：如果是 BASE 模式，模型只吃影像，不吃數值特徵
        if CURRENT_MODE == MODE_BASE:
            return self.model(image)
        return self.model(image, qus)


def prepare_dataloader(fold, batch_size):
    """准备数据加载器"""
    train_idx, val_idx = list(convDataset.splits)[fold - 1]
    
    # 修正呼叫方式，配合最新版的 build_imagelist
    train_list = convDataset.build_imagelist(train_idx)
    train_ds = convDataset.PDFFDataset(train_list, isTrain=True)
    
    train_image_targets = np.array([item.pdffClass for item in train_ds.dataList])
    unique_classes, counts = np.unique(train_image_targets, return_counts=True)
    class_weight_dict = {}
    for cls, count in zip(unique_classes, counts):
        class_weight_dict[cls] = 1.0 / count
    samples_weight = np.array([class_weight_dict[t] for t in train_image_targets])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    print(f"sampler len is {len(sampler)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=True)

    return train_loader, len(train_ds)


def setup_model_and_optimizer(device):
    """根據 convUtils 的當前設定，動態建立對應的模型與優化器"""
    if CURRENT_MODEL == MODEL_CONVNEXT:
        pretrained_convnext = models.convnext_tiny(weights='DEFAULT')
        if CURRENT_MODE == MODE_BASE:
            multi_model = convModel.BaseConv(pretrained_convnext)
            optimizer = optim.AdamW(multi_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        elif CURRENT_MODE == MODE_MULTI:
            multi_model = convModel.MultiModalConv(pretrained_convnext, num_scalars=NUM_SCALARS)
            optimizer, _ = convVal.create_optimizer_conv(multi_model, requires_grad=True, T_max=100)
        elif CURRENT_MODE == MODE_MULTI_ATTN:
            multi_model = convModel.MultiModalAttnConv(pretrained_convnext, num_scalars=NUM_SCALARS, num_qus_types=NUM_QUS_TYPES)
            optimizer, _ = convVal.create_optimizer_conv_Attn(multi_model, requires_grad=True, T_max=100)
            
    elif CURRENT_MODEL == MODEL_MEDVIT:
        pretrained_med = MedViT_small()  
        pretrained_med.load_state_dict(torch.load(MEDVIT_LOAD_PRETEAINMODEL_PATH), strict=False)
        if CURRENT_MODE == MODE_BASE:
            multi_model = convModel.BaseMed(pretrained_med)
            optimizer = optim.AdamW(multi_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        elif CURRENT_MODE == MODE_MULTI:
            multi_model = convModel.MultiModalMed(pretrained_med, num_scalars=NUM_SCALARS)
            optimizer, _ = convVal.create_optimizer_med(multi_model, requires_grad=True, T_max=100)
        elif CURRENT_MODE == MODE_MULTI_ATTN:
            multi_model = convModel.MultiModalAttnMed(pretrained_med, num_scalars=NUM_SCALARS, num_qus_types=NUM_QUS_TYPES)
            optimizer, _ = convVal.create_optimizer_med_attn(multi_model, requires_grad=True, T_max=100)
            
    elif CURRENT_MODEL == MODEL_VGG:
        pretrained_vgg = models.vgg16(weights='DEFAULT')  
        if CURRENT_MODE == MODE_BASE:
            multi_model = convModel.BaseVGG(pretrained_vgg)
            optimizer = optim.AdamW(multi_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        elif CURRENT_MODE == MODE_MULTI:
            multi_model = convModel.MultiModalVGG(pretrained_vgg, num_scalars=NUM_SCALARS)
            optimizer, _ = convVal.create_optimizer_vgg(multi_model, requires_grad=True, T_max=100)
        elif CURRENT_MODE == MODE_MULTI_ATTN:
            multi_model = convModel.MultiModalAttnVGG(pretrained_vgg, num_scalars=NUM_SCALARS, num_qus_types=NUM_QUS_TYPES)
            optimizer, _ = convVal.create_optimizer_vgg_attn(multi_model, requires_grad=True, T_max=100)

    # 🌟 【魔法修復區】強行解除 Optimizer 的 Scheduler 標籤
    # PyTorch Scheduler 會在 param_groups 留下 'initial_lr'，我們把它刪掉騙過 LRFinder
    for param_group in optimizer.param_groups:
        if 'initial_lr' in param_group:
            del param_group['initial_lr']

    multi_model = multi_model.to(device)
    return multi_model, optimizer


def run_lr_finder_single_batch(fold=1, batch_size=32, num_iter=200, start_lr=1e-8, end_lr=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    train_loader, num_samples = prepare_dataloader(fold, batch_size)
    wrapped_loader = LRFinderDataWrapper(train_loader)
    print(f"\n加载数据 (batch_size={batch_size})...")
    print(f"训练样本数: {num_samples}")
    print(f"每个 epoch 的批次数: {len(train_loader)}")

    print(f"\n初始化模型... ({CURRENT_MODEL} / {CURRENT_MODE})")
    multi_model, optimizer = setup_model_and_optimizer(device)
    wrapped_model = LRFinderModelWrapper(multi_model).to(device)
    criterion = nn.MSELoss()

    print(f"\n创建 LR Finder...")
    lr_finder = LRFinder(wrapped_model, optimizer, criterion, device=device)

    print("\n" + "="*60)
    print(f"开始学习率搜索 (batch_size={batch_size})")
    print("="*60)
    
    # 🌟 【防呆機制】：將 diverge_th 放寬到 100
    # 避免 MSE Loss 在初期的正常批次震盪被誤判為模型崩潰而提早結束
    lr_finder.range_test(wrapped_loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter, step_mode="exp", diverge_th=1000)
    
    history = lr_finder.history
    lrs = history['lr']
    losses = history['loss']
    
    print("\n" + "="*60)
    print(f"学习率搜索结果 (batch_size={batch_size})")
    print("="*60)
    
    min_loss_idx = np.argmin(losses)
    min_loss = losses[min_loss_idx]
    min_loss_lr = lrs[min_loss_idx]
    
    print(f"总迭代次数: {len(losses)}")
    print(f"初始 loss: {losses[0]:.6f}")
    print(f"最小 loss: {min_loss:.6f}")
    print(f"最大 loss: {max(losses):.6f}")
    print(f"最终 loss: {losses[-1]:.6f}")
    print()
    
    suggested_lr = None
    try:
        loss_grad = np.gradient(losses)
        skip_start = 10
        skip_end = 5
        
        if len(loss_grad) > (skip_start + skip_end):
            valid_grad = loss_grad[skip_start:-skip_end]
            valid_lrs = lrs[skip_start:-skip_end]
            steepest_idx = np.argmin(valid_grad)
            suggested_lr = valid_lrs[steepest_idx]
        else:
            steepest_idx = np.argmin(loss_grad)
            suggested_lr = lrs[steepest_idx]
            
        if suggested_lr is not None:
            print(f"推荐学习率 (最陡峭处): {suggested_lr:.2e}")
    except Exception as e:
        print(f"推荐学习率计算失败: {e}")
        suggested_lr = None

    print(f"最小 loss 对应的学习率: {min_loss_lr:.2e}")
    stats = {
        'batch_size': batch_size,
        'lrs': np.array(lrs),
        'losses': np.array(losses),
        'min_loss': min_loss,
        'min_loss_lr': min_loss_lr,
        'suggested_lr': suggested_lr,
    }
    
    return lr_finder, stats


def plot_comparison(stats_list, figsize=(14, 6)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'v']
    
    for i, stats in enumerate(stats_list):
        batch_size = stats['batch_size']
        lrs = stats['lrs']
        losses = stats['losses']
        
        skip_start = 10
        skip_end = 5
        plot_lrs = lrs[skip_start:-skip_end]
        plot_losses = losses[skip_start:-skip_end]
        
        ax1.semilogx(plot_lrs, plot_losses, 
                    label=f'BS={batch_size}',
                    color=colors[i % len(colors)],
                    linewidth=2,
                    marker=markers[i % len(markers)],
                    markersize=4,
                    markevery=max(1, len(plot_lrs) // 20))
        
        if stats['suggested_lr'] is not None:
            idx = np.argmin(np.abs(plot_lrs - stats['suggested_lr']))
            ax1.plot(plot_lrs[idx], plot_losses[idx], 
                    marker='*', markersize=15,
                    color=colors[i % len(colors)],
                    markeredgecolor='red', markeredgewidth=1.5)
        
        min_idx = np.argmin(plot_losses)
        linear_end_idx = min(min_idx + 50, len(plot_lrs))
        
        ax2.plot(range(linear_end_idx), plot_losses[:linear_end_idx],
                label=f'BS={batch_size}',
                color=colors[i % len(colors)],
                linewidth=2,
                marker=markers[i % len(markers)],
                markersize=4,
                markevery=max(1, linear_end_idx // 20))
    
    ax1.set_xlabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('LR Finder - Log Scale', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11, loc='best')
    
    ax2.set_xlabel('Iteration ', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('LR Finder - Linear Scale', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    return fig


def main(fold=1, batch_sizes=[16, 32], num_iter=200, start_lr=1e-8, end_lr=1):
    print("\n" + "="*60)
    print(f"LR Finder - 多 Batch Size 对比 ({CURRENT_MODEL})")
    print("="*60)
    print(f"Fold: {fold}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"搜索范围: {start_lr:.2e} ~ {end_lr:.2e}")
    print(f"迭代次数: {num_iter}")
    print("="*60 + "\n")
    
    stats_list = []
    for batch_size in batch_sizes:
        print(f"\n{'#'*60}")
        print(f"# Batch Size: {batch_size}")
        print(f"{'#'*60}\n")
        
        lr_finder, stats = run_lr_finder_single_batch(
            fold=fold,
            batch_size=batch_size,
            num_iter=num_iter,
            start_lr=start_lr,
            end_lr=end_lr
        )
        stats_list.append(stats)
    
    print("\n" + "="*60)
    print("生成对比图表...")
    print("="*60 + "\n")
    
    fig = plot_comparison(stats_list)
    output_path = f"./picture/lr_finder_BS_comparison_fold{fold}.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存到: {output_path}\n")
    plt.show()
    
    return stats_list

if __name__ == "__main__":
    FOLD = 1
    BATCH_SIZES = [16, 32]  
    NUM_ITER = 200
    START_LR = 1e-8
    END_LR = 1
    
    if len(sys.argv) > 1: FOLD = int(sys.argv[1])
    if len(sys.argv) > 2: NUM_ITER = int(sys.argv[2])
    
    stats_list = main(
        fold=FOLD,
        batch_sizes=BATCH_SIZES,
        num_iter=NUM_ITER,
        start_lr=START_LR,
        end_lr=END_LR
    )