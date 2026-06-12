import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch_lr_finder import LRFinder, TrainDataLoaderIter

# 导入项目模块
import convDataset
import convModel
from convUtils import *
import torchvision.models as models


# 1. DataLoader 封裝器：負責把 4 個變數打包成套件規定的 2 個 (Inputs, Targets)
class LRFinderDataWrapper(TrainDataLoaderIter):
    def __init__(self, dataloader):
        # 必須呼叫父類別的初始化，把真正的 dataloader 傳給它
        super().__init__(dataloader)
        
    def inputs_labels_from_batch(self, batch_data):
        # 🌟 這個函數是套件留給我們的官方後門！
        # 每抓一個 batch，套件就會呼叫這裡，我們負責把你的 4 個變數拆開
        image, targets, patient_ids, taitsi = batch_data
        # 🌟 新增這一行：強制將標籤維度從 [BatchSize] 轉換為 [BatchSize, 1]
        targets = targets.view(-1, 1)
        
        # 然後依照套件規定的格式回傳: (inputs, labels)
        return (image, taitsi), targets

# 2. Model 封裝器 (這個完全不用動，維持原樣)
class LRFinderModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, inputs):
        # 接收到 Tuple，拆包後送給真實模型
        image, taitsi = inputs
        return self.model(image, taitsi)
    


def create_ConvNeXt_model(device):
    """创建 ConvNext 模型"""
    torch.manual_seed(100)  # 确保 ConvNext 的权重初始化一致
    pretrained_convnext = models.convnext_tiny(weights='DEFAULT')
    multi_model = convModel.MultiModalConv(pretrained_convnext, num_scalars=10)
    multi_model = multi_model.to(device)
    
    return multi_model


def create_optimizer(model, base_lr=1e-8, lr_factor=0.1):
    optimizer = optim.AdamW([
        {'params': model.image_extractor.parameters(), 'lr': base_lr * lr_factor},
        {'params': model.scalar_norm.parameters(),     'lr': base_lr},
        {'params': model.fc_features.parameters(),     'lr': base_lr},
        {'params': model.fc_decision.parameters(),     'lr': base_lr},
    ], weight_decay=3e-2)
    return optimizer


def prepare_dataloader(fold, batch_size):
    """准备数据加载器"""
    train_idx, val_idx = list(convDataset.splits)[fold - 1]
    
    train_list = convDataset.build_imagelist(train_idx, IMG_FOLDER, MASK_FOLDER, convDataset.labels_reg, convDataset.tai_values, convDataset.tsi_values)
    
    train_ds = convDataset.PDFFDataset(train_list, isTrain=True)
    
    # 计算类别权重用于采样
    train_image_targets = np.array([item.pdffClass for item in train_list])
    unique_classes, counts = np.unique(train_image_targets, return_counts=True)
    class_weight_dict = {}
    for cls, count in zip(unique_classes, counts):
        class_weight_dict[cls] = 1.0 / count
    samples_weight = np.array([class_weight_dict[t] for t in train_image_targets])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=True)
    
    return train_loader, len(train_ds)


def run_lr_finder_single_batch(fold=1, batch_size=32, num_iter=200, start_lr=1e-8, end_lr=1, lr_factor=0.1):
    """
    运行单个 batch size 的学习率搜索
    
    Args:
        fold: 使用第几折的数据
        batch_size: batch 大小
        num_iter: 搜索迭代次数
        start_lr: 起始学习率
        end_lr: 结束学习率
        lr_factor: 学习率倍数
    
    Returns:
        lr_finder 对象和统计信息
    """
    
    # 1. 设定运算装置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 2. 获取数据
    train_loader, num_samples = prepare_dataloader(fold, batch_size)
    # 【黃金修正 1】：包裝 DataLoader
    wrapped_loader = LRFinderDataWrapper(train_loader)
    print(f"\n加载数据 (batch_size={batch_size})...")
    print(f"训练样本数: {num_samples}")
    print(f"每个 epoch 的批次数: {len(train_loader)}")
    
    
    # 3. 构建模型
    print("\n初始化模型...")
    multi_model = create_ConvNeXt_model(device)
    # 【黃金修正 2】：包裝 Model
    wrapped_model = LRFinderModelWrapper(multi_model).to(device)


    # 4. 设定优化器和损失函数
    criterion = nn.MSELoss()
    optimizer = create_optimizer(multi_model, base_lr=1e-8, lr_factor=lr_factor)

    # 5. 创建 LR Finder（支持 ConvNext 特征提取器的低学习率）
    print(f"\n创建 LR Finder (ConvNext 学习率倍数: {lr_factor})...")
    # 【黃金修正 3】：直接使用原生 LRFinder
    lr_finder = LRFinder(wrapped_model, optimizer, criterion, device=device)

    # 6. 运行 range_test
    print("\n" + "="*60)
    print(f"开始学习率搜索 (batch_size={batch_size})")
    print("="*60)
    
    # 原生 range_test 會自動幫你做 EMA 平滑化，並且在 Loss 爆炸時早停！
    lr_finder.range_test(wrapped_loader, end_lr=end_lr, num_iter=num_iter,step_mode="exp")
    # 7. 获取历史记录
    history = lr_finder.history
    lrs = history['lr']
    losses = history['loss']

    
    # 8. 输出统计信息
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
    
# 🌟 修正點：手動計算「最陡峭下降點 (Steepest Gradient)」作為推薦學習率
    suggested_lr = None
    try:
        
        # 計算 Loss 曲線的一階導數（斜率/梯度）
        loss_grad = np.gradient(losses)
        
        # 為了避免頭尾的極端雜訊，我們忽略前 10 個與最後 5 個點
        skip_start = 10
        skip_end = 5
        
        if len(loss_grad) > (skip_start + skip_end):
            valid_grad = loss_grad[skip_start:-skip_end]
            valid_lrs = lrs[skip_start:-skip_end]
            
            # 尋找梯度最小（負最多、下降最快）的那個點
            steepest_idx = np.argmin(valid_grad)
            suggested_lr = valid_lrs[steepest_idx]
        else:
            # 如果迭代次數太少，就直接全域尋找
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
    """
    在同一张图上绘制多个 batch size 的学习率曲线
    
    Args:
        stats_list: 包含多个统计字典的列表
        figsize: 图表尺寸
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 定义颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'v']
    
    # 绘制对数坐标
    for i, stats in enumerate(stats_list):
        batch_size = stats['batch_size']
        lrs = stats['lrs']
        losses = stats['losses']
        
        # 跳过前几个和后几个点以获得更好的视图
        skip_start = 10
        skip_end = 5
        
        plot_lrs = lrs[skip_start:-skip_end]
        plot_losses = losses[skip_start:-skip_end]
        
        # 左图：log 刻度学习率
        ax1.semilogx(plot_lrs, plot_losses, 
                    label=f'BS={batch_size}',
                    color=colors[i % len(colors)],
                    linewidth=2,
                    marker=markers[i % len(markers)],
                    markersize=4,
                    markevery=max(1, len(plot_lrs) // 20))
        
        # 标记推荐点
        if stats['suggested_lr'] is not None:
            idx = np.argmin(np.abs(plot_lrs - stats['suggested_lr']))
            ax1.plot(plot_lrs[idx], plot_losses[idx], 
                    marker='*', markersize=15,
                    color=colors[i % len(colors)],
                    markeredgecolor='red', markeredgewidth=1.5)
        
        # 右图：线性刻度学习率（放大前面的部分）
        # 只显示到最小 loss 之后的部分
        min_idx = np.argmin(plot_losses)
        linear_end_idx = min(min_idx + 50, len(plot_lrs))
        
        ax2.plot(range(linear_end_idx), plot_losses[:linear_end_idx],
                label=f'BS={batch_size}',
                color=colors[i % len(colors)],
                linewidth=2,
                marker=markers[i % len(markers)],
                markersize=4,
                markevery=max(1, linear_end_idx // 20))
    
    # 设置左图（log 刻度）
    ax1.set_xlabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('LR Finder - Log Scale', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11, loc='best')
    
    # 设置右图（线性刻度）
    ax2.set_xlabel('Iteration ', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('LR Finder - Linear Scale', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    return fig


def main(fold=1, batch_sizes=[16, 32], num_iter=200, 
         start_lr=1e-8, end_lr=1, lr_factor=0.1):
    """
    主函数：对比多个 batch size 的学习率搜索
    
    Args:
        fold: 使用第几折的数据
        batch_sizes: 要测试的 batch size 列表
        num_iter: 搜索迭代次数
        start_lr: 起始学习率
        end_lr: 结束学习率
        lr_factor: 学习率倍数
    """
    
    print("\n" + "="*60)
    print("ConvNext LR Finder - 多 Batch Size 对比")
    print("="*60)
    print(f"Fold: {fold}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"搜索范围: {start_lr:.2e} ~ {end_lr:.2e}")
    print(f"迭代次数: {num_iter}")
    print(f"ConvNext 学习率倍数: {lr_factor}")
    print("="*60 + "\n")
    
    # 对每个 batch size 运行搜索
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
            end_lr=end_lr,
            lr_factor=lr_factor
        )
        
        stats_list.append(stats)
    
    # 绘制对比图
    print("\n" + "="*60)
    print("生成对比图表...")
    print("="*60 + "\n")
    
    fig = plot_comparison(stats_list)
    
    # 保存图表
    output_path = f"./picture/conv_lr_finder_batch_comparison_fold{fold}.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存到: {output_path}\n")
    
    # 输出汇总信息
    print("\n" + "="*60)
    print("汇总结果")
    print("="*60)
    
    for stats in stats_list:
        print(f"\nBatch Size: {stats['batch_size']}")
        print(f"  - 最小 loss: {stats['min_loss']:.6f}")
        print(f"  - 最小 loss 对应的 lr: {stats['min_loss_lr']:.2e}")
        if stats['suggested_lr'] is not None:
            print(f"  - 推荐学习率: {stats['suggested_lr']:.2e}")
        else:
            print(f"  - 推荐学习率: 无")
    
    print("\n" + "="*60)
    print("建议:")
    print("="*60)
    
    # 比较两个 batch size 的结果
    if len(stats_list) >= 2:
        bs1_stats = stats_list[0]
        bs2_stats = stats_list[1]
        
        print(f"\n对比 Batch Size {bs1_stats['batch_size']} vs {bs2_stats['batch_size']}:")
        bs1_lr = bs1_stats['suggested_lr'] if bs1_stats['suggested_lr'] is not None else None
        bs2_lr = bs2_stats['suggested_lr'] if bs2_stats['suggested_lr'] is not None else None
        
        if bs1_lr is not None:
            print(f"  - BS{bs1_stats['batch_size']} 推荐 lr: {bs1_lr:.2e}")
        else:
            print(f"  - BS{bs1_stats['batch_size']} 推荐 lr: 无法计算")
        
        if bs2_lr is not None:
            print(f"  - BS{bs2_stats['batch_size']} 推荐 lr: {bs2_lr:.2e}")
        else:
            print(f"  - BS{bs2_stats['batch_size']} 推荐 lr: 无法计算")
    
    
    plt.show()
    
    return stats_list


if __name__ == "__main__":
    # 默认配置
    FOLD = 1
    BATCH_SIZES = [16, 32]  # 对比两个 batch size
    NUM_ITER = 200
    START_LR = 1e-8
    END_LR = 1
    LR_FACTOR = 0.1  # ConvNext 特征提取器为其他层的 1/10
    
    # 可以通过命令行参数修改
    import sys
    if len(sys.argv) > 1:
        FOLD = int(sys.argv[1])
    if len(sys.argv) > 2:
        NUM_ITER = int(sys.argv[2])
    
    stats_list = main(
        fold=FOLD,
        batch_sizes=BATCH_SIZES,
        num_iter=NUM_ITER,
        start_lr=START_LR,
        end_lr=END_LR,
        lr_factor=LR_FACTOR
    )
