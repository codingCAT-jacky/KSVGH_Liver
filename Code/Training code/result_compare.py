"""
results_compare.py
讀取 convTrain.py 訓練過程中累積寫入的 fold_results.csv，
整理成「架構 x 模態 x 5-Fold MAE」表格，並對指定的兩個模型組合做 paired t-test。

使用前提：
    - convTrain.py 已經分別跑過 MultiModalConv（MODE_MULTI）與
      MultiModalAttnConv（MODE_MULTI_ATTN）各 5 折，
      結果都已 append 進 ./outcome/fold_results.csv

用法:
    python results_compare.py
"""

import os
import csv
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_rel


# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
FOLD_RESULTS_CSV = "./outcome/fold_results.csv"

# 要比較的兩組設定 (model, mode, num_qus_types, use_segment_mask)
# 預設比較 MultiModalConv vs MultiModalAttnConv，固定 ConvNeXt backbone
COMPARISON_PAIRS = [
    {
        "label": "MultiModalConv vs MultiModalAttnConv",
        "config_a": ("model_convnext", "multi",      2, "False"),
        "config_b": ("model_convnext", "multiAttn",  2, "False"),
    },
]


def load_results(csv_path):
    """
    讀取 fold_results.csv，回傳 dict:
        key   = (model, mode, num_qus_types, use_segment_mask)
        value = {fold: best_mae, ...}

    若同一組設定的同一折出現多次紀錄 (例如重新訓練過)，取最後一筆 (最新時間戳)。
    """
    if not os.path.isfile(csv_path):
        print(f"找不到結果檔案: {csv_path}")
        print("請先用 convTrain.py 訓練並產生 fold_results.csv")
        return {}

    records = defaultdict(dict)  # key -> {fold: best_mae}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["mode"], row["num_qus_types"], row["use_segment_mask"], row["lr_decay"])
            fold = int(row["fold"])
            mae = float(row["best_mae"])
            # 後面的紀錄覆蓋前面的 (取最新一次訓練結果)
            records[key][fold] = mae

    return records


def print_summary_table(records):
    """印出所有組合的 5-Fold MAE 總表"""
    print("=" * 100)
    print("所有模型組合的 5-Fold MAE 總表")
    print("=" * 100)
    header = f"{'Model':<18}{'Mode':<12}{'QUS':<6}{'Seg':<7}"
    for i in range(1, 6):
        header += f"{'Fold'+str(i):<10}"
    header += f"{'Mean':<10}{'Std':<10}{'N_Folds':<8}"
    print(header)
    print("-" * 100)

    for key, fold_dict in sorted(records.items()):
        model, mode, qus, seg = key
        maes_in_order = [fold_dict.get(f, None) for f in range(1, 6)]
        valid_maes = [v for v in maes_in_order if v is not None]

        row = f"{model:<18}{mode:<12}{qus:<6}{seg:<7}"
        for v in maes_in_order:
            row += f"{v:.4f}    " if v is not None else f"{'--':<10}"
        if valid_maes:
            row += f"{np.mean(valid_maes):<10.4f}{np.std(valid_maes, ddof=1) if len(valid_maes)>1 else 0:<10.4f}{len(valid_maes):<8}"
        else:
            row += f"{'--':<10}{'--':<10}{0:<8}"
        print(row)
    print("=" * 100)


def compare_two_configs(records, label, config_a, config_b):
    """
    對兩組設定做 paired t-test。
    必須兩者都有完整 5 折結果，且 fold 編號要能一一配對 (同樣的病人切分)。
    """
    print("\n" + "=" * 100)
    print(f"比較: {label}")
    print("=" * 100)

    if config_a not in records:
        print(f"⚠ 找不到設定 A 的結果: {config_a}")
        return
    if config_b not in records:
        print(f"⚠ 找不到設定 B 的結果: {config_b}")
        return

    folds_a = records[config_a]
    folds_b = records[config_b]

    common_folds = sorted(set(folds_a.keys()) & set(folds_b.keys()))

    if len(common_folds) < 2:
        print(f"⚠ 共同 fold 數量不足 ({len(common_folds)})，無法做 paired t-test (至少需要 2 折)")
        return

    if len(common_folds) < 5:
        print(f"⚠ 注意：只有 {len(common_folds)}/5 折同時擁有結果，"
              f"建議補完整 5 折再下結論。目前先以這 {len(common_folds)} 折計算。")

    maes_a = np.array([folds_a[f] for f in common_folds])
    maes_b = np.array([folds_b[f] for f in common_folds])

    print(f"\nConfig A: model={config_a[0]}, mode={config_a[1]}, qus_types={config_a[2]}, segment_mask={config_a[3]}")
    print(f"Config B: model={config_b[0]}, mode={config_b[1]}, qus_types={config_b[2]}, segment_mask={config_b[3]}")
    print(f"\n{'Fold':<8}{'Config A MAE':<16}{'Config B MAE':<16}{'Diff (A-B)':<14}")
    for f, a, b in zip(common_folds, maes_a, maes_b):
        print(f"{f:<8}{a:<16.4f}{b:<16.4f}{a-b:<14.4f}")

    mean_a, std_a = maes_a.mean(), maes_a.std(ddof=1) if len(maes_a) > 1 else 0.0
    mean_b, std_b = maes_b.mean(), maes_b.std(ddof=1) if len(maes_b) > 1 else 0.0

    print(f"\nConfig A: mean={mean_a:.4f}, std={std_a:.4f}")
    print(f"Config B: mean={mean_b:.4f}, std={std_b:.4f}")

    t_stat, p_value = ttest_rel(maes_a, maes_b)
    print(f"\nPaired t-test: t={t_stat:.4f}, p={p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        better = "A" if mean_a < mean_b else "B"
        better_config = config_a if better == "A" else config_b
        print(f"\n✅ 差異具統計顯著性 (p < {alpha})")
        print(f"   → {('Config ' + better)} (model={better_config[0]}, mode={better_config[1]}) "
              f"的 MAE 顯著較低，可視為較優模型")
    else:
        print(f"\n⚠ 差異未達統計顯著性 (p >= {alpha})")
        print(f"   → 不能斷定兩者哪個真的比較好，目前的差異可能只是隨機波動")


def main():
    records = load_results(FOLD_RESULTS_CSV)

    if not records:
        return

    print_summary_table(records)

    for pair in COMPARISON_PAIRS:
        compare_two_configs(records, pair["label"], pair["config_a"], pair["config_b"])


if __name__ == "__main__":
    main()