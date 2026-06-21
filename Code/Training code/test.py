import numpy as np
import convDataset

def tenHighestPDFFPatient():
    print("\n" + "*"*60)
    print(" 🏥 脂肪肝最嚴重病患 (Top 10) 數值分析")
    print("*"*60)

    # 1. 直接從 convDataset 取得已解析好的全體數據
    labels_pct = convDataset.labels_pct
    tai_values = convDataset.tai_values
    tsi_values = convDataset.tsi_values
    swe_values = convDataset.swe_values
    ezhri_values = convDataset.ezhri_values

    N = len(labels_pct)

    # 2. 將所有病患的資料打包在一起，方便排序與過濾
    patients = []
    for i in range(N):
        patients.append({
            'id': i,                 # 記錄原本在資料夾中的 Index
            'pdff': labels_pct[i],
            'tai': tai_values[i],
            'tsi': tsi_values[i],
            'swe': swe_values[i],
            'ezhri': ezhri_values[i]
        })

    # =========================================================
    # 第一部分：取出 PDFF 最高，且擁有 SWE 數值的前 10 名
    # =========================================================
    # 過濾掉 SWE 為缺漏值 (-1) 的病患
    valid_swe_patients = [p for p in patients if -1 not in p['swe']]
    
    # 依照 PDFF 數值由大到小排序，並切片取前 10 名
    top10_swe = sorted(valid_swe_patients, key=lambda x: x['pdff'], reverse=True)[:10]

    print("\n[ 第一部分 ] 脂肪肝最嚴重的 10 位病患 (擁有 TAI, TSI, SWE 數值)")
    print("-" * 75)
    
    tai_means, tsi_means, swe_means = [], [], []
    for rank, p in enumerate(top10_swe, 1):
        # 將每個人各自的 5 個量測點取平均
        t_mean = np.mean(p['tai'])
        ts_mean = np.mean(p['tsi'])
        s_mean = np.mean(p['swe'])
        
        tai_means.append(t_mean)
        tsi_means.append(ts_mean)
        swe_means.append(s_mean)
        
        print(f"Top {rank:<2}: 病患 Index {p['id']:<3} | PDFF: {p['pdff']:>5.2f}% | "
              f"TAI: {t_mean:>5.2f} | TSI: {ts_mean:>5.2f} | SWE: {s_mean:>5.2f}")

    print("-" * 75)
    print(f"📌 這 10 位病患的【總平均】:")
    print(f"   ➤ PDFF 平均 : {np.mean([p['pdff'] for p in top10_swe]):.2f}%")
    print(f"   ➤ TAI 總平均: {np.mean(tai_means):.2f}")
    print(f"   ➤ TSI 總平均: {np.mean(tsi_means):.2f}")
    print(f"   ➤ SWE 總平均: {np.mean(swe_means):.2f}")


    # =========================================================
    # 第二部分：取出 PDFF 最高，且擁有 EZHRI 數值的前 10 名
    # =========================================================
    print("\n\n" + "="*75)
    # 過濾掉 EZHRI 為缺漏值 (-1) 的病患
    valid_ezhri_patients = [p for p in patients if -1 not in p['ezhri']]
    
    # 依照 PDFF 數值由大到小排序，並切片取前 10 名
    top10_ezhri = sorted(valid_ezhri_patients, key=lambda x: x['pdff'], reverse=True)[:10]

    print("\n[ 第二部分 ] 脂肪肝最嚴重的 10 位病患 (擁有 EZHRI 數值)")
    print("-" * 75)
    
    ezhri_means = []
    for rank, p in enumerate(top10_ezhri, 1):
        # EZHRI 只有 1 個點，但為了格式統一還是取 mean
        e_mean = np.mean(p['ezhri'])
        ezhri_means.append(e_mean)
        
        print(f"Top {rank:<2}: 病患 Index {p['id']:<3} | PDFF: {p['pdff']:>5.2f}% | EZHRI: {e_mean:>5.2f}")

    print("-" * 75)
    print(f"📌 這 10 位病患的【總平均】:")
    print(f"   ➤ PDFF 平均  : {np.mean([p['pdff'] for p in top10_ezhri]):.2f}%")
    print(f"   ➤ EZHRI總平均: {np.mean(ezhri_means):.2f}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    tenHighestPDFFPatient()

