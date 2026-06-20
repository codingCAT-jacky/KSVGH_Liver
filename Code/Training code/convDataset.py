import albumentations as A
from usaugment.albumentations import DepthAttenuation, GaussianShadow, HazeArtifact, SpeckleReduction
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import os
from sklearn.model_selection import StratifiedKFold
import random
import cv2
from convUtils import *
import torch
import numpy as np


# 建立 Dataset & DataLoader
with open(PDFF_FILE, 'r') as f:
    labels_pct = [float(line.strip().replace('%','')) for line in f]  # 0..100
with open(TAITSI_FILE, 'r') as f:
    tai_values, tsi_values = [], []  
    for line in f:
        elements = line.strip().split()
        # 1. 轉成 float，2. 不要多包 []
        tai_row = [float(x) for x in elements[0:5]]
        tsi_row = [float(x) for x in elements[5:10]]
        tai_values.append(tai_row) 
        tsi_values.append(tsi_row) 
with open(SWE_FILE, 'r') as f:
    swe_values = []
    for line in f:
        swe_row = []
        elements = line.strip().split()
        for item in elements:
            try:
                num = float(item) # 嘗試轉換為浮點數
                swe_row.append(num)
            except ValueError:
                swe_row = [-1 for i in range(5)]
                break
        swe_values.append(swe_row) 
with open(EZHRI_FILE, 'r') as f:
    ezhri_values = []
    for line in f:
        ezhri_row = []
        elements = line.strip().split()
        try:
            num = float(elements[0]) # 嘗試轉換為浮點數
            ezhri_row.append(num) #取第一個就好
        except ValueError:
            ezhri_row = [-1]
        ezhri_values.append(ezhri_row) 
# 將 List 轉成 NumPy Array，這樣後面的減法除法才不會報錯
tai_values = np.array(tai_values, dtype=np.float32)
tsi_values = np.array(tsi_values, dtype=np.float32)
swe_values = np.array(swe_values, dtype=np.float32)
ezhri_values = np.array(ezhri_values, dtype=np.float32)
labels_reg  = np.array([v/100.0 for v in labels_pct], dtype=np.float32)   
labels_cls  = np.array([pdff_to_class(v/100.0) for v in labels_pct], int) 
N = len(labels_reg)
print(f"Total folders found: {len(labels_reg)}")


# 不分層
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kf.split(np.arange(N), y=labels_cls))


class imageData:
    def __init__(self, imgpath, maskpath, value, patient_id=None, tai=None, tsi=None, swe=None, ezhri=None):
        self.imgpath = imgpath      # 圖片路徑
        self.maskpath = maskpath
        self.value = value    # 對應數值
        self.pdffClass = pdff_to_class(value)  # 對應類別
        self.patient_id = patient_id  # 新增：紀錄這張圖屬於哪個病人
        self.tai = tai  # 新增：紀錄這張圖的 TAI 值
        self.tsi = tsi  # 新增：紀錄這張圖的 TSI 值
        self.swe = swe if -1 not in swe else None
        self.ezhri = ezhri if -1 not in ezhri else None

def _sample_gaussian_shadow():
    return GaussianShadow(
        p=1.0,
        strength=random.uniform(0.25, 0.8),
        sigma_x=random.uniform(0.01, 0.2),
        sigma_y=random.uniform(0.01, 0.2),
    )

def _sample_depth_attenuation():
    return DepthAttenuation(
        p=1.0,
        attenuation_rate=random.uniform(0.0, 3.0),
    )

def build_trivial2():
    ops_pool = [None, _sample_gaussian_shadow, _sample_depth_attenuation]

    def _trivial2(image, scan_mask):
        # 抽兩個（可重複），並即時產生隨機參數的實例
        choices = [random.choice(ops_pool) for _ in range(2)]
        ops = [fn() for fn in choices if fn is not None]

        # 若兩個都抽到 None，就相當於 identity
        comp = A.Compose(ops, additional_targets={"scan_mask": "mask"})

        out = comp(image = image, scan_mask = scan_mask)
        return (out["image"])
    return _trivial2

def build_imagelist(idx_list):
    out = []
    imgfolders = sorted(os.listdir(IMG_FOLDER))
    maskfolders = sorted(os.listdir(MASK_FOLDER))

    for i in idx_list:
        imgf = imgfolders[i]
        maskf = maskfolders[i]
        imgfolder_path = os.path.join(IMG_FOLDER, imgf)
        maskfolder_path = os.path.join(MASK_FOLDER, maskf)
        y_reg = float(labels_reg[i])
        tai = tai_values[i]  
        tsi = tsi_values[i]  
        swe = swe_values[i]
        ezhri = ezhri_values[i]
        for img in sorted(os.listdir(imgfolder_path)):
            if img.lower().endswith('.png'):
                # 修改這裡：傳入 patient_id=i
                out.append(imageData(os.path.join(imgfolder_path, img), 
                                     os.path.join(maskfolder_path, img), 
                                     y_reg, 
                                     patient_id=i,
                                     tai=tai,
                                     tsi=tsi,
                                     swe=swe,
                                     ezhri=ezhri))
    return out

trivial2 = build_trivial2()
base_transform = A.Compose([
    A.Resize(height=224, width=224), 
    # A.Normalize(mean=(0.5,), std=(0.5,)),
    # A.Normalize(mean=(0.449,), std=(0.226,), max_pixel_value=1.0), # 喚醒 ImageNet 肌肉記憶的鑰匙
    ToTensorV2(transpose_mask=True),
])



# 定義 Dataset
# eng = matlab.engine.start_matlab()
class PDFFDataset(Dataset):
    def __init__(self, imageDataList, isTrain):
        valid = []
        for imgData in imageDataList:
            if (imgData.swe is None and NUM_QUS_TYPES>=3) or (imgData.ezhri is None and NUM_QUS_TYPES==4):
                continue
            valid.append(imgData)
        self.dataList = valid
        self.isTrain = isTrain

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, idx):
        img  = cv2.imread(self.dataList[idx].imgpath)
        img = img.astype(np.float32) / 255.0
        # img  = cv2.imread(self.dataList[idx].imgpath, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.dataList[idx].maskpath, cv2.IMREAD_GRAYSCALE)
        mask = np.clip(mask, None, 1)

        if self.isTrain:
            # 先做 TA-style 的兩步增強
            img = trivial2(image=img, scan_mask=mask)

        # 2) 先在 NumPy 階段做「中心裁切」（安全邊界）
        H, W, _ = img.shape
        centerx, centery = W // 2, H // 2
        half_w, half_h   = 270, 270
        x1 = max(0, centerx - half_w)
        y1 = max(0, centery - half_h)
        x2 = min(W, centerx + half_w)   
        y2 = min(H, centery + half_h)
        # 若圖片比裁切框小，這裡會自動被夾在界內
        img  = img[y1:y2, x1:x2, :].astype(np.float32)
        img01  = img[..., 0]   
        # img  = img[y1:y2, x1:x2]
        # img = img.astype(np.float32) / 255.0

        # 前處理denoise
        # filtered_img = usFilter.matlabSRAD(img01, eng)
        # filtered_img = np.clip(filtered_img, 0.0, 1.0)
        # # 變回 HWC 給 Albumentations  
        # filtered_img = filtered_img[..., None].astype(np.float32)
       
        # 再做固定前處理
        out = base_transform(image=img01)
        # out = base_transform(image=img)
        img_t = out["image"]             # torch.Tensor [1,Hc,Wc]
        
        y = torch.tensor(self.dataList[idx].value, dtype=torch.float32)  
        patient_id = self.dataList[idx].patient_id

        # QUS
        taitsi = torch.tensor(list(self.dataList[idx].tai) + list(self.dataList[idx].tsi) , dtype=torch.float32)
        swe = torch.tensor(list(self.dataList[idx].swe) , dtype=torch.float32) if self.dataList[idx].swe is not None else torch.zeros(5, dtype=torch.float32)
        ezhri = torch.tensor(list(self.dataList[idx].ezhri) , dtype=torch.float32) if self.dataList[idx].ezhri is not None else torch.zeros(1, dtype=torch.float32)
        return img_t, y, patient_id, taitsi, swe, ezhri
    

