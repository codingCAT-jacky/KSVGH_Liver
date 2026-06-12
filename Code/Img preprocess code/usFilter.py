import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import matlab.engine
from skimage.metrics import mean_squared_error 
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from scipy.ndimage import sobel

def compute_snr(image1, image2):
    roi1 = image1[600:750, 300:500]
    roi2 = image2[600:750, 300:500]
    noise = roi1 - roi2

    signal_power = np.mean(roi1**2)- np.mean(roi1)**2
    noise_power = np.mean(noise**2)- np.mean(noise)**2
    snr = (signal_power / (noise_power + 1e-8))  # 避免除以零
    return snr

def matlabAF(img, eng):
    # 將影像轉為 MATLAB 格式
    img_mat = matlab.double(img.tolist())

    # 執行 AF
    win_vec = matlab.double([[3, 3]])  # 1x2 向量要用雙層 list
    J_mat = eng.wiener2(img_mat, win_vec, nargout=1)
    J_img = np.array(J_mat, dtype=np.float64)  # [0,1]

    # 將結果轉回 NumPy 陣列
    despeckled_img = np.array(J_img, dtype=np.uint8)

    return despeckled_img

def matlabSRAD(img, eng):
    img_mat = matlab.double(img.astype(np.float64).tolist())
    J = eng.specklefilt(img_mat, nargout=1)      # 期望返回 double [0,1]
    return np.array(J, dtype=np.float32)         # (H,W), 0~1

def matlabWaveLet(img, eng):
# 套用 adaptive filter
    I_d = (img.astype(np.float64) / 255.0).tolist()
    
    # 一行流：wdenoise2
    I_mat = matlab.double(I_d)
    Ld = eng.wdenoise2(eng.log(eng.plus(I_mat, 2.220446049250313e-16)), 3,
                    'Wavelet','sym4',
                    'DenoisingMethod','Bayes',
                    'NoiseEstimate','LevelDependent', nargout=1)
    J = np.exp(np.array(Ld))               # 回到 [0,~]
    filtered_img = (255*np.clip(J/np.max(J),0,1)).astype(np.uint8)
    
    return filtered_img

def matlabCLAHE(img, eng):
    # 將影像轉為 MATLAB 格式
    img_mat = matlab.uint8(img.tolist())
    numtiles = matlab.double([8.0, 8.0])
    # 執行 CLAHE
    J_mat = eng.adapthisteq( img_mat, 'NumTiles', numtiles, 'ClipLimit', 0.01,  'Distribution', 'uniform',  nargout=1)
   
    J_img = np.array(J_mat, dtype=np.uint8) 

    return J_img

def speckle_index(I, eps=1e-12):
    mu  = I.mean()
    sd  = I.std(ddof=1)
    SI  = sd / (mu + eps)
    return float(SI)

def edge_preservation_index(I_before, I_after, mask_or_pct=90):
    I_before = np.asarray(I_before, np.float64)
    I_after  = np.asarray(I_after,  np.float64)

    # Sobel 梯度幅值
    gx1 = sobel(I_before, axis=1); gy1 = sobel(I_before, axis=0)
    gx2 = sobel(I_after,  axis=1); gy2 = sobel(I_after,  axis=0)
    G1 = np.hypot(gx1, gy1); G2 = np.hypot(gx2, gy2)

    # 邊緣集合
    if isinstance(mask_or_pct, np.ndarray) and mask_or_pct.dtype==bool:
        E = mask_or_pct
    else:
        p = float(mask_or_pct)
        thr = np.percentile(G1, p)
        E = G1 >= thr

    g1 = G1[E] - G1[E].mean()
    g2 = G2[E] - G2[E].mean()
    denom = np.sqrt((g1*g1).sum() * (g2*g2).sum()) + np.finfo(np.float64).eps
    epi_corr  = (g1*g2).sum() / denom              # -1~1
    epi_ratio = G2[E].mean() / (G1[E].mean() + 1e-12)  # ~1 最佳
    return float(epi_corr), float(epi_ratio)

# min max更穩健：用百分位避免極端值干擾（推薦）
def minmax01_percentile(img, p_low=1, p_high=99, eps=1e-6):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [p_low, p_high])
    img = np.clip(img, lo, hi)
    return (img - lo) / (hi - lo + eps)

# # apply filter
# input_base_dir = "nckuSmallPng"
# output_base_dir = "nckuAFPng"
# count = 0
# # 確保輸出根資料夾存在
# os.makedirs(output_base_dir, exist_ok=True)

# # 處理每個子資料夾 NCKU0001 ~ NCKU0200
# for folder_idx in range(1, 201): 
#         folder_name = f"NCKU{folder_idx:04d}"
#         input_folder = os.path.join(input_base_dir, folder_name)
#         output_folder = os.path.join(output_base_dir, folder_name)

#         # 確保輸出子資料夾存在
#         os.makedirs(output_folder, exist_ok=True)
#         eng = matlab.engine.start_matlab()

#         # 處理該資料夾下的所有圖片
#         for filename in os.listdir(input_folder):
#             if filename.lower().endswith((".png")):
#                 input_path = os.path.join(input_folder, filename)
#                 img = np.array(Image.open(input_path).convert('L'))

#                 # 套用 filter
#                 minmaximg  = minmax01_percentile(img) * 255
#                 minmaximg = minmaximg.astype(np.uint8)
#                 filtered_img = matlabSRAD(minmaximg, eng)
#                 # 儲存結果
#                 output_path = os.path.join(output_folder, filename)
#                 cv2.imshow("DICOM filtered_img", filtered_img)  # 或 pixel_bgr
#                 cv2.imshow("DICOM minmaximg", minmaximg)  # 或 pixel_bgr
#                 cv2.imshow("ori Image1", img)  # 或 pixel_bgr
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
#                 # cv2.imwrite(output_path, filtered_img)
#                 # ssim_val = structural_similarity(img, filtered_img)  # skimage>=0.19，灰階不用指定 channel_axis
#                 # SI_val  = speckle_index(filtered_img)
#                 # epi_corr, epi_ratio = edge_preservation_index(img, filtered_img)

#                 # with open("./numeric/nckuSRADSnr.txt", "a", encoding="utf-8") as f:
#                 #     f.write(f"{output_path} | SSIM={ssim_val:.3f} | SI={SI_val:.3f} | epi={epi_corr:.3f}, {epi_ratio:.3f}\n")
#         print(f"✅ 處理完成: {folder_name}")



def singleDir():
    input_base_dir = "nckuBigPng/NCKU0036"
    output_base_dir = "nckuAFPng/NCKU0036"
    # 確保輸出子資料夾存在
    eng = matlab.engine.start_matlab()
    for filename in os.listdir(input_base_dir):
        if filename.lower().endswith((".png")):
            input_path = os.path.join(input_base_dir, filename)
            img = np.array(Image.open(input_path).convert('L'))
            # 套用 filter
            
            filtered_img = matlabAF(input_path, eng)

            # 儲存結果
            output_path = os.path.join(output_base_dir, filename)
            # filtered_img = np.array(Image.open(output_path).convert('L'))
            cv2.imwrite(output_path, filtered_img)