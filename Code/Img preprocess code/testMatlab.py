import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from skimage.metrics import mean_squared_error 
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# 1. 啟動 MATLAB 引擎
eng = matlab.engine.start_matlab()

# # 2) 在 Python 讀入灰階影像
# I1 = np.array(Image.open('gray1.png').convert('L'))  # uint8, shape (H, W)

# # 3) 轉成 MATLAB 可讀型別
# #    specklefilt 接受 uint8/uint16 或 double（0~1）。這裡示範維持 uint8：
# I1_mat = matlab.uint8(I1.tolist())

# # 4) 在 MATLAB 執行 specklefilt（回傳 1 個輸出）
# J1_mat = eng.specklefilt(I1_mat, nargout=1)

# # 5) 轉回 NumPy
# J1 = np.array(J1_mat, dtype=np.uint8)  # 若 specklefilt 回 double，可改 dtype=np.float64

# # 6) 在 Python 顯示結果
# plt.figure()
# plt.imshow(J1, cmap='gray')
# plt.title(r"1.4 Filtered Back-projection ($\theta = 0:0.3:180$, Ram-Lak)")
# plt.axis('off')
# plt.show()

img_path = "./barfet2.jpg"      # 你的輸入影像
I1 = np.array(Image.open(img_path).convert('L'))  # uint8, shape (H, W)

I1_float = I1.astype(np.float64) / 255.0               # [0,1]
I1_mat   = matlab.double(I1_float.tolist())            # 給 MATLAB 的 double

# 3) 在 MATLAB 端產生高斯雜訊影像（double[0,1]）
#    注意：imnoise 的 mean=0, var=0.025（相當大，依需求可調）
noise_mat = eng.imnoise(I1_mat, 'speckle')
noise_mat1 = eng.imnoise(I1_mat, 'gaussian')
noise_mat2 = eng.imnoise(I1_mat, 'salt & pepper')

# 轉回 NumPy（仍是 [0,1] 的 float）
noise = np.array(noise_mat, dtype=np.float64)
noise1 = np.array(noise_mat1, dtype=np.float64)
noise2 = np.array(noise_mat2, dtype=np.float64)
# 4) 不同視窗大小做 wiener2（在 MATLAB 端跑），回傳結果轉回 NumPy
wins = [ 8, 9, 10, 11, 12, 13]  # 你原本 i+2 的那四組
filtered_list = []
for w in wins:
    win_vec = matlab.double([[w, w]])  # 1x2 向量要用雙層 list
    J_mat = eng.wiener2(noise_mat, win_vec, nargout=1)
    J = np.array(J_mat, dtype=np.float64)  # [0,1]
    filtered_list.append((w, J))

# 5) 計算指標（**建議用原圖 I1_float 作為參考**）
print("=== Metrics vs. Reference (clean image) ===")
for w, J in filtered_list:
    psnr_val = peak_signal_noise_ratio(I1_float, J, data_range=1.0)
    ssim_val = structural_similarity(I1_float, J, data_range=1.0)  # skimage>=0.19，灰階不用指定 channel_axis
    mse_val  = mean_squared_error(I1_float, J)
    print(f"W={w:2d} | PSNR={psnr_val:.3f} dB | SSIM={ssim_val:.6f} | MSE={mse_val:.6e}")

# （可選）也可看「去噪前後」：Noisy vs Reference
noisy_psnr = peak_signal_noise_ratio(I1_float, noise, data_range=1.0)
noisy_ssim = structural_similarity(I1_float, noise, data_range=1.0)
noisy_mse  = mean_squared_error(I1_float, noise)
print(f"\nNoisy vs Ref | PSNR={noisy_psnr:.3f} dB | SSIM={noisy_ssim:.6f} | MSE={noisy_mse:.6e}")

# 6) 視覺化（用 cmap='gray' 避免假彩）
plt.figure(figsize=(10, 7))
plt.subplot(2, 3, 4); plt.imshow(noise, cmap='gray'); plt.title("speckle"); plt.axis('off')
plt.subplot(2, 3, 5); plt.imshow(noise1, cmap='gray'); plt.title("gaussian"); plt.axis('off')
plt.subplot(2, 3, 6); plt.imshow(noise2, cmap='gray'); plt.title("salt & pepper"); plt.axis('off')
plt.subplot(2, 3, 2); plt.imshow(I1, cmap='gray'); plt.title("origin"); plt.axis('off')

# for i, (w, J) in enumerate(filtered_list, start=1):
#     plt.subplot(2, 3, i)
#     plt.imshow(J, cmap='gray')
#     plt.title(f"wiener2 {w}x{w}")
#     plt.axis('off')

plt.tight_layout()
plt.show()



