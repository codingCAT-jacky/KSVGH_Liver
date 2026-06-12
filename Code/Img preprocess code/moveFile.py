import os
import shutil
import cv2
from PIL import Image
# from usFilter import matlabAF, matlabSRAD
# from skimage.metrics import mean_squared_error 
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity
# import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

# # 目標資料夾
# base_dir1 = "sonoBigPng"
# base_dir2 = "nckuPng"
# # 確保根資料夾存在
# os.makedirs(base_dir2, exist_ok=True)

# # 建立 NCKU001 ~ NCKU144
# for i in range(1, 201):
#     folder_name = f"NCKU{i:04d}"  # 自動補零到 4 位數
#     folder_path = os.path.join(base_dir1, folder_name)
#     os.makedirs(folder_path, exist_ok=True)
#     print(f"Created: {folder_path}")
    
# print("全部資料夾建立完成！")

# move
count = 0
for sonoRoot, sonoDirs, files in os.walk("./ncku"):
    count += 1
    # print(f"sonoRoot is : {sonoRoot}")
    if count > 200 and len(sonoDirs) > 0:
        for sonoDir in sonoDirs:
            dir_path = os.path.join(sonoRoot, sonoDir)
            print(f"sonoDir is : {sonoDir}, sonoRoot is : {sonoRoot}")
            for subRoot, subDirs, subfiles in os.walk(dir_path):
                for subfile in subfiles:
                    if subfile.endswith('.dcm'):
                        # 原始檔案位置
                        src_file = os.path.join(subRoot, subfile)
                        
                        # 設定新檔案路徑（這裡改名為 newDir.dcm）
                        new_file_name = str(sonoDir) + str(subfile)
                        dst_file = os.path.join(sonoRoot, new_file_name)

                        
                        # 移動並改名
                        shutil.move(src_file, dst_file)
                        print(f"src_file is : {src_file}")
                        print(f"dst_file is : {dst_file}")
print("所有資料夾已建立完成。")


# 計算總檔案數量
# count = 0
# arr1 = []
# arr2 = []
# for sonoRoot, sonoDirs, files in os.walk(base_dir1):
#     arr1.append(len(files))
# for sonoRoot, sonoDirs, files in os.walk(base_dir2):
#     arr2.append(len(files))
# for i in range(90):
#     if arr1[i] != arr2[i]:
#         print(f"Folder {i+1:03d} has different file counts: {arr1[i]} vs {arr2[i]}")

# print(f"Total files in {base_dir1}: {len(arr1)} and {base_dir1}: {len(arr2)}")
# 1639 + 1502


# apply filter
# 原始資料夾與輸出資料夾
# input_base_dir = "nckuPng"
# output_base_dir = "nckuWaveletPng"

# # 確保輸出根資料夾存在
# os.makedirs(output_base_dir, exist_ok=True)

# # 處理每個子資料夾 NCKU0001 ~ NCKU0144
# for folder_idx in range(1, 200): 
#         folder_name = f"NCKU{folder_idx:04d}"
#         input_folder = os.path.join(input_base_dir, folder_name)
#         output_folder = os.path.join(output_base_dir, folder_name)

#         # 確保輸出子資料夾存在
#         os.makedirs(output_folder, exist_ok=True)

#         # 處理該資料夾下的所有圖片
#         for filename in os.listdir(input_folder):
#             if filename.lower().endswith((".png")):
#                 input_path = os.path.join(input_folder, filename)

#                 # 套用 adaptive filter
#                 eng = matlab.engine.start_matlab()
#                 img = np.array(Image.open(input_path).convert('L'))  # 轉為灰階
#                 I_d = (img.astype(np.float64) / 255.0).tolist()
                
#                 # 一行流：wdenoise2
#                 I_mat = matlab.double(I_d)
#                 Ld = eng.wdenoise2(eng.log(eng.plus(I_mat, 2.220446049250313e-16)), 3,
#                                 'Wavelet','sym4',
#                                 'DenoisingMethod','Bayes',
#                                 'NoiseEstimate','LevelDependent', nargout=1)
#                 J = np.exp(np.array(Ld))               # 回到 [0,~]
#                 filtered_img = (255*np.clip(J/np.max(J),0,1)).astype(np.uint8)


#                 # 儲存結果
#                 output_path = os.path.join(output_folder, filename)
#                 cv2.imwrite(output_path, filtered_img)
#                 psnr_val = peak_signal_noise_ratio(img, filtered_img, data_range=1.0)
#                 ssim_val = structural_similarity(img, filtered_img, data_range=1.0)  # skimage>=0.19，灰階不用指定 channel_axis
#                 mse_val  = mean_squared_error(img, filtered_img)
                
#                 with open("./numeric/nckuWaveletSnr.txt", "a", encoding="utf-8") as f:
#                     f.write(f"file={output_path} | PSNR={psnr_val:.3f} dB | SSIM={ssim_val:.6f} | MSE={mse_val:.6e}\n")

#         print(f"✅ 處理完成: {folder_name}")

# print("全部處理完成！")

# img = cv2.imread("nckuPng\\NCKU0002\\1_1_1_100.png", cv2.IMREAD_GRAYSCALE)
# print(f"shape is : {img.shape}")
