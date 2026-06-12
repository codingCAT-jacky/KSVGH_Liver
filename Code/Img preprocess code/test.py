import cv2
# import pydicom
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
# from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_color_lut, convert_color_space
import os
from pathlib import Path
import SimpleITK as sitk
# import shutil
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
import pandas as pd


# 讀取 DICOM 檔案
# full_path = "C:\\Users\\JackyLai2\\Documents\計畫\\fatty liver\\ncku\\NCKU0197\\1_1_1_600.dcm"
# filepath = "C:\\Users\\JackyLai2\\Downloads\\image\\I0000029.qus"
# # 1. 讀取資料
# with open(filepath, 'rb') as f:
#     f.seek(76)
#     raw_data = np.frombuffer(f.read(), dtype=np.int16)

# # 折疊並轉置為 [深度, 掃描線]
# map_2d = raw_data.reshape((560, 8192)).T

# # 2. 隨機抽取中間的一條掃描線 (A-line)
# aline = map_2d[:, 280]

# # 3. 進行快速傅立葉轉換 (FFT)
# N = len(aline)
# yf = fft(aline)

# # 假設機台的類比數位轉換器 (ADC) 採樣頻率為 40 MHz (超音波常見規格)
# # 注意：即使真實採樣頻率不是 40MHz，頻譜的「鐘形波峰」特徵也不會改變
# fs = 40e6  
# xf = fftfreq(N, 1/fs)[:N//2]

# # 取得真實振幅
# amplitude = 2.0/N * np.abs(yf[0:N//2])

# # 4. 畫出頻譜圖
# plt.figure(figsize=(10, 6))
# plt.plot(xf / 1e6, amplitude, color='blue') # 將 X 軸轉為 MHz
# plt.title("Frequency Spectrum of a Single A-line (Proof of Raw RF)", fontsize=14)
# plt.xlabel("Frequency (MHz)", fontsize=12)
# plt.ylabel("Amplitude", fontsize=12)
# plt.grid(True)

# # 限制 X 軸範圍看清楚主要頻帶 (0 ~ 15 MHz)
# plt.xlim(0, 15)
# plt.show()
# dicom_file = sitk.ReadImage(full_path2)
# pixel_array = sitk.GetArrayFromImage(dicom_file)     # 形狀通常為 (1, H, W)
# pixel_array = pixel_array.squeeze()      


# # 步驟一：正規化至 0–255
# pixel_norm = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
# pixel_uint8 = pixel_norm.astype(np.uint8)
# pixel_bgr = cv2.cvtColor(pixel_uint8, cv2.COLOR_RGB2BGR)
# print("pixel array shape is : dim is: len shape is:", pixel_array.shape, pixel_array.ndim, len(pixel_array.shape))
# print("pixel norm type is : ", type(pixel_norm))
# print("pixel_uint8 type is : ", type(pixel_uint8))
# print(f"pixel_bgr ndim is : {pixel_bgr.ndim}")
# print(f"pixel_bgr shape is : {pixel_bgr.shape}")
# cv2.imshow("DICOM Image", pixel_bgr)  # 或 pixel_bgr
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # 印出部分 DICOM 資訊
# print("SOPClass:", dicom_file.SOPClassUID.name)
# print("Patient ID:", dicom_file.PatientID)
# print("Study Date:", dicom_file.StudyDate)
# print("Modality:", dicom_file.Modality)
# print(f"ImageType: {dicom_file.ImageType}")
# print(f"Series Description: {dicom_file.SeriesDescription}")

def countClass():
    cntClass = [0,0,0,0]
    bins = [0.064, 0.174, 0.221]
    
    with open("./numeric/nckuPdff.txt", 'r') as f:
        allLabels = [float(line.strip().replace('%', '')) for line in f.readlines()]
        for label in allLabels:
            tmp = np.digitize(label/100, bins, right=False)
            print(tmp)
            cntClass[tmp] += 1 # 產生 0,1,2,3] += 1
        print(f"Class counts: {cntClass}")
# countClass()            

def remove_dirs_without_files(root: str):
    root = Path(root)
    removed = 0
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(root)):
        if i==0:
            continue  # 跳過 root 資料夾本身    
        # 若此資料夾「沒有任何檔案」，而且子資料夾在這輪已處理完
        if len(filenames) == 0:
            os.rmdir(dirpath)     
            print(f"🗑️ 刪除空資料夾: {dirpath}")
# remove_dirs_without_files("./nckuBigPng")


def fileNum(root: str):
    count = 0
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(root)):
        for filename in filenames:
            if filename.lower().endswith((".png")):
                count += 1
    return count
# print(f"目前 nckuBigPng 資料夾共有 {fileNum('./nckuPng/CMask')} 張圖片")

def countSI(root: str):
    count = 0
    total_si = 0.0
    total_ssim = 0.0
    total_epi = 0.0

    with open(root, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "SI=" in line:
                # 取出 "SI=" 後面到下一個 '|' 或逗號的部分
                si_part = line.split("SI=")[1].split("|")[0].split(",")[0].strip()
                # ssim_part = line.split("SSIM=")[1].split("|")[0].split(",")[0].strip()
                # epi_part = line.split("epi=")[1].split("|")[0].split(",")[0].strip()

                si_val = float(si_part)
                # ssim_val = float(ssim_part)
                # epi_val = float(epi_part)
   
                total_si += si_val
                # total_ssim += ssim_val
                # total_epi += epi_val
     
                count += 1
                # if count<10:
                #     print(f"第 {count+1} 筆 SSIM: {ssim_val}")


    print(f"總 SI: {total_si}")
    # print(f"總 SSIM: {total_ssim}")
    # print(f"總 EPI: {total_epi}")

    print(f"平均 SI: {total_si/count if count>0 else 'N/A'}")
    # print(f"平均 SSIM: {total_ssim/count if count>0 else 'N/A'}")
    # print(f"平均 EPI: {total_epi/count if count>0 else 'N/A'}")
# countSI("./numeric/nckuSnr.txt")

def cropAPic():
    full_pathDcm = "./ncku/NCKU0084/1_1_1_50.dcm"
    output_path = "./nckuBigPng/NCKU0084/1_1_1_50.png"
    dicom_file = sitk.ReadImage(full_pathDcm)
    pixel_array = sitk.GetArrayFromImage(dicom_file)     # 形狀通常為 (1, H, W)
    pixel_array = pixel_array.squeeze()      
    pixel_norm = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
    pixel_uint8 = pixel_norm.astype(np.uint8)
  
    width = 360
    height = 360

    h, w = pixel_array.shape[0], pixel_array.shape[1]
    cx, cy = w // 2 - 40, h // 2 
    cropped = pixel_array[cy - height // 2 : cy + height // 2, cx - width // 2 : cx + width // 2]
    cv2.imwrite(output_path, cropped)
    # cv2.imshow("DICOM Image", pixel_uint8)  # 或 pixel_bgr
    # cv2.imshow("DICOM Image2", cropped)  # 或 pixel_bg
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# cropAPic()

def convertToGray():
    # root = "D:\\newU\\NCKU0004\\0520"
    root2 = "C:\\Users\\JackyLai2\\Documents\\計畫\\fatty liver\\ncku\\NCKU0001\\SAG"
    for img in os.listdir(root2):
        if img.lower().endswith((".dcm")):
            path = os.path.join(root2, img)
            dicom_file = sitk.ReadImage(path)
            pixel_array = sitk.GetArrayFromImage(dicom_file)     # 形狀通常為 (1, H, W)
            print(f"pixel arr shape is : {pixel_array.shape}, dim is : {pixel_array.ndim}")
            pixel_array = pixel_array.squeeze()      
            pixel_norm = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
            pixel_uint8 = pixel_norm.astype(np.uint8)
            pixel_bgr = cv2.cvtColor(pixel_uint8, cv2.COLOR_RGB2BGR)
            height, width = pixel_bgr.shape[:2]

            # 2. 計算右側裁剪區域
            # 這裡我們假設要裁剪並放大圖片的右半邊 (從寬度的一半到最右邊)
            # 您可以調整 crop_start_x 的計算方式來選擇不同比例的右側區域
            # crop_start_x = width // 4
            # 裁剪操作：[所有y行, x從起始點到最後, 所有通道]
            # crop_image = pixel_bgr[:, crop_start_x:width, :]

            # 3. 計算放大後的尺寸
            # 例如：放大 2.0 倍
            # scale_factor = 3.0
            # new_width = int(crop_image.shape[1] * scale_factor)
            # new_height = int(crop_image.shape[0] * scale_factor)

            # 4. 進行放大 (調整尺寸)
            # 使用 cv2.resize 函式，並指定插值方法。
            # cv2.INTER_CUBIC 或 cv2.INTER_LANCZOS4 在放大時通常能提供較好的品質
            # enlarged_image = cv2.resize(crop_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # 5. 儲存圖片
            # output_filename = 'enlarged_right_side.jpg'
            # cv2.imwrite(output_filename, enlarged_image)

            
            cv2.imshow("DICOM Image2", pixel_bgr)  # 或 pixel_bg
            cv2.waitKey(0)
            # gray_image = cv2.cvtColor(pixel_norm, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite(path, gray_image)
convertToGray()

def moveFilesToDir():
    source_root = r"C:\Users\JackyLai2\Documents\計畫\fatty liver\nckuPng\OriginPng\NCKU0201"
    cnt = 0
    for dirpath, dirnames, filenames in os.walk(source_root):
        cnt += 1
        # if cnt <=200:
        #     continue
        for filename in filenames:
            if "0808a" in filename:
                source_path = os.path.join(dirpath, filename)
                newFileName = "0927" + filename.strip("0808a")
                dst_path = os.path.join(dirpath, newFileName)
                print(f"source path is : {source_path}, dst path is : {dst_path}")
                os.rename(source_path, dst_path)
                # shutil.move(source_path, dst_path)
                # print(f"Moved: {source_path} -> {dst_path}")
# moveFilesToDir()

def cutColorImage():
    filepath = './nckuPng'
    for sonoRoot, sonoDirs, files in os.walk(filepath):
            count += 1
            newDir = ""
            if count > 1:
                newDir = ".\\nckuCPng\\" +  sonoRoot.strip('./ncku')
                os.makedirs(newDir, exist_ok=True)

            print(f"sonoRoot is : {sonoRoot}")
            for index, file in enumerate(files):
                img = cv2.imread("./nckuPng/NCKU0039/1_1_1_300.png")              # 原圖
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # # 1) 稍微模糊，讓二值化更穩
                # gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

                # 2) 二值化：把非黑的地方抓出來
                _, bw = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

                h, w = bw.shape

                # 3) 只取中間區域來找輪廓，避開左邊的表格跟右邊的色條
                x1 = int(w * 0.05)    # 左切一點
                x2 = int(w * 0.95)    # 右切一點
                bw_mid = bw[:, x1:x2]

                # 4) 找輪廓  
                contours, _ = cv2.findContours(bw_mid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 沒有找到就結束
                if len(contours) == 0:
                    raise RuntimeError("no contour found")

                # 取面積最大的那個（通常就是其中一張超音波圖）
                cnt = max(contours, key=cv2.contourArea)

                # 5) 取得直線 bounding box
                x, y, ww, hh = cv2.boundingRect(cnt)

                # 注意：我們是在裁過的區域(bw_mid)裡面算的，要把 x 加回去
                x += x1

                # 6) 裁原圖
                cropped = img[y:y+hh, x:x+ww]
                cv2.imshow("img", img)
                cv2.imshow("cropped.png", cropped)
                cv2.imshow("bw", bw)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
# cutColorImage()

def crt():
    cnt = 0
    for sonoRoot, sonoDirs, files in os.walk("C:\\Users\\JackyLai2\\Documents\\計畫\\fatty liver\\segem\\BDataset\\sample001\\liver-slice"):

        for index, file in enumerate(files):
            full_path = os.path.join(sonoRoot, file)  
            if "json" in full_path:
                print(f"json path is {full_path}")
            # img = cv2.imread(full_path)
            # h, w, _ = img.shape
            # centerx = w//2
            # centery = 370
            # widthd2, heighte2 = 200, 200
            # cropped =  img[centery-heighte2:centery+heighte2, centerx-widthd2:centerx+widthd2, :]
            # cv2.imshow("bw", cropped)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # if cropped.shape != (400, 400, 3):
            #     print(f"mask {cropped.shape} , path {full_path} ")
            #     cnt+=1
    print(f"cnt is {cnt}")
# crt()

def minSize():
    path = "./nckuPng/CPng"
    minH = 1000
    minW = 1000
    for sonoRoot, sonoDirs, files in os.walk(path):

        minFile = ""
        for index, file in enumerate(files):
            if file.lower().endswith((".png")):
                full_path = os.path.join(sonoRoot, file)  
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                h, w = img.shape
                width = min(540, w)
                height = min(560, h)
                midImg = img[h//2-height//2:h//2+height//2, w//2-width//2:w//2+width//2]
                cv2.imshow("img2", midImg)
                cv2.waitKey(0)
                # if h<minH or w<minW:
                #     print(f"minFile is {full_path}, minH is {minH}, minW is {minW}, h is {h}, w is {w}")
                #     minH = min(h, minH)
                #     minW = min(w, minW)
    print(f"minH is {minH}, minW is {minW}")
# minSize()

def printSize():
    path = "./nckuPng/CPng/NCKU0104/1_1_1_26.png"
    path2 = "./nckuPng/CPng/NCKU0127/7A1_1_1_1.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
    h, w = img.shape
    h2, w2 = img2.shape
    print(f"file: {path}, height: {h}, width: {w}")
    print(f"file: {path2}, height: {h2}, width: {w2}")
# printSize()

def getDataFromExcel():
    # 1. 讀取 CSV 檔案，將 header 設定為 1 (略過第一列的分類，使用第二列的詳細欄位名稱)
    file_path = './ncku/shotList.xlsx'
    df = pd.read_csv(file_path, header=1)

    # 2. 自動抓取包含關鍵字的欄位名稱，避免因換行符號 (\n) 導致抓錯欄位
    sex_col = [col for col in df.columns if 'Sex' in col][0]
    height_col = [col for col in df.columns if 'Height' in col][0]
    bmi_col = [col for col in df.columns if 'BMI' in col][0]
    skin_col = [col for col in df.columns if 'skin-to-liver' in col][0]

    # 3. 確保資料型別為數值，並將無效值轉為 NaN
    df[height_col] = pd.to_numeric(df[height_col], errors='coerce')
    df[bmi_col] = pd.to_numeric(df[bmi_col], errors='coerce')
    df[skin_col] = pd.to_numeric(df[skin_col], errors='coerce')

    # 4. 篩選男生與女生的資料
    male_df = df[df[sex_col].str.contains('M', na=False)]
    female_df = df[df[sex_col].str.contains('F', na=False)]

    # 5. 定義計算與印出統計量的函式
    def print_stats(name, series):
        series = series.dropna() # 排除空值
        print(f"=== {name} ===")
        print(f"平均數 (Mean) : {series.mean():.2f}")
        print(f"標準差 (Std)  : {series.std():.2f}")
        print(f"最大值 (Max)  : {series.max():.2f}")
        print(f"最小值 (Min)  : {series.min():.2f}\n")

    # 6. 計算並顯示結果
    print_stats("男生身高", male_df[height_col])
    print_stats("女生身高", female_df[height_col])
    print_stats("BMI (全體)", df[bmi_col])
    print_stats("Skin-to-liver capsular distance (mm) (全體)", df[skin_col])


def validate_ten_floats_per_line(file_path):
        error_lines = []
        # 建議指定編碼，避免跨平台讀取錯誤
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, start=1):
                # 移除頭尾的空白與換行字元
                line = line.strip()

                # 將該行依據空白字元（空格或 Tab）進行分割
                # 如果你的資料是用逗號分隔的 (CSV)，請改成 line.split(',')
                elements = line.split()

                # 檢查條件 1：數量是否剛好為 10
                if len(elements) != 10:
                    print(f"第 {line_num} 行數量錯誤: 找到 {len(elements)} 個元素，預期 10 個")
                    continue

                # 檢查條件 2：這 10 個元素是否都能轉換為浮點數
                has_invalid_float = False
                invalid_items = []
                
                for item in elements:
                    try:
                        float(item) # 嘗試轉換為浮點數
                    except ValueError:
                        has_invalid_float = True
                        invalid_items.append(item)
                
                if has_invalid_float:
                    print(f"第 {line_num} 行包含非數字內容: {invalid_items}")
# validate_ten_floats_per_line("./numeric/nckuTAITSI.txt")


def detectYellow():
                    # 矩形尺寸
                    # h, w = pixel_bgr.shape[:2]
                    width = 360
                    height = 360
                    # # 偵測輪廓
                    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


                    # # 裁減TAI
                    # countRoi = 0
                    # for contour in contours:
                    #     # 擬合為多邊形（近似矩形）
                    #     epsilon = 0.02 * cv2.arcLength(contour, True)
                    #     approx = cv2.approxPolyDP(contour, epsilon, True)

                    #     # 若為四邊形，印出角點
                    #     if len(approx) == 4:
                    #         area = cv2.contourArea(approx)
                    #         if area>10000:
                    #             if countRoi == 1:
                    #                 # 裁減出ROI
                    #                 points = approx.reshape(4, 2).astype(np.float32)
                    #                 rect = order_points(points)
                    #                 upper_mid = (rect[0] + rect[1]) / 2
                    #                 lower_mid = (rect[2] + rect[3]) / 2
                    #                 mid = (upper_mid + lower_mid) / 2
                    #                 cx, cy = int(mid[0]), int(mid[1])
                    #                 cropped = inpaintImg[cy - height // 2 : cy + height // 2, cx - width // 2 : cx + width // 2]
                    #                 # dst = np.array([[0, 0], [width - 1, 0],[width - 1, height - 1],[0, height - 1]], dtype="float32")
                    #                 # M = cv2.getPerspectiveTransform(rect, dst)
                    #                 # warped = cv2.warpPerspective(pixel_bgr, M, (int(width), int(height)))
            
                    #                 # 畫出來確認
                    #                 cv2.imshow("cropped", cropped)  # 或 pixel_bgr
                    #                 cv2.imshow("TAI", pixel_bgr)  # 或 pixel_bgr
                    #                 # coutoursImg = cv2.drawContours(pixel_bgr, [approx], -1, (0, 0, 255), 2) 
                    #                 # cv2.imshow("coutoursImg", coutoursImg)
                    #                 # cv2.imshow("warped", warped)
                    #                 # cv2.imwrite(output_jpg_path, cropped)  # 存成 png
                    #                 cv2.waitKey(0)
                    #                 cv2.destroyAllWindows()
                    #             else:
                    #                 countRoi += 1
                    
                    # # 裁減b mode
                    # if countRoi == 0:
                    #     # 從 masked 圖片中裁出這個區域
                    #     cx, cy = w // 2 + 55, h // 2 + 50
                    #     cropped = inpaintImg[cy - height // 2 : cy + height // 2, cx - width // 2 : cx + width // 2]

                        # 顯示與儲存結果
                        # cv2.imwrite(output_jpg_path, cropped)
                        # cv2.imshow("cropped", cropped)  # 或 pixel_bgr
                        # cv2.imshow("DICOM Image1", pixel_bgr)  # 或 pixel_bgr
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                    # 印出部分 DICOM 資訊
                    # print("Patient ID:", dicom_file.PatientID)
                    # print("Study Date:", dicom_file.StudyDate)
                    # print("Modality:", dicom_file.Modality)
                    # print(f"ImageType: {dicom_file.ImageType}")
                    # print(f"Series Description: {dicom_file.SeriesDescription}")

