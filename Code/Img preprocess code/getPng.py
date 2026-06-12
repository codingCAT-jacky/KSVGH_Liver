import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import SimpleITK as sitk


def is_bottom_arc(mask, cnt, bottom_ratio=0.6, residual_thresh=130.0):
    """
    回傳 True = 底部像弧形；False = 底部破損 
    bottom_ratio: 只拿 mask 高度的後面這一段來看底部 (0~1)
    residual_thresh: 殘差門檻，越小越嚴格
    """

    h, w = mask.shape[:2]

    # 2) 只取靠下的點
    xs = cnt[:, 0, 0]
    ys = cnt[:, 0, 1]
    bottom_mask = ys > (h * bottom_ratio)
    # print(f"bottom_mask is : {bottom_mask}")
    xs_bottom = xs[bottom_mask]
    ys_bottom = ys[bottom_mask]

    # 太少點就判失敗
    if len(xs_bottom) < 20:
        return False, None

    # 3) 擬合一個圓，OpenCV 有 minEnclosingCircle
    center, radius = cv2.minEnclosingCircle(np.column_stack((xs_bottom, ys_bottom)))
    cx, cy = center

    # 4) 算每個點到這個圓的距離差（實際半徑 - 擬合半徑）
    dists = np.sqrt((xs_bottom - cx)**2 + (ys_bottom - cy)**2)
    residual = np.abs(dists - radius)  # 每個點偏離圓多少
    mean_res = residual.mean()
    print(f"dists shape is : {dists.shape}, radius is : {radius}, mean_res is : {mean_res}")

    # 5) 用門檻判斷
    is_arc = mean_res < residual_thresh
    return is_arc

def croppe_image(pixel_bgr):
    grayImg = pixel_bgr if len(pixel_bgr.shape) == 2 else cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2GRAY)

    # 2) 二值化：把非黑的地方抓出來
    _, bw = cv2.threshold(grayImg, 1, 255, cv2.THRESH_BINARY)
    h, w = grayImg.shape

    # 3) 只取中間區域來找輪廓，避開左邊的表格跟右邊的色條 opening：先侵蝕再膨脹
    x1 = int(w * 0.05)    # 左切一點
    x2 = int(w * 0.95)    # 右切一點
    bw_mid = bw[:, x1:x2]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 7))  # 寬3、高25你可以調
    eroded  = cv2.erode(bw_mid, kernel, iterations=1)
    opened_mask  = cv2.dilate(eroded, kernel, iterations=1)

    # 4) 找輪廓  
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 沒有找到就結束
    if len(contours) == 0:
        raise RuntimeError("no contour found")

    # 取面積最大的那個（通常就是其中一張超音波圖）
    cnt = max(contours, key=cv2.contourArea)

    # 5) 取得直線 bounding box # 注意：我們是在裁過的區域(bw_mid)裡面算的，要把 x 加回去
    x, y, ww, hh = cv2.boundingRect(cnt)
    x += x1
    cropped = pixel_bgr[y:y+hh, x:x+ww]
    
    # 6) 裁原圖
    # width, height = 960, 704
    # topLeft = [y, (x + ww // 2) - (width // 2)]
    # cropped = pixel_bgr[topLeft[0]:topLeft[0]+height, topLeft[1]:topLeft[1]+width]
    
    # if( is_bottom_arc(bw, cnt) ):
    #     cropped = pixel_bgr[y:y+hh, x:x+ww]
    # else:
    #     cv2.imshow("pixel_bgr", pixel_bgr)
    #     cv2.imshow("bw", bw)
    #     cv2.imshow("cropped", cropped)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     # cv2.line(pixel_bgr,(topLeft[1], int(topLeft[0]+h*0.6)), (topLeft[1]+w, int(topLeft[0]+h*0.6)),(0,0,255),5)  # 繪製線條
    # print(f"cropped shape is : {cropped.shape} hh is : {hh}, ww is : {ww}")

    # cv2.imshow("pixel_bgr", pixel_bgr)
    # cv2.imshow("opened_mask", opened_mask)
    # cv2.imshow("cropped", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped

# 載入 DICOM 檔案
def open_dicom_image(filepath):
    try:
        count = 0
        for sonoRoot, sonoDirs, files in os.walk(filepath):
            count += 1
            newDir = ""
            
            if count > 290:
                strip = sonoRoot.strip('D:\\newU\\')
        
                if strip.count("\\") >= 1:
                    first, rest = strip.split("\\", 1)
                    stripResult = first + "\\" + rest.replace("\\", "")
                    newDir = ".\\nckuPng\\newOriginPng\\" +  stripResult
                else:
                    stripResult = strip
                    newDir = ".\\nckuPng\\newOriginPng\\" +  stripResult + "\\"
                    os.makedirs(newDir, exist_ok=True)
                
            else:
                continue
            # print(f"sonoRoot is : {sonoRoot}")
            # print(f"strip is : {strip}")
            # print(f"stripResult is : {stripResult}")
            # print(f"newDir is : {newDir}")
            for index, file in enumerate(files):
                full_path = os.path.join(sonoRoot, file) 

                if ".dcm" in full_path:  
                    # 提取影像資料
                    dicom_file = sitk.ReadImage(full_path)
                    pixel_array = sitk.GetArrayFromImage(dicom_file)     # 形狀通常為 (1, H, W)
                    pixel_array = pixel_array.squeeze()      


                    # 步驟一：正規化至 0–255
                    pixel_norm = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
                    pixel_uint8 = pixel_norm.astype(np.uint8)
                    pixel_bgr = cv2.cvtColor(pixel_uint8, cv2.COLOR_RGB2BGR)
                    
                    # 設定中心點 矩形尺寸
                    # h, w = pixel_bgr.shape[:2]
                    # cx, cy = w // 2 + 55, h // 2 
                    # # 建立 mask
                    # mask = np.ones((h, w), dtype=np.uint8)
                    # cv2.fillPoly(mask, pts, 0)
                    # # 套用 mask 到原圖
                    # result = cv2.bitwise_or(pixel_bgr, pixel_bgr, mask=mask)
                    # 從 masked 圖片中裁出這個區域
                    # cropped = pixel_bgr[cy - height // 2 : cy + height // 2, cx - width // 2 : cx + width // 2]

                    
                    # cropped = croppe_image(pixel_bgr)

                    
                    # 顯示與儲存結果
                    output_jpg_path = newDir + file.strip('.dcm') + ".png"
                    # print(f"output_jpg_path is : {output_jpg_path}")
                    # cv2.imshow(full_path, cropped)
                    cv2.imwrite(output_jpg_path, pixel_bgr)  # 存成 png
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {e}")

dicom_file_path = './ncku'
dicom_file_path2 = "D:\\newU"
# open_dicom_image(dicom_file_path)
open_dicom_image(dicom_file_path2)