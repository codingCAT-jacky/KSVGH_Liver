import cv2
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import SimpleITK as sitk
import math

def order_points(pts):
    # 依照左上、右上、右下、左下順序排序
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def circle_from_3pts(A, B, C):
    A, B, C = map(np.asarray, (A, B, C))
    ax, ay = A; bx, by = B; cx, cy = C
    d = 2 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(d) < 1e-9:
        return None, None  # 三點幾乎共線，外接圓不穩定
    ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by)) / d
    uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax)) / d
    center = np.array([ux, uy], dtype=np.float32)
    r = np.linalg.norm(center - A)
    return center, r

def fit_circle(points, weights=None): #more than 3 points
    P = np.asarray(points, dtype=np.float64)
    x, y = P[:,0], P[:,1]
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x*x + y*y
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=np.float64))
        A = A * w[:,None]; b = b * w
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)  # [A,B,C]
    A_, B_, C_ = sol
    cx, cy = A_, B_
    r = np.sqrt(cx*cx + cy*cy + C_)

    return (float(cx), float(cy)), float(r)

def crop_image(pixel_bgr):
    # 1) 二值化：把非黑的地方抓出來
    grayImg = pixel_bgr if len(pixel_bgr.shape) == 2 else cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2GRAY)
    h, w = grayImg.shape
    cnt, finalx, finaly, finalww, finalhh, nfinalx, nfinaly, nfinalww, nfinalhh, yfinalx, yfinaly, yfinalww, yfinalhh = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    finalMask, cropped = [], []
    imageDamaged, containsYellow = False, False

    #) 2 偵測是否有大面積黃色ROI
    hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    ycontours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(ycontours) > 0:
        maxycnt = max(ycontours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(maxycnt, True)
        approx = cv2.approxPolyDP(maxycnt, epsilon, True)
        area = cv2.contourArea(approx)
        containsYellow = True if area > 10000 else False

    # 3) 切圖片
    if containsYellow: 
        # 對含有兩張超音波的圖片：先侵蝕再膨脹
        _, bw = cv2.threshold(grayImg, 10, 255, cv2.THRESH_BINARY)
        bw_mid = bw[:, int(w * 0.10):int(w * 0.95)]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 7))  # 寬3、高25你可以調
        bw_mid  = cv2.erode(bw_mid, kernel, iterations=1)
        # 取面積最大的那個 取得直線 bounding box 
        contours, _ = cv2.findContours(bw_mid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        yfinalx, yfinaly, yfinalww, yfinalhh = cv2.boundingRect(cnt)
        yfinalx += int(w * 0.10)
        cnt[:, :, 0] += int(w * 0.10) 
    else:         
    # 對可能損壞的圖片 做兩次mask
        _, bw = cv2.threshold(grayImg, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
    # mask2 = np.zeros((h, w), dtype=np.uint8)
    # cv2.drawContours(mask2, [cnt.astype(np.int32)], contourIdx=-1, color=255, thickness=1)
    # cv2.imshow("pixel_bgr ", pixel_bgr)  # 或 pixel_bgr
    # cv2.imshow("bw ", bw)  # 或 pixel_bgr
    # cv2.imshow("mask2 ", mask2)  # 或 pixel_bgr
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 3.1 找上半圓的三個點
    tmpx, tmpy, tmpww, boundingHeight = cv2.boundingRect(cnt)
    pts = cnt.reshape(-1, 2)
    upper_pts = pts[pts[:, 1] <= h * 0.5] # 取上半段點
    minUpperY = np.min(upper_pts[:, 1])
    maxUpperY = np.max(upper_pts[:, 1])
    minUpperPts = upper_pts[upper_pts[:, 1] == minUpperY]
    firstPts = minUpperPts[0]
    topLeft, topRight, arcHeight, arcPtCnt, higherCorner = 0, 0, 0, 0, 0
    foundR, foundL = False, False
    middleRMid, middleLMid = None, None  # 初始化變數，避免未定義錯誤

    pointsOnArcL, pointsOnArcR = [], []
    pointsWeight = [4,4]
    foundTwoPts = False
    while not foundTwoPts:
        for pts in minUpperPts:
            if  np.abs(pts[0] - firstPts[0]) > 100:  # 間距夠大代表找到兩個點了
                topRight = pts if pts[0] > firstPts[0] else firstPts
                topLeft = firstPts if pts[0] > firstPts[0] else pts
                foundTwoPts = True
                break
        minUpperY += 1
        while(len(upper_pts[upper_pts[:, 1] == minUpperY]) == 0):
            minUpperY += 1
            if minUpperY > maxUpperY:
                imageDamaged, foundTwoPts = True, True  #其實沒有找到兩點 但找不到點了 直接跳出去
                break
        minUpperPts = upper_pts[upper_pts[:, 1] == minUpperY]
        firstPts = minUpperPts[0] if len(minUpperPts) > 0  else [0,0]
    
    higherCorner = topRight[1] if topLeft[1]>topRight[1]  else topLeft[1]
    pointsOnArcL.append(topLeft.copy())
    pointsOnArcR.append(topRight.copy())
    itToLeft, itToRight = (topLeft + topRight) // 2, (topLeft + topRight) // 2
    while itToLeft[0] > topLeft[0] and itToRight[0] < topRight[0]:
        # 找圓弧上的右邊點
        while not foundR and itToRight[0] < topRight[0]: 
            midPts = upper_pts[upper_pts[:, 0] == itToRight[0]]
            if len(midPts) > 0:
                middleRMid = midPts[np.argmin(midPts[:, 1])]
                # if not (middleRMid[1] > abs(topLeft[0] - topRight[0])): #過於偏下
                #     foundR = True
                if len(pointsOnArcR)==1:
                    foundR = True if (middleRMid[1] < abs(topLeft[0] - topRight[0])) else False #過於偏下
                elif (middleRMid[1]-pointsOnArcR[len(pointsOnArcR)-1][1]) < 2*abs(pointsOnArcR[len(pointsOnArcR)-1][1]-pointsOnArcR[len(pointsOnArcR)-2][1]):
                    foundR = True
            itToRight[0] += 1
        # 找圓弧上的左邊點
        while not foundL and itToLeft[0] > topLeft[0]: 
            itToLeft[0] -= 1
            midPts = upper_pts[upper_pts[:, 0] == itToLeft[0]]
            if len(midPts) > 0:
                middleLMid = midPts[np.argmin(midPts[:, 1])]
                # if not (middleLMid[1] > abs(topLeft[0] - topRight[0])): #過於偏下
                #     foundL = True
                if len(pointsOnArcL)==1:
                    foundL = True if (middleLMid[1] < abs(topLeft[0] - topRight[0])) else False #過於偏下
                elif (middleLMid[1]-pointsOnArcL[len(pointsOnArcL)-1][1]) < 2*abs(pointsOnArcL[len(pointsOnArcL)-1][1]-pointsOnArcL[len(pointsOnArcL)-2][1]):
                    foundL = True
        # 一次加兩個點
        if foundR and foundL:
            pointsOnArcL.append(middleLMid.copy())
            pointsOnArcR.append(middleRMid.copy())
            pointsWeight.append(1)
            pointsWeight.append(1)
            foundR, foundL = False, False
            if arcPtCnt==0:
                arcPtCnt+=1
                arcHeight = (middleLMid[1] + middleRMid[1]) // 2 
            break #只取兩個點

    
    # 3.2 以4個點找出圓心和半徑
    if not imageDamaged:
        # center, smallRadius = circle_from_3pts(topLeft, topRight, middleRMid)
        center, smallRadius = fit_circle(pointsOnArcL+pointsOnArcR, pointsWeight)
        centerX, centerY = int(center[0]), int(center[1])
        
        #3.2 以這個圓心畫出兩個「完整扇形」mask 
        bigRadius = int(smallRadius) + boundingHeight - (arcHeight-higherCorner)
        start_angle = int(math.degrees(math.atan2(topLeft[1] - centerY, topLeft[0] - centerX))) # 左端
        end_angle   = int(math.degrees(math.atan2(topRight[1] - centerY, topRight[0] - centerX)))  # 右端
        maskSmall = np.zeros(grayImg.shape, np.uint8)
        maskBig = np.zeros(grayImg.shape, np.uint8)
        cv2.ellipse(maskSmall, (centerX, centerY), (int(smallRadius), int(smallRadius)), 0, start_angle, end_angle, 255, -1)
        cv2.ellipse(maskBig,  (centerX, centerY), (bigRadius, bigRadius), 0, start_angle, end_angle, 255, -1)

        #3.3 取得最終mask 用新mask再標一次邊框 
        finalMask = cv2.subtract(maskBig, maskSmall)  # 自動飽和裁切到 [0,255]，不會出現 1
        newContours, _ = cv2.findContours(finalMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        newCnt = max(newContours, key=cv2.contourArea)
        nfinalx, nfinaly, nfinalww, nfinalhh = cv2.boundingRect(newCnt)
        finalx, finaly, finalww, finalhh = (nfinalx, nfinaly, nfinalww, nfinalhh) if not containsYellow else (yfinalx, yfinaly, yfinalww, yfinalhh)
        finalMask = finalMask[finaly:finaly+finalhh, finalx:finalx+finalww]
        cropped = pixel_bgr[finaly:finaly+finalhh, finalx:finalx+finalww]
        if abs(start_angle - end_angle) < 60 or cropped.shape[0] < 500 or cropped.shape[1] < 500:
            imageDamaged = True
    # print(f"num of points on arc is {len(pointsOnArcL+pointsOnArcR)}, cropped shape is {cropped.shape}")
    # cv2.circle(pixel_bgr,(centerX, centerY), bigRadius, (255, 255, 0), 1)    
    # cv2.circle(pixel_bgr,(centerX, centerY), int(smallRadius), (255, 255, 0), 1) 
    # for point in pointsOnArcL+pointsOnArcR:
    #     cv2.circle(pixel_bgr,(point[0], point[1]), 1, (0, 255, 255), 1)
       
        
    # if imageDamaged:
    #                 print(f"start_angle is : {start_angle}, end_angle is : {end_angle}, ")
    #                 print(f"topLeft is : {topLeft}, topRight is : {topRight}, middleRMid is : {middleRMid}, middleLMid is :{middleLMid}")
    #                 mask2 = np.zeros((h, w), dtype=np.uint8)
    #                 cv2.drawContours(mask2, [cnt.astype(np.int32)], contourIdx=-1, color=255, thickness=1)
    #                 cv2.imshow("pixel_bgr", pixel_bgr)  # 或 pixel_bgr
    #                 cv2.imshow("cropped", cropped)  # 或 pixel_bgr
    #                 cv2.imshow("bw ", bw)  # 或 pixel_bgr
    #                 cv2.imshow("mask2 ", mask2)  # 或 pixel_bgr
    #                 cv2.waitKey(0)
    #                 cv2.destroyAllWindows()

    # cv2.imshow("maskBig ", maskBig)  # 或 pixel_bgr
    # cv2.imshow("maskSmall ", maskSmall)  # 或 pixel_bgr
    # cv2.imshow("finalMask ", finalMask)  # 或 pixel_bgr
    # cv2.imshow("cropped ", cropped)  # 或 pixel_bgr
    # cv2.imshow("pixel_bgr ", pixel_bgr)  # 或 pixel_bgr
    # cv2.imshow("SHOWIMG ", showImg)  # 或 pixel_bgr
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped, finalMask, imageDamaged

# 載入 DICOM 檔案
def open_dicom_image(filepath):
    count = 0
    for sonoRoot, sonoDirs, files in os.walk(filepath):
        # 1. 資料夾設定
        newDir = ".\\nckuPng\\CPng\\" +  sonoRoot.strip('./ncku')
        newMaskDir = ".\\nckuPng\\CMask\\" +  sonoRoot.strip('./ncku')
        os.makedirs(newDir, exist_ok=True)
        os.makedirs(newMaskDir, exist_ok=True)
        
        count += 1
        if count <= 219:
            continue
        print(f"newDir is : {newDir}")
        # 2. 標準化檔案
        for index, file in enumerate(files):
            full_path = os.path.join(sonoRoot, file)    
            if "dcm" in full_path:
                output_jpg_path = newDir + "\\" + file.strip('.dcm') + ".png"
                output_mask_path = newMaskDir + "\\" + file.strip('.dcm') + ".png"
                # print(f"file is : {output_jpg_path}") 
                # 讀取 DICOM 檔案 
                dicom_file = sitk.ReadImage(full_path)
                pixel_array = sitk.GetArrayFromImage(dicom_file)     # 形狀通常為 (1, H, W)
                pixel_array = pixel_array.squeeze()         
                # 步驟一：正規化至 0–255
                pixel_norm = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
                pixel_uint8 = pixel_norm.astype(np.uint8)
                pixel_bgr = cv2.cvtColor(pixel_uint8, cv2.COLOR_RGB2BGR)
                
                # impainting
                croppedFirst, finalMask, imageDamaged =  crop_image(pixel_bgr)
                if imageDamaged:
                    with open("damagedMask.txt", "a") as f:
                        f.write(f"{output_jpg_path}\n")
                    # cv2.imshow("damaged"+output_jpg_path, pixel_bgr)  # 或 pixel_bgr
                    # cv2.imshow("damaged"+croppedFirst, croppedFirst)  # 或 pixel_bgr
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                else:
                    hsv = cv2.cvtColor(croppedFirst, cv2.COLOR_BGR2HSV)
                    lower_yellow = np.array([20, 100, 100])
                    upper_yellow = np.array([40, 255, 255])
                    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                    # 擴大一點點，順便吃到反鋸齒邊緣像素
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                    dilmask = cv2.dilate(mask, kernel, iterations=2)
                    inpaintImg = cv2.inpaint(croppedFirst, dilmask, 2, cv2.INPAINT_TELEA)
                    cv2.imwrite(output_jpg_path, inpaintImg)
                    cv2.imwrite(output_mask_path, finalMask)

                # cv2.imshow(output_jpg_path, inpaintImg)  # 或 pixel_bgr
                # cv2.imshow("final mask out", finalMask)  # 或 pixel_bgr
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

dicom_file_path = './ncku'
open_dicom_image(dicom_file_path)

