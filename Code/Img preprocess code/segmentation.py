import cv2
import numpy as np
from roboflow import Roboflow
import os


def largest_axis_aligned_rect_in_mask(mask):
    """
    給 0/255 的二值 mask，回傳 (x, y, w, h) 為軸對齊最大內接矩形。
    """
    # 轉成 0/1
    M = (mask > 0).astype(np.uint8)
    H, W = M.shape

    # heights[i][j] = 以 (i,j) 為底，向上連續 1 的高度
    heights = np.zeros((H, W), dtype=np.int32)
    heights[0] = M[0]
    for i in range(1, H):
        heights[i] = (heights[i-1] + 1) * M[i]

    best_area, best = 0, (0, 0, 0, 0)

    # 對每一列做最大直方圖矩形（單調棧 O(W)）
    for i in range(H):
        h = heights[i]
        stack = []
        j = 0
        while j <= W:
            cur = h[j] if j < W else 0
            if not stack or cur >= h[stack[-1]]:
                stack.append(j); j += 1
            else:
                top = stack.pop()
                height = h[top]
                left  = stack[-1] + 1 if stack else 0
                right = j - 1
                width = right - left + 1
                area  = height * width
                if area > best_area:
                    best_area = area
                    # 矩形右下角在第 i 列，高度=height，寬=width
                    x = left
                    y = i - height + 1
                    w = width
                    h_rect = height
                    best = (x, y, w, h_rect)

    return best  # (x, y, w, h)


# 1. 設定參數 (請替換成您的真實資料)
API_KEY = "iKY1vNXkgGZWElyfp1kg"  # 請在此填入您的 Private API Key
IMAGE_PATH = "./nckuPng/OriginPng/NCKU0173/1_1_1_100.png" # 您要測試的圖片路徑

# 2. 初始化 Roboflow 並載入模型
rf = Roboflow(api_key=API_KEY)
# 根據您提供的網址：universe.roboflow.com/nuubms410liver/liver-8hjm6/model/3
project = rf.workspace("nuubms410liver").project("liver-8hjm6")
model = project.version(3).model

filepath = "./nckuPng/CPng"
for sonoRoot, sonoDirs, files in os.walk(filepath):
    # 1. 資料夾設定
    newDir = ".\\nckuPng\\BondingBoxPng" +  sonoRoot.strip('./nckuPng/CPng')
    os.makedirs(newDir, exist_ok=True)
    print(f"newDir is : {newDir}")
    # 2. 標準化檔案
    for index, file in enumerate(files):
        full_path = os.path.join(sonoRoot, file)    

        if "png" in full_path:
            # 3. 進行預測 (Inference) confidence=40 代表信心度超過 40% 才顯示，overlap=30 代表重疊度閾值
            prediction = model.predict(full_path, confidence=80).json()

            # 4. 讀取圖片準備繪圖
            image = cv2.imread(full_path)
            
            # print("預測結果：")
            # 5. 解析預測結果並繪製標記點
            for detection in prediction['predictions']:
                class_name = detection['class']
                confidence = detection['confidence']
                # print(f"- 偵測到: {class_name} (信心度: {confidence:.1%})")

                if confidence < 0.8:
                    with open("lowConfidence.txt", "a") as f:
                            f.write(f"{full_path} confidence:{confidence:.1%}\n")

                # 檢查是否有 'points' (分割模型的特徵)
                if 'points' in detection:
                    # 提取所有點的座標 [{'x': 10, 'y': 20}, ...]
                    points = detection['points']
                    x_center = detection['x']
                    y_center = detection['y']
                    width = detection['width']
                    height = detection['height']
                    # 轉換成 OpenCV 接受的 numpy 格式 (整數)
                    pts_list = [[int(p['x']), int(p['y'])] for p in points]
                    pts_np = np.array(pts_list, np.int32)
                    pts_np = pts_np.reshape((-1, 1, 2)) # Reshape 成 (點數, 1, 2)
                    

                    # --- 座標轉換 ---
                    # OpenCV 的 rectangle 需要「左上角」和「右下角」座標，且必須是整數
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # 畫出多邊形線條 (True 代表封閉圖形, 綠色 (0, 255, 0), 線寬 2)
                    # cv2.polylines(image, [pts_np], isClosed=True, color=(0, 255, 0), thickness=2)
                    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    # cv2.fillPoly(mask, [pts_np], 255)
                    # x, y, w, h = largest_axis_aligned_rect_in_mask(mask)
                    # 顏色是 BGR，這裡用黃色 (0,255,255)，線寬 2
                    # cv2.rectangle(image, (x, y) , (x + w, y + h), (0, 255, 255), 2)
                    # cropped = image[y:y+h, x:x+w]
                    cropped = image[y1:y2, x1:x2]
                    # cv2.imshow("cropped", cropped)
                    # cv2.waitKey(0)
                    output_jpg_path = newDir + "\\" + file
                    cv2.imwrite(output_jpg_path, cropped)
                    

                    
                    
                    # 標上文字
                    # x_text = pts_list[0][0]
                    # y_text = max(0, pts_list[0][1] - 10)
                    # cv2.putText(image, f"confidence: {confidence:.2%}", (x_text, y_text), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 6. 展示結果
cv2.imshow("Roboflow Inference Result", image)
cv2.waitKey(0) # 按任意鍵關閉視窗
cv2.destroyAllWindows()