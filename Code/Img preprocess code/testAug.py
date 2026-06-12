from shutil import copy
import albumentations as A
from usaugment.albumentations import DepthAttenuation, GaussianShadow, HazeArtifact, SpeckleReduction
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

transform = A.Compose(
    [
        # DepthAttenuation(p=1, attenuation_rate=1.0, max_attenuation=0.0), #如果max_attenuation = 0：衰減最為劇烈，亮度從探頭處的1降到遠方的 0。如果= 1，影像亮度都保持不變。
        GaussianShadow(p=1.0, strength=0.8, sigma_x=0.08, sigma_y=0.08), #機率 強度 影子在x軸的標準差(擴散程度) 影子在y軸的標準差
    ],
    additional_targets={"scan_mask": "mask"}
)
# image = Image.open("./nckuPng/nckuCPng/NCKU0001/1_1_1_1000.png")
# image = image.convert('RGB') 
# scan_mask = Image.open("./nckuPng/nckuCMask/NCKU0001/1_1_1_1000.png").convert('L') 
# scan_mask = scan_mask.convert('RGB') 
# plt.imshow(image)
# plt.show()
# plt.imshow(scan_mask)
# plt.show()

tMaskEx = cv2.imread("./picture/byra2018_liver_ultrasound_mask.png", cv2.IMREAD_GRAYSCALE)
tMask_clippedEx = np.clip(tMaskEx, None, 1)
tMask = cv2.imread("./nckuPng/CMask/NCKU0001/1_1_1_1000.png",  cv2.IMREAD_GRAYSCALE)
tMask_clipped = np.clip(tMask, None, 1)

tImgEx = cv2.imread("./picture/byra2018_liver_ultrasound.png", cv2.IMREAD_GRAYSCALE)
copytImgEx = tImgEx / 255.0
copytImgEx = np.stack([copytImgEx, copytImgEx, copytImgEx], axis=-1)
# tImg = cv2.imread("./nckuPng/CPng/NCKU0001/1_1_1_1000.png",  cv2.IMREAD_GRAYSCALE)
tImg = cv2.imread("./firstaug.jpg",  cv2.IMREAD_GRAYSCALE)
copytImg = tImg / 255.0
copytImg = np.stack([copytImg, copytImg, copytImg], axis=-1)

# cv2.imshow("pixel_bgr ", tImg)  # 或 pixel_bgr
# cv2.waitKey(0)
# cv2.destroyAllWindows()

transformedImgEx = transform(image=copytImgEx, scan_mask=tMask_clippedEx)
transformedImg = transform(image=copytImg, scan_mask=tMask_clipped)


# print(cropped.shape)
# for channel in cropped:
#     for row in channel:
#         for val in row:
#             if val>1:
#                 print("not 0 to 1")
# print(f"type {type(transformedImg)}, keys {transformedImg.keys()}")

      
print(f"type img {type(tImg)}")
cv2.imshow("processed ", transformedImg["image"])  # 或 pixel_bgr
cv2.imshow("origin ", tImg)  # 或 pixel_bgr
# cv2.imshow("Ex processed ", transformedImgEx["image"])  # 或 pixel_bgr
# cv2.imshow("Ex origin ", tImgEx)  # 或 pixel_bgr
# cv2.imshow("origin mask ", tMask)  # 或 pixel_bgr
# cv2.imshow("mask ", transformedImg["scan_mask"]*255)  # 或 pixel_bgr
# cv2.imshow("cropped", cropped)
# cv2.imwrite("firstaug.jpg", transformedImg["image"]*255)
cv2.imwrite("secondaug.jpg", transformedImg["image"]*255)
# cv2.imwrite("ori.jpg", tImg )
cv2.waitKey(0)



