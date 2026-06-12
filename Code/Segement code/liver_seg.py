# SPDX-License-Identifier: AGPL-3.0-only


#    Copyright (C) 2024 Zone24x7, Inc  
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License version 3 as
#    published by the Free Software Foundation. 
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License version 3.0 for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import argparse
import numpy as np
import os
import torch.nn as nn
from utils.utils import *
from dataset_liverusrecon import *
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

IMG_SIZE = 384

def overlay_red_mask(original_gray, mask_gray, alpha=0.5):
    """
    將灰階 Mask 以紅色半透明覆蓋到原始灰階影像上。
    
    參數:
        original_gray (np.ndarray): 原始灰階圖，形狀 (H, W)
        mask_gray (np.ndarray): 模型預測的灰階或二值化遮罩，形狀 (H, W)
        alpha (float): 紅色的透明度 (0.0 ~ 1.0)，越大紅色越不透明
    回傳:
        np.ndarray: 疊加完成的 BGR 彩色影像
    """
    # 1. 將單通道灰階底圖，轉換為三通道的 BGR 彩色圖
    # 這樣我們才有空間可以塞入紅色的像素
    bg_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    
    # 2. 建立一個形狀相同的「純紅」圖層
    # 注意：OpenCV 的通道順序是 B, G, R，所以紅色在索引 2
    red_layer = np.zeros_like(bg_bgr)
    red_layer[:, :, 2] = 255  
    
    # 3. 找出 Mask 中需要標記的範圍（排除背景的 0）
    roi = mask_gray > 0
    
    # 4. 只針對 Mask 涵蓋的區域進行 Alpha 混和
    result = bg_bgr.copy()
    result[roi] = bg_bgr[roi] * (1 - alpha) + red_layer[roi] * alpha
    
    # 確保最終數值型態為 OpenCV 接受的 uint8
    return result.astype(np.uint8)


class weighted_mse_loss_WITHSCALE(nn.Module):
    def __init__(self, weights):
        super(weighted_mse_loss_WITHSCALE, self).__init__()
        self.weights = torch.from_numpy(weights).float().cuda()

    def forward(self, out, label):
        weights = self.weights.unsqueeze(dim=0)
        # weights = torch.cat([weights, torch.FloatTensor([[1.]]).cuda()], dim=1)
        pct_var = (out - label)**2
        output = pct_var * weights
        loss = output.mean()
        return loss

def transunet(): 
    
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 2
    config_vit.n_skip = 3
    config_vit.patches.size = (16, 16)
    if 'R50-ViT-B_16'.find('R50') != -1:
        # IMG_SIZE 是 tuple (192, 192)
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))
    net = ViT_seg(config_vit, img_size=IMG_SIZE, num_classes=config_vit.n_classes).cuda()
    # net.load_from(weights=np.load(config_vit.pretrained_path))
    return net

def save_prediction_image(preds, batch_id, save_folder_name):

    xx=["ANTAX","MIDLINE","MCL"]
    desired_path = os.path.join(save_folder_name,  batch_id)

    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    for i in range(preds.shape[0]):
        img = preds[i, :, :] * 255
        img_np = img.astype('uint8')
        img_resized = cv2.resize(img_np, (1280, 876)) # 恢复到原始尺寸
        export_name = xx[i] + '.png'
        cv2.imwrite(os.path.join(desired_path, export_name), img_resized)


def Inference(seg_model, data_test):
    """推理分割模型，仅输出分割掩码"""
    pred_list = []
    
    for batch_id, (images) in enumerate(data_test):
        with torch.no_grad():
            bs = images.shape[0]
            images = images.view(-1, 1, IMG_SIZE, IMG_SIZE)
            images = images.cuda()
            print(f"images shape: {images.shape}")
            # 分割模型
            output = seg_model(images)
            output = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).float().cpu().data.numpy()
            pred_list.append(pred)
    # print(len(pred_list))
    # print(pred_list[0].shape)
    return np.array(pred_list)


def main_function():
    seg_model_path = "C:\\Users\\JackyLai2\\Documents\\計畫\\fatty liver\\segem\\models\\seg_model_epoch_100.pkl"
    save_path_pred = ".\\nckuPng\\predicted_liver_masks"
    data_dir = ".\\nckuPng\\CPng"

    if not os.path.exists(save_path_pred):
        os.mkdir(save_path_pred)

    seed_value = 1234
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

    inference = OnlyTest(data_dir, save_path_pred)
    inference_load = torch.utils.data.DataLoader(dataset=inference,
                                    num_workers=0, batch_size=1, shuffle=False)
    
    seg_model = transunet()
    print(f"Loading segmentation model weights from: {seg_model_path}")
    seg_model.load_state_dict(torch.load(seg_model_path))
    
    mask_out = Inference(seg_model, inference_load) 
    print(f"mask_out shape: {mask_out.shape}")  # (N, Hc, Wc) 或 (N, C, Hc, Wc)，取决于模型输出

    # 保存分割掩码
    img_list = inference.get_img_list()
    for index, (pred_images) in enumerate(mask_out):
        export_name = img_list[index].dst_path
        img_original_size = img_list[index].original_size
        # print(f"Original image size for index {index}: {img_original_size}")
        print(f"Saving predicted mask to: {export_name}")
        img = pred_images[0, :, :] * 255
        img_np = img.astype('uint8')
        img_resized = cv2.resize(img_np, (img_original_size[1], img_original_size[0])) # 恢复到原始尺寸
        cv2.imwrite(export_name, img_resized)

        # overlay
        # file_name = img_list[index].img_path
        # ori_img = cv2.imread(file_name, 0).astype(np.float32)
        # overlay_img = overlay_red_mask(ori_img, img_resized, alpha=0.5)
        # cv2.imwrite("overlay_result.png", overlay_img)
        
if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.cuda)

    main_function()


