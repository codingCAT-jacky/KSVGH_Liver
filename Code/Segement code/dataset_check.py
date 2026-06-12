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
import os
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from utils.utils import *

IMG_SIZE = (384, 384)  # 统一输出尺寸

class ImgObj:
    def __init__(self, img_path, dst_path, original_size=None):
        self.img_path = img_path
        self.dst_path = dst_path
        self.original_size = original_size  # 用于记录原始尺寸

class OnlyTest(Dataset):
    def __init__(self, file_root_dir, dst_root_dir):
        self.img_list = []
        cnt = 0
        for sonoRoot, sonoDirs, files in os.walk(file_root_dir):
            cnt += 1
            if cnt < 50:
                for index, file in enumerate(files):
                    if "png" in file:
                        full_path = os.path.join(sonoRoot, file) 
                        dst_patient_dir = sonoRoot.replace(file_root_dir, "", 1).lstrip(".\\")
                        dst_path = os.path.join(dst_root_dir, dst_patient_dir, file)
                        self.img_list.append(ImgObj(full_path, dst_path))
                        os.makedirs(os.path.join(dst_root_dir, dst_patient_dir), exist_ok=True)
                        print(f"Added image: {full_path} with destination: {dst_path}")

    def __getitem__(self, index):
        file_name = self.img_list[index].img_path
        img = cv2.imread(file_name, 0).astype(np.float32)
        self.img_list[index].original_size = img.shape  # 记录原始尺寸
        # print(f"Read image: {file_name} with shape: {img.shape}")
        
        img_resized = cv2.resize(img, IMG_SIZE)
        img_list = [img_resized, img_resized, img_resized]  # 复制成三通道

        img_data = np.array(img_list)
        img_data = normalization2(img_data, max=1, min=0)
        img_as_tensor = torch.from_numpy(img_data).float()

        return (img_as_tensor)
    
    def __len__(self):
        return len(self.img_list)
    
    def get_img_list(self):
        return self.img_list