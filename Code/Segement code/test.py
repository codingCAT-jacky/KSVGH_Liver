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
import trimesh
import torch.nn as nn
from networks.para_model import para_model
from utils.utils import *
from dataset_liverusrecon import *
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PCA_NUM = 50
input_ch_num = 3
IMG_SIZE = 384

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
        export_name = xx[i] + '.png'
        cv2.imwrite(os.path.join(desired_path, export_name), img_np)


def Inference(models, data_test):

    model1 = models[0]
    model2 = models[1]
    out_pca_list = []
    pred_list = []
    for batch_id, (images) in enumerate(data_test):
        with torch.no_grad():
            bs = images.shape[0]
            images = images.view(-1, 1, IMG_SIZE, IMG_SIZE)
            images = Variable(images.cuda())
            # seg model
            output1 = model1(images)

            # para model
            output1 = torch.softmax(output1, dim=1)
            pred1 = output1
            output1 = torch.argmax(output1, dim=1).float()
            output1 = output1.view(bs, -1, IMG_SIZE, IMG_SIZE)
            output2 = model2(output1)

            # return outputs
            pred1 = torch.argmax(pred1, dim=1).float().cpu().data.numpy()
            pred_list.append(pred1)
            out_pca_list.append(output2[0, :].cpu().numpy())
    return np.array(pred_list), np.array(out_pca_list)


def main_function(input_path_test, save_path):
    seg_model_path = os.path.join( "./models", "seg_model_epoch_100.pkl")


    save_path_pred = os.path.join(save_path, "predicted_liver_masks")
    save_path_obj = os.path.join(save_path, "predicted_liver_shapes")
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path_pred):
        os.mkdir(save_path_pred)
    if not os.path.exists(save_path_obj):
        os.mkdir(save_path_obj)

    seed_value = 1234
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)



    inference = OnlyTest(input_path_test)
    inference_load = torch.utils.data.DataLoader(dataset=inference,
                                    num_workers=0, batch_size=1, shuffle=False)
   
    seg_model = transunet()
    print(f"Loading segmentation model weights from: {seg_model_path}")
    seg_model.load_state_dict(torch.load(seg_model_path))
    
    mask_out = Inference(seg_model, inference_load) 


    # 保存分割掩码
    data_name = inference.get_data_name()
    for i in range(mask_out.shape[0]):
        save_prediction_image(mask_out[i, :], data_name[i], save_path_pred)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Test the model.")
    arg_parser.add_argument(
        "--inference",
        "-t",
        dest="inference_directory",
        required=True,
        help="The inferencing data directory. "
    )
    arg_parser.add_argument(
        "--save",
        "-s",
        dest="save_directory",
        required=True,
        help="The save directory.",
    )

    args = arg_parser.parse_args()
    main_function(args.inference_directory, args.save_directory)