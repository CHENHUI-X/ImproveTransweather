
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
import datetime
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.train_data_functions import TrainData
from utils.val_data_functions import ValData

from utils.utils import PSNR , SSIM , validation
from torchvision.models import vgg16 ,convnext_base


import numpy as np
import random
from tqdm import tqdm
# from transweather_model import SwingTransweather
from models.SwingTransweather_model import SwingTransweather
from utils.utils import Logger
from torchsummary import summary

# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, convnext_model):
        super(LossNetwork, self).__init__()
        self.convnext_layers = convnext_model
        self.layer_name_mapping = [1,3,5,7]
        self.mse_loss = F.mse_loss

    def output_features(self, x):
        output = []
        for name, module in self.convnext_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)

        return output

    def forward(self,  sw_fm , gt):
        # Denoised image ( B,3,256,256 )
        # Ground True ( B,3,256,256 )
        loss = []
        conv_fm = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(sw_fm, conv_fm):
            loss.append(self.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)


conv = convnext_base(pretrained=True).features
# download model to  C:\Users\CHENHUI/.cache\torch\hub\checkpoints\vgg16-397923af.pth
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in conv.parameters():
    param.requires_grad = False
loss_network = LossNetwork(conv)
loss_network.eval()
print(summary(conv,input_size=(3,256,256)))

for name, module in conv.features._modules.items():
    print(name,module)


'''

7 stochastic_depth           [-1, 1024, 8, 8]    《=》 convd32x


5 stochastic_depth          [-1, 512, 16, 16]     《=》   convd16x  
 
 
3 stochastic_depth          [-1, 256, 32, 32]     《=》   convd8x   
 
1 stochastic_depth          [-1, 128, 64, 64]     《=》   convd4x   : 暂时不要 


'''

# 两个地方  ， 训练 ， 一个验证