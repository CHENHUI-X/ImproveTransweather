from torchvision.models import convnext_tiny , vgg16
import os
from torchsummary.torchsummary import  summary
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
import datetime
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.switch_backend('agg')
# --- Define the perceptual loss network --- #
convnext = vgg16(pretrained=True).features

convnext = convnext.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in convnext.parameters():
    param.requires_grad = False
print(summary(convnext,input_size=(3,256,256)))
print(convnext)