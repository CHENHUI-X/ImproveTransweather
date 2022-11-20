
# =========================================================
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

from scripts.train_data_functions import TrainData
from scripts.val_data_functions import ValData

from scripts.utils import PSNR, SSIM, validation_gpu
from torchvision.models import vgg16
from models.perceptual import LossNetwork

import numpy as np
import random
from tqdm import tqdm
# from transweather_model import SwingTransweather
from models.transweather_model import Transweather
from scripts.utils import Logger
plt.switch_backend('agg')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=64, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=64, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=200, type=int)

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs


#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, lambda_loss))


train_data_dir = './data/train/'
val_data_dir = './data/test/'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = Transweather()


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]

vgg_model = vgg_model.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

# --- Load the network weight --- #
if not os.path.exists('./{}/'.format(exp_name)):
    os.mkdir('./{}/'.format(exp_name))

try:
    net.load_state_dict(torch.load('./{}/best'.format(exp_name)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')


# pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# print("Total_params: {}".format(pytorch_total_params))
loss_network = LossNetwork(vgg_model)
loss_network.eval()

# --- Load training data and validation/test data --- #

### The following file should be placed inside the directory "./data/train/"

labeled_name = 'train.txt'

### The following files should be placed inside the directory "./data/test/"

# val_filename = 'val_list_rain800.txt'
# val_filename1 = 'raindroptesta.txt'
val_filename = 'test.txt'

# --- Load training data and validation/test data --- #
train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,labeled_name), batch_size=train_batch_size, shuffle=True, num_workers=4)

## Uncomment the other validation data loader to keep an eye on performance 
## but note that validating while training significantly increases the train time 

val_data_loader = DataLoader(ValData(crop_size,val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)
# val_data_loader1 = DataLoader(ValData(val_data_dir,val_filename1), batch_size=val_batch_size, shuffle=False, num_workers=8)
# val_data_loader2 = DataLoader(ValData(val_data_dir,val_filename2), batch_size=val_batch_size, shuffle=False, num_workers=8)

psnr = PSNR()
ssim = SSIM()
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'num of parameter is {pytorch_total_params}')

step = 0
lendata = len(train_data_loader)
for epoch in range(epoch_start,num_epochs):
    psnr_list = []
    start_time = time.time()
    epoch_loss = 0
    epoch_psnr = 0
    epoch_ssim = 0
    # adjust_learning_rate(optimizer, epoch)
    loop = tqdm(train_data_loader, desc="Progress bar : ")
#-------------------------------------------------------------------------------------------------------------
    for batch_id, train_data in enumerate(loop):

        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        pred_image = net(input_image)

        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt)

        loss = smooth_loss + lambda_loss * perceptual_loss

        loss.backward()
        optimizer.step()

        step_psnr, step_ssim = \
            psnr.to_psnr(pred_image.detach(), gt.detach()), ssim.to_ssim(pred_image.detach(), gt.detach())

        loop.set_postfix(
            {'Epoch': f'{epoch + 1} / {num_epochs}', 'Step': f'{step + 1}', 'Steploss': '{:.4f}'.format(loss.item())})
        epoch_loss += loss.item()
        epoch_psnr += step_psnr
        epoch_ssim += step_ssim
        step = step + 1

    epoch_loss /= lendata
    epoch_psnr /= lendata
    epoch_ssim /= lendata

    print('----Epoch: [{}/{}], EpochAveLoss: {:.4f}, EpochAvePSNR: {:.4f}, EpochAveSSIM: {:.4f}----'
          .format(epoch + 1, num_epochs, epoch_loss, epoch_psnr, epoch_ssim)
          )
    if (epoch + 1) % 5 == 0:
        # --- Use the evaluation model in testing --- #
        val_loss, val_psnr, val_ssim = validation_gpu(net, val_data_loader, device=device, loss_network=loss_network,
                                                      ssim=ssim, psnr=psnr, lambda_loss=lambda_loss, )
        print('--- ValLoss : {:.4f} , Valpsnr : {:.4f} , Valssim : {:.4f}'.format(val_loss, val_psnr, val_ssim))

# if __name__ == '__main__':
#     training()
