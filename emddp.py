
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
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.train_data_functions import TrainData
from utils.val_data_functions import ValData
# from utils import to_psnr, print_log, validation, adjust_learning_rate
from utils.utils import PSNR , SSIM , validation
from torchvision.models import vgg16
from models.perceptual import LossNetwork

import numpy as np
import random
from tqdm import tqdm
# from transweather_model import SwingTransweather
from models.SwingTransweather_model import SwingTransweather
from utils.utils import Logger

# ================================ Parse hyper-parameters  ================================= #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default = 2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=1, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default = 0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str,default='checkpoint')
parser.add_argument('-seed', help='set random seed', default=666, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default= 2 , type=int)
parser.add_argument("--pretrained", help='whether have a pretrained model', type=int,default=0)
parser.add_argument("--isresume", help='if you have a pretrained model , you can continue train it ', type=int,default=0)
parser.add_argument("--time_str", help='where the logging file and tensorboard you want continue', type=str,default=None)

args = parser.parse_args()
learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
pretrained = args.pretrained
isresume = args.isresume
time_str = args.time_str

# ==============================================================================

def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)
    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
# ================================ Set seed  ================================= #
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

# =============  Load training data and validation/test data  ============ #

train_data_dir = './data/train/'
val_data_dir = './data/test/'
### The following file should be placed inside the directory "./data/train/"
labeled_name = 'allweather_subset_train.txt'
### The following files should be placed inside the directory "./data/test/"
# val_filename = 'val_list_rain800.txt'
# val_filename1 = 'raindroptesta.txt'
val_filename = 'allweather_subset_test.txt'

# ================== Define the model nad  loss network  ===================== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SwingTransweather().to(device) # GPU or CPU
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# download model to  C:\Users\CHENHUI/.cache\torch\hub\checkpoints\vgg16-397923af.pth
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg_model).to(device)
loss_network.eval()

# =====================================  DDP model setup   ==================================== #
init_distributed()

net = net.cuda()
loss_network = loss_network.cuda()
# Convert BatchNorm to SyncBatchNorm.
net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

local_rank = int(os.environ['LOCAL_RANK'])
net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
loss_network = nn.parallel.DistributedDataParallel(loss_network, device_ids=[local_rank])

trainset = TrainData(crop_size, train_data_dir, labeled_name)
testset = ValData(crop_size,val_data_dir,val_filename)

train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                            sampler = train_sampler, num_workers = 8,)
test_sampler =  DistributedSampler(dataset=testset, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size,
                                            shuffle = False, sampler = test_sampler, num_workers=8)


# -----Some parameters------
step = 0
# if step_start : step = step + step_start
lendata = len(train_data_loader)
num_epochs = num_epochs + epoch_start

# --------- train model ! ---------
for epoch in range(epoch_start,num_epochs): # default epoch_start = 0

    # adjust_learning_rate(optimizer, epoch)
    loop = tqdm(train_data_loader, desc="Progress bar : ")

    train_sampler.set_epoch(epoch) # TODO : why this ?
    # 如果不调用set_epoch, 那么每个epoch都会使用第1个epoch的indices, 因为epoch数没有改变, 随机种子seed一直是初始状态
#-------------------------------------------------------------------------------------------------------------
    for batch_id, train_data in enumerate(loop):

        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        # print(input_image.shape)
        gt = gt.to(device)
        print( f'logging on {local_rank}')
        # --- Forward + Backward + Optimize --- #
        net.to(device).train()
        pred_image = net(input_image)