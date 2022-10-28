
import torch
import argparse

from torch.utils.data import DataLoader

from val_data_functions import ValData
from utils import PSNR , SSIM , validation ,load_best_model , PollExecutorSaveImg ,save_img
import numpy as np
import random
from tqdm import tqdm
# from transweather_model import SwingTransweather
from SwingTransweather_model import SwingTransweather
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for visualization')
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=32, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str,default='./checkpoint')
parser.add_argument('-seed', help='set random seed', default=666, type=int)


args = parser.parse_args()

crop_size = args.crop_size
val_batch_size = args.val_batch_size
exp_name = args.exp_name


#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

# --- Load training data and validation/test data --- #
train_data_dir = './data/train/'
val_data_dir = './data/test/'
### The following file should be placed inside the directory "./data/train/"
labeled_name = 'allweather_subset_train.txt'
### The following files should be placed inside the directory "./data/test/"
# val_filename = 'val_list_rain800.txt'
# val_filename1 = 'raindroptesta.txt'
val_filename = 'allweather_subset_test.txt'

# --- Load validation/test data --- #
val_data_loader = DataLoader(ValData(crop_size,val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=4)

# --- Gpu device --- #
if torch.cuda.is_available():
    GPU = True
    device = torch.device("cuda:0")
    device_ids = [Id for Id in range(torch.cuda.device_count())]
else:
    GPU = False
    device = torch.device("cpu")

net = SwingTransweather().to(device)
net = load_best_model(net,exp_name=exp_name ).eval()# GPU or CPU

# -----Some parameters------
total_step = 0
step = 0
lendata = len(val_data_loader)
psnr = PSNR()
ssim = SSIM()

loop = tqdm(val_data_loader, desc="Progress bar : ")
with torch.no_grad():
    for batch_id, val_data in enumerate(loop):
        input_image, gt, img_names = val_data
        input_image = input_image.to(device)
        # print(input_image.shape)
        gt = gt.to(device)
        # --- Forward + Backward + Optimize --- #
        pred_image = net(input_image).to('cpu')
        PollExecutorSaveImg(
             iamge_names = img_names  , images = pred_image*255 , n_files = len(input_image)
        )



