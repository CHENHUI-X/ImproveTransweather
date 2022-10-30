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
from torch.utils.tensorboard import SummaryWriter

from train_data_functions import TrainData
from val_data_functions import ValData
# from utils import to_psnr, print_log, validation, adjust_learning_rate
from utils import PSNR , SSIM , validation
from torchvision.models import vgg16
from perceptual import LossNetwork


import numpy as np
import random
from tqdm import tqdm
# from transweather_model import SwingTransweather
from SwingTransweather_model import SwingTransweather
from  utils import Logger
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

# --- Parse hyper-parameters  --- #
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

# --- Load training data and validation/test data --- #
train_data_dir = './data/train/'
val_data_dir = './data/test/'
### The following file should be placed inside the directory "./data/train/"
labeled_name = 'allweather_subset_train.txt'
### The following files should be placed inside the directory "./data/test/"
# val_filename = 'val_list_rain800.txt'
# val_filename1 = 'raindroptesta.txt'
val_filename = 'allweather_subset_test.txt'

# --- Load training data and validation/test data --- #
train_data_loader = DataLoader(TrainData(crop_size, train_data_dir, labeled_name), batch_size=train_batch_size, shuffle=True)
val_data_loader = DataLoader(ValData(crop_size,val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)

# --- Gpu device --- #
if torch.cuda.is_available() :
    GPU = True
    device = torch.device('cuda:0')
    net = SwingTransweather().to(device)
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    if len(device_ids) > 0 :
        net = nn.DataParallel(net, device_ids=device_ids)
        print('-' * 50)
        print('Train model on multi GPU with multi threads ! \n')
else:
    GPU = False
    device = torch.device("cpu")
    net = SwingTransweather().to(device)

# --- Build optimizer --- #
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

# --- Build learning rate scheduler --- #
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,max_lr=0.01,
#     total_steps = num_epochs *( len(train_data_loader) // train_batch_size + 1)
# )#注意,这个OneCycleLR会导致无论你optim中的lr设置是啥,最后起作用的还是max_lr
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, T_0=300, T_mult=1, eta_min=0.001, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10 , gamma = 0.99)

# --- Previous PSNR and SSIM in testing --- #
psnr = PSNR()
ssim = SSIM()


if pretrained:
    if not os.path.exists('./{}/'.format(exp_name)):
        os.mkdir('./{}/'.format(exp_name))
    try:
        print('--- Loading model weight... ---')
        # original saved file with DataParallel
        state_dict = torch.load('./{}/best_model.pth'.format(exp_name))
        # state_dict = {
        #     "net": net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     "epoch": epoch,
        #     'scheduler': scheduler.state_dict()
        # }
        net.load_state_dict(state_dict['net'])
        print('--- Loading model successfully! ---')
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))
        old_val_loss, old_val_psnr, old_val_ssim = validation(
            net, val_data_loader, device=device,
            loss_network=loss_network, ssim=ssim, psnr=psnr, lambda_loss=lambda_loss
        )
        print(' old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))
        if isresume:
            optimizer.load_state_dict(state_dict['optimizer'])
            epoch_start = state_dict['epoch']  # Do not need + 1
            step_start = state_dict['step']
            scheduler.load_state_dict(state_dict['scheduler'])
            print(f" Let's continue training the model from epoch {epoch_start} !")

            assert args.time_str is None, 'If you want to resume, you must specify a timestamp !'

            # -----Logging------
            time_str = args.time_str
            step_logger = Logger(timestamp=time_str, filename=f'train-step.txt').initlog()
            epoch_logger = Logger(timestamp=time_str, filename=f'train-epoch.txt').initlog()
            val_logger = Logger(timestamp=time_str, filename=f'val-epoch.txt').initlog()

            writer = SummaryWriter(f'logs/tensorboard/{time_str}')  # tensorboard writer
        else:
            # 否则就是 有pretrain的model，但是只是作为比较，不是继续在此基础上进行训练，那么就需要新的logging
            curr_time = datetime.datetime.now()
            time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d_%H:%M:%S')
            step_logger = Logger(timestamp=time_str, filename=f'train-step.txt').initlog()
            epoch_logger = Logger(timestamp=time_str, filename=f'train-epoch.txt').initlog()
            val_logger = Logger(timestamp=time_str, filename=f'val-epoch.txt').initlog()

            writer = SummaryWriter(f'logs/tensorboard/{time_str}')  # tensorboard writer
        del state_dict
        torch.cuda.empty_cache()

    except:
        raise FileExistsError

else:  # 如果没有pretrained的model，那么就新建logging
    if not os.path.exists('./{}/'.format(exp_name)):
        os.mkdir('./{}/'.format(exp_name)) # checkpoint
    old_val_psnr, old_val_ssim = 0.0, 0.0
    print('-' * 50)
    print('Do not continue training an already pretrained model , '
          'if you need , please specify the parameter pretrained = 1 .\n'
          'Now will be train the model from scratch ! ')

    # -----Logging------
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d_%H:%M:%S')
    step_logger = Logger(timestamp=time_str, filename=f'train-step.txt').initlog()
    epoch_logger = Logger(timestamp=time_str, filename=f'train-epoch.txt').initlog()
    val_logger = Logger(timestamp=time_str, filename=f'val-epoch.txt').initlog()
    writer = SummaryWriter(f'logs/tensorboard/{time_str}')  # tensorboard writer
    # -------------------
    step_start = 0

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# download model to  C:\Users\CHENHUI/.cache\torch\hub\checkpoints\vgg16-397923af.pth
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model).to(device)
loss_network.eval()

# -----Some parameters------
step = 0
if step_start : step = step + step_start
lendata = len(train_data_loader)
num_epochs = num_epochs + epoch_start

# --------- train model ! ---------
for epoch in range(epoch_start,num_epochs): # default epoch_start = 0
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
        # print(input_image.shape)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        pred_image = net(input_image)

        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt)
        # ssim_loss = ssim.to_ssim_loss(pred_image,gt)
        loss = smooth_loss + lambda_loss * perceptual_loss
        # loss = ssim_loss + lambda_loss * perceptual_loss
        loss.backward()
        optimizer.step()
        step_psnr, step_ssim = \
            psnr.to_psnr(pred_image.detach(), gt.detach()), ssim.to_ssim(pred_image.detach(), gt.detach())

        # 只 master 进程做 logging，否则输出会很乱
        if args.local_rank == 0:
            loop.set_postfix(
                {'Epoch': f'{epoch + 1} / {num_epochs}', 'Step': f'{step + 1}', 'Steploss': '{:.4f}'.format(loss.item())})

            writer.add_scalar('TrainingStep/step-loss', loss.item(), step + 1)
            writer.add_scalar('TrainingStep/step-SSIM',step_psnr, step + 1)
            writer.add_scalar('TrainingStep/step-PSNR', step_ssim, step + 1)
            writer.add_scalar(
                'TrainingStep/lr', scheduler.get_last_lr()[0] , step + 1
            ) # logging lr for every step
            step_logger.writelines(
                f'Epoch: {epoch + 1} / {num_epochs} - Step: {step + 1}'
                + ' - steploss: {:.4f} - stepPSNR: {:.4f} - stepSSIM: {:.4f}\n'.format(
                    loss.item(),step_psnr,step_ssim
                )
            )

            if step % 50 == 0 : step_logger.flush()

        epoch_loss += loss.item()
        epoch_psnr += step_psnr
        epoch_ssim += step_ssim
        step = step + 1


    scheduler.step()  # Adjust learning rate for every epoch
    epoch_loss /= lendata
    epoch_psnr /= lendata
    epoch_ssim /= lendata
    # 只 master 进程做 logging，否则输出会很乱
    if args.local_rank == 0:
        print('----Epoch: [{}/{}], EpochAveLoss: {:.4f}, EpochAvePSNR: {:.4f}, EpochAveSSIM: {:.4f}----'
              .format(epoch + 1, num_epochs, epoch_loss ,epoch_psnr,epoch_ssim)
              )
        writer.add_scalar('TrainingEpoch/epoch-loss', epoch_loss, epoch + 1 )
        writer.add_scalar('TrainingEpoch/epoch-PSNR', epoch_psnr, epoch + 1 )
        writer.add_scalar('TrainingEpoch/epoch-SSIM', epoch_ssim, epoch + 1 )

        epoch_logger.writelines('Epoch [{}/{}], EpochAveLoss: {:.4f}, EpochAvePSNR: {:.4f} EpochAveSSIM: {:.4f}\n'.format(
            epoch + 1 , num_epochs, epoch_loss , epoch_psnr, epoch_ssim
        ))
        if ( epoch + 1 ) % 1 == 0 : epoch_logger.flush()

        # --- Save the  parameters --- #
        model_to_save = net.module if hasattr(net, "module") else net
        ## Take care of distributed/parallel training
        '''
        If you have an error about load model in " Missing key(s) in state_dict: " , 
        maybe you can  reference this url :
        https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/7
        '''
        checkpoint = {
            "net": model_to_save.cpu().state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch + 1,
            'step' : step + 1,
            'scheduler': scheduler.state_dict()
        }
        torch.save(checkpoint , './{}/latest_model.pth'.format(exp_name))

        # --- Use the evaluation model in testing --- #
        val_loss, val_psnr, val_ssim = validation(
            net, val_data_loader, device=device,
            loss_network=loss_network, ssim=ssim, psnr=psnr, lambda_loss=lambda_loss
        )
        writer.add_scalar('Validation/loss', val_loss, epoch + 1 )
        writer.add_scalar('Validation/PSNR', val_psnr, epoch + 1 )
        writer.add_scalar('Validation/SSIM', val_ssim, epoch + 1 )
        #  logging
        val_logger.writelines('Epoch [{}/{}], ValEpochAveLoss: {:.4f}, ValEpochAvePSNR: {:.4f} ValEpochAveSSIM: {:.4f}\n'.format(
            epoch + 1 , num_epochs, val_loss, val_psnr, val_ssim
        ))
        # val_psnr1, val_ssim1 = validation(net, val_data_loader1, device, exp_name)
        # val_psnr2, val_ssim2 = validation(net, val_data_loader2, device, exp_name)

    one_epoch_time = time.time() - start_time
    # print("Rain 800")
    # print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, exp_name)
    # print("Rain Drop")
    # print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr1, val_ssim1, exp_name)
    # print("Test1")
    # print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr2, val_ssim2, exp_name)

    # --- update the network weight --- #

    if val_psnr >= old_val_psnr:
        model_to_save = net.module if hasattr(net, "module") else net
        ## Take care of distributed/parallel training
        '''
        If you have an error about load model in " Missing key(s) in state_dict: " , 
        maybe you can  reference this url :
        https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/7
        '''
        checkpoint = {
            "net": model_to_save.cpu().state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch + 1,
            'step': step + 1,
            'scheduler': scheduler.state_dict()
        }
        torch.save(checkpoint, './{}/best_model.pth'.format(exp_name))
        print('Update the best model !')
        old_val_psnr = val_psnr

    # Note that we find the best model based on validating with raindrop data.

step_logger.close()
epoch_logger.close()


