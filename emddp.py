import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import datetime
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shutil

plt.switch_backend('agg')

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.train_data_functions import TrainData
from utils.val_data_functions import ValData
# from utils import to_psnr, print_log, validation, adjust_learning_rate
from utils.utils import PSNR, SSIM, validation
from torchvision.models import vgg16, convnext_base
from models.perceptual import LossNetwork

import numpy as np
import random
from tqdm import tqdm
# from transweather_model import SwingTransweather
from models.SwingTransweather_model import SwingTransweather
from utils.utils import Logger, init_distributed, is_main_process, torch_distributed_zero_first

from apex import amp

# ================================ Parse hyper-parameters  ================================= #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('--learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('--crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('--train_batch_size', help='Set the training batch size', default=16, type=int)
parser.add_argument('--epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('--step_start', help='Starting step number of the resume training', default=0, type=int)

parser.add_argument('--lambda_loss', help='Set the lambda in loss function', default=0.05, type=float)
parser.add_argument('--val_batch_size', help='Set the validation/test batch size', default=32, type=int)
parser.add_argument('--exp_name', help='directory for saving the networks of the experiment', type=str,
                    default='checkpoint')
parser.add_argument('--seed', help='set random seed', default=666, type=int)
parser.add_argument('--num_epochs', help='number of epochs', default=2, type=int)
parser.add_argument('--isapex', help='Automatic Mixed-Precision', default=0, type=int)
parser.add_argument("--pretrained", help='whether have a pretrained model', type=int, default=0)
parser.add_argument("--isresume", help='if you have a pretrained model , you can continue train it ', type=int,
                    default=0)
parser.add_argument("--time_str", help='where the logging file and tensorboard you want continue', type=str,
                    default=None)
parser.add_argument("--local_rank", help='where the logging file and tensorboard you want continue', type=str,
                    default=None)

args = parser.parse_args()
learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
step_start = args.step_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
pretrained = args.pretrained
isresume = args.isresume
time_str = args.time_str
isapex = args.isapex

local_rank = int(os.environ['LOCAL_RANK'])

# ==============================================================================


init_distributed()

# ================================ Set seed  ================================= #
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

if is_main_process(local_rank):
    print('Seed:\t{}'.format(seed))
    print('--- Hyper-parameters for training ---')
    print(
        'learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}'.format(
            learning_rate,
            crop_size,
            train_batch_size,
            val_batch_size,
            lambda_loss))

# =============  Load training data and validation/test data  ============ #

train_data_dir = './data/train/'
val_data_dir = './data/test/'
### The following file should be placed inside the directory "./data/train/"
# labeled_name = 'allweather_subset_train.txt'
labeled_name = 'train.txt'
### The following files should be placed inside the directory "./data/test/"
# val_filename = 'val_list_rain800.txt'
# val_filename1 = 'raindroptesta.txt'
# val_filename = 'allweather_subset_test.txt'
val_filename = 'test.txt'
# ================== Define the model nad  loss network  ===================== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SwingTransweather().to(device)  # GPU or CPU

with torch_distributed_zero_first(local_rank=local_rank):
    # vgg_model = vgg16(pretrained=True).features[:16]
    # vgg_model = vgg_model.to(device)
    # # download model to  C:\Users\CHENHUI/.cache\torch\hub\checkpoints\vgg16-397923af.pth
    # # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
    # for param in vgg_model.parameters():
    #     param.requires_grad = False
    # loss_network = LossNetwork(vgg_model).to(device)
    # loss_network.eval()

    conv = convnext_base(pretrained=True).features
    # download model to  C:\Users\CHENHUI/.cache\torch\hub\checkpoints\vgg16-397923af.pth
    # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
    for param in conv.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(conv).to(device)
    loss_network.eval()

# ==========================  Build optimizer  ========================= #
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

# ================== Build learning rate scheduler  ===================== #
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,max_lr=0.01,
#     total_steps = num_epochs *( len(train_data_loader) // train_batch_size + 1)
# )#注意,这个OneCycleLR会导致无论你optim中的lr设置是啥,最后起作用的还是max_lr
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, T_0=300, T_mult=1, eta_min=0.001, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

# ================== Previous PSNR and SSIM in testing  ===================== #
psnr = PSNR()
ssim = SSIM()

# ================== Molde checkpoint  ===================== #
if not os.path.exists('./{}/'.format(exp_name)):
    os.mkdir('./{}/'.format(exp_name))

# ================== Load model or resume from checkpoint  ===================== #

with torch_distributed_zero_first(local_rank):
    if pretrained:
        try:
            print('--- Loading model weight... ---')
            # original saved file with DataParallel
            state_dict = torch.load('./{}/best_model.pth'.format(exp_name), map_location=device)

            # state_dict = {
            #     "net": net.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     "epoch": epoch,
            #     'scheduler': scheduler.state_dict()
            # }

            if is_main_process(local_rank):  # 只有主进程需要读取已经有的模型进行psnr的初始值计算
                net.load_state_dict(state_dict['net'])
                print('--- Loading model successfully! ---')
                pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                print("Total_params: {}".format(pytorch_total_params))

                val_data_loader = DataLoader(ValData(crop_size, val_data_dir, val_filename), batch_size=val_batch_size,
                                             shuffle=False, num_workers=8)
                old_val_loss, old_val_psnr, old_val_ssim = validation(
                    net, val_data_loader, device=device,
                    loss_network=loss_network, ssim=ssim, psnr=psnr, lambda_loss=lambda_loss
                )
                del val_data_loader
                print(' old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

            if isresume:
                optimizer.load_state_dict(state_dict['optimizer'])
                epoch_start = state_dict['epoch']  # Do not need + 1
                step_start = state_dict['step']
                scheduler.load_state_dict(state_dict['scheduler'])

                if is_main_process(local_rank):  # 只有master进程做logging
                    print(f" Let's continue training the model from epoch {epoch_start} !")
                    assert args.time_str is not None, 'If you want to resume, you must specify a timestamp !'
                    # -----Logging-----
                    time_str = args.time_str
                    step_logger = Logger(timestamp=time_str, filename=f'train-step.txt').initlog()
                    epoch_logger = Logger(timestamp=time_str, filename=f'train-epoch.txt').initlog()
                    val_logger = Logger(timestamp=time_str, filename=f'val-epoch.txt').initlog()
                    writer = SummaryWriter(f'logs/tensorboard/{time_str}')  # tensorboard writer

            del state_dict
            torch.cuda.empty_cache()

        except:
            raise FileNotFoundError
        finally:
            ...
    else:
        # 如果没有pretrained的model，那么就新建logging

        if is_main_process(local_rank):
            old_val_psnr, old_val_ssim = 0.0, 0.0
            print('-' * 50)
            print('Do not continue training an already pretrained model , '
                  'if you need , please specify the parameter ** pretrained | isresume | time_str ** .\n'
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

# ================  Amp, short for Automatic Mixed-Precision ================
if isapex:
    print(f" Let's using  Automatic Mixed-Precision to speed traing !")
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

# =====================================  DDP model setup   ==================================== #

net = net.cuda()
loss_network = loss_network.cuda()
# Convert BatchNorm to SyncBatchNorm.
net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)
# loss_network = nn.parallel.DistributedDataParallel(loss_network, device_ids=[local_rank])

trainset = TrainData(crop_size, train_data_dir, labeled_name)
testset = ValData(crop_size, val_data_dir, val_filename)

train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                sampler=train_sampler, num_workers=4)
test_sampler = DistributedSampler(dataset=testset, shuffle=False)
val_data_loader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size,
                                              sampler=test_sampler, num_workers=4)

# -----Some parameters------
step = 0
step = step + step_start
lendata = len(train_data_loader)
num_epochs = num_epochs + epoch_start
# --------- train model ! ---------
for epoch in range(epoch_start, num_epochs):  # default epoch_start = 0
    start_time = time.time()
    epoch_loss = 0
    epoch_psnr = 0
    epoch_ssim = 0
    # adjust_learning_rate(optimizer, epoch)
    loop = train_data_loader
    if is_main_process(local_rank):
        loop = tqdm(train_data_loader, desc="Progress bar : ")

    train_sampler.set_epoch(epoch)  # TODO : why this ?
    test_sampler.set_epoch(epoch)
    # 如果不调用set_epoch, 那么每个epoch都会使用第1个epoch的indices, 因为epoch数没有改变, 随机种子seed一直是初始状态
    # -------------------------------------------------------------------------------------------------------------
    for id, train_data in enumerate(loop):

        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        # print(input_image.shape)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.to(device).train()

        pred_image, sw_fm = net(input_image)

        pred_image.to(device)
        sw_fm = [i.to(device) for i in sw_fm]

        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(sw_fm, gt)
        # ssim_loss = ssim.to_ssim_loss(pred_image,gt)
        loss = smooth_loss + lambda_loss * perceptual_loss
        # loss = ssim_loss + lambda_loss * perceptual_loss
        if isapex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        else:
            loss.backward()
            optimizer.step()

        if is_main_process(local_rank):
            loop.set_postfix(
                {'Epoch': f'{epoch + 1} / {num_epochs}', 'Step': f'{step + 1}',
                 'Steploss': '{:.4f}'.format(loss.item())})

            step_psnr, step_ssim = \
                psnr.to_psnr(pred_image.detach(), gt.detach()), ssim.to_ssim(pred_image.detach(), gt.detach())
            writer.add_scalar('TrainingStep/step-loss', loss.item(), step + 1)
            writer.add_scalar('TrainingStep/step-SSIM', step_psnr, step + 1)
            writer.add_scalar('TrainingStep/step-PSNR', step_ssim, step + 1)
            writer.add_scalar(
                'TrainingStep/lr', scheduler.get_last_lr()[0], step + 1
            )  # logging lr for every step
            step_logger.writelines(
                f'Epoch: {epoch + 1} / {num_epochs} - Step: {step + 1}'
                + ' - steploss: {:.4f} - stepPSNR: {:.4f} - stepSSIM: {:.4f}\n'.format(
                    loss.item(), step_psnr, step_ssim
                )
            )
            if step % 50 == 0: step_logger.flush()
            epoch_loss += loss.item()
            epoch_psnr += step_psnr
            epoch_ssim += step_ssim
            step = step + 1

    scheduler.step()  # Adjust learning rate for every epoch
    # with torch_distributed_zero_first(local_rank):
    if is_main_process(local_rank):
        epoch_loss /= lendata
        epoch_psnr /= lendata
        epoch_ssim /= lendata
        print('----Epoch: [{}/{}], EpochAveLoss: {:.4f}, EpochAvePSNR: {:.4f}, EpochAveSSIM: {:.4f}----'
              .format(epoch + 1, num_epochs, epoch_loss, epoch_psnr, epoch_ssim)
              )
        writer.add_scalar('TrainingEpoch/epoch-loss', epoch_loss, epoch + 1)
        writer.add_scalar('TrainingEpoch/epoch-PSNR', epoch_psnr, epoch + 1)
        writer.add_scalar('TrainingEpoch/epoch-SSIM', epoch_ssim, epoch + 1)

        epoch_logger.writelines(
            'Epoch [{}/{}], EpochAveLoss: {:.4f}, EpochAvePSNR: {:.4f} EpochAveSSIM: {:.4f}\n'.format(
                epoch + 1, num_epochs, epoch_loss, epoch_psnr, epoch_ssim
            ))
        epoch_logger.flush()

        # --- Save the  parameters --- #
        model_to_save = net.module if hasattr(net, "module") else net
        ## Take care of distributed/parallel training
        '''
        If you have an error about load model in " Missing key(s) in state_dict: " ,
        maybe you can  reference this url :
        https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/7
        '''
        checkpoint = {
            "net": model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch + 1,
            'step': step + 1,
            'scheduler': scheduler.state_dict()
        }
        torch.save(checkpoint, './{}/latest_model.pth'.format(exp_name))

        # --- Use the evaluation model in testing  for every 10 epoch--- #
        if (epoch + 1) % 10 == 0:
            val_loss, val_psnr, val_ssim = validation(
                net, val_data_loader, device=device,
                loss_network=loss_network, ssim=ssim, psnr=psnr, lambda_loss=lambda_loss)
            writer.add_scalar('Validation/loss', val_loss, epoch + 1)
            writer.add_scalar('Validation/PSNR', val_psnr, epoch + 1)
            writer.add_scalar('Validation/SSIM', val_ssim, epoch + 1)
            # logging
            val_logger.writelines(
                'Epoch [{}/{}], ValEpochAveLoss: {:.4f}, ValEpochAvePSNR: {:.4f} ValEpochAveSSIM: {:.4f}\n'.format(
                    epoch + 1, num_epochs, val_loss, val_psnr, val_ssim
                ))
            val_logger.flush()

            one_epoch_time = time.time() - start_time

            if val_psnr >= old_val_psnr:
                torch.save(checkpoint, './{}/best_model.pth'.format(exp_name))
                # shutil.copy2(
                #     './{}/latest_model.pth'.format(exp_name),
                #     './{}/best_model.pth'.format(exp_name),
                # )
                print('Update the best model !')
                old_val_psnr = val_psnr

if is_main_process(local_rank):
    step_logger.close()
    epoch_logger.close()
    print('=================================== END TRAIN ===================================')


