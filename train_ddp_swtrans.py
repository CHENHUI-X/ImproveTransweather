import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
import datetime
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from scripts.train_data_functions import TrainData
from scripts.val_data_functions import ValData
# from utils import to_psnr, print_log, validation, adjust_learning_rate
from scripts.utils import PSNR, SSIM, validation_ddp, validation_gpu, synthetic_loss
from torchvision.models import convnext_base, convnext_tiny, vgg16
from torchvision.models import convnext_tiny
from models.perceptual import LossNetwork

import numpy as np
import random
from tqdm import tqdm

from models.SwingTransweather_model import SwingTransweather
from models.FocalResWeather import SwingTransweather
from models.acc_unet import ACC_UNet

from scripts.utils import Logger, init_distributed, is_main_process, torch_distributed_zero_first

# ================================ Parse hyper-parameters  ================================= #
parser = argparse.ArgumentParser( description = 'Hyper-parameters for network' )
parser.add_argument( '--learning_rate', help = 'Set the learning rate', default = 2e-4, type = float )
parser.add_argument( '--crop_size', help = 'Set the crop_size', default = [256, 256], nargs = '+', type = int )
parser.add_argument( '--train_batch_size', help = 'Set the training batch size', default = 32, type = int )
parser.add_argument( '--epoch_start', help = 'Starting epoch number of the training', default = 0, type = int )
parser.add_argument( '--step_start', help = 'Starting step number of the resume training', default = 0, type = int )

parser.add_argument( '--alpha_loss', help = 'Set the alpha in loss function for perceptual_loss', default = 0.04, type = float)
parser.add_argument( '--beta_loss', help = 'Set the beta in loss function for ssim_loss', default = 0.01, type = float )
parser.add_argument( '--gamma_loss', help = 'Set the gamma in loss function for identity_loss', default = 0.01, type = float )

parser.add_argument( '--val_batch_size', help = 'Set the validation/test batch size', default = 32, type = int )
parser.add_argument( '--exp_name', help = 'directory for saving the networks of the experiment', type = str,default = 'checkpoint')
parser.add_argument( '--seed', help = 'set random seed', default = 666, type = int )
parser.add_argument( '--num_epochs', help = 'number of epochs', default = 2, type = int )
parser.add_argument( '--isapex', help = 'Automatic Mixed-Precision', default = 1, type = int )
parser.add_argument( "--pretrained", help = 'whether have a pretrained model', type = int, default = 0 )
parser.add_argument(
    "--isresume", help = 'if you have a pretrained model , you can continue train it ', type = int,
    default = 0
)
parser.add_argument(
    "--time_str", help = 'where the logging file and tensorboard you want continue', type = str,
    default = None
)
parser.add_argument(
    "--local_rank", help = 'For DDP training model ', type = str,
    default = None
)

parser.add_argument( "--step_size", help = 'step size of step lr scheduler', type = int, default = 10 )
parser.add_argument( "--step_gamma", help = 'gamma of step lr scheduler', type = float, default = 0.999 )

args = parser.parse_args()
learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
step_start = args.step_start
alpha = args.alpha_loss
beta = args.beta_loss
gamma = args.gamma_loss

val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
pretrained = args.pretrained
isresume = args.isresume
time_str = args.time_str
isapex = args.isapex
step_size = args.step_size
step_gamma = args.step_gamma

local_rank = int( os.environ['LOCAL_RANK'] )
world_size = int( os.environ['WORLD_SIZE'] )


# ================ Initialize the distribution environment ==============
init_distributed()

# ================================ Set seed  ================================= #
seed = args.seed
if seed is not None:
    np.random.seed( seed )
    torch.manual_seed( seed )
    torch.cuda.manual_seed_all( seed )
    random.seed( seed )

# =============  Load training data and validation/test data  ============ #

train_data_dir = './data/train/'
val_data_dir = './data/test/'
### The following file should be placed inside the directory "./data/train/"
labeled_name = 'allweather_subset_train.txt'
# labeled_name = 'train.txt'
### The following files should be placed inside the directory "./data/test/"
# val_filename = 'val_list_rain800.txt'
# val_filename1 = 'raindroptesta.txt'
val_filename = 'allweather_subset_test.txt'
# val_filename = 'test.txt'

# ================== Define the model nad  loss network  ===================== #

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
net = ACC_UNet(n_channels = 3 , n_classes = 3).to( device )  # GPU or CPU
# net = torch.compile( net )

with torch_distributed_zero_first( local_rank = local_rank ):
    # vgg_model = vgg16(pretrained=True).features[:16]
    # vgg_model = vgg_model.to(device)
    # # download model to  C:\Users\CHENHUI/.cache\torch\hub\checkpoints\vgg16-397923af.pth
    # # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
    # for param in vgg_model.parameters():
    #     param.requires_grad = False
    # perceptual_loss_network = LossNetwork(vgg_model).to(device)
    # perceptual_loss_network.eval()

    conv = convnext_tiny( pretrained = True ).features
    # download model to  C:\Users\CHENHUI/.cache\torch\hub\checkpoints\vgg16-397923af.pth
    # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
    for param in conv.parameters():
        param.requires_grad = False

    perceptual_loss_network = LossNetwork( conv ).to( device )
    perceptual_loss_network.eval()

# ==========================  Build optimizer  ========================= #
optimizer = torch.optim.AdamW( net.parameters(), lr = learning_rate )

# ================== Build learning rate scheduler  ===================== #
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,max_lr=0.01,
#     total_steps = num_epochs *( len(train_data_loader) // train_batch_size + 1)
# )#注意,这个OneCycleLR会导致无论你optim中的lr设置是啥,最后起作用的还是max_lr
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, T_0=300, T_mult=1, eta_min=0.001, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR( optimizer, step_size = step_size, gamma = step_gamma )

# ================== Previous PSNR and SSIM in testing  ===================== #
psnr = PSNR()
ssim = SSIM()

# ================  Amp, short for Automatic Mixed-Precision ================
if isapex:
    use_amp = True
    scaler = torch.cuda.amp.GradScaler( enabled = use_amp )
    if is_main_process( local_rank ):
        print( f"--- Let's using  Automatic Mixed-Precision to speed training !" )

# ================== Molde checkpoint  ===================== #
if not os.path.exists( './{}/'.format( exp_name ) ):
    os.mkdir( './{}/'.format( exp_name ) )

# ================== Load model or resume from checkpoint  ===================== #

with torch_distributed_zero_first( local_rank ):
    if pretrained:
        assert time_str is not None, 'Must specify a model timestamp'
        try:
            print( f'--- GPU:{local_rank} Loading best model weight...' )
            # original saved file with DataParallel
            best_state_dict = torch.load( './{}/{}/best_model.pth'.format( exp_name ,time_str ), map_location = device )

            # state_dict = {
            #     "net": net.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     "epoch": epoch,
            #     'scheduler': scheduler.state_dict()
            # }
            print( f'--- GPU:{local_rank} Loading model successfully!' )
            if is_main_process( local_rank ):  # only master process need calculate old psnr
                net.load_state_dict( best_state_dict['net'] )
                pytorch_total_params = sum( p.numel() for p in net.parameters() if p.requires_grad )
                print( "--- Total_params: {}".format( pytorch_total_params ) )

                val_data_loader = DataLoader(
                    ValData( crop_size, val_data_dir, val_filename ), batch_size = val_batch_size,
                    shuffle = False, num_workers = 8
                )
                old_val_loss, old_val_psnr, old_val_ssim = validation_gpu(
                    net, val_data_loader, device = device,
                    perceptual_loss_network = perceptual_loss_network,
                    ssim = ssim, psnr = psnr,
                    alpha = alpha, beta = beta, gama = gamma
                )
                del val_data_loader  # only master processing
            del best_state_dict  # for all processing

            if isresume:

                # Need load the latest trained model for continue training .
                last_state_dict = torch.load( './{}/{}/latest_model.pth'.format( exp_name ,time_str), map_location = device )

                net.load_state_dict( last_state_dict['net'] )
                optimizer.load_state_dict( last_state_dict['optimizer'] )
                epoch_start = last_state_dict['epoch']  # Do not need + 1
                step_start = last_state_dict['step']
                scheduler.load_state_dict( last_state_dict['scheduler'] )
                if isapex:
                    scaler.load_state_dict( last_state_dict['amp_scaler'] )

                del last_state_dict

                if is_main_process( local_rank ):  # master process logging
                    print( f"--- Let's continue training the model from epoch {epoch_start} !" )
                    # -----Logging-----
                    time_str = args.time_str
                    step_logger = Logger( timestamp = time_str, filename = f'train-step.txt' ).initlog()
                    epoch_logger = Logger( timestamp = time_str, filename = f'train-epoch.txt' ).initlog()
                    val_logger = Logger( timestamp = time_str, filename = f'val-epoch.txt' ).initlog()
                    writer = SummaryWriter( f'logs/tensorboard/{time_str}' )  # tensorboard writer
            else:
                # if not resume but have pretrained , it means we need train a model from scratch
                # the older model just used to compared , then we also need crate new logging

                if is_main_process( local_rank ):  # 只有master进程做logging
                    curr_time = datetime.datetime.now()
                    time_str = datetime.datetime.strftime( curr_time, r'%Y_%m_%d_%H_%M_%S' )
                    step_logger = Logger( timestamp = time_str, filename = f'train-step.txt' ).initlog()
                    epoch_logger = Logger( timestamp = time_str, filename = f'train-epoch.txt' ).initlog()
                    val_logger = Logger( timestamp = time_str, filename = f'val-epoch.txt' ).initlog()

                    writer = SummaryWriter( f'logs/tensorboard/{time_str}' )  # tensorboard writer

            torch.cuda.empty_cache()
        except Exception:
            raise FileNotFoundError

    else:
        # if we do not have a pretrained model , then we create a new logger
        if is_main_process( local_rank ):
            old_val_psnr, old_val_ssim = 0.0, 0.0
            print( '=' * 50 )
            print(
                '--- Now will be training the model from scratch ! '
                'if you need to continue train an already pretrained model,'
                ' please specify the parameter ** pretrained | isresume | time_str ** .\n'
            )
            # -----Logging------
            curr_time = datetime.datetime.now()
            time_str = datetime.datetime.strftime( curr_time, '%Y_%m_%d_%H_%M_%S' )
            step_logger = Logger( timestamp = time_str, filename = f'train-step.txt' ).initlog()
            epoch_logger = Logger( timestamp = time_str, filename = f'train-epoch.txt' ).initlog()
            val_logger = Logger( timestamp = time_str, filename = f'val-epoch.txt' ).initlog()
            writer = SummaryWriter( f'logs/tensorboard/{time_str}' )  # tensorboard writer
            # -------------------
            step_start = 0

# =====================================  DDP model setup   ==================================== #
net = net.cuda()
perceptual_loss_network = perceptual_loss_network.cuda()
# Convert BatchNorm to SyncBatchNorm.
net = nn.SyncBatchNorm.convert_sync_batchnorm( net )

# the parameter "broadcast_buffers" is set False for if you only use one GPU for DDP ,else it should be True
net = nn.parallel.DistributedDataParallel(
    net, device_ids = [local_rank], find_unused_parameters = True, broadcast_buffers = True if world_size > 1 else False
)
# perceptual_loss_network = nn.parallel.DistributedDataParallel(perceptual_loss_network, device_ids=[local_rank])

trainset = TrainData( crop_size, train_data_dir, labeled_name )
testset = ValData( crop_size, val_data_dir, val_filename )

train_sampler = DistributedSampler( dataset = trainset, shuffle = True )
train_data_loader = torch.utils.data.DataLoader(
    trainset, batch_size = train_batch_size,
    sampler = train_sampler, num_workers = 4
)
test_sampler = DistributedSampler( dataset = testset, shuffle = False )
val_data_loader = torch.utils.data.DataLoader(
    testset, batch_size = val_batch_size,
    sampler = test_sampler, num_workers = 4
)

# ================================  Set parameters and save them and Synchronize all processes =============================== #

step = 0
step = step + step_start
num_epochs = num_epochs + epoch_start
lendata = len( train_data_loader )

if is_main_process( local_rank ):
    pytorch_total_params = sum( p.numel() for p in net.parameters() if p.requires_grad )
    parameter_logger = Logger( timestamp = time_str, filename = f'parameters.txt', mode = 'w+' ).initlog()
    print( '--- Hyper-parameters for training...' )
    parameter = '--- seed: {}\n' \
                '--- learning_rate: {}\n' \
                '--- total_epochs: {}\n' \
                '--- total_params: {}\n' \
                '--- crop_size: {}\n' \
                '--- train_batch_size: {}\n' \
                '--- val_batch_size: {}\n' \
                '--- alpha: {}\n' \
                '--- beta: {}\n' \
                '--- gamma: {}\n' \
                '--- lrscheduler_step_size: {}\n' \
                '--- lrscheduler_step_gamma: {}\n'.format(
        seed, learning_rate, num_epochs, pytorch_total_params,
        crop_size, train_batch_size, val_batch_size,
        alpha, beta, gamma, step_size, step_gamma
    )

    print( parameter )
    parameter_logger.writelines( parameter )
    parameter_logger.close()
# --------- train model ! ---------
if is_main_process( local_rank ): print( '=' * 25, ' Begin training model ! ', '=' * 25, )
dist.barrier()
for epoch in range( epoch_start, num_epochs ):  # default epoch_start = 0
    start_time = time.time()
    epoch_loss = 0
    epoch_psnr = 0
    epoch_ssim = 0
    # adjust_learning_rate(optimizer, epoch)
    loop = train_data_loader
    if is_main_process( local_rank ):
        loop = tqdm( train_data_loader, desc = "--- Progress bar : " )

    train_sampler.set_epoch( epoch )
    test_sampler.set_epoch( epoch )
    # 如果不调用set_epoch, 那么每个epoch都会使用第1个epoch的indices, 因为epoch数没有改变, 随机种子seed一直是初始状态
    # -------------------------------------------------------------------------------------------------------------
    for id, train_data in enumerate( loop ):

        input_image, gt, imgid = train_data
        input_image = input_image.to( device )
        # print(input_image.shape)
        gt = gt.to( device )

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad( set_to_none = True )  # set_to_none = True here can modestly improve performance

        # --- Forward + Backward + Optimize --- #
        if isapex:
            with torch.autocast( device_type = 'cuda', dtype = torch.float16, enabled = use_amp ):
                net.to( device ).train()
                pred_image, fm = net( input_image )
                gt_pred, _ = net( gt )
                pred_image.to( device )
                gt_pred.to( device )
                fm = [i.to( device ) for i in fm]
                # smooth_loss = F.smooth_l1_loss(pred_image, gt).mean()
                # perceptual_loss = perceptual_loss_network(pred_image,gt,fm).mean()
                # loss = smooth_loss + alpha * perceptual_loss
                loss = synthetic_loss(
                    pred_image, gt, gt_pred, fm,
                    perceptual_loss_network, ssim,
                    alpha, beta, gamma
                )

            scaler.scale( loss ).backward()
            scaler.step( optimizer )
            scheduler.step()
            scaler.update()
        else:
            net.to( device ).train()
            pred_image, fm = net( input_image )
            gt_pred, _ = net( gt )
            pred_image.to( device )
            gt_pred.to( device )
            fm = [i.to( device ) for i in fm]
            # smooth_loss = F.smooth_l1_loss(pred_image, gt).mean()
            # perceptual_loss = perceptual_loss_network(pred_image,gt,fm).mean()
            # loss = smooth_loss + alpha * perceptual_loss
            loss = synthetic_loss(
                pred_image, gt, gt_pred, fm,
                perceptual_loss_network, ssim,
                alpha, beta, gamma
            )
            '''
            Note : the loss do not synchronize but the gradient 
                   is auto synchronize when loss execute backward .
            '''
            # # ssim_loss = ssim.to_ssim_loss(pred_image,gt)
            # loss = smooth_loss + alpha * perceptual_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            

        # calculate psnr and ssim
        step_psnr, step_ssim = \
            psnr.to_psnr( pred_image.detach(), gt.detach() ), ssim.to_ssim( pred_image.detach(), gt.detach() )

        # collection result to GPU:0
        dist.barrier()  # synchronize processing
        torch.distributed.reduce( loss, 0, op = torch.distributed.ReduceOp.AVG )
        torch.distributed.reduce( step_psnr, 0, op = torch.distributed.ReduceOp.AVG )
        torch.distributed.reduce( step_ssim, 0, op = torch.distributed.ReduceOp.AVG )

        if is_main_process( local_rank ):

            # logging
            loop.set_postfix(
                {'Epoch': f'{epoch + 1} / {num_epochs}', 'Step': f'{step + 1}',
                 'Steploss': '{:.4f}'.format( loss.item() )
                 }
            )
            writer.add_scalar( 'TrainingStep/step-loss', loss.item(), step + 1 )
            writer.add_scalar( 'TrainingStep/step-PSNR', step_psnr, step + 1 )
            writer.add_scalar( 'TrainingStep/step-SSIM', step_ssim, step + 1 )
            writer.add_scalar(
                'TrainingStep/lr', scheduler.get_last_lr()[0], step + 1
            )  # logging lr for every step
            step_logger.writelines(
                f'Epoch: {epoch + 1} / {num_epochs} - Step: {step + 1}'
                + ' - steploss: {:.4f} - stepPSNR: {:.4f} - stepSSIM: {:.4f}\n'.format(
                    loss.item(), step_psnr, step_ssim
                )
            )

            epoch_loss += loss.item()
            epoch_psnr += step_psnr
            epoch_ssim += step_ssim
            step = step + 1
            if step % 50 == 0: step_logger.flush()


    if is_main_process( local_rank ):
        epoch_loss /= lendata  # here epoch loss have reduced on GPU:0
        epoch_psnr /= lendata
        epoch_ssim /= lendata
        print(
            '--- Epoch: [{}/{}], EpochAveLoss: {:.4f}, EpochAvePSNR: {:.4f}, EpochAveSSIM: {:.4f}----'
            .format( epoch + 1, num_epochs, epoch_loss, epoch_psnr, epoch_ssim )
        )
        writer.add_scalar( 'TrainingEpoch/epoch-loss', epoch_loss, epoch + 1 )
        writer.add_scalar( 'TrainingEpoch/epoch-PSNR', epoch_psnr, epoch + 1 )
        writer.add_scalar( 'TrainingEpoch/epoch-SSIM', epoch_ssim, epoch + 1 )

        epoch_logger.writelines(
            'Epoch [{}/{}], EpochAveLoss: {:.4f}, EpochAvePSNR: {:.4f} EpochAveSSIM: {:.4f}\n'.format(
                epoch + 1, num_epochs, epoch_loss, epoch_psnr, epoch_ssim
            )
        )
        epoch_logger.flush()

        # --- Save the  parameters --- #
        model_to_save = net.module if hasattr( net, "module" ) else net
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
            'scheduler': scheduler.state_dict(),
            'amp_scaler': scaler.state_dict() if isapex else None
        }
        torch.save( checkpoint, './{}/{}/latest_model.pth'.format( exp_name,time_str ) )

    # --- Use the evaluation model in testing  for every 5 epoch--- #
    if ((epoch + 1) % 5 == 0) or (epoch == num_epochs - 1):

        '''
        - here when you want to evaluate the test data on a specific device (lets say GPU:0,and you have 2 GPU),
          you can use "local_model = net.module" to evaluate : 
          please see https://github.com/pytorch/pytorch/issues/54059  for more details .
          as this situation , your test data must not be swap by DDP, otherwise GPU:0 can only get half size data .
          
        - flowing is DDP validation .
        '''

        val_loss, val_psnr, val_ssim = validation_ddp(
            net, val_data_loader, device = device,
            perceptual_loss_network = perceptual_loss_network,
            ssim = ssim, psnr = psnr,
            alpha = alpha, beta = beta, gama = gamma,
            local_rank = local_rank
        )
        # collection val result to GPU:0
        dist.barrier()  # synchronize processing
        torch.distributed.reduce( val_loss, 0, op = torch.distributed.ReduceOp.AVG )
        torch.distributed.reduce( val_psnr, 0, op = torch.distributed.ReduceOp.AVG )
        torch.distributed.reduce( val_ssim, 0, op = torch.distributed.ReduceOp.AVG )

        if is_main_process( local_rank ):
            print( '--- ValLoss : {:.4f} , Valpsnr : {:.4f} , Valssim : {:.4f}'.format( val_loss, val_psnr, val_ssim ) )
            # logging
            writer.add_scalar( 'Validation/loss', val_loss, epoch + 1 )
            writer.add_scalar( 'Validation/psnr', val_psnr, epoch + 1 )
            writer.add_scalar( 'Validation/ssim', val_ssim, epoch + 1 )
            # logging
            val_logger.writelines(
                'Epoch [{}/{}], ValEpochAveLoss: {:.4f}, ValEpochAvePSNR: {:.4f} ValEpochAveSSIM: {:.4f}\n'.format(
                    epoch + 1, num_epochs, val_loss, val_psnr, val_ssim
                )
            )
            val_logger.flush()

            if val_psnr >= old_val_psnr:
                checkpoint = {
                    "net": model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch + 1,
                    'step': step + 1,
                    'scheduler': scheduler.state_dict(),
                    'amp_scaler': scaler.state_dict() if isapex else None
                }
                torch.save( checkpoint, './{}/{}/best_model.pth'.format( exp_name ,time_str ) )
                # shutil.copy2(
                #     './{}/latest_model.pth'.format(exp_name),
                #     './{}/best_model.pth'.format(exp_name),
                # )
                print( '--- Update the best model !' )
                old_val_psnr = val_psnr

    dist.barrier()

if is_main_process( local_rank ):
    step_logger.close()
    epoch_logger.close()
    val_logger.close()
    writer.close()
print(
    f'=================================== END TRAIN IN PROCESSING DEVICE {local_rank} ==================================='
)
dist.barrier()
