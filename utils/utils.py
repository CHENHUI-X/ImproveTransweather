
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
random.seed(666)
import shutil
import numpy as np
import math
from tqdm import tqdm
from pytorch_msssim import ssim
# https://github.com/VainF/pytorch-msssim
from concurrent.futures import ProcessPoolExecutor
import cv2
import matplotlib.pyplot as plt
import re
class Logger():
    def __init__(self, timestamp : str ,  filename : str, log_path = './logs/loss/'):
        self.log_path = log_path + timestamp # './logs/loss/2022-10-29_15:14:33'
        os.makedirs(self.log_path, exist_ok=True)
        self.log_file = self.log_path + '/' + filename # './logs/loss/2022-10-29_15:14:33/xxx.txt'

        self.logger = open(file=self.log_file, mode='a+')
    def initlog(self):
        return self.logger
    def close(self):
        self.logger.close()


# Process image
def split_train_test(img_dir : str = './allweather_2'):

    Inputdir = img_dir+'/Input/' # Input or input
    Outputdir = img_dir+'/Output/' # Output or gt
    imgfiles = []
    for file in os.listdir(Inputdir):
        if file.endswith(".png") or file.endswith('.jpg'):
            imgfiles.append(file)
            # print(os.path.join(Inputdir, file))
    imgnum = len(imgfiles)
    train_index = random.sample(range(imgnum), 10,)
    test_index = list(set(range(imgnum)) - set(train_index))

    train_input_dir = img_dir + '/train/input' # train image input
    os.makedirs(train_input_dir,exist_ok=True)
    train_gt_dir = img_dir + '/train/gt'  # train image gt
    os.makedirs(train_gt_dir, exist_ok=True)
    with open(img_dir + '/train/train.txt',mode='w+') as f:
        for trainind in train_index:
            shutil.copy2(
                Inputdir +  imgfiles[trainind],
                train_input_dir
            )
            shutil.copy2(
                Outputdir + imgfiles[trainind],
                train_gt_dir
            )
            f.writelines('/input/' + imgfiles[trainind] +'\n')
    # ----------------------------------------
    test_input_dir = img_dir + '/test/input'  # test image input
    os.makedirs(test_input_dir, exist_ok=True)
    test_gt_dir = img_dir + '/test/gt'  # test image gt
    os.makedirs(test_gt_dir, exist_ok=True)
    with open(img_dir + '/test/test.txt',mode='w+') as f:
        for testind in test_index:
            shutil.copy2(
                Inputdir + imgfiles[testind],
                test_input_dir,
            )
            shutil.copy2(
                Outputdir + imgfiles[testind],
                test_gt_dir,
            )
            f.writelines('/input/' + imgfiles[testind] +'\n')

# Calculate PSNR
class PSNR(object):
    def to_psnr(self , pred: torch.Tensor, grtruth: torch.Tensor ,  data_range = 1.0):
        assert pred.shape == grtruth.shape , 'Shape of pre image not equals to gt image !'
        if data_range < 255 :
            pred *= 255
            grtruth *= 255
        mse = torch.mean((pred - grtruth) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

class SSIM(object):
    def to_ssim(self, pred: torch.Tensor, grtruth: torch.Tensor ,
                data_range = 1.0 ,size_average=True):
        assert pred.shape == grtruth.shape, 'Shape of pre image not equals to gt image !'
        ssim_out = ssim(pred, grtruth, data_range=data_range, size_average = size_average)
        return ssim_out

    def to_ssim_loss(self, pred: torch.Tensor, grtruth: torch.Tensor ,
                  data_range = 1.0 ,size_average=True):
        # this can used as loss function
        assert pred.shape == grtruth.shape, 'Shape of pre image not equals to gt image !'
        ssim_out = ssim(pred, grtruth, data_range=data_range, size_average=size_average)
        ssim_loss = 1 - ssim_out
        return ssim_loss

@torch.no_grad()
def validation(net, val_data_loader, device='cuda:0', **kwargs):
    loop = tqdm(val_data_loader, desc="----Validation : ")
    net.to(device).eval()
    loss_network = kwargs['loss_network'].to(device)
    ssim = kwargs['ssim']
    psnr = kwargs['psnr']
    lambda_loss = kwargs['lambda_loss']

    lendata = len(val_data_loader)
    val_loss = 0
    val_psnr = 0
    val_ssim = 0
    for batch_id, test_data in enumerate(loop):
        input_image, gt, imgname = test_data
        input_image = input_image.to(device)
        gt = gt.to(device)
        pred_image = net(input_image).to(device)
        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt)
        # ssim_loss = ssim.to_ssim_loss(pred_image,gt)
        loss = smooth_loss + lambda_loss * perceptual_loss
        val_loss += loss.item()
        val_ssim += ssim.to_ssim(pred_image, gt)
        val_psnr += psnr.to_psnr(pred_image, gt)

    val_loss /= lendata
    val_ssim /= lendata
    val_psnr /= lendata
    print('----ValLoss : {:.4f} , Valpsnr : {:.4f} , Valssim : {:.4f}'.format(val_loss,val_psnr,val_ssim))
    net.train()
    return val_loss,val_psnr,val_ssim
@torch.no_grad()
def load_best_model(net , exp_name = 'checkpoint' ):
    if not os.path.exists('./{}/'.format(exp_name)):
        # os.mkdir('./{}/'.format(exp_name))
        raise FileNotFoundError
    try:
        print('--- Loading model weight... ---')
        # original saved file with DataParallel
        state_dict = torch.load('./{}/best_model.pth'.format(exp_name))
        # checkpoint = {
        #     "net": net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     "epoch": epoch,
        #     'step': step,
        #     'scheduler': scheduler.state_dict()
        # }
        net.load_state_dict(state_dict['net'])

        print('--- Loading model successfully! ---')
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))
        return net
    except :
        print('--- Loading model weight... ---')
        state_dict = torch.load('./{}/best_model.pth'.format(exp_name))
        '''
            If you have an error about load model in " Missing key(s) in state_dict: " , please reference this url 
            https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/7
        '''
        # original saved file with DataParallel
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict['net'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)
        print('--- Loading model successfully! ---')
        del state_dict , new_state_dict
        torch.cuda.empty_cache()
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))
        return net

# save data to a file
def save_img( img_name , img , filepath = './data/test/pred/', ):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    # print( img.numpy().shape)
    cv2.imwrite(filepath + img_name.split('/')[-1], img.numpy())
    # plt.imshow(np.ndarray(img))
    # plt.show()

def PollExecutorSaveImg(iamge_names , images , n_files = 8 ):
    # create the process pool
    with ProcessPoolExecutor(8) as exe:
        # submit tasks to generate files
        _ = [exe.submit(save_img, iamge_names[i], images[i].permute(1,2,0)) for i in range(n_files)]

if __name__ == '__main__':
    # split_train_test(r'D:/下载/Allweather_subset')
    # test
    # psnrobj = PSNR()
    # print(
    #     psnrobj.to_psnr(
    #         torch.ones((8, 3 , 256, 256 )), torch.ones((8, 3 , 256, 256 ))*0.978
    #     )
    # )
    # ssimobj = SSIM()
    # print(
    #     ssimobj.to_ssim(
    #         torch.ones((8, 3 , 256, 256 )), torch.ones((8, 3 , 256, 256 )) * 0.978
    #     )
    # )
    ...


