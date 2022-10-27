
import torch
import torch.nn as nn
import os
import random
random.seed(666)
import shutil
import numpy as np
import math

from pytorch_msssim import ssim
# https://github.com/VainF/pytorch-msssim

class Logger():
    def __init__(self,filename : str, log_path = 'logs/loss/'):
        os.makedirs(log_path, exist_ok=True)
        self.log_path = log_path + filename
    def initlog(self):
        self.looger =  open(file = self.log_path,mode='a')
        return self.looger
    def close(self):
        self.looger.close()

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



if __name__ == '__main__':
    # split_train_test(r'D:/下载/Allweather_subset')
    psnrobj = PSNR()
    print(
        psnrobj.to_psnr(
            torch.ones((8, 3 , 256, 256 )), torch.ones((8, 3 , 256, 256 ))*0.978
        )
    )
    ssimobj = SSIM()
    print(
        ssimobj.to_ssim(
            torch.ones((8, 3 , 256, 256 )), torch.ones((8, 3 , 256, 256 )) * 0.978
        )
    )





