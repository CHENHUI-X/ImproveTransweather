import os
import shutil
# https://github.com/VainF/pytorch-msssim
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

import cv2
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

# for processing image  : python3 scripts/utils.py
# from ssim import _fspecial_gauss_1d, ssim

# for train model
from scripts.ssim import _fspecial_gauss_1d, ssim


class Logger():
    def __init__(self, timestamp: str, filename: str, log_path='./logs/loss/'):
        self.log_path = log_path + timestamp  # './logs/loss/2022-10-29_15:14:33'
        os.makedirs(self.log_path, exist_ok=True)
        self.log_file = self.log_path + '/' + filename  # './logs/loss/2022-10-29_15:14:33/xxx.txt'

        self.logger = open(file=self.log_file, mode='a+')

    def initlog(self):
        return self.logger

    def close(self):
        self.logger.close()


# Process image directory to standard
def images_organize(img_dir: str = './allweather_2', Istrain = True):
    print('=========================== Processing images ... ===========================')
    Inputdir = img_dir + '/input'  # Input or input
    Outputdir = img_dir + '/gt'  # Output or gt

    _ = 'train' if Istrain else 'test'

    new_input_dir = f'./data/{_}/input'  #  image input
    new_gt_dir = f'./data/{_}/gt'  #  image gt
    # copy the whole directory
    shutil.copytree(Inputdir, new_input_dir)
    shutil.copytree(Outputdir, new_gt_dir)
    '''
        Note : must copy directory firstly that ensure the path is created , and then create the txt .
               if you create  destination  folder firstly, the copytree function will raise an error about 
               " file is exit " 
    '''

    # get image file name for this dataset
    with open(f'./data/{_}/{_}.txt', mode='w+') as f:
        for file in os.listdir(Outputdir): # the data pair txt based on only output
            if file.endswith(".png") or file.endswith('.jpg'):
                f.writelines('input/' + file + '\n')


    print('===========================          END          ===========================')
# ==================================================================================================

# save data to a file
def save_img(img_name, img, filepath='./data/test/pred/', ):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    # print( img.numpy().shape)
    cv2.imwrite(filepath + img_name.split('/')[-1], img.numpy())
    # plt.imshow(np.ndarray(img))
    # plt.show()


def PollExecutorSaveImg(iamge_names, images, n_files=8):
    # create the process pool
    with ProcessPoolExecutor(8) as exe:
        # submit tasks to generate files
        _ = [exe.submit(save_img, iamge_names[i], images[i].permute(1, 2, 0)) for i in range(n_files)]

# ===================================================================================================
# ===================================================================================================
# Calculate PSNR
class PSNR(object):
    def to_psnr(self, pred: torch.Tensor, grtruth: torch.Tensor, data_range=1.0):
        assert pred.shape == grtruth.shape, 'Shape of pre image not equals to gt image !'

        if data_range < 255:
            pred *= 255
            grtruth *= 255
        mse = torch.mean((pred.type(torch.cuda.FloatTensor) - grtruth) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


# Calculate SSIM
class SSIM(object):
    def __init__(
          self,
          data_range=1.0,
          size_average=True,
          win_size=11,
          win_sigma=1.5,
          channel=3,
          spatial_dims=2,
          K=(0.01, 0.03),
          nonnegative_ssim=False,
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def to_ssim(self, pred: torch.Tensor, grtruth: torch.Tensor, ):
        return ssim(
            pred.type(torch.cuda.FloatTensor),
            grtruth,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim, )

    def to_ssim_loss(self, pred: torch.Tensor, grtruth: torch.Tensor,
                     data_range=1.0, size_average=True):
        return 1 - ssim(pred.type(torch.cuda.FloatTensor),
                        grtruth,
                        data_range=self.data_range,
                        size_average=self.size_average,
                        win=self.win,
                        K=self.K,
                        nonnegative_ssim=self.nonnegative_ssim, )

# ====================================================================================
# ====================================================================================
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
        pred_image, sw_fm = net(input_image)

        pred_image.to(device)
        sw_fm = [i.to(device) for i in sw_fm]

        smooth_loss = F.smooth_l1_loss(pred_image, gt).mean()
        perceptual_loss = loss_network(sw_fm, gt).mean()
        # ssim_loss = ssim.to_ssim_loss(pred_image,gt)
        loss = smooth_loss + lambda_loss * perceptual_loss
        val_loss += loss.item()
        val_ssim += ssim.to_ssim(pred_image, gt)
        val_psnr += psnr.to_psnr(pred_image, gt)

    val_loss /= lendata
    val_ssim /= lendata
    val_psnr /= lendata
    print('----ValLoss : {:.4f} , Valpsnr : {:.4f} , Valssim : {:.4f}'.format(val_loss, val_psnr, val_ssim))
    net.train()
    return val_loss, val_psnr, val_ssim


@torch.no_grad()
def load_best_model(net, exp_name='checkpoint'):
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
    except:
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
        del state_dict, new_state_dict
        torch.cuda.empty_cache()
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))
        return net

# ============================================================================================
# ============================================================================================
# ================================  DDP function ================================
def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://"  # default

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


def is_main_process(rank):
    return rank == 0


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    yield

    if local_rank == 0:
        torch.distributed.barrier()


if __name__ == '__main__':
    images_organize(r'/root/autodl-tmp/Transweather/data')
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


