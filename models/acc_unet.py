"""
ACC-UNet architecture using PyTorch
"""

import torch


class ChannelSELayer(torch.nn.Module):
    """
    Implements Squeeze and Excitation
    """

    def __init__(self, num_channels):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
        """

        super(ChannelSELayer, self).__init__()

        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # 每个通道一个score

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio

        self.fc1 = torch.nn.Linear(
            num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(
            num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)

    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        out = self.act(self.fc1(self.gp_avg_pool(
            inp).view(batch_size, num_channels)))
        # 就是对score 维度使用 linear 进行放缩
        out = self.sigmoid(self.fc2(out))

        out = torch.mul(inp, out.view(
            batch_size, num_channels, 1, 1))  # 在通道维度上进行乘法

        out = self.bn(out)
        out = self.act(out)

        return out


class HANCLayer(torch.nn.Module):
    """
    Implements Hierarchical Aggregation of Neighborhood Context operation
    """

    def __init__(self, in_chnl, out_chnl, k):
        """
        Initialization

        Args:
            in_chnl (int): number of input channels
            out_chnl (int): number of output channels
            k (int): value of k in HANC
        """

        super(HANCLayer, self).__init__()

        self.k = k

        self.cnv = torch.nn.Conv2d(
            (2 * k - 1) * in_chnl, out_chnl, kernel_size=(1, 1))
        self.act = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm2d(out_chnl)

    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        x = inp

        if self.k == 1:
            x = inp

        elif self.k == 2:
            x = torch.concat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(
                        torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=2)(
                        torch.nn.MaxPool2d(2)(x)),
                ],
                dim=2,
            )

        # 自身 + mean(x,2) + mean(x,4) + + mean(max,2) + mean(max,4)
        elif self.k == 3:
            x = torch.concat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(
                        torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(
                        torch.nn.AvgPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=2)(
                        torch.nn.MaxPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(
                        torch.nn.MaxPool2d(4)(x)),
                ],
                dim=2,
            )

        elif self.k == 4:
            x = torch.concat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(
                        torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(
                        torch.nn.AvgPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(
                        torch.nn.AvgPool2d(8)(x)),
                    torch.nn.Upsample(scale_factor=2)(
                        torch.nn.MaxPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(
                        torch.nn.MaxPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(
                        torch.nn.MaxPool2d(8)(x)),
                ],
                dim=2,
            )

        elif self.k == 5:
            x = torch.concat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(
                        torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(
                        torch.nn.AvgPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(
                        torch.nn.AvgPool2d(8)(x)),
                    torch.nn.Upsample(scale_factor=16)(
                        torch.nn.AvgPool2d(16)(x)),
                    torch.nn.Upsample(scale_factor=2)(
                        torch.nn.MaxPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(
                        torch.nn.MaxPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(
                        torch.nn.MaxPool2d(8)(x)),
                    torch.nn.Upsample(scale_factor=16)(
                        torch.nn.MaxPool2d(16)(x)),
                ],
                dim=2,
            )

        x = x.view(batch_size, num_channels * (2 * self.k - 1), H, W)

        x = self.act(self.bn(self.cnv(x)))

        return x


class Conv2d_batchnorm(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1),
        activation="LeakyReLU",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        return self.sqe(self.activation(x))


class Conv2d_channel(torch.nn.Module):
    """
    2D pointwise Convolutional layers        
    """

    def __init__(self, num_in_filters, num_out_filters):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=(1, 1),
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        return self.sqe(self.activation(x))


class HANCBlock(torch.nn.Module):
    """
    Encapsulates HANC block
    """

    def __init__(self, n_filts, out_channels, k=3, inv_fctr=3):
        """
        Initialization

        Args:
            n_filts (int): number of filters
            out_channels (int): number of output channel
            activation (str, optional): activation function. Defaults to 'LeakyReLU'.
            k (int, optional): k in HANC. Defaults to 1.
            inv_fctr (int, optional): inv_fctr in HANC. Defaults to 4.
        """

        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            n_filts, n_filts * inv_fctr, kernel_size=1)
        self.norm1 = torch.nn.BatchNorm2d(n_filts * inv_fctr)

        self.conv2 = torch.nn.Conv2d(
            n_filts * inv_fctr,
            n_filts * inv_fctr,
            kernel_size=3,
            padding=1,
            groups=n_filts * inv_fctr,
        )
        self.norm2 = torch.nn.BatchNorm2d(n_filts * inv_fctr)

        self.hnc = HANCLayer(n_filts * inv_fctr, n_filts, k)

        self.norm = torch.nn.BatchNorm2d(n_filts)

        self.conv3 = torch.nn.Conv2d(n_filts, out_channels, kernel_size=1)
        self.norm3 = torch.nn.BatchNorm2d(out_channels)

        self.sqe = ChannelSELayer(out_channels)

        self.activation = torch.nn.LeakyReLU()

    def forward(self, inp):

        x = self.conv1(inp)  # up channel num to n_filts * inv_fctr
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.hnc(x)  # transform to n_filts

        x = self.norm(x + inp)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.sqe(x)

        return x


class ResPath(torch.nn.Module):
    """
    Implements ResPath-like modified skip connection

    """

    def __init__(self, in_chnls, n_lvl):
        """
        Initialization

        Args:
            in_chnls (int): number of input channels
            n_lvl (int): number of blocks or levels
        """

        super(ResPath, self).__init__()

        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        self.sqes = torch.nn.ModuleList([])

        self.bn = torch.nn.BatchNorm2d(in_chnls)
        self.act = torch.nn.LeakyReLU()
        self.sqe = torch.nn.BatchNorm2d(in_chnls)

        for i in range(n_lvl):
            self.convs.append(
                torch.nn.Conv2d(in_chnls, in_chnls,
                                kernel_size=(3, 3), padding=1)
            )
            self.bns.append(torch.nn.BatchNorm2d(in_chnls))
            self.sqes.append(ChannelSELayer(in_chnls))

    def forward(self, x):

        for i in range(len(self.convs)):
            x = x + self.sqes[i](
                self.act(
                    self.bns[i](
                        self.convs[i](x)
                    )
                )
            )

        return self.sqe(self.act(self.bn(x)))


class MLFC(torch.nn.Module):
    """
    Implements Multi Level Feature Compilation

    """

    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, lenn=1):
        """
        Initialization

        Args:
            in_filters1 (int): number of channels in the first level
            in_filters2 (int): number of channels in the second level
            in_filters3 (int): number of channels in the third level
            in_filters4 (int): number of channels in the fourth level
            lenn (int, optional): number of repeats. Defaults to 1.
        """

        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4
        self.in_filters = (
            in_filters1 + in_filters2 + in_filters3 + in_filters4
        )  # total number of channels

        self.no_param_up = torch.nn.Upsample(
            scale_factor=2)  # used for upsampling
        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])

        for i in range(lenn):
            self.cnv_blks1.append(
                Conv2d_batchnorm(self.in_filters, in_filters1, (1, 1))
            )
            self.cnv_mrg1.append(Conv2d_batchnorm(
                2 * in_filters1, in_filters1, (1, 1)))
            self.bns1.append(torch.nn.BatchNorm2d(in_filters1))
            self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))

            self.cnv_blks2.append(
                Conv2d_batchnorm(self.in_filters, in_filters2, (1, 1))
            )
            self.cnv_mrg2.append(Conv2d_batchnorm(
                2 * in_filters2, in_filters2, (1, 1)))
            self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
            self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

            self.cnv_blks3.append(
                Conv2d_batchnorm(self.in_filters, in_filters3, (1, 1))
            )
            self.cnv_mrg3.append(Conv2d_batchnorm(
                2 * in_filters3, in_filters3, (1, 1)))
            self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
            self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

            self.cnv_blks4.append(
                Conv2d_batchnorm(self.in_filters, in_filters4, (1, 1))
            )
            self.cnv_mrg4.append(Conv2d_batchnorm(
                2 * in_filters4, in_filters4, (1, 1)))
            self.bns4.append(torch.nn.BatchNorm2d(in_filters4))
            self.bns_mrg4.append(torch.nn.BatchNorm2d(in_filters4))

        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1)
        self.sqe2 = ChannelSELayer(in_filters2)
        self.sqe3 = ChannelSELayer(in_filters3)
        self.sqe4 = ChannelSELayer(in_filters4)

    def forward(self, x1, x2, x3, x4):

        batch_size, _, h1, w1 = x1.shape # ( C, H, W) 
        _, _, h2, w2 = x2.shape # ( C * 2 , H // 2, W // 2)
        _, _, h3, w3 = x3.shape # ( C * 4 , H // 4, W // 4) 
        _, _, h4, w4 = x4.shape # ( C * 8 , H // 8, W // 8)

        for i in range(len(self.cnv_blks1)):
            x_c1 = self.act(
                self.bns1[i](
                    self.cnv_blks1[i](
                        torch.cat(
                            [
                                x1,
                                self.no_param_up(x2),
                                self.no_param_up(self.no_param_up(x3)),
                                self.no_param_up(self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            ) # trans C + C * 2 + C * 4 + C * 8 into C
            x_c2 = self.act(
                self.bns2[i](
                    self.cnv_blks2[i](
                        torch.cat(
                            [
                                self.no_param_down(x1),
                                (x2),
                                (self.no_param_up(x3)),
                                (self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            ) # trans C + C * 2 + C * 4 + C * 8 into 2*C
            x_c3 = self.act(
                self.bns3[i](
                    self.cnv_blks3[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(x1)),
                                self.no_param_down(x2),
                                (x3),
                                (self.no_param_up(x4)),
                            ],
                            dim=1,
                        )
                    )
                )
            ) # trans C + C * 2 + C * 4 + C * 8 into 4*C 
            x_c4 = self.act(
                self.bns4[i](
                    self.cnv_blks4[i](
                        torch.cat(
                            [
                                self.no_param_down(
                                    self.no_param_down(self.no_param_down(x1))),
                                self.no_param_down(self.no_param_down(x2)),
                                self.no_param_down(x3),
                                x4,
                            ],
                            dim=1,
                        )
                    )
                )
            ) # trans C + C * 2 + C * 4 + C * 8 into 8*C

            x_c1 = self.act(
                self.bns_mrg1[i](
                    self.cnv_mrg1[i](
                        torch.cat([x_c1, x1], dim=2).view(
                            batch_size, 2 * self.in_filters1, h1, w1)
                    )
                    + x1
                )
            ) # 2*C -> C
            x_c2 = self.act(
                self.bns_mrg2[i](
                    self.cnv_mrg2[i](
                        torch.cat([x_c2, x2], dim=2).view(
                            batch_size, 2 * self.in_filters2, h2, w2)
                    )
                    + x2
                )
            )  # 4*C -> 2*C
            x_c3 = self.act(
                self.bns_mrg3[i](
                    self.cnv_mrg3[i](
                        torch.cat([x_c3, x3], dim=2).view(
                            batch_size, 2 * self.in_filters3, h3, w3)
                    )
                    + x3
                )
            ) # 8*C -> 4*C
            x_c4 = self.act(
                self.bns_mrg4[i](
                    self.cnv_mrg4[i](
                        torch.cat([x_c4, x4], dim=2).view(
                            batch_size, 2 * self.in_filters4, h4, w4)
                    )
                    + x4
                )
            ) # 16*C -> 8*C

        x1 = self.sqe1(x_c1)
        x2 = self.sqe2(x_c2)
        x3 = self.sqe3(x_c3)
        x4 = self.sqe4(x_c4)

        return x1, x2, x3, x4 # 实际最后Channel数目没变


class ACC_UNet(torch.nn.Module):
    """
    ACC-UNet model
    """

    def __init__(self, n_channels, n_classes, n_filts=32):
        """
        Initialization

        Args:
            n_channels (int): number of channels of the input image.
            n_classes (int): number of output classes
            n_filts (int, optional): multiplier of the number of filters throughout the model.
                                     Increase this to make the model wider.
                                     Decrease this to make the model ligher.
                                     Defaults to 32.
        """

        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.pool = torch.nn.MaxPool2d(2)

        self.cnv11 = HANCBlock(n_channels, n_filts, k=3, inv_fctr=3)
        self.cnv12 = HANCBlock(n_filts, n_filts, k=3, inv_fctr=3)

        self.cnv21 = HANCBlock(n_filts, n_filts * 2, k=3, inv_fctr=3)
        self.cnv22 = HANCBlock(n_filts * 2, n_filts * 2, k=3, inv_fctr=3)

        self.cnv31 = HANCBlock(n_filts * 2, n_filts * 4, k=3, inv_fctr=3)
        self.cnv32 = HANCBlock(n_filts * 4, n_filts * 4, k=3, inv_fctr=3)

        self.cnv41 = HANCBlock(n_filts * 4, n_filts * 8, k=2, inv_fctr=3)
        self.cnv42 = HANCBlock(n_filts * 8, n_filts * 8, k=2, inv_fctr=3)

        self.cnv51 = HANCBlock(n_filts * 8, n_filts * 16, k=1, inv_fctr=3)
        self.cnv52 = HANCBlock(n_filts * 16, n_filts * 16, k=1, inv_fctr=3)

        self.rspth1 = ResPath(n_filts, 4) # 第一个 level 是 4个 Respath
        self.rspth2 = ResPath(n_filts * 2, 3)
        self.rspth3 = ResPath(n_filts * 4, 2)
        self.rspth4 = ResPath(n_filts * 8, 1)

        self.mlfc1 = MLFC(n_filts, n_filts * 2, n_filts *
                          4, n_filts * 8, lenn=1)
        self.mlfc2 = MLFC(n_filts, n_filts * 2, n_filts *
                          4, n_filts * 8, lenn=1)
        self.mlfc3 = MLFC(n_filts, n_filts * 2, n_filts *
                          4, n_filts * 8, lenn=1)

        self.up6 = torch.nn.ConvTranspose2d(
            n_filts * 16, n_filts * 8, kernel_size=(2, 2), stride=2)
        self.cnv61 = HANCBlock(n_filts * 8 + n_filts * 8,
                               n_filts * 8, k=2, inv_fctr=3)
        self.cnv62 = HANCBlock(n_filts * 8, n_filts * 8, k=2, inv_fctr=3)

        self.up7 = torch.nn.ConvTranspose2d(
            n_filts * 8, n_filts * 4, kernel_size=(2, 2), stride=2)
        self.cnv71 = HANCBlock(n_filts * 4 + n_filts * 4,
                               n_filts * 4, k=3, inv_fctr=3)
        self.cnv72 = HANCBlock(n_filts * 4, n_filts * 4, k=3, inv_fctr=34)

        self.up8 = torch.nn.ConvTranspose2d(
            n_filts * 4, n_filts * 2, kernel_size=(2, 2), stride=2)
        self.cnv81 = HANCBlock(n_filts * 2 + n_filts * 2,
                               n_filts * 2, k=3, inv_fctr=3)
        self.cnv82 = HANCBlock(n_filts * 2, n_filts * 2, k=3, inv_fctr=3)

        self.up9 = torch.nn.ConvTranspose2d(
            n_filts * 2, n_filts, kernel_size=(2, 2), stride=2)
        self.cnv91 = HANCBlock(n_filts + n_filts, n_filts, k=3, inv_fctr=3)
        self.cnv92 = HANCBlock(n_filts, n_filts, k=3, inv_fctr=3)

        # if n_classes == 1:
        #     self.out = torch.nn.Conv2d(n_filts, n_classes, kernel_size=(1, 1))
        #     self.last_activation = torch.nn.Sigmoid()
        # else:
        #     self.out = torch.nn.Conv2d(
        #         n_filts, n_classes + 1, kernel_size=(1, 1))
        #     self.last_activation = None
        self.out = torch.nn.Conv2d(n_filts, n_classes, kernel_size=(1, 1))
        self.last_activation = torch.nn.Sigmoid()
        

    def forward(self, x):

        x1 = x  # ( 3, H, W)

        x2 = self.cnv11(x1)  # ( C, H, W) , C default = 32
        x2 = self.cnv12(x2)  # ( C, H, W)

        x2p = self.pool(x2)  # ( C, H // 2, W // 2)

        x3 = self.cnv21(x2p)  # ( C * 2 , H // 2, W // 2)
        x3 = self.cnv22(x3)  # ( C * 2 , H // 2, W // 2)

        x3p = self.pool(x3)  # ( C * 2 , H // 4, W // 4)

        x4 = self.cnv31(x3p)  # ( C * 4 , H // 4, W // 4)
        x4 = self.cnv32(x4)  # ( C * 4 , H // 4, W // 4)

        x4p = self.pool(x4)  # ( C * 4 , H // 8, W // 8)

        x5 = self.cnv41(x4p)  # ( C * 8 , H // 8, W // 8)
        x5 = self.cnv42(x5)  # ( C * 8 , H // 8, W // 8)

        x5p = self.pool(x5)  # ( C * 8 , H // 16, W // 16)

        x6 = self.cnv51(x5p)  # ( C * 16 , H // 16, W // 16) 
        x6 = self.cnv52(x6)  # ( C * 16 , H // 16, W // 16) # 中间最下边的 block
        

        x2 = self.rspth1(x2)  # ( C, H, W)
        x3 = self.rspth2(x3)  # ( C * 2 , H // 2, W // 2)
        x4 = self.rspth3(x4)  # ( C * 4 , H // 4, W // 4)
        x5 = self.rspth4(x5)  # ( C * 8 , H // 8, W // 8)

        x2, x3, x4, x5 = self.mlfc1(x2, x3, x4, x5) 
        x2, x3, x4, x5 = self.mlfc2(x2, x3, x4, x5)
        x2, x3, x4, x5 = self.mlfc3(x2, x3, x4, x5)# 各自的channel数目没变, 这里只是把各个level的 map进行了fuse

        x7 = self.up6(x6) # ( C * 16 , H // 16, W // 16) -> ( C * 8 , H // 8, W // 8)
        x7 = self.cnv61(torch.cat([x7, x5], dim=1)) # C * 8 * 2 -> C * 8
        x7 = self.cnv62(x7)

        x8 = self.up7(x7) # ( C * 8 , H // 8, W // 8) -> ( C * 4 , H // 4, W // 4)
        x8 = self.cnv71(torch.cat([x8, x4], dim=1)) # C * 4 * 2 -> C * 4
        x8 = self.cnv72(x8)

        x9 = self.up8(x8) # ( C * 2 , H // 2, W // 2)
        x9 = self.cnv81(torch.cat([x9, x3], dim=1)) # C * 2 * 2 -> C * 2
        x9 = self.cnv82(x9)

        x10 = self.up9(x9) # ( C  , H , W )
        x10 = self.cnv91(torch.cat([x10, x2], dim=1))
        x10 = self.cnv92(x10)
        logits = self.last_activation(self.out(x10))


        return x + logits


if __name__ == '__main__':
    print(torch.cuda.is_available())
    torch.manual_seed(666)
    img = torch.rand(size=(1, 3, 224, 224)).to('cuda:0')
    net = ACC_UNet(n_channels=3, n_classes=3).to('cuda:0')
    out = net(img)
