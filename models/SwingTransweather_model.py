
import warnings
from functools import partial

import torch.nn.functional
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .base_networks import *


class EncoderSwTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size = 7,in_chans = 3, embed_dims=[128, 256, 512, 1024],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, mlpdrop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer = None, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], block_num = 4, window_size = 8, input_resolution=[64, 32, 16, 8]):
        '''

        :param img_size:
        :param patch_size:
        :param in_chans:
        :param embed_dims:  embed_dims = [64, 128, 320, 512] 外边传进来的参数
        :param num_heads:
        :param mlp_ratios:
        :param qkv_bias:
        :param qk_scale:
        :param mlpdrop_rate:
        :param attn_drop_rate:
        :param drop_path_rate:
        :param norm_layer:
        :param depths:   TransformerSubBlock layer num in a BLock , that many Block form the Transformer Encoder
        :param sr_ratios:  attention key scaler factor
        :param block_num: 4
        :param window_size:  8
        :param input_resolution: [64,32,16,8]
        '''
        super().__init__()
        self.embed_dims = embed_dims
        # patch embedding definitions
        # A special patch embedding , just for process original image

        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size = 4,
                                       in_chans=in_chans,embed_dim=embed_dims[0], norm_layer=norm_layer)
        # (3,256,256) -> ( 96 , 64, 64)

        # patch merging : dowm sample
        self.patch_embed2 = PatchMerging(input_resolution = to_2tuple(input_resolution[0]),
                                         current_dim = embed_dims[0],norm_layer=norm_layer)
        # ( 96 , 64, 64) -> ( 192 , 32, 32)

        self.patch_embed3 = PatchMerging(input_resolution=to_2tuple(input_resolution[1]),
                                         current_dim=embed_dims[1], norm_layer=norm_layer)
        # ( 192 , 32, 32) -> ( 384 , 16, 16)

        self.patch_embed4 = PatchMerging(input_resolution=to_2tuple(input_resolution[2]),
                                         current_dim=embed_dims[2], norm_layer=norm_layer)
        # ( 384 , 16, 16) -> ( 768 , 8 , 8)
    # ===========================================================================================


        # self.patch_embed1 = OverlapPatchEmbed(
        #     img_size=img_size, patch_size = patch_size, stride = 4, in_chans = in_chans, embed_dim=embed_dims[0])
        # # A special patch embedding , just for process original image
        #
        # self.patch_embed2 = OverlapPatchEmbed(
        #     img_size=img_size // 4,patch_size = 2, stride = 2,in_chans=embed_dims[0],embed_dim=embed_dims[1])
        #
        # self.patch_embed3 = OverlapPatchEmbed(
        #     img_size=img_size // 8,patch_size = 2, stride = 2,in_chans=embed_dims[1],embed_dim=embed_dims[2])
        #
        # self.patch_embed4 = OverlapPatchEmbed(
        #     img_size=img_size // 16,patch_size = 2, stride = 2,in_chans=embed_dims[2],embed_dim=embed_dims[3])

        ###########################################################################################
        # for Intra-patch transformer blocks
        # 注意这里不要看img_size的具体数据去推导后续的尺寸，这里指定的是224，并且实例化类的时候
        # 用的还是默认的尺寸，但是实际输入的尺寸是256
        self.mini_patch_embed1 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                   in_chans=embed_dims[0],
                                                   embed_dim=embed_dims[1])
        self.mini_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                   in_chans=embed_dims[1],
                                                   embed_dim=embed_dims[2])
        self.mini_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                   in_chans=embed_dims[2],
                                                   embed_dim=embed_dims[3])
        self.mini_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 32, patch_size=3, stride=2,
                                                   in_chans=embed_dims[3],
                                                   embed_dim=embed_dims[3])

        ###########################################################################################
        # main  encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        # self.block1 = nn.ModuleList([TransformerSubBlock(
        #     dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[0])
        #     for i in range(depths[0])])

        self.block1 = nn.ModuleList([SwinTransformerBlock(
            dim = embed_dims[0], input_resolution=to_2tuple(input_resolution[0]),
            num_heads = num_heads[0], window_size=8,
            shift_size = window_size // 2 if (i % 2 == 0) else 0,
            mlp_ratio= mlp_ratios[0], qkv_bias = True, qk_scale=None,
            mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer, fused_window_process=False
        ) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        # intra-patch encoder

        # self.patch_block1 = nn.ModuleList([TransformerSubBlock(
        #     dim=embed_dims[1], num_heads=num_heads[0],
        #     mlp_ratio = mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate,
        #     drop_path=dpr[cur + i], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[0])
        #     for i in range(depths[0])])
        #

        self.patch_block1 = nn.ModuleList([SpatialTransformerBlock(
            feature_size = input_resolution[1],in_channels=embed_dims[1], num_heads = 8,
            mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], norm_layer=norm_layer)
            for i in range(depths[0])])


        self.pnorm1 = norm_layer(embed_dims[1])

        # main  encoder
        cur += depths[0]
        # self.block2 = nn.ModuleList([TransformerSubBlock(
        #     dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[1])
        #     for i in range(depths[1])])

        self.block2 = nn.ModuleList([SwinTransformerBlock(
            dim=embed_dims[1], input_resolution=to_2tuple(input_resolution[1]), num_heads=num_heads[1], window_size=8,
            shift_size=window_size // 2 if (i % 2 == 0) else 0,
            mlp_ratio=mlp_ratios[1], qkv_bias=True, qk_scale=None,
            mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer, fused_window_process=False
        ) for i in range(depths[1])])

        self.norm2 = norm_layer(embed_dims[1])

        # intra-patch encoder
        #
        # self.patch_block2 = nn.ModuleList([TransformerSubBlock(
        #     dim=embed_dims[2], num_heads=num_heads[1],
        #     mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate,
        #     drop_path=dpr[cur + i], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[1])
        #     for i in range(depths[1])])

        self.patch_block2 = nn.ModuleList([SpatialTransformerBlock(
            feature_size = input_resolution[2], in_channels=embed_dims[2] ,
            num_heads = 2,mlp_ratio=mlp_ratios[1],
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], norm_layer=norm_layer)
            for i in range(depths[1])])

        self.pnorm2 = norm_layer(embed_dims[2])

        # main  encoder
        cur += depths[1]
        # self.block3 = nn.ModuleList([TransformerSubBlock(
        #     dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[2])
        #     for i in range(depths[2])])

        self.block3 = nn.ModuleList([SwinTransformerBlock(
            dim=embed_dims[2], input_resolution=to_2tuple(input_resolution[2]), num_heads=num_heads[2], window_size=8,
            shift_size=window_size // 2 if (i % 2 == 0) else 0,
            mlp_ratio=mlp_ratios[2], qkv_bias=True, qk_scale=None,
            mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer, fused_window_process=False
        ) for i in range(depths[2])])

        self.norm3 = norm_layer(embed_dims[2])

        # intra-patch encoder

        # self.patch_block3 = nn.ModuleList([TransformerSubBlock(
        #     dim = embed_dims[3], num_heads=num_heads[2],
        #     mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     mlpdrop = mlpdrop_rate, attn_drop=attn_drop_rate,
        #     drop_path=dpr[cur + i], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[2])
        #     for i in range(depths[2])])

        self.patch_block3 = nn.ModuleList([SpatialTransformerBlock(
            feature_size = input_resolution[3],in_channels=embed_dims[3] ,num_heads = 1,
            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], norm_layer=norm_layer)
            for i in range(depths[2])])

        self.pnorm3 = norm_layer(embed_dims[3])

        # main  encoder
        cur += depths[2]

        # self.block4 = nn.ModuleList([TransformerSubBlock(
        #     dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[3])
        #     for i in range(depths[3])])

        # self.block4 = nn.ModuleList([SwinTransformerBlock(
        #     dim=embed_dims[3], input_resolution=to_2tuple(input_resolution[3]), num_heads=num_heads[3], window_size=8,
        #     shift_size=window_size // 2 if (i % 2 == 0) else 0,
        #     mlp_ratio=mlp_ratios[3], qkv_bias=True, qk_scale=None,
        #     mlpdrop=mlpdrop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
        #     norm_layer=norm_layer, fused_window_process=False
        # ) for i in range(depths[3])])
        # self.norm4 = norm_layer(embed_dims[3])

        cur += depths[3]
        # =================================================================================
        # Out patch embedding
        self.patch_embed = nn.ModuleList([
            self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4
        ])

        # Intra patch embedding
        self.mini_patch_embed = nn.ModuleList([
            self.mini_patch_embed1, self.mini_patch_embed2, self.mini_patch_embed3 ,# self.mini_patch_embed4
        ])  # Actually do not need the mini_patch_embed4

        # Outer Block
        self.block = nn.ModuleList([
            self.block1, self.block2, self.block3 , #self.block4
        ])

        # Outer Norm
        self.norm = nn.ModuleList([
            self.norm1, self.norm2, self.norm3 ,# self.norm4
        ])

        # Intra Block
        self.patch_block = nn.ModuleList([
            self.patch_block1, self.patch_block2, self.patch_block3
        ])
        # Intra Norm
        self.pnorm = nn.ModuleList([
            self.pnorm1, self.pnorm2, self.pnorm3
        ])

        # active function
        self.active = nn.GELU()



    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        '''
        A summary illustration for the structure , see :
        https://drive.google.com/file/d/147DS5lJN4k76q9eKdXoVVo9xrqJnbFz4/view?usp=sharing
        '''
        B = x.shape[0]  # ( B , C , H , W )
        outs = []
        # Patch embedding the original image
        outer_patched_x, outer_H, outer_W = self.patch_embed[0](x)  # ( B , H//patch_size * W//patch_size , C )
        outBlock = outer_patched_x.permute(0, 2, 1).reshape(B, -1, outer_H, outer_W)
        # (B , C , H//patch_size , W//patch_size )

        outs.append(outBlock)

        ###################################################################################
        #                                   Block1 ~ Block3
        ###################################################################################
        input = outer_patched_x  # current input is previous output

        for i in range(3):
            # print('输入：' , input.shape)
            intra_branch_input, outer_branch_input = input, input  # do not directly modify `input`

            # ======================================================================================
            # Outer Transformer Block
            outer_short_cut = outer_branch_input  # # for shortcut
            for subBlock in self.block[i]:
                outer_branch_input = subBlock(outer_branch_input, outer_H, outer_W)
                outer_branch_input = self.active(outer_branch_input)

            outBlock = self.norm[i](outer_branch_input) + outer_short_cut

            # projection size , now size of output of outer block is equals the size of output of intra block

            # outBlock = outBlock.reshape(B, outer_H, outer_W, -1).permute(0, 3, 1, 2).contiguous()

            outBlock, intra_H, intra_W = self.patch_embed[i + 1](outBlock)
            outBlock = outBlock.permute(0, 2, 1).reshape(B, -1, intra_H, intra_W)
            # ======================================================================================

            # ======================================================================================
            # intra patch embedding :
            intra_patched_x, intra_H, intra_W = self.mini_patch_embed[i](
                intra_branch_input.permute(0, 2, 1).reshape(B, self.embed_dims[i], outer_H, outer_W)
            )
            # （ B , embed_dim , outer_H, outer_W ）-> ( B,  intra_H *  intra_W , embed_dim[next])
            # intra_H = outer_H // stride

            intra_short_cut = intra_patched_x  # # for shortcut : truly input of intra block

            # Intra Transformer Block
            for subBlock in self.patch_block[i]:
                intra_patched_x = subBlock(intra_patched_x, intra_H, intra_W)
                intra_patched_x = self.active(intra_patched_x)

            intraBlock = self.pnorm[i](intra_patched_x) + intra_short_cut
            intraBlock = intraBlock.reshape(B, intra_H, intra_W, -1).permute(0, 3, 1, 2).contiguous()
            # ======================================================================================

            output = outBlock + intraBlock  # shape with ( B , C , H , W )

            input = output.reshape(B, intra_H * intra_W, -1).contiguous()  # current out is next input , it's a cycle

            outer_H, outer_W = intra_H, intra_W  # Update the next input size

            outs.append(output)  # store the feature

        ###################################################################################
        #                                      Block4
        # here maybe do not need the block 4 , because it just implement that (B,1024,8,8 ) -> ( B,1024,8,8)
        # TODO : Ignore the block 4
        ###################################################################################

        # short_cut = output
        # outBlock = output.view(outBlock.shape[0], outBlock.shape[1], -1).permute(0, 2, 1)
        #
        # for i, blk in enumerate(self.block[-1]):
        #     outBlock = blk(outBlock, intra_H, intra_W)
        # # print('block 4 output shape : ', outBlock.shape)
        # outBlock = self.norm[-1](outBlock)
        # outBlock = outBlock.reshape(B, intra_H, intra_W, -1).permute(0, 3, 1, 2).contiguous() + short_cut
        # outs.append(outBlock)

        return outs  # Return 5 feature map with different shape ( note , last and second last shape is equipment )

    def forward(self, x):
        x = self.forward_features(x)
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size = 224, patch_size = 7, stride = 4, in_chans = 3, embed_dim = 96 ):
        super().__init__()
        padding =  0 if patch_size % 2 == 0 else patch_size // 2
        self.batch_norm = nn.BatchNorm2d(in_chans)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride=stride,
                              padding = padding , groups = in_chans if embed_dim % in_chans == 0 else 1)

        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        # pdb.set_trace()
        # print('data：', x.shape,x.device)
        # print('model：', next( self.proj.parameters()).device)
        x = self.batch_norm(x)
        x = self.proj(x)
        # ( N, in_chans, H, W ) -> ( N, embed_dim, H // stride, W // stride )

        N, out_chans, newH, newW = x.shape
        x = x.flatten(2).transpose(1, 2)
        # ( N, embed_dim,  H // stride, W // stride ) -> ( N,  H // stride *  W // stride , embed_dim)

        x = self.norm(x)

        return x, newH, newW  # H // stride, W // stride

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size = 224, patch_size = 4, in_chans =  3, embed_dim=96, norm_layer=None):
        super().__init__()

        # patch num in a row or column ,eg. 224/4 = 56
        # that's number of patches contained in each line
        patches_resolution = [img_size// patch_size, img_size // patch_size] # （56 ，56）
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution

        self.num_patches = patches_resolution[0] * patches_resolution[1] # 56 * 56
        # total path in a channel

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size)
        # 可以看到，这里使用的是patch_size的核 和 patch_size的步长来实现几个像素作为一个patch
        # 这样经过embedding之后，后续操作的对象就不是像素上的尺寸了，是直接 一 个 patch 做一个操作像素
        # 和VIT里边的不一样，VIT是直接物理上把图片按8*8划分，然后拉直进行后续操作，所以这里和VIT还不一样
        # 后来的ConvNext也是和这个操作一样。
        # (B , 3 , 224 ,224) - > (B , 96 , 56 ,56 )
        if norm_layer is not None:
            # default nn.LayerNorm
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C

        # ** flatten(2) , 2 means that start flatten dim is 2 , which is w and h **
        # input ( B , 3 , 224 , 224) -> ( B , 96 , 56 , 56) -> ( B , 56*56 , 96)

        if self.norm is not None:
            x = self.norm(x)
        return x , H//self.patch_size ,W//self.patch_size


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature. eg 56
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, current_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = current_dim
        self.reduction = nn.Linear(4 * current_dim, 2 * current_dim, bias=False)
        self.norm = norm_layer(4 * current_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution # eg 56
        B, L, C = x.shape # ( B , 56 * 56 , C )
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C) # eg ( B , 56 , 56 , C )
        '''
        assume x : 6 * 6 
                        *  *  *  *  *  * 
                        *  *  *  *  *  *
                        *  *  *  *  *  *
                        *  *  *  *  *  *
                        *  *  *  *  *  *
                        *  *  *  *  *  *  
        '''

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        '''
            x0 
                        +  *  +  *  +  * 
                        *  *  *  *  *  *
                        +  *  +  *  +  *
                        *  *  *  *  *  *
                        +  *  +  *  +  *
                        *  *  *  *  *  *   
        '''
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        '''
            x1 
                        *  *  *  *  *  * 
                        +  *  +  *  +  *
                        *  *  *  *  *  *
                        +  *  +  *  +  *
                        *  *  *  *  *  *
                        +  *  +  *  +  *  
        '''

        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        '''
            x2 
                         *  +  *  +  *  + 
                         *  *  *  *  *  *
                         *  +  *  +  *  +
                         *  *  *  *  *  *
                         *  +  *  +  *  +
                         *  *  *  *  *  *         
        '''
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        '''
            x3 
                        *  *  *  *  *  * 
                        *  +  *  +  *  +
                        *  *  *  *  *  *
                        *  +  *  +  *  +
                        *  *  *  *  *  *
                        *  +  *  +  *  +  
        '''
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x, H//2, W//2

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops



def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                      and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


############################################################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # the shape of out as same as input
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio )
            # size // sr_ratio
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):

        B, N, C = x.shape  # x shape with e.g. ( B,  h  *  w  , embed_dim)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # ( B , N , num_heads , C // self.num_heads) -> ( B ,num_heads ,  N , C // self.num_heads)

        if self.sr_ratio > 1:
            # 就是现在每个query不再与N个key做內积（查询），而是只与少量的Key(N / sr_ratio^2 ,当然对应value值也是这么多 )做內积
            # 但是query的个数不变，所以得到的还是N个sentence分别与那些key得到的权重，不过key的个数少一点。
            # 这里的思想和residual net 很像，用K个中心vector代表所有数据

            x_ = x.permute(0, 2, 1).reshape(B, C, H ,W )
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # sr : （ B , C  , H , W ) -> (B , C , h , w ) , where h = H // sr_ratio
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention_dec(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # learnerabled
        self.task_query = nn.Parameter(torch.randn(1, 16, dim))
        # here the input of decoder shape is ( B , image size // 64 , dim) , 16 = 256 // 64
        # also , you can define an arbitrary size for task query , because you can the interpolate
        # function to adapt it to correct size for decoder .
        # https://blog.csdn.net/weixin_47156261/article/details/116840741

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size = sr_ratio, stride = sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):

        B, N, C = x.shape  # (B,4*4,512)
        task_q = self.task_query

        # This is because we fix the task parameters to be of a certain dimension, so with varying batch size, we just stack up the same queries to operate on the entire batch
        if B > 1:
            task_q = task_q.unsqueeze(0).repeat(B, 1, 1, 1)
            task_q = task_q.squeeze(1)

        q = self.q(task_q).reshape(
            B, task_q.shape[1], self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # q = torch.nn.functional.interpolate(q, size=(v.shape[2], v.shape[3]))
        # https://blog.csdn.net/weixin_47156261/article/details/116840741

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Spatial_Attention(nn.Module):
    def __init__(self, embedding_dim , num_heads = 8, qkv_bias=False, qk_scale=None, attn_drop = 0., proj_drop=0.,sr_ratio=1):
        super().__init__()
        dim = embedding_dim
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):

        B, N, C = x.shape # here Actually is （B , embedding , H * W ）
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # ( B , N , num_heads , C // self.num_heads) -> ( B ,num_heads ,  N , C // self.num_heads)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block_dec(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention_dec(
            dim,
            num_heads = num_heads, qkv_bias = qkv_bias, qk_scale = qk_scale,
            attn_drop = attn_drop, proj_drop = drop, sr_ratio = sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # If do not specific the parameter out_features dim in mlp ,
        # it will be equals in_features dim


    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class TransformerSubBlock(nn.Module):
    # A transformer block has a series of TransformerSubBlock
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, mlpdrop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=mlpdrop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlpdrop)


    def forward(self, x, H, W):
        # x shape with e.g.  ( B,  h // stride *  w // stride , embed_dim)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class SpatialTransformerBlock(nn.Module):
    # A transformer block has a series of TransformerSubBlock
    def __init__(self, feature_size, in_channels ,num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, mlpdrop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.hidden_dim = 64
        dim = feature_size * feature_size # default 16 * 16
        self.norm1 = norm_layer(dim)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.batch_norm2 = nn.BatchNorm2d(self.hidden_dim)

        self.reduce = nn.Conv2d(in_channels = in_channels ,out_channels = self.hidden_dim,kernel_size=1)
        self.increase = nn.Conv2d(in_channels = self.hidden_dim ,out_channels = in_channels,kernel_size=1)
        self.attn = Spatial_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=mlpdrop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, H, W):

        B , N , C = x.shape
        assert N == H * W  , 'Have a error about the input size'
        # here x shape actually should be（B , embedding , H * W ） for spatial attention
        shortcut = x
        x = x.permute(0,2,1).reshape(B,C,H,W)
        x = self.reduce(self.batch_norm1(x)) # ( B , C ,H ,W ) -> ( B ,64 , H , W )
        x = x.reshape(B,64,H*W) # reshape to ( B , 64 , N ) ,then see N as embedding dim
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))# B , 64 , N
        x = x.reshape(B, 64, H , W)
        x = self.increase(self.batch_norm2(x)) # ( B ,64 , H , W ) ->  ( B , C ,H ,W )

        x = x.reshape(B,C,N).permute(0,2,1).contiguous() # B , N , C
        x = shortcut + x
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution: tuple, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, mlpdrop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        # input_resolution = [ 64 , 32 , 16 , 8 ]
        # Corresponding [ H/4 , H/8 , H/16 , H/32 ] ,H = 224
        self.num_heads = num_heads

        self.window_size = window_size  # 8
        self.shift_size = shift_size  # 0 if (i % 2 == 0) else window_size // 2
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=mlpdrop)
        '''
            attention 操作是一样的 , 只不过输入不一样,当shift_size > 0 时,会对输入的x进行roll操作
            具体实现见forward函数,另外,当shift_size > 0时,还要有相应的mask矩阵.
            在得到 Q@K^T 的矩阵之后 再把 mask 加到上边
        '''

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlpdrop)

        if self.shift_size > 0:
            # Need calculate attention mask for SW-MSA
            # see detail https://github.com/microsoft/Swin-Transformer/issues/38
            H, W = self.input_resolution  # default path size = 4 , image size = 224 ,then H = 56

            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            # e.g. shape with ( 1 , 56 , 56 ,1 )
            # default window_size = 7 , total 8*8 window
            # shift_size = window_size//2

            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            # Is equality : (slice(0, -7, None), slice(-7, -3, None), slice(-3, None, None))
            # slice(start, stop[, step])
            '''
            a = list(range(5))
             out : [0, 1, 2, 3, 4]
            b = slice(1,4,1)
             out : slice(1, 4, 1)
            a[b]
             out : [1, 2, 3]
            '''

            # 这里在给几个区域划分标号
            # 标号为 0,1,2,3,4,5,6,7,8 ,从左到右,从上到下
            cnt = 0
            for h in h_slices:
                '''
                    slice(0, -7, None)
                    slice(-7, -3, None)
                    slice(-3, None, None)
                '''
                for w in w_slices:
                    '''
                        slice(0, -7, None)
                        slice(-7, -3, None)
                        slice(-3, None, None)
                    '''
                    img_mask[:, h, w, :] = cnt  # 标号
                    # e.g. shape with ( 1 , 56 , 56 ,1 )
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            # nW, window_size, window_size, 1
            # ( total num of window , window_size , window_size ,1) , B = 1
            # ( 8 * 8 , 7 , 7 , 1 )
            # 这个地方就是 ‘1张’ image
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # ( 8 * 8 , 7 * 7 ）
            # 总共 8 * 8 个窗口， 每个窗口内 7 * 7 个 patch相互之间计算attention

            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # 广播减法
            # # ( 8 * 8 , 7 * 7 , 7 * 7）

            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            '''
            # 拉直 , 做广播减法 , 以右下角4个区域为例, 4 5 7 8
            >>> a = torch.Tensor([4,5,7,8])
            >>> b = a.reshape(-1,1)
            >>> a - b
            >>>           4    5    7    8
                tensor([[ 0.,  1.,  3.,  4.],   4
                        [-1.,  0.,  2.,  3.],   5
                        [-3., -2.,  0.,  1.],   7
                        [-4., -3., -1.,  0.]])  8
            # 可以看到,对角线为0,正好表示 区域4 和 区域4 要做attention ,
            # 而4 与其他位置不为0 ,不需要attention , 同理

            # 这里分了9个区域,实际上4个区域即可
            # See https://github.com/microsoft/Swin-Transformer/issues/194
            '''

        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        # self.attn_mask = nn.Parameter(attn_mask, requires_grad=False)

        self.fused_window_process = fused_window_process

    def forward(self, x, H, W):
        assert (H, W) == self.input_resolution, 'The input_resolution do not equals to current H , W '
        # H, W = self.input_resolution
        B, L, C = x.shape
        # eg. shape( B , 56*56 , 96) ,56 =  224 / 4 means that "input_resolution"
        # 4 means patch size ,which pixel num of a row of a patch

        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)  # eg , shape( B , 56 , 56 , 96)

        # cyclic shift
        if self.shift_size > 0:
            # in a block  , i for layer index , the shift_size =  0 if (i % 2 == 0) else window_size // 2
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                '''
                    torch.roll 
                    https://blog.csdn.net/weixin_42899627/article/details/116095067
                    >>> x  
                        tensor([[0, 1, 2],
                                [3, 4, 5],
                                [6, 7, 8]])
                    >>> shifted_x = torch.roll(x, shifts=(-1,-1),dims=(0,1))
                        tensor([[4, 5, 3],
                                [7, 8, 6],
                                [1, 2, 0]])   

                    the shifted attention visualization 
                        https://github.com/microsoft/Swin-Transformer/issues/38 
                '''
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)
                # input shape( B , 56 , 56 , 96)
                # # ( B * total num of window , window_size , window_size ,C)
                # e.g ( B * 8 * 8  , 7 , 7 , 96 ) for window size = 7 ,patch size = 4
                # then it have 56 * 56 patch , and a window contains 7*7patch

            else:
                # x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
                raise NotImplementedError
        else:
            # Do not slide
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)
            # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # nW*B, window_size*window_size, C
        # # ( B * total num of window , window_size * window_size ,C)
        # input ( B * 8 * 8  , 7 , 7 , 96 ) - > ( B * 8 * 8  , 7 * 7 , 96 )
        # then it have 56 * 56 patch , and a window contains 7*7 patch
        # shape like ( batch  , seq_len , dimension )

        # W-MSA/SW-MSA

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # nW*B, window_size*window_size, C
        # here nW = 8 * 8

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        # 因为前边为了计算attention方便,对shift之后的map进行了局部移位,现在移回去
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                # x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
                raise NotImplementedError
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size ， default  7

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # E.g. ( B , 56 , 56 , 96 )
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    '''
            B , 
            num patch of a raw within a window  , 
            window_size, 
            num patch of a raw within a window   , 
            window size , 
            C
    '''
    # (B , num patch of a raw within a window  , window_size, num patch of a raw within a window , window size , C )
    # ( B , 8 , 7 , 8 , 7 , 96)
    # ( 0 , 1 , 2 ,  3 ,  4 , 5 )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # ( B * total num of window , window_size , window_size ,C)
    # ( B * 8 * 8 , 7 , 7 , 96 )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)  # that is ( B * 8 * 8 , 7 , 7 , 96 )
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window. default 7
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww  default  7
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # sqrt(dim K = Q)

        # define a parameter table of relative position bias
        # see https://svainzhu.com/2022/02/Swin-T.html
        # 把每个patch看成一个像素 , M = 7
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        '''
            >>> coords_h = torch.arange(3)
            >>> coords_w = torch.arange(3)
            >>> coords_w
            out[0] : 
                    tensor([0, 1, 2])
            >>> torch.meshgrid([coords_h, coords_w])
            out[1] : 
                     (tensor([[0, 0, 0],
                              [1, 1, 1],
                              [2, 2, 2]]),
                      tensor([[0, 1, 2],
                              [0, 1, 2],
                              [0, 1, 2]]))
            >>> coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            out[2] : 
                    tensor([[[0, 0, 0],
                             [1, 1, 1],
                             [2, 2, 2]],
                            [[0, 1, 2],
                             [0, 1, 2],
                             [0, 1, 2]]])

        '''
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        '''
            >>> coords_flatten
            out[0]:
                    tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2],
                            [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        '''

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        '''
            # here coords_flatten[:, :, None] <=> coords_flatten.unsqueeze(-1)
            # so the " coords_flatten[:, :, None] "  is like :
                                tensor([[[0],
                                         [0],
                                         [0],
                                         [1],
                                         [1],
                                         [1],
                                         [2],
                                         [2],
                                         [2]],
                                        [[0],
                                         [1],
                                         [2],
                                         [0],
                                         [1],
                                         [2],
                                         [0],
                                         [1],
                                         [2]]])
            # so the " coords_flatten[:, None, : ] "  is like : 
                    tensor([[[0, 0, 0, 1, 1, 1, 2, 2, 2]],
                            [[0, 1, 2, 0, 1, 2, 0, 1, 2]]])

        '''
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 行标

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # 将得到的相对索引(x,y)合并变为一个新的索引 : x + y , 同时这个索引表不需要变动,注册为 buffer
        self.register_buffer("relative_position_index", relative_position_index)
        # self.relative_position_index = nn.Parameter(relative_position_index, requires_grad=False)
        #  =========================================================================================

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # for q , k , v
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)  # Dropout ratio of output. Default: 0.0

        trunc_normal_(self.relative_position_bias_table, std=.02)  # 将bias控制在0附近
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # shape like ( batch  , seq_len , dimension )
        # e.g ( B * 8 * 8  , 7 * 7 , 96 ) for window size = 7 ,patch size = 4
        # then it have 56 * 56 patch , and a window contains 7*7 patch , a patch have 4 * 4 pixel
        # so have 8 * 8 window

        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        # self.qkv(x) shape with ( B * 8 * 8  , 7 * 7 , 96 * 3 )

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # shape with ( B * 8 * 8 ,  3  , 7 * 7  ,   32 )

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # ( B_ ,  num_heads , N  , N)
        # ( B * 8 * 8 ,  3  , 7 * 7  , 7 * 7)
        # where N_ij is weights of patch_i and patch_j

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # Q*K^T + B
        # ( B_ ,  num_heads , N , N )
        # ( B * 8 * 8 ,  3  , 7 * 7  , 7 * 7)
        # where N_ij is weights of patch_i and patch_j

        if mask is not None:
            nW = mask.shape[0]

            # input shape with # ( 8 * 8 , 7 * 7 , 7 * 7 ）
            # (B_ // nW, nW, self.num_heads, N, N)  ： （ B , 8 * 8 , 3 , 7 * 7 , 7 * 7 )
            # mask.unsqueeze(1).unsqueeze(0) : torch.Size([1, 64, 1, 49, 49])
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)  # mask
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # (attn @ v) =
        #       ( B_ ,  3 , 56*56 , 56*56 ) * ( B_, 3 , 56 * 56 , 32 )
        #           = ( B_, 3 , 56 * 56 , 32 )
        # 再reshape , ( B_ , 56 * 56 , C )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DecoderSwTransformer(nn.Module):
    def __init__(self, img_size = 256 ,embed_dims=[64, 128, 256, 512],num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, mlpdrop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths

        # patch_embed
        # 这里不用看img_size，和这个参数半毛钱关系没有，只需要看里边具体干什么就行了
        # 这里stride=2 ，就是把尺寸减半，dim不变
        # 事实上，这里的输入img_size也不是原来的1/16，正常到这里应该是 8 = 256/32
        # self.patch_embed1 = OverlapPatchEmbed(img_size=img_size//16, patch_size=3, stride=2, in_chans=embed_dims[-1],
        #                                       embed_dim=embed_dims[-1])
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size // 32, patch_size=3, stride=2, in_chans=embed_dims[-1],
                                              embed_dim=embed_dims[-1])
        # transformer decoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block_dec(
            dim=embed_dims[-1], num_heads=num_heads[-1], mlp_ratio=mlp_ratios[-1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=mlpdrop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[-1])
            for i in range(depths[-1])])  # depths[-1] = 3
        self.norm1 = norm_layer(embed_dims[-1])
        self.active = nn.ReLU()
        cur += depths[-1]


    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        # x shape with [(B, 128, 32, 32) , (B, 320, 16, 16) , (B, 512, 8, 8) ,( B, 512, 8, 8) ]

        x = x[-1]

        # 可以看到仅仅是把encoder最后一层的输出(block4的输出）作为输入，输入到了decoder的Transformer里边，
        # 而其它层的输出则是作为feature输入到了后续的conv projection
        B = x.shape[0]

        # stage 1
        x, H, W = self.patch_embed1(x)
        shortcut = x
        # input shape (8, 512, 8, 8) -> out shape ( 8 , 4 * 4 ,1024 )

        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
            x = self.active(x)

        x = shortcut + self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # torch.Size([B, 1024, 4, 4])]

        return x  # outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class SwTenc(EncoderSwTransformer):
    def __init__(self, **kwargs):
        super(SwTenc, self).__init__(
            img_size = 256 ,embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16],
            mlp_ratios=[2, 2, 2, 2], qkv_bias = True, mlpdrop_rate = 0.1, attn_drop_rate = 0.1,
            drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 6, 2],sr_ratios=[4,2,2,1]
            ,input_resolution=[64, 32, 16, 8])


class SwTdec(DecoderSwTransformer):
    def __init__(self, **kwargs):
        super(SwTdec, self).__init__(
            embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios = [2, 2, 2, 2],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2,2,2,3], sr_ratios = [4,2,2,1],
            mlpdrop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)


class convprojection(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection, self).__init__()

        self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.dense_5 = nn.Sequential(ResidualBlock(512),ResidualBlock(512))
        self.convd16x = UpsampleConvLayer(512, 256, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(256),ResidualBlock(256))
        self.convd8x = UpsampleConvLayer(256 , 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128),ResidualBlock(128))

        # ***************** make convd4x output channel from 64 -> 128 *****************
        self.convd4x = UpsampleConvLayer(128 , 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64),ResidualBlock(64))

        self.convd2x = UpsampleConvLayer(64 , 32, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential(ResidualBlock(32),ResidualBlock(32))
        self.convd1x = UpsampleConvLayer(32 , 8, kernel_size=4, stride=2)
        self.conv_output = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)


    def forward(self, x1, x2):
        # x1 : list , shape with
        # [(B, 128, 64, 64)，(B, 256, 32, 32), (B, 512, 16, 16), (B, 1024, 8, 8), (B, 1024, 8, 8)]
        # come from encoder

        # x2 : shape with (B, 1024, 4, 4)
        # come from decoder ,thus it is actually equipment decoder(x1[-1])

        # 可以看到仅仅是把encoder最后一层的输出作为输入，输入到了decoder的Transformer里边，
        # 而其它层的输出则是作为feature输入到了后续的conv projection
        res32x0 = self.convd32x(x2)
        # (B, 1024, 8, 8)
        res32x = self.dense_5(res32x0) + x1[3]
        # res32x = self.dense_5(res32x)

        res16x0 = self.convd16x(res32x)
        # (8, 512, 16, 16)
        res16x = self.dense_4(res16x0) + x1[2]
        # res16x = self.dense_4(res16x)

        res8x0 = self.convd8x(res16x)  # output  [8, 256, 32, 32]
        res8x = self.dense_3(res8x0) + x1[1]
        # res8x = self.dense_3(res8x)

        # make convd4x output channel from 64 -> 128
        res4x0 = self.convd4x(res8x)  # [8, 128, 64, 64]
        res4x = self.dense_2(res4x0) + x1[0]
        # res4x = self.dense_2(res4x)

        res2x = self.convd2x(res4x)  # [8, 64, 128, 128]
        res2x = self.dense_1(res2x)

        x = self.convd1x(res2x)  # ( 8 , 32 ,256 ,256)
        x = self.conv_output(x)  # ( 8 , 3 , 256 ,256)
        # print(x.shape)

        return x, ( res4x0, res8x0, res16x0)


## The following is original network found in paper which solves all-weather removal problems
## using a single model

class SwingTransweather(nn.Module):

    def __init__(self, **kwargs):
        super(SwingTransweather, self).__init__()

        self.STenc = SwTenc()

        self.Tdec = SwTdec()

        self.convtail = convprojection()

        self.active = nn.Sigmoid()

    def forward(self, x):
        x1 = self.STenc(x)
        # list : shape with
        # (8, 128, 64, 64),(8, 256, 32, 32),(8, 512, 16, 16),(8, 1024, 8, 8),(8, 1024, 8, 8)
        # print(x1.device, x1.shape)
        x2 = self.Tdec(x1)  # shape with torch.Size([8, 1024, 4, 4])

        x, sw_fm = self.convtail(x1, x2)

        clean = self.active(x)

        return clean, sw_fm