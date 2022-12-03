
# --- Imports --- #
import torch
import torch.nn.functional as F


# --- Perceptual loss network  --- #


# class LossNetwork(torch.nn.Module):
#     def __init__(self, vgg_model):
#         super(LossNetwork, self).__init__()
#         self.vgg_layers = vgg_model
#         self.layer_name_mapping = {
#             '3': "relu1_2",
#             '8': "relu2_2",
#             '15': "relu3_3"
#         }
#         self.mse_loss = F.mse_loss
#
#     def output_features(self, x):
#         output = {}
#         for name, module in self.vgg_layers._modules.items():
#             x = module(x)
#             if name in self.layer_name_mapping:
#                 output[self.layer_name_mapping[name]] = x
#         return list(output.values())
#
#     def forward(self, pred_im, gt):
#         # Denoised image ( B,3,256,256 )
#         # Ground True ( B,3,256,256 )
#         loss = []
#         pred_im_features = self.output_features(pred_im)
#         gt_features = self.output_features(gt)
#         for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
#             loss.append(self.mse_loss(pred_im_feature, gt_feature))
#
#         return sum(loss)/len(loss)



# --- Perceptual loss network  --- #


class LossNetwork(torch.nn.Module):
    def __init__(self, convnext_model):
        super(LossNetwork, self).__init__()
        self.convnext_layers = convnext_model
        self.layer_name_mapping = [3,8,15] # 目前只要3 5 7
        self.mse_loss = F.mse_loss

    def output_features(self, x):
        output = []
        for name, module in self.convnext_layers._modules.items():
            x = module(x)
            if int(name) in self.layer_name_mapping:
                output.append(x)

        return output

    def forward(self, pr , gt , sw_fm ):
        # Denoised image ( B,3,256,256 )
        # Ground True ( B,3,256,256 )
        loss1 = []
        loss2 = []
        conv_fm_gt = self.output_features(gt)
        conv_fm_pr = self.output_features(pr)

        # for sw_feature, gt_feature in zip(sw_fm, conv_fm_gt):
        #     loss1.append(self.mse_loss(sw_feature, gt_feature))
        for pr_feature, gt_feature in zip(conv_fm_pr, conv_fm_gt):
            loss2.append(self.mse_loss(pr_feature, gt_feature))

        return  sum(loss2)/len(loss2) # sum(loss1)/len(loss1) +