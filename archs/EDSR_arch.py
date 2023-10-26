# import torch
# from torch import nn as nn
import paddle
import paddle.nn as nn
from models.archs.arch_util import  make_layer,ResidualBlockNoBN
import models.archs.arch_util as arch_util
# from models.archs.arch_util import ARCH_REGISTRY


# @ARCH_REGISTRY.register()
class EDSR(nn.Layer):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 in_nc,
                 out_nc,
                 num_feat=64,
                 num_block=16,
                 scale=4,
                 res_scale=1,
                 img_range=255.,
                 conv=arch_util.default_conv,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR, self).__init__()

        self.img_range = img_range
        # self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.mean =paddle.to_tensor(rgb_mean).reshape([1, 3, 1, 1])

        self.conv_first = nn.Conv2D(in_nc, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat,res_scale=res_scale)
        self.conv_after_body = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
        self.upsample = arch_util.Upsampler(conv,scale, num_feat)
        self.conv_last = nn.Conv2D(num_feat, out_nc, 3, 1, 1)

    def forward(self, x):

        # print(x.shape)
        self.mean = self.mean.astype(x.dtype)

        x = (x - self.mean) * self.img_range
        # print(x.shape)
        x = self.conv_first(x)
        # print(x.shape)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x