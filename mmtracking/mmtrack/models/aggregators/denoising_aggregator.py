import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from ..builder import AGGREGATORS
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d


class DenseLayer(nn.Module):
    """Dense layer

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.

    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c_in, h, w).

        Returns:
            Tensor: Forward results, tensor with shape (n, c_in+c_out, h, w).
        """
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    """Residual Dense Block of Residual Dense Network

    Args:
        in_channels (int): Channel number of inputs.
        channel_growth (int): Channels growth in each layer.
        num_layers (int): Layer number in the Residual Dense Block.
    """

    def __init__(self, in_channels, channel_growth, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[
            DenseLayer(in_channels + channel_growth * i, channel_growth)
            for i in range(num_layers)
        ])

        # local feature fusion
        self.lff = nn.Conv2d(
            in_channels + channel_growth * num_layers,
            in_channels,
            kernel_size=1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return x + self.lff(self.layers(x))  # local residual learning


class ModulatedDCNPack(ModulatedDeformConv2d):
    """Modulated Deformable Convolutional Pack.

    Different from the official DCN, which generates offsets and masks from
    the preceding features, this ModulatedDCNPack takes another different
    feature to generate masks and offsets.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset, val=0, bias=0)

    def forward(self, x, extra_feat):
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class TemporalAttentionFusion(nn.Module):
    """
    Temporal attention fusion.
    """
    def __init__(self, channels, mid_channels, emb_nums=3):
        super().__init__()
        self.channels = channels
        self.mid_channels = mid_channels
        self.emb_nums = emb_nums

        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1)

        self.offset_conv = nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, padding=1)
        self.dcn_pack = ModulatedDCNPack(mid_channels, mid_channels, 3, padding=1, deform_groups=8)
        self.emb_conv = nn.Sequential(*[nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
                                        for _ in range(emb_nums)])

        self.conv2 = nn.Conv2d(mid_channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))

        feat = []
        for i in range(x.shape[0]):
            x_ref = x[[i]]
            x_ref = x_ref.repeat(x.shape[0], 1, 1, 1)
            x_cat = torch.cat([x, x_ref], dim=1)
            x_set = self.offset_conv(x_cat)
            x_dcn = self.dcn_pack(x, x_set)
            x_cor = self.emb_conv(x_dcn * x_ref)
            x_cor = torch.softmax(x_cor, dim=0)
            x_fus = torch.sum(x_cor * x, dim=0, keepdim=True)
            feat.append(x_fus)
        out = torch.cat(feat, dim=0)

        out = self.relu(self.conv2(out))
        return out


@AGGREGATORS.register_module()
class DenoisingAggregator(nn.Module):
    """
    For denoising aggregator.
    """
    def __init__(self,
                 in_channels=64,
                 mid_channels=64,
                 num_blocks=1,
                 num_layers=6,
                 channel_growth=64,
                 emb_nums=3,
                 with_rdb=True,
                 with_taf=True,):

        super().__init__()
        self.with_rdb = with_rdb
        self.with_taf = with_taf

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        if self.with_rdb:
            self.rdbs = nn.Sequential(
                *[RDB(mid_channels, channel_growth, num_layers) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        if self.with_taf:
            self.taf = TemporalAttentionFusion(mid_channels,
                                               mid_channels=mid_channels,
                                               emb_nums=emb_nums)
        self.conv3 = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.conv1(x))
        x_f = x
        if self.with_rdb:
            x = self.rdbs(x)
        x = self.relu(self.conv2(x + x_f))
        x_m = x
        if self.with_taf:
            x = self.taf(x)
        x = self.relu(self.conv3(x + x_m))
        out = x + identity
        return out
