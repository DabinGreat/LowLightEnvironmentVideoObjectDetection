import torch.nn as nn
from mmdet.models.backbones.resnet import Bottleneck, ResNet
from ..builder import CLEANER


@CLEANER.register_module()
class ResCleaner(ResNet):
    """
    ResNet cleaner for feature extracting to supervise noise feature.

    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
    }

    def __init__(self,
                 **kwargs):
        super(ResCleaner, self).__init__(**kwargs)


@CLEANER.register_module()
class ResRAWCleaner(ResNet):
    """
    ResNet cleaner for feature extracting to supervise noise feature.

    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
    }

    def __init__(self,
                 **kwargs):
        super(ResRAWCleaner, self).__init__(in_channels=4, **kwargs)