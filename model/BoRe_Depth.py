import torch
from pytorch_lightning import LightningModule

from .DepthNet import DepthNet
from .PoseNet import PoseNet


class BoRe_Depth(LightningModule):
    def __init__(self):
        super(BoRe_Depth, self).__init__()

        # model
        self.depth_net = DepthNet()
        self.pose_net = PoseNet()
