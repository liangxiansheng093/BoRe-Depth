import torch
from pytorch_lightning import LightningModule

from .DepthNet import DepthNet
from .PoseNet import PoseNet
from .Seg_Encoder import Seg_Encoder


seg_model = Seg_Encoder()
seg_model.load_state_dict(torch.load("checkpoints/semantic.pth")["state_dict"], strict=False)


class BoRe_Depth(LightningModule):
    def __init__(self):
        super(BoRe_Depth, self).__init__()

        # model
        self.depth_net = DepthNet()
        self.pose_net = PoseNet()
