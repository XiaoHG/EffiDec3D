from torch.cuda.amp import autocast, GradScaler

from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.networks.nets import UNETR, SwinUNETR
from networks.swin_unetr_effidec3d import SwinUNETR as SwinUNETRv2
from networks.MedNeXt.mednextv1.create_mednext_v1 import create_mednext_v1
from networks.UXNet_3D.network_backbone import UXNET, UXNET_EffiDec3D
from networks.swin_unetr_effidec3d import SwinUNETR_EffiDec3D
from networks.MedNeXt.mednextv1.create_mednextv1_effidec3d import create_mednextv1_effidec3d

from networks.unetr_pp.synapse.unetr_pp_synapse import UNETR_PP
from networks.nnunet.network_architecture.generic_UNet import Generic_UNet
from networks.SegFormer3D.segformer3d import SegFormer3D 
from networks.SlimUNETR.SlimUNETR import SlimUNETR
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceCELoss
from monai_utils.inferers.utils import sliding_window_inference_1out
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    Activations,
    )

import torch
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms import data_loader, data_transforms
from monai.networks.blocks import UnetOutBlock
import torch.nn as nn
import torch.nn.functional as F

from ptflops import get_model_complexity_info

import csv
import os
import numpy as np
import scipy.ndimage as ndimage
from medpy import metric
from tqdm import tqdm
import argparse

def test_integration():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UXNET_EffiDec3D(
        in_chans=1, out_chans=2,
        feat_size=[48, 96, 192, 384],
        n_decoder_channels=48,
        resolution_factor=2,
        use_hu_attention=True
    ).to(device)
    x = torch.randn(1, 1, 128, 128, 64).to(device)
    out = model(x)
    print(f"Output shape: {out.shape}")
    loss = out.mean()
    loss.backward()
    print("✓ 集成成功，梯度流正常")

if __name__ == "__main__":
    test_integration()