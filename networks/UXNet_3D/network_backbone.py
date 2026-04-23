#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:15:47 2024

@author: Md Mostafijur Rahman
"""

import sys
from typing import Tuple
import numpy as np
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from networks.UXNet_3D.uxnet_encoder import uxnet_conv

from ip_units.hugca import HUDensityCrossAttention

import logging
logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class ModifiedUnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size,
        upsample_kernel_size ,
        norm_name,
        res_block = False,
        skip_aggregation = 'concatenation', 
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.
            skip_aggregation: type of skip aggregation, addition or concatenation 
        """

        super().__init__()
        self.skip_aggregation = skip_aggregation
        in_out_channels = out_channels
        if self.skip_aggregation =='concatenation':
            in_out_channels = out_channels + out_channels
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                in_out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                in_out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)

        # xiaohg
        if skip is None:
            return out

        if self.skip_aggregation=='concatenation':
            out = torch.cat((out, skip), dim=1)
        else:
            out = out + skip
        out = self.conv_block(out)
        return out

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


# class ResBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self,
#                  in_planes: int,
#                  planes: int,
#                  spatial_dims: int = 3,
#                  stride: int = 1,
#                  downsample: Union[nn.Module, partial, None] = None,
#     ) -> None:
#         """
#         Args:
#             in_planes: number of input channels.
#             planes: number of output channels.
#             spatial_dims: number of spatial dimensions of the input image.
#             stride: stride to use for first conv layer.
#             downsample: which downsample layer to use.
#         """
#
#         super().__init__()
#
#         conv_type: Callable = Conv[Conv.CONV, spatial_dims]
#         norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
#
#         self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
#         self.bn1 = norm_type(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = norm_type(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         residual = x
#
#         out: torch.Tensor = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


class UXNET(nn.Module):

    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        # self.classification = False
        # self.vit = ViT(
        #     in_channels=in_channels,
        #     img_size=img_size,
        #     patch_size=self.patch_size,
        #     hidden_size=hidden_size,
        #     mlp_dim=mlp_dim,
        #     num_layers=self.num_layers,
        #     num_heads=num_heads,
        #     pos_embed=pos_embed,
        #     classification=self.classification,
        #     dropout_rate=dropout_rate,
        #     spatial_dims=spatial_dims,
        # )
        self.uxnet_3d = uxnet_conv(
            in_chans= self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    
    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        #print([outs[0].shape,outs[1].shape,outs[2].shape,outs[3].shape])
        # print(outs[0].size())
        # print(outs[1].size())
        # print(outs[2].size())
        # print(outs[3].size())
        enc1 = self.encoder1(x_in)
        #print('enc1:', enc1.size())
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        #print('enc2:', enc2.size())
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        #print('enc3:', enc3.size())
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        #print('enc4:', enc4.size())
        # dec4 = self.proj_feat(outs[3], self.hidden_size, self.feat_size)
        enc_hidden = self.encoder5(outs[3])
        #print('enc_hidden:', enc_hidden.size())
        dec3 = self.decoder5(enc_hidden, enc4)
        #print('dec3:', dec3.size())
        dec2 = self.decoder4(dec3, enc3)
        #print('dec2:', dec2.size())
        dec1 = self.decoder3(dec2, enc2)
        #print('dec1:', dec1.size())
        dec0 = self.decoder2(dec1, enc1)
        #print('dec0:', dec0.size())
        out = self.decoder1(dec0)
        #print('out:', out.size())     
        # feat = self.conv_proj(dec4)
        
        return self.out(out)

# class UXNET_EffiDec3D(nn.Module):

#     def __init__(
#         self,
#         in_chans=1,
#         out_chans=13,
#         depths=[2, 2, 2, 2],
#         feat_size=[48, 96, 192, 384],
#         n_decoder_channels=48,
#         drop_path_rate=0,
#         layer_scale_init_value=1e-6,
#         hidden_size: int = 768,
#         norm_name: Union[Tuple, str] = "instance",
#         conv_block: bool = True,
#         res_block: bool = True,
#         skip_aggregation: str = 'concatenation',
#         resolution_factor: int = 2,
#         spatial_dims=3
#     ) -> None:
#         """
#         Args:
#             in_channels: dimension of input channels.
#             out_channels: dimension of output channels.
#             img_size: dimension of input image.
#             feature_size: dimension of network feature size.
#             hidden_size: dimension of hidden layer.
#             mlp_dim: dimension of feedforward layer.
#             num_heads: number of attention heads.
#             pos_embed: position embedding layer type.
#             norm_name: feature normalization type and arguments.
#             conv_block: bool argument to determine if convolutional block is used.
#             res_block: bool argument to determine if residual block is used.
#             dropout_rate: faction of the input units to drop.
#             spatial_dims: number of spatial dims.

#         """

#         super().__init__()

#         # in_channels: int,
#         # out_channels: int,
#         # img_size: Union[Sequence[int], int],
#         # feature_size: int = 16,
#         # if not (0 <= dropout_rate <= 1):
#         #     raise ValueError("dropout_rate should be between 0 and 1.")
#         #
#         # if hidden_size % num_heads != 0:
#         #     raise ValueError("hidden_size should be divisible by num_heads.")
#         self.hidden_size = hidden_size
#         # self.feature_size = feature_size
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.depths = depths
#         self.drop_path_rate = drop_path_rate
#         self.feat_size = feat_size
#         self.n_decoder_channels = n_decoder_channels
#         self.resolution_factor = resolution_factor
#         self.cls_head_in_channels = n_decoder_channels
#         self.n_channels_enc2_dec3 = min(self.n_decoder_channels,self.feat_size[0])
#         self.layer_scale_init_value = layer_scale_init_value
#         self.out_indice = []
#         for i in range(len(self.feat_size)):
#             self.out_indice.append(i)

#         self.spatial_dims = spatial_dims

#         # self.classification = False
#         # self.vit = ViT(
#         #     in_channels=in_channels,
#         #     img_size=img_size,
#         #     patch_size=self.patch_size,
#         #     hidden_size=hidden_size,
#         #     mlp_dim=mlp_dim,
#         #     num_layers=self.num_layers,
#         #     num_heads=num_heads,
#         #     pos_embed=pos_embed,
#         #     classification=self.classification,
#         #     dropout_rate=dropout_rate,
#         #     spatial_dims=spatial_dims,
#         # )
#         self.uxnet_3d = uxnet_conv(
#             in_chans= self.in_chans,
#             depths=self.depths,
#             dims=self.feat_size,
#             drop_path_rate=self.drop_path_rate,
#             layer_scale_init_value=1e-6,
#             out_indices=self.out_indice
#         )
#         if self.resolution_factor <= 1:
#             self.encoder1 = UnetrBasicBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.in_chans,
#                 out_channels=self.n_channels_enc2_dec3,
#                 kernel_size=3,
#                 stride=1,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#         if self.resolution_factor <= 2:
#             self.encoder2 = UnetrBasicBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[0],
#                 out_channels=self.n_channels_enc2_dec3,
#                 kernel_size=3,
#                 stride=1,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#         if self.resolution_factor <= 4:
#             self.encoder3 = UnetrBasicBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[1],
#                 out_channels=self.n_decoder_channels,
#                 kernel_size=3,
#                 stride=1,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#         if self.resolution_factor <= 8:
#             self.encoder4 = UnetrBasicBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[2],
#                 out_channels=self.n_decoder_channels,
#                 kernel_size=3,
#                 stride=1,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#         if self.resolution_factor <= 16:
#             self.encoder5 = UnetrBasicBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[3],
#                 out_channels=self.n_decoder_channels,
#                 kernel_size=3,
#                 stride=1,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#             self.cls_head_in_channels = self.n_decoder_channels
#         if self.resolution_factor <= 8:
#             self.decoder5 = ModifiedUnetrUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.n_decoder_channels,
#                 out_channels=self.n_decoder_channels,
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#                 skip_aggregation=skip_aggregation,
#             )
#             self.cls_head_in_channels = self.n_decoder_channels
#         if self.resolution_factor <= 4:
#             self.decoder4 = ModifiedUnetrUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.n_decoder_channels,
#                 out_channels=self.n_decoder_channels,
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#                 skip_aggregation=skip_aggregation,
#             )
#             self.cls_head_in_channels = self.n_decoder_channels
#         if self.resolution_factor <= 2:
#             self.decoder3 = ModifiedUnetrUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.n_decoder_channels,
#                 out_channels=self.n_channels_enc2_dec3,
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#                 skip_aggregation=skip_aggregation
#             )
#             self.cls_head_in_channels = self.n_channels_enc2_dec3
#         if self.resolution_factor <= 1:
#             self.decoder2 = ModifiedUnetrUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.n_channels_enc2_dec3,
#                 out_channels=self.n_channels_enc2_dec3,
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#                 skip_aggregation=skip_aggregation
#             )
#             self.decoder1 = UnetrBasicBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.n_channels_enc2_dec3,
#                 out_channels=self.n_channels_enc2_dec3,
#                 kernel_size=3,
#                 stride=1,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#             self.cls_head_in_channels = self.n_channels_enc2_dec3
#         self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.cls_head_in_channels, out_channels=self.out_chans)
#         # self.conv_proj = ProjectionHead(dim_in=hidden_size)

#     def proj_feat(self, x, hidden_size, feat_size):
#         new_view = (x.size(0), *feat_size, hidden_size)
#         x = x.view(new_view)
#         new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
#         x = x.permute(new_axes).contiguous()
#         return x
    
#     def forward(self, x_in):

#         # Check for invalid resolution_factor
#         if self.resolution_factor > 16:
#             print("Invalid resolution_factor for this model. Must be <= 16.")
#             return sys.exit() 

#         outs = self.uxnet_3d(x_in)
#         #print([outs[0].shape,outs[1].shape,outs[2].shape,outs[3].shape])
#         # print(outs[0].size())
#         # print(outs[1].size())
#         # print(outs[2].size())
#         # print(outs[3].size())
#         # enc1 = self.encoder1(x_in)
#         # print(enc1.size())
#         # Encoder Pass
#         enc1, enc2, enc3, enc4, enc_hidden, result = None, None, None, None, None, None

#         if self.resolution_factor <= 1:
#             enc1 = self.encoder1(x_in)
#         if self.resolution_factor <= 2:
#             x2 = outs[0]  # Highest resolution from backbone
#             enc2 = self.encoder2(x2) if hasattr(self, 'encoder2') else x2
#         if self.resolution_factor <= 4:
#             x3 = outs[1]
#             enc3 = self.encoder3(x3) if hasattr(self, 'encoder3') else x3
#         if self.resolution_factor <= 8:
#             x4 = outs[2]
#             enc4 = self.encoder4(x4) if hasattr(self, 'encoder4') else x4
#         if self.resolution_factor <= 16:
#             enc_hidden = self.encoder5(outs[3]) 
#             result = enc_hidden  # Temporary variable to store the output result

#         # Decoder Pass (start from 8x resolution)

#         if self.resolution_factor <= 8:
#             dec3 = self.decoder5(enc_hidden, enc4 if hasattr(self, 'encoder4') else None, x_in)
#             result = dec3
#         if self.resolution_factor <= 4:
#             dec2 = self.decoder4(dec3, enc3 if hasattr(self, 'encoder3') else None, x_in)
#             result = dec2
#         if self.resolution_factor <= 2:
#             dec1 = self.decoder3(dec2, enc2 if hasattr(self, 'encoder2') else None, x_in)
#             result = dec1
#         if self.resolution_factor <= 1:
#             dec0 = self.decoder2(dec1, enc1 if hasattr(self, 'encoder1') else None, x_in)
#             result = self.decoder1(dec0)

#         ## feat = self.conv_proj(dec4)
#         # Return the final result, passed through the output layer
#         return self.out(result)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

# 假设您的其他组件（uxnet_conv, UnetrBasicBlock, UnetOutBlock）已存在
# 并且 ModifiedUnetrUpBlock 可按需修改或保留原样（我们这里采用跳过内部融合的方式）

class UXNET_EffiDec3D(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        n_decoder_channels=48,
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        skip_aggregation: str = 'concatenation',  # 仅用于原始路径，当use_hu_attention=False时生效
        resolution_factor: int = 2,
        spatial_dims=3,
        use_hu_attention: bool = True,   # 是否启用HU引导注意力
        hu_attn_num_heads: int = 4,
    ) -> None:
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.feat_size = feat_size
        self.n_decoder_channels = n_decoder_channels
        self.resolution_factor = resolution_factor
        self.n_channels_enc2_dec3 = min(self.n_decoder_channels, self.feat_size[0])
        self.use_hu_attention = use_hu_attention

        self.out_indice = list(range(len(self.feat_size)))

        # 编码器主干
        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            out_indices=self.out_indice
        )

        # 编码器各层后处理块
        if self.resolution_factor <= 1:
            self.encoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.in_chans,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            )
        if self.resolution_factor <= 2:
            self.encoder2 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[0],
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            )
        if self.resolution_factor <= 4:
            self.encoder3 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.n_decoder_channels,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            )
        if self.resolution_factor <= 8:
            self.encoder4 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.n_decoder_channels,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            )
        if self.resolution_factor <= 16:
            self.encoder5 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[3],
                out_channels=self.n_decoder_channels,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            )

        # 解码器上采样层（仅包含上采样操作，不融合）
        self.upsample5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) if resolution_factor <= 8 else None
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) if resolution_factor <= 4 else None
        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) if resolution_factor <= 2 else None
        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) if resolution_factor <= 1 else None

        # 解码器融合后的卷积块（替代原 ModifiedUnetrUpBlock 中的卷积部分）
        if self.resolution_factor <= 8:
            self.conv5 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_decoder_channels * 2 if skip_aggregation == 'concatenation' else self.n_decoder_channels,
                out_channels=self.n_decoder_channels,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            ) if not use_hu_attention else UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_decoder_channels,  # 注意力输出通道数不变
                out_channels=self.n_decoder_channels,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            )
        if self.resolution_factor <= 4:
            self.conv4 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_decoder_channels * 2 if skip_aggregation == 'concatenation' else self.n_decoder_channels,
                out_channels=self.n_decoder_channels,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            ) if not use_hu_attention else UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_decoder_channels,
                out_channels=self.n_decoder_channels,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            )
        if self.resolution_factor <= 2:
            self.conv3 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_decoder_channels * 2 if skip_aggregation == 'concatenation' else self.n_decoder_channels,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            ) if not use_hu_attention else UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_decoder_channels,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            )
        if self.resolution_factor <= 1:
            self.conv2 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_channels_enc2_dec3 * 2 if skip_aggregation == 'concatenation' else self.n_channels_enc2_dec3,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            ) if not use_hu_attention else UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_channels_enc2_dec3,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            )
            self.decoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_channels_enc2_dec3,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3, stride=1,
                norm_name=norm_name, res_block=res_block,
            )

        # 新增：HU引导交叉注意力模块
        if self.use_hu_attention:
            self.hu_attns = nn.ModuleDict()
            ch_deep = self.n_decoder_channels
            ch_shallow = self.n_channels_enc2_dec3
            if self.resolution_factor <= 16:
                self.hu_attns['lvl5'] = HUDensityCrossAttention(dim=ch_deep, num_heads=hu_attn_num_heads)
            if self.resolution_factor <= 8:
                self.hu_attns['lvl4'] = HUDensityCrossAttention(dim=ch_deep, num_heads=hu_attn_num_heads)
            if self.resolution_factor <= 4:
                self.hu_attns['lvl3'] = HUDensityCrossAttention(dim=ch_deep, num_heads=hu_attn_num_heads)
            if self.resolution_factor <= 2:
                self.hu_attns['lvl2'] = HUDensityCrossAttention(dim=ch_shallow, num_heads=hu_attn_num_heads)
            # lvl1 通常没有跳跃连接，不添加

        # 最终输出层
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=self.n_channels_enc2_dec3 if self.resolution_factor <= 1 else self.n_decoder_channels,
            out_channels=self.out_chans
        )

    def _get_ct_down(self, ct_input, target_feat):
        """将输入CT图下采样至目标特征图空间尺寸"""
        return F.interpolate(ct_input, size=target_feat.shape[2:], mode='trilinear', align_corners=False)

    def forward(self, x_in):
        # x_in: (B, 1, H, W, D) 归一化到[0,1]
        outs = self.uxnet_3d(x_in)

        enc1 = enc2 = enc3 = enc4 = enc_hidden = None
        if self.resolution_factor <= 1:
            enc1 = self.encoder1(x_in)
        if self.resolution_factor <= 2:
            enc2 = self.encoder2(outs[0])
        if self.resolution_factor <= 4:
            enc3 = self.encoder3(outs[1])
        if self.resolution_factor <= 8:
            enc4 = self.encoder4(outs[2])
        if self.resolution_factor <= 16:
            enc_hidden = self.encoder5(outs[3])

        x = enc_hidden

        # ----- lvl5 -----
        if self.resolution_factor <= 8:
            # 上采样（如果有）
            if self.upsample5 is not None:
                x = self.upsample5(x)
            # 此时 x 空间尺寸与 enc4 相同
            if self.use_hu_attention and 'lvl5' in self.hu_attns and enc4 is not None:
                ct_down = self._get_ct_down(x_in, x)   # 与 x, enc4 同尺寸
                x = self.hu_attns['lvl5'](x, enc4, ct_down)
            else:
                # 原始融合方式
                if enc4 is not None:
                    x = torch.cat([x, enc4], dim=1) if skip_aggregation == 'concatenation' else x + enc4
            x = self.conv5(x)

        # ----- lvl4 -----
        if self.resolution_factor <= 4:
            if self.upsample4 is not None:
                x = self.upsample4(x)
            if self.use_hu_attention and 'lvl4' in self.hu_attns and enc3 is not None:
                ct_down = self._get_ct_down(x_in, x)
                x = self.hu_attns['lvl4'](x, enc3, ct_down)
            else:
                if enc3 is not None:
                    x = torch.cat([x, enc3], dim=1) if skip_aggregation == 'concatenation' else x + enc3
            x = self.conv4(x)

        # ----- lvl3 -----
        if self.resolution_factor <= 2:
            if self.upsample3 is not None:
                x = self.upsample3(x)
            if self.use_hu_attention and 'lvl3' in self.hu_attns and enc2 is not None:
                ct_down = self._get_ct_down(x_in, x)
                x = self.hu_attns['lvl3'](x, enc2, ct_down)
            else:
                if enc2 is not None:
                    x = torch.cat([x, enc2], dim=1) if skip_aggregation == 'concatenation' else x + enc2
            x = self.conv3(x)

        # ----- lvl2 -----
        if self.resolution_factor <= 1:
            if self.upsample2 is not None:
                x = self.upsample2(x)
            if self.use_hu_attention and 'lvl2' in self.hu_attns and enc1 is not None:
                ct_down = self._get_ct_down(x_in, x)
                x = self.hu_attns['lvl2'](x, enc1, ct_down)
            else:
                if enc1 is not None:
                    x = torch.cat([x, enc1], dim=1) if skip_aggregation == 'concatenation' else x + enc1
            x = self.conv2(x)
            x = self.decoder1(x)

        return self.out(x)