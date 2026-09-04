from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock, get_conv_layer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Union, Tuple


class FlexibleUnetUpBlock(nn.Module):
    """
    Added the option to set skip_channels.
    
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            skip_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip=None):
        out = self.transp_conv(inp)
        if skip is not None:
            out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out
    
    
class Encoder(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 1,
        filters: Sequence[int] = (16,32,64,128,256,312),
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        res_block: bool = True,
        spatial_dims: int = 3
        ):
        super().__init__()
        
        block = UnetResBlock if res_block else UnetBasicBlock
        
        num_blocks = len(filters)
        kernel_sizes = [3] * num_blocks
        strides = [1] + [2] * (num_blocks - 1)
        
        self.down_blocks = nn.ModuleList()
        
        last_out_channels = in_channels
        
        for i in range(num_blocks):
            
            this_out_channels = filters[i]
            self.down_blocks.append(
                block(  
                    spatial_dims=spatial_dims,
                    in_channels=last_out_channels,
                    out_channels=this_out_channels,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    norm_name=norm_name,
                    act_name=act_name
                )
            )
            last_out_channels = this_out_channels
            
    def forward(self, x):
        
        out_feats = []
        for block in self.down_blocks:
            x = block(x)
            out_feats.append(x)
        
        return out_feats
    
    
class Decoder(nn.Module):
    
    def __init__(
        self,
        out_channels: int = 1,
        skip_features: Sequence[int] = (312, 256, 128, 64, 32, 16),
        filters: Sequence[int] | None = None,
        no_skip_levels: int = 0, 
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        num_output_heads: int = 4,
        upsample_outputs: bool = True,
        spatial_dims: int = 3,
    ): 
        
        '''
        Correct len of skip features still required, but value of last no_skip_levels entries don't matter
        '''
        super().__init__()
        
        self.upsample_outputs = upsample_outputs
        
        if filters is None:
            filters = skip_features[1:]
        assert len(filters) == len(skip_features) - 1
        num_blocks = len(filters)
        self.num_blocks = num_blocks
        self.no_skip_levels = no_skip_levels
        
        self.up_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        
        last_out_channels = skip_features[0]
        for i in range(num_blocks):
            
            self.up_blocks.append(
                FlexibleUnetUpBlock(  
                    spatial_dims=spatial_dims,
                    in_channels=last_out_channels,
                    out_channels=filters[i],
                    skip_channels=skip_features[i+1] if i<num_blocks-no_skip_levels else 0,
                    upsample_kernel_size=2,
                    kernel_size=3,
                    stride=1, # Not used (upsample stride is same as upsample_kernel_size)
                    norm_name=norm_name,
                    act_name=act_name
                )
            )
            last_out_channels = filters[i]
            
            blocks_remaining = num_blocks - i
            if num_output_heads >= blocks_remaining:
                out_block = UnetOutBlock(
                        spatial_dims=spatial_dims,
                        in_channels=last_out_channels,
                        out_channels=out_channels
                    )
            else:
                out_block = None
            self.out_blocks.append(out_block)
            
    def forward(self, x):
        
        outs = []
        x = x[::-1]
        inp = x[0]
        
        for i, (skip, up_block, out_block) in enumerate(zip(x[1:], self.up_blocks, self.out_blocks)):
            if i<self.num_blocks-self.no_skip_levels:
                x_feat = up_block(inp, skip)
            else:
                x_feat = up_block(inp)
            
            if out_block:
                out_lr = out_block(x_feat)
                outs.append(out_lr)
                
            inp = x_feat
        
        if self.upsample_outputs:
            target_shape = outs[-1].shape[2:]
            for i, out_lr in enumerate(outs[:-1]):
                out_hr = F.interpolate(out_lr, size=target_shape, mode="nearest")
                outs[i] = out_hr
        
        return outs