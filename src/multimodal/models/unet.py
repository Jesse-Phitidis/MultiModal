import torch.nn as nn
from typing import Sequence, Union, Tuple
from multimodal.models.blocks import Encoder, Decoder


class UNet(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        filters: Sequence[int] = (16,32,64,128,256,312),
        filters_decoder: Sequence[int] | None = None,
        no_skip_levels: int = 0,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        res_block: bool = True,
        num_output_heads: int = 4,
        upsample_outputs: bool = True,
        spatial_dims: int = 3,
    ):
        super().__init__()
        
        self.encoder = Encoder(
            in_channels=in_channels,
            filters=filters,
            norm_name=norm_name,
            act_name=act_name,
            res_block=res_block,
            spatial_dims=spatial_dims
        )
        
        self.decoder = Decoder(
            out_channels=out_channels,
            skip_features=filters[::-1],
            filters=filters_decoder,
            no_skip_levels=no_skip_levels,
            norm_name=norm_name,
            act_name=act_name,
            num_output_heads=num_output_heads,
            upsample_outputs=upsample_outputs,
            spatial_dims=spatial_dims
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))