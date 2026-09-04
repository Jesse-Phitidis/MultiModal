import torch
import torch.nn as nn
from typing import Sequence, Union, Tuple
from multimodal.models.blocks import Encoder, Decoder

class HeMIS(nn.Module):
    
    def __init__(
        self,
        num_encoders: int,
        encoder_in_channels: int = 1,
        encoder_filters: Sequence[int] = (16,32,64,128,256,312),
        encoder_res_block: bool = True,
        decoder_out_channels: int = 1,
        decoder_filters: Sequence[int] | None = None,
        decoder_num_output_heads: int = 4,
        decoder_upsample_outputs: bool = True,
        unet_norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        unet_act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        spatial_dims: int = 3
    ):
        super().__init__()
        
        self.encoder_filters = encoder_filters
            
        self.encoders = nn.ModuleList()
        for _ in range(num_encoders):
            self.encoders.append(
                Encoder(
                    in_channels=encoder_in_channels,
                    filters=encoder_filters,
                    norm_name=unet_norm_name,
                    act_name=unet_act_name,
                    res_block=encoder_res_block,
                    spatial_dims=spatial_dims
                )
            )
        
        # HeMIS concatenates Mean and Variance feature maps, so skip features are doubled
        hemis_skip_features = [f * 2 for f in encoder_filters[::-1]]
        
        self.decoder = Decoder(
            out_channels=decoder_out_channels,
            skip_features=hemis_skip_features,
            filters=decoder_filters,
            no_skip_levels=0, 
            norm_name=unet_norm_name,
            act_name=unet_act_name,
            num_output_heads=decoder_num_output_heads,
            upsample_outputs=decoder_upsample_outputs,
            spatial_dims=spatial_dims   
        )
        
        
    def forward(self, x):
        
        # Calculate which channels are present (nonzero) to act as our mask
        spatial_input_shape = [i for i in x.shape[2:]]
        dims = [2 + i for i in range(len(spatial_input_shape))]
        is_nonzero = x.abs().sum(dim=dims) > 0 # [B, C]
        
        # Encode each channel of x if not empty
        x_feats = self.encode_nonzero_channels(x, is_nonzero, spatial_input_shape) # list[list]
            
        # Get features for each resolution level
        x_feats_T = [[sublist[i] for sublist in x_feats] for i in range(len(x_feats[0]))] # encoders[resolutions] -> resolutions[encoders]
        
        # Prepare availability counts for moment calculations
        num_avail = is_nonzero.sum(dim=1) # [B]
        num_avail_safe = torch.clamp(num_avail, min=1)
        var_denom = torch.clamp(num_avail - 1, min=1)
        
        final_feats = []
        for feats_at_res in x_feats_T:
            feats_at_res = torch.stack(feats_at_res, dim=1) # [B, num_encoders, f, H/x, W/x, D/x]
            
            # Reshape masks and counts for broadcasting
            res_dims = len(feats_at_res.shape[2:]) # 4 (feature dim + spatial dims)
            
            # Mask broadcasts against [B, C, f, H/x, W/x, D/x] -> Needs 6 dims: [B, C, 1, 1, 1, 1]
            bcast_mask = is_nonzero.view(x.shape[0], x.shape[1], *([1] * res_dims))
            
            # Moment counts broadcast against [B, f, H/x, W/x, D/x] -> Needs 5 dims: [B, 1, 1, 1, 1]
            bcast_avail_safe = num_avail_safe.view(x.shape[0], *([1] * res_dims))
            bcast_var_denom = var_denom.view(x.shape[0], *([1] * res_dims))
            bcast_num_avail = num_avail.view(x.shape[0], *([1] * res_dims))
            
            # 1. Mean calculation
            feats_masked = feats_at_res * bcast_mask 
            mean_feats = feats_masked.sum(dim=1) / bcast_avail_safe # [B, f, H/x, W/x, D/x]
            
            # 2. Variance calculation
            # Subtract mean ONLY from present modalities
            diff = (feats_at_res - mean_feats.unsqueeze(1)) * bcast_mask
            sq_diff = diff ** 2
            var_feats = sq_diff.sum(dim=1) / bcast_var_denom # [B, f, H/x, W/x, D/x]
            
            # Variance is zero if only 1 (or 0) modalities are present
            var_feats = torch.where(bcast_num_avail > 1, var_feats, torch.zeros_like(var_feats))
            
            # 3. Concatenate mean and variance along the feature dimension
            hemis_feats = torch.cat([mean_feats, var_feats], dim=1) # [B, 2*f, H/x, W/x, D/x]
            final_feats.append(hemis_feats)
        
        # Decode
        out = self.decoder(final_feats)
        
        return out
            
    def encode_nonzero_channels(self, x, is_nonzero, spatial_input_shape):
        B, C = x.shape[0], x.shape[1]
        encoder_out_shapes = [
            (B, feats, *[s//(2**i) for s in spatial_input_shape]) for i, feats in enumerate(self.encoder_filters)
            ]

        x_out_lst = []
        for i in range(C):
            x_c = x[:,i:i+1]
            is_nonzero_c = is_nonzero[:,i] # [B]
            x_c_out_all_lst = []
            if torch.all(is_nonzero_c==False):
                for out_shape in encoder_out_shapes:
                    # ADDED dtype=x.dtype here
                    x_c_out = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
                    x_c_out_all_lst.append(x_c_out)
            else:
                x_c = x_c[is_nonzero_c] # [Bnz,1,H,W,D]
                x_c_out_nonzero_lst = self.encoders[i](x_c) # [[Bnz,f1,H,W,D], [Bnz,f2,H/2,W/2,D/2], ...] f=features
                for j, x_c_out_nonzero in enumerate(x_c_out_nonzero_lst):
                    # ADDED dtype=x.dtype here
                    x_c_out = torch.zeros(encoder_out_shapes[j], device=x_c_out_nonzero.device, dtype=x_c_out_nonzero.dtype) 
                    x_c_out[is_nonzero_c] = x_c_out_nonzero # [B,fx,H,W,D]
                    x_c_out_all_lst.append(x_c_out)
            x_out_lst.append(x_c_out_all_lst)
            
        return x_out_lst