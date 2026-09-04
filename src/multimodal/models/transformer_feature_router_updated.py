from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.blocks import SABlock, TransformerBlock, MLPBlock, PatchEmbeddingBlock
from monai.networks.layers import trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Union, Tuple, Literal
from multimodal.models.blocks import Encoder, Decoder
from einops import rearrange
from einops.layers.torch import Rearrange


class WindowBlock(nn.Module):
    
    def __init__(self, window_size: Sequence[int]) -> None:
        ''' 
        Args:
            window_size: sequence of ints, window size for each spatial dimension. Length should be 2 for 2D and 3 for 3D.
        '''
        super().__init__()
        
        self.window_size = window_size
        self.dims = len(self.window_size)
        assert self.dims in [2,3]
        
        if self.dims==2:
            self.rearrange = Rearrange(
                "B E C (H hws) (W wws) -> (B H W) (E hws wws) C", 
            hws=window_size[0], wws=window_size[1]
            )
        
        if self.dims==3:
            self.rearrange = Rearrange(
                "B E C (H hws) (W wws) (D dws) -> (B H W D) (E hws wws dws) C", 
            hws=window_size[0], wws=window_size[1], dws=window_size[2]
            )
            
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        '''
        Args:
            x: (B, E, C, H, W, D) input tensor.
            mask: (B, E, 1, 1, 1, 1) boolean mask of missing modalities.
        
        Returns:
            x: (B*nwin, ws^3*E, C)
            mask: (B*nwin, ws^3*E, 1)
        '''
        
        mask = mask.expand(-1, -1, -1, *x.shape[3:])
        x = self.rearrange(x)
        mask = self.rearrange(mask)
        
        return x, mask
    
    
class UnwindowBlock(nn.Module):

    def __init__(self, window_size: Sequence[int], shape: Sequence[int]) -> None:
        ''' 
        Args:
            window_size: sequence of ints, window size for each spatial dimension. Length should be 2 for 2D and 3 for 3D.
            shape: sequence of ints, spatial shape of the input tensor. Length should be 2 for 2D and 3 for 3D.
        '''
        super().__init__()
        
        self.window_size = window_size
        self.dims = len(self.window_size)
        assert self.dims in [2,3]
        
        if self.dims==2:
            self.rearrange = Rearrange(
                "(B H_ws W_ws) (E hws wws) C -> B E C (H_ws hws) (W_ws wws)", 
            hws=window_size[0], wws=window_size[1], H_ws=shape[0]//window_size[0], W_ws=shape[1]//window_size[1]
            )
        
        if self.dims==3:
            self.rearrange = Rearrange(
                "(B H_ws W_ws D_ws) (E hws wws dws) C -> B E C (H_ws hws) (W_ws wws) (D_ws dws)", 
            hws=window_size[0], wws=window_size[1], dws=window_size[2], H_ws=shape[0]//window_size[0], W_ws=shape[1]//window_size[1], D_ws=shape[2]//window_size[2]
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        '''
        Args:
            x: (B*nwin, ws^3*E, C) input.
        
        Returns:
            x: (B, E, C, H, W, D)
        '''
        
        x = self.rearrange(x)
        
        return x
    
    
class TransformerFeatureRouter(nn.Module):
    
    ''' 
    This module takes in a (B, E, C, H, W, D) tensor, where E is the number of experts.
    It jointly processes the features of the experts and returns a softmax over the E dimension
    which can be used to weight the features of each expert for subsequent processing stages.
    Reduction options include: no reduction; reduction over the channel dimension; reduction
    over the spatial dimensions; or reduction of channel and spatial dimensions.
    
    '''
    
    def __init__(
        self,
        num_experts: int,
        in_channels: int, # per expert
        input_shape: Sequence[int], # ignored if reduction in ["all", "spatial"]
        patch_size: Sequence[int], # ignored if reduction in ["all", "spatial"]
        embedding_dim: int,
        mlp_dim: int,
        window_size: Sequence[int], # ignored if reduction in ["all", "spatial"]
        num_heads: int,
        num_blocks: int,
        reduction: Literal["all", "channel", "spatial", "none"] = "none",
        apply_mask: bool = True,
        add_expert_embedding: bool = True
    ):
        super().__init__()
        
        spatial_dims = len(input_shape)
        assert spatial_dims in [2,3]
        
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.reduction = reduction
        self.apply_mask = apply_mask
        self.add_expert_embedding = add_expert_embedding
        
        # common layers
        experts_into_batch = Rearrange("B E C H W -> (B E) C H W") if spatial_dims == 2 else Rearrange("B E C H W D -> (B E) C H W D")
        experts_out_of_batch = Rearrange("(B E) C H W -> B E C H W", E=num_experts) if spatial_dims == 2 else Rearrange("(B E) C H W D -> B E C H W D", E=num_experts)
        
        ### in blocks depending on reduction
        self.in_blocks: nn.Module = nn.ModuleList()
        
        if reduction in ["all", "spatial"]:
            # set variables and common layers
            window_size = [1]*spatial_dims
            patched_shape = [1]*spatial_dims
            embedding_dim_to_last = Rearrange("B E C 1 1 -> B E 1 1 C") if spatial_dims==2 else Rearrange("B E C 1 1 1 -> B E 1 1 1 C")
            embedding_dim_from_last = Rearrange("B E 1 1 C -> B E C 1 1") if spatial_dims==2 else Rearrange("B E 1 1 1 C -> B E C 1 1 1")
            # if we don't want spatial dims, lets get rid of them first
            pool_class = nn.AdaptiveAvgPool2d if spatial_dims == 2 else nn.AdaptiveAvgPool3d
            pool_shape = (1,1) if spatial_dims == 2 else (1,1,1)
            self.in_blocks.append(experts_into_batch)
            self.in_blocks.append(pool_class(pool_shape))
            self.in_blocks.append(experts_out_of_batch)
            # project to embedding dim with linear layer
            self.in_blocks.append(embedding_dim_to_last)
            self.in_blocks.append(nn.Linear(in_channels, embedding_dim))
            self.in_blocks.append(embedding_dim_from_last)
            
        if reduction in ["channel", "none"]:
            # set vairables and common layers
            patched_shape = [s//p for s, p in zip(input_shape, patch_size)]
            # here we will do a common patch embedding step
            self.in_blocks.append(experts_into_batch)
            conv_class = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
            self.in_blocks.append(
                conv_class(
                    in_channels=in_channels, out_channels=embedding_dim, 
                    kernel_size=patch_size, stride=patch_size
                    )
                )
            self.in_blocks.append(experts_out_of_batch)
        
        # set up learnable embeddings for possible expert and possible spatial position and missing embedding
        if add_expert_embedding:
            self.expert_embedding = nn.Parameter(torch.zeros(1, num_experts, embedding_dim, *([1]*spatial_dims)))
            trunc_normal_(self.expert_embedding, mean=0.0, std=0.02, a=-2.0, b=2.0)
        if reduction in ["none", "channel"]:
            self.positional_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dim, *patched_shape))
            trunc_normal_(self.positional_embedding, mean=0.0, std=0.02, a=-2.0, b=2.0)
        if not apply_mask:
            self.missing_embedding = nn.Parameter(torch.zeros(embedding_dim, *([1]*spatial_dims)))
            trunc_normal_(self.missing_embedding, mean=0.0, std=0.02, a=-2.0, b=2.0)
        
        self.window = WindowBlock(window_size=window_size)
        
        ### transformer blocks
        self.transformer_blocks: nn.Module = nn.ModuleList()
        
        for _ in range(num_blocks):
            self.transformer_blocks.append(
                TransformerBlock(
                hidden_size=embedding_dim, mlp_dim=mlp_dim, num_heads=num_heads, use_flash_attention=True
                )
            )
            
        self.unwindow = UnwindowBlock(window_size=window_size, shape=patched_shape)
        
        ### out blocks depending on reduction
        self.out_blocks: nn.Module = nn.ModuleList()
        
        if reduction in ["all", "spatial"]:
            self.out_blocks.append(embedding_dim_to_last)
            self.out_blocks.append(nn.Linear(embedding_dim, 1 if reduction=="all" else in_channels))
            self.out_blocks.append(embedding_dim_from_last)
            
        if reduction in ["channel", "none"]:
            self.out_blocks.append(experts_into_batch)
            self.out_blocks.append(
                conv_class(
                    in_channels=embedding_dim,
                    out_channels=1 if reduction=="channel" else in_channels, 
                    kernel_size=1, stride=1
                    )
            )
            self.out_blocks.append(nn.Upsample(size=input_shape, mode="bilinear" if spatial_dims==2 else "trilinear"))
            self.out_blocks.append(experts_out_of_batch)

    def forward(self, x):
        
        # ensure tensor not list of tensors from encoders
        if not isinstance(x, torch.Tensor):
            x = torch.stack(x, dim=1)
        
        # get mask for missing modalities
        is_zero = x.abs().sum(dim=list(range(2,len(x.shape))), keepdim=True) == 0 # (B, E, 1, 1, 1, 1)
        not_zero = ~is_zero
        
        # prepare spatial shapes and embedding dim (prep depends on reduction type)
        for block in self.in_blocks:
            x = block(x)
            
        # add missing embedding where missing if we aren't masking out missing modalities
        if not self.apply_mask:
            x[is_zero.flatten(1)] = self.missing_embedding.to(x.dtype)
        
        # add positional embedding if we haven't reduced spatial dim
        if self.reduction in ["none", "channel"]:
            x = x + self.positional_embedding
        
        # add modality embeddings
        if self.add_expert_embedding:
            x = x + self.expert_embedding
        
        # format for windowed attention in transformer
        x, mask = self.window(x, not_zero)
        
        # process with transformer blocks
        for block in self.transformer_blocks:
            if self.apply_mask:
                x = block(x, mask.squeeze(-1))
            else:
                x = block(x)
        
        # unwindow and return spatial dims
        x = self.unwindow(x)
        
        # perform reduction and necessary related ops
        for block in self.out_blocks:
            x = block(x)
        
        # softmax over modality dimension where modalities are present
        if self.apply_mask:
            x[is_zero.flatten(1)] = -torch.inf
        x = torch.softmax(x, dim=1)
        
        return x
    
    
class EncodersTransformerFeatureRouterDecoder(nn.Module):
    
    def __init__(
        self,
        input_shape: Sequence[int] | int,
        num_encoders: int,
        encoder_in_channels: int = 1,
        encoder_filters: Sequence[int] = (32,64,128,256,320,320),
        encoder_res_block: bool = True,
        decoder_out_channels: int = 1,
        decoder_filters: Sequence[int] | None = None,
        decoder_num_output_heads: int = 4,
        decoder_upsample_outputs: bool = True,
        unet_norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        unet_act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        router_patch_sizes: Sequence[int | Sequence[int]] = (10, 10, 10, 10, 5, 5),
        router_embedding_dim: Sequence[int] = (32,64,128,256,320,320),
        router_mlp_dim: Sequence[int] = (32,64,128,256,320,320),
        router_window_size: Sequence[int | Sequence[int]] = (2, 2, 2, 2, 2, 1),
        router_num_heads: int = 8,
        router_num_blocks: Sequence[int] = (2,2,2,2,2,2),
        router_reduction: Literal["all", "channel", "spatial", "none"] = "none",
        router_apply_mask: bool = True,
        router_add_expert_embedding: bool = True,
        spatial_dims: int = 3
    ):
        super().__init__()
        
        self.num_encoders = num_encoders
        self.encoder_filters = encoder_filters
        self.spatial_dims = spatial_dims
        self.router_apply_mask = router_apply_mask
        self.router_add_expert_embedding = router_add_expert_embedding
        
        if isinstance(input_shape, int):
            input_shape = [input_shape] * spatial_dims
        self.input_shape = input_shape
        
        if router_embedding_dim is None:
            router_embedding_dim = encoder_filters
            
        if router_mlp_dim is None:
            router_mlp_dim = [2*x for x in encoder_filters]
            
        if isinstance(router_patch_sizes[0], int):
            router_patch_sizes = [[x]*spatial_dims for x in router_patch_sizes]
            
        if isinstance(router_window_size[0], int):
            router_window_size = [[x]*spatial_dims for x in router_window_size]
        
        # initialise encoders (one for each modality)
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
            
        # initialise routers (one for each resolution level)
        self.routers = nn.ModuleList()
        for i in range(len(encoder_filters)):
            shape = [s//(2**i) for s in input_shape]
            
            self.routers.append(
                TransformerFeatureRouter(
                    num_experts=num_encoders,
                    in_channels=encoder_filters[i],
                    input_shape=shape,
                    patch_size=router_patch_sizes[i],
                    embedding_dim=router_embedding_dim[i],
                    mlp_dim=router_mlp_dim[i],
                    window_size=router_window_size[i],
                    num_heads=router_num_heads,
                    num_blocks=router_num_blocks[i],
                    reduction=router_reduction,
                    apply_mask=router_apply_mask,
                    add_expert_embedding=router_add_expert_embedding
                )
            )
        
        # initialize joint decoder
        self.decoder = Decoder(
            out_channels=decoder_out_channels,
            skip_features=encoder_filters[::-1],
            filters=decoder_filters,
            no_skip_levels=0, 
            norm_name=unet_norm_name,
            act_name=unet_act_name,
            num_output_heads=decoder_num_output_heads,
            upsample_outputs=decoder_upsample_outputs,
            spatial_dims=spatial_dims   
        )

    def forward(self, x, return_weights: bool = False):
        # x is shape:[B, num_encoders, H, W, D] (i.e. each modality is a channel)
        
        # encode only non-zero channels
        x_feats = self.encode_nonzero_channels(x) # list[list]: [encoder_idx][resolution_idx]
        
        # transpose to [resolution_idx][encoder_idx]
        x_feats_T = [[sublist[i] for sublist in x_feats] for i in range(len(x_feats[0]))]
        
        final_feats = []
        all_weights = []
        
        # iterate through resolutions (from highest to bottleneck)
        for i, (feats_at_res, router) in enumerate(zip(x_feats_T, self.routers)):
            # stack features from each encoder 
            feats_at_res_stack = torch.stack(feats_at_res, dim=1) # [B, num_encoders, C, H, W, D]
            
            # group normalisation so no expert can dominate too much
            feats_at_res_stack = feats_at_res_stack.flatten(start_dim=0, end_dim=1)
            feats_at_res_stack = F.group_norm(feats_at_res_stack, num_groups=1)
            feats_at_res_stack = feats_at_res_stack.unflatten(dim=0, sizes=(x.shape[0], self.num_encoders))
            
            # compute weights based on the features using the router for this level
            weights = router(feats_at_res_stack) # [B, num_encoders, (C or 1), (H or 1), (W or 1), (D or 1)] depending on reduction mode
            
            # apply weights - weights sum to 1 in dim 1 after router softmax, so this is a weighted average over the experts
            weighted_feat = (weights * feats_at_res_stack).sum(dim=1)
            final_feats.append(weighted_feat)
            
            if return_weights:
                all_weights.append(weights)
        
        # decode the weighted features
        out = self.decoder(final_feats)
        
        if return_weights:
            return out, all_weights
        return out

    def encode_nonzero_channels(self, x):
        """this method re-batches based on modalities present and runs a (sub-) batch per modality"""
        B, C = x.shape[0], x.shape[1]
        encoder_out_shapes = [(B, feats, *[s//(2**i) for s in self.input_shape]) for i, feats in enumerate(self.encoder_filters)]

        dims = [2 + i for i in range(self.spatial_dims)]
        # find the all zero batches and modalities
        is_nonzero = x.abs().sum(dim=dims) > 0 # [B, num_encoders]
        
        x_out_lst = []
        for i in range(C):
            x_c = x[:, i:i+1]
            is_nonzero_c = is_nonzero[:, i]
            x_c_out_all_lst = []
            
            if not torch.any(is_nonzero_c):
                # all batches for this modality are zero - no need to run encoder, just fill with zeros of the correct shape
                for out_shape in encoder_out_shapes:
                    x_c_out_all_lst.append(torch.zeros(out_shape, device=x.device))
            else:
                # run encoder only on batches with non-zero items for this modality
                x_c_nonzero = x_c[is_nonzero_c]
                x_c_out_nonzero_lst = self.encoders[i](x_c_nonzero)
                
                for j, x_c_out_nz in enumerate(x_c_out_nonzero_lst):
                    x_c_out = torch.zeros(encoder_out_shapes[j], device=x.device, dtype=x_c_out_nz.dtype)
                    x_c_out[is_nonzero_c] = x_c_out_nz
                    x_c_out_all_lst.append(x_c_out)
            x_out_lst.append(x_c_out_all_lst)
            
        return x_out_lst