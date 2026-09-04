import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceLoss, DeepSupervisionLoss
from monai.utils import pytorch_after
from typing import Sequence, Any, Union


class MyDiceCELoss(DiceCELoss):
    
    '''
    This class provides three improvements:
    
    1. The input weights do not need to be tensors already.
    2. The option to normalise given weights in a sensible way.
    3. The option to force BCE to be used for multilabel settings. 
    4. An option for reduction='per_batch' which will return a batch of losses.
    '''
    
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Any | None = None,
        multilabel: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        dice_weight: Sequence[float] | None = None,
        ce_weight: Sequence[float] | None = None,
        sensible_weight_norm: bool = False,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` and ``weight`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``. only used by the `DiceLoss`, not for the `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            multilabel: use nn.BCEWithLogitsLoss for each channel to accomodate a multilabel setting.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``, ``"mean_per_batch"``, ``"sum_per_batch"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
                - ``"mean_per_batch"``: mean except over batch dimension, i.e. the output will be a vector of shape (B,).
                - ``"sum_per_batch"``: sum except over batch dimension, i.e. the output will be a vector of shape (B,).

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            dice_weight: the weights for monai.losses.DiceCELoss. If include_background=False then it should not include
                a value for the first channel.
            ce_weight: the weight param for torch.nn.CrossEntropyLoss or the pos_weight param for torch.nn.BCEWithLogitsLoss 
                if the prediction is single channel or if multilabel=True.
            sensible_weight_norm: normalise given weights so they sum to the number of classes (after possibly removing 
                background for DiceLoss).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.multilabel = multilabel
        self.sensible_weight_norm = sensible_weight_norm

        if dice_weight is not None:
            dice_weight = torch.tensor(dice_weight)
        if ce_weight is not None:
            ce_weight = torch.tensor(ce_weight)
        if dice_weight is not None and sensible_weight_norm:
            assert len(dice_weight) != 1, "Cannot use sensible_weight_norm for Dice on a single channel"
            dice_weight = (dice_weight / dice_weight.sum()) * len(dice_weight)
        if ce_weight is not None and sensible_weight_norm:
            assert len(ce_weight) != 1, "Cannot use sensible weight norm with BCE since background per channel is always given weight of 1"
            ce_weight = (ce_weight / ce_weight.sum()) * len(ce_weight)
        else:
            dice_weight = dice_weight
            ce_weight = ce_weight
        
        if reduction in ["mean_per_batch", "sum_per_batch"]:
            reduction = 'none'
            self.reduction_per_batch = True
        elif reduction in ["mean", "sum"]:
            reduction = reduction
            self.reduction_per_batch = False
        else:
            raise ValueError(f"reduction should be one of 'mean', 'sum', 'mean_per_batch' or 'sum_per_batch' but got {reduction}")

        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=dice_weight,
        )
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        self.binary_cross_entropy = nn.BCEWithLogitsLoss(pos_weight=ce_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss = self.dice(input, target)
        if input.shape[1] == 1 or self.multilabel:
            ce_loss = self.bce(input, target)
        else:
            ce_loss = self.ce(input, target)

        if self.reduction_per_batch:
            reduce_func = torch.sum if self.reduction == "sum" else torch.mean
            dice_loss = reduce_func(dice_loss, dim=[i for i in range(1, len(dice_loss.shape))])
            ce_loss = reduce_func(ce_loss, dim=[i for i in range(1, len(ce_loss.shape))])
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss
    
    

class MyDeepSupervisionLoss(DeepSupervisionLoss):
    
    '''
    Add an argument to normalise the weights to sum to one like for nnU-Net.
    Add option to downsample the target with max pooling instead of interpolation.
    Handle unreduced losses of any shape.
    '''
    
    def __init__(self, norm: bool = False, max_pool: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.norm = norm
        self.max_pool = max_pool
        
    def get_weights(self, *args, **kwargs) -> list[float]:
        weights = super().get_weights(*args, **kwargs)
        if not self.norm:
            return weights
        return (np.array(weights) / np.sum(weights)).tolist()

    def get_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape[2:] != target.shape[2:]:
            dims = len(input.shape[2:])
            if self.max_pool:
                max_pool = F.max_pool2d if dims == 2 else F.max_pool3d
                kernel = tuple(np.array(target.shape[2:]) // np.array(input.shape[2:]))
                target = max_pool(target, kernel)
            else:
                target = F.interpolate(target, size=input.shape[2:], mode=self.interp_mode)
        return self.loss(input, target)  # type: ignore[no-any-return]
    
    def forward(self, input: Union[None, torch.Tensor, list[torch.Tensor]], target: torch.Tensor) -> torch.Tensor:
        if isinstance(input, (list, tuple)):
            weights = self.get_weights(levels=len(input))
            losses = []
            for l in range(len(input)):
                losses.append(weights[l] * self.get_loss(input[l].float(), target))
            loss = torch.stack(losses, dim=0).sum(dim=0)
            return loss
        if input is None:
            raise ValueError("input shouldn't be None.")

        return self.loss(input.float(), target)  # type: ignore[no-any-return]