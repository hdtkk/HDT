
from typing import Optional

import torch
import torch.nn as nn
from gluonts.core.component import validated


class Scaler:
    def __call__(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError



class StdScaler(Scaler):
    """
    Computes a std scaling  value along dimension ``dim``, and scales the data accordingly.

    Parameters
    ----------
    dim
        dimension along which to compute the scale
    keepdim
        controls whether to retain dimension ``dim`` (of length 1) in the
        scale tensor, or suppress it.
    minimum_scale
        default scale that is used for elements that are constantly zero
        along dimension ``dim``.
    """

    @validated()
    def __init__(
        self,
        dim: int = -1,
        keepdim: bool = False,
        minimum_scale: float = 1e-5,
    ) -> None:
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    def __call__(
        self, data: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            data.shape == weights.shape
        ), "data and weights must have same shape"
        with torch.no_grad():
            denominator = weights.sum(self.dim, keepdim=self.keepdim)
            denominator = denominator.clamp_min(1.0)
            loc = (data * weights).sum(
                self.dim, keepdim=self.keepdim
            ) / denominator

            variance = (((data - loc) * weights) ** 2).sum(
                self.dim, keepdim=self.keepdim
            ) / denominator
            scale = torch.sqrt(variance + self.minimum_scale)
            return (data - loc) / scale, loc, scale

def configure_optimizers(self):
    decay, no_decay = set(), set()
    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    for mn, m in self.model.transformer.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn

            if pn.endswith("bias"):
                no_decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=1e-4, betas=(0.9, 0.95))
        return optimizer