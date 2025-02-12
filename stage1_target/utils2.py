from typing import List, Optional, Type
import inspect

import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
# pred_length=24
# factor=10
# d_model = 64
# batch = 32
# router = torch.randn(pred_length, factor, d_model)
# y = repeat(router, 'pred_length factor d_model -> (repeat pred_length) factor d_model', repeat=batch)
# print(y.shape)
def copy_parameters(
        net_source: torch.nn.Module,
        net_dest: torch.nn.Module,
        strict: Optional[bool] = True,
) -> None:
    """
    Copies parameters from one network to another.

    Parameters
    ----------
    net_source
        Input network.
    net_dest
        Output network.
    strict:
        whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    """

    net_dest.load_state_dict(net_source.state_dict(), strict=strict)
    # net_dest.pred_scaled = net_source.scaled

    # print('1', net_dest.pred_scaled)
    # print('2', net_source.scaled)


def get_forward_input_names(module: Type[torch.nn.Module]):
    params = inspect.signature(module.forward).parameters
    param_names = [k for k, v in params.items() if not str(v).startswith("*")]
    assert param_names[0] == "self", (
        "Expected first argument of forward to be `self`, "
        f"but found `{param_names[0]}`"
    )
    return param_names[1:]  # skip: self


def weighted_average(
        x: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,

    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Parameters
    ----------
    x
        Input tensor, of which the average must be computed.
    weights
        Weights tensor, of the same shape as `x`.
    dim
        The dim along which to average `x`

    Returns
    -------
    Tensor:
        The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(
            weights != 0, x * weights, torch.zeros_like(x)
        )
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
                   weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
               ) / sum_weights
    else:
        return x.mean(dim=dim)


def lagged_sequence_values(
        indices: List[int],
        prior_sequence: torch.Tensor,
        sequence: torch.Tensor,
) -> torch.Tensor:
    """
    Constructs an array of lagged values from a given sequence.

    Parameters
    ----------
    indices
        Indices of the lagged observations. For example, ``[0]`` indicates
        that, at any time ``t``, the will have only the observation from
        time ``t`` itself; instead, ``[0, 24]`` indicates that the output
        will have observations from times ``t`` and ``t-24``.
    prior_sequence
        Tensor containing the input sequence prior to the time range for
        which the output is required (shape: ``(N, H, C)``).
    sequence
        Tensor containing the input sequence in the time range where the
        output is required (shape: ``(N, T, C)``).

    Returns
    -------
    Tensor
        A tensor of shape ``(N, T, L)``: if ``I = len(indices)``,
        and ``sequence.shape = (N, T, C)``, then ``L = C * I``.
    """
    assert max(indices) <= prior_sequence.shape[1], (
        f"lags cannot go further than prior sequence length, found lag"
        f" {max(indices)} while prior sequence is only"
        f"{prior_sequence.shape[1]}-long"
    )

    full_sequence = torch.cat((prior_sequence, sequence), dim=1)

    lags_values = []
    for lag_index in indices:
        begin_index = -lag_index - sequence.shape[1]
        end_index = -lag_index if lag_index > 0 else None
        lags_values.append(full_sequence[:, begin_index:end_index, ...])

    lags = torch.stack(lags_values, dim=-1)
    return lags.reshape(lags.shape[0], lags.shape[1], -1)


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        yield from self.iterable
