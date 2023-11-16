"""Utilities to describe gradients."""
from typing import Dict, Union
import torch
from torch.nn import Module

def grad_norm(module: Module, norm_type: Union[float, int, str], group_separator: str='/') -> Dict[str, float]:
    if False:
        return 10
    "Compute each parameter's gradient's norm and their overall norm.\n\n    The overall norm is computed over all gradients together, as if they\n    were concatenated into a single vector.\n\n    Args:\n        module: :class:`torch.nn.Module` to inspect.\n        norm_type: The type of the used p-norm, cast to float if necessary.\n            Can be ``'inf'`` for infinity norm.\n        group_separator: The separator string used by the logger to group\n            the gradients norms in their own subfolder instead of the logs one.\n\n    Return:\n        norms: The dictionary of p-norms of each parameter's gradient and\n            a special entry for the total p-norm of the gradients viewed\n            as a single vector.\n\n    "
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")
    norms = {f'grad_{norm_type}_norm{group_separator}{name}': p.grad.data.norm(norm_type) for (name, p) in module.named_parameters() if p.grad is not None}
    if norms:
        total_norm = torch.tensor(list(norms.values())).norm(norm_type)
        norms[f'grad_{norm_type}_norm_total'] = total_norm
    return norms