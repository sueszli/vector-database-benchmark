"""Provides phony for arbitrary dependency in a autograd graph."""
from typing import Dict, List, Tuple
import torch
from torch import Tensor
from .stream import default_stream, use_stream
__all__: List[str] = ['get_phony']
_phonies: Dict[Tuple[torch.device, bool], Tensor] = {}

def get_phony(device: torch.device, *, requires_grad: bool) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    "Get a phony. Phony is tensor without space.\n\n    It is useful to make arbitrary dependency in a autograd graph because it doesn't require any\n    gradient accumulation.\n\n    .. note::\n\n        Phonies for each device are cached. If an autograd function gets a phony\n        internally, the phony must be detached to be returned. Otherwise, the\n        autograd engine will mutate the cached phony in-place::\n\n            class Phonify(torch.autograd.Function):\n                @staticmethod\n                def forward(ctx, input):\n                    phony = get_phony(input.device, requires_grad=False)\n                    return phony.detach()  # detach() is necessary.\n\n    "
    key = (device, requires_grad)
    try:
        phony = _phonies[key]
    except KeyError:
        with use_stream(default_stream(device)):
            phony = torch.empty(0, device=device, requires_grad=requires_grad)
        _phonies[key] = phony
    return phony