"""Portal keeps a tensor in the pocket plane. The tensor becomes hidden to the
autograd engine. The shared context of three functions (:class:`PortalBlue`,
:class:`PortalOrange`, and :class:`PortalCopy`) out of the computation graph is
one of the most important feature of :mod:`torchpipe.skip`.

The metaphor is inspired by Portal™ from Valve.

"""
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from ..copy import Context as CopyContext
from ..copy import Copy
from ..phony import get_phony
from ..stream import AbstractStream, get_device
__all__: List[str] = []

class Portal:
    """A portal for a tensor."""

    def __init__(self, tensor: Optional[Tensor], tensor_life: int) -> None:
        if False:
            i = 10
            return i + 15
        self.put_tensor(tensor, tensor_life)
        self.grad: Optional[Tensor] = None

    def blue(self) -> Tensor:
        if False:
            return 10
        'Creates a :class:`PortalBlue` which hides the underlying tensor from\n        the autograd engine.\n\n        Join the returning phony to the main lane of the autograd graph to\n        assure the correct backpropagation::\n\n            PortalBlue --+\n                         |\n            ---------- Join --\n\n        '
        tensor = self.use_tensor()
        if tensor is None:
            return get_phony(torch.device('cpu'), requires_grad=False)
        return PortalBlue.apply(self, tensor)

    def orange(self, phony: Tensor) -> Optional[Tensor]:
        if False:
            while True:
                i = 10
        'Creates a :class:`PortalOrange` which retrieves the hidden tensor\n        without losing ability of backpropagation.\n\n        Give a phony forked from the main lane of an autograd graph::\n\n                +-- PortalOrange --+\n                |                  |\n            -- Fork --------- f(a, b) --\n\n        '
        self.check_tensor_life()
        if self.tensor is None:
            return self.use_tensor()
        return PortalOrange.apply(self, phony)

    def copy(self, prev_stream: AbstractStream, next_stream: AbstractStream, phony: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Copies the hidden tensor by a :class:`PortalCopy`.\n\n        Give a phony and use the returning phony to keep backpropagation::\n\n                +-- PortalCopy --+\n                |                |\n            -- Fork ---------- Join --\n\n        '
        if self.tensor is None:
            return get_phony(torch.device('cpu'), requires_grad=False)
        return PortalCopy.apply(self, prev_stream, next_stream, phony)

    def check_tensor_life(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.tensor_life <= 0:
            raise RuntimeError('tensor in portal has been removed')

    def put_tensor(self, tensor: Optional[Tensor], tensor_life: int) -> None:
        if False:
            while True:
                i = 10
        'Stores a tensor into this portal.'
        self.tensor_life = tensor_life
        if tensor_life > 0:
            self.tensor = tensor
        else:
            self.tensor = None

    def use_tensor(self) -> Optional[Tensor]:
        if False:
            for i in range(10):
                print('nop')
        'Retrieves the underlying tensor and decreases the tensor  life. When\n        the life becomes 0, it the tensor will be removed.\n        '
        self.check_tensor_life()
        tensor = self.tensor
        self.tensor_life -= 1
        if self.tensor_life <= 0:
            self.tensor = None
        return tensor

    def put_grad(self, grad: Tensor) -> None:
        if False:
            while True:
                i = 10
        'Stores a gradient into this portal.'
        self.grad = grad

    def use_grad(self) -> Tensor:
        if False:
            print('Hello World!')
        'Retrieves and removes the underlying gradient. The gradient is\n        always ephemeral.\n        '
        if self.grad is None:
            raise RuntimeError('grad in portal has been removed or never set')
        grad = self.grad
        self.grad = None
        return grad

class Context(CopyContext):
    portal: Portal

class PortalBlue(torch.autograd.Function):
    """Hides a tensor from the autograd engine by a :class:`Portal`."""

    @staticmethod
    def forward(ctx: Context, portal: Portal, tensor: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        ctx.portal = portal
        phony = get_phony(tensor.device, requires_grad=False)
        return phony.detach()

    @staticmethod
    def backward(ctx: Context, grad_phony: Tensor) -> Tuple[None, Tensor]:
        if False:
            for i in range(10):
                print('nop')
        grad = ctx.portal.use_grad()
        return (None, grad)

class PortalOrange(torch.autograd.Function):
    """Retrieves the hidden tensor from a :class:`Portal`."""

    @staticmethod
    def forward(ctx: Context, portal: Portal, phony: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        ctx.portal = portal
        tensor = portal.use_tensor()
        assert tensor is not None
        return tensor.detach()

    @staticmethod
    def backward(ctx: Context, grad: Tensor) -> Tuple[None, None]:
        if False:
            for i in range(10):
                print('nop')
        ctx.portal.put_grad(grad)
        return (None, None)

class PortalCopy(torch.autograd.Function):
    """Copies the hidden tensor in a :class:`Portal`. It replaces the hidden
    tensor with copied one.
    """

    @staticmethod
    def forward(ctx: Context, portal: Portal, prev_stream: AbstractStream, next_stream: AbstractStream, phony: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        ctx.portal = portal
        assert portal.tensor is not None
        (portal.tensor,) = Copy.forward(ctx, prev_stream, next_stream, portal.tensor)
        phony = get_phony(get_device(next_stream), requires_grad=False)
        return phony.detach()

    @staticmethod
    def backward(ctx: Context, grad_phony: Tensor) -> Tuple[None, None, None, None]:
        if False:
            i = 10
            return i + 15
        portal = ctx.portal
        assert portal.grad is not None
        (_, _, portal.grad) = Copy.backward(ctx, portal.grad)
        return (None, None, None, None)