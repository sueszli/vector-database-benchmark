from typing import List
import torch
from torch import Tensor
from torch._ops import ops
__all__ = ['FloatFunctional', 'FXFloatFunctional', 'QFunctional']

class FloatFunctional(torch.nn.Module):
    """State collector class for float operations.

    The instance of this class can be used instead of the ``torch.`` prefix for
    some operations. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> f_add = FloatFunctional()
        >>> a = torch.tensor(3.0)
        >>> b = torch.tensor(4.0)
        >>> f_add.add(a, b)  # Equivalent to ``torch.add(a, b)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.activation_post_process = torch.nn.Identity()

    def forward(self, x):
        if False:
            return 10
        raise RuntimeError('FloatFunctional is not intended to use the ' + "'forward'. Please use the underlying operation")
    'Operation equivalent to ``torch.add(Tensor, Tensor)``'

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        r = torch.add(x, y)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.add(Tensor, float)``'

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        if False:
            while True:
                i = 10
        r = torch.add(x, y)
        return r
    'Operation equivalent to ``torch.mul(Tensor, Tensor)``'

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        r = torch.mul(x, y)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.mul(Tensor, float)``'

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        r = torch.mul(x, y)
        return r
    'Operation equivalent to ``torch.cat``'

    def cat(self, x: List[Tensor], dim: int=0) -> Tensor:
        if False:
            return 10
        r = torch.cat(x, dim=dim)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``relu(torch.add(x,y))``'

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        r = torch.add(x, y)
        r = torch.nn.functional.relu(r)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.matmul(Tensor, Tensor)``'

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        r = torch.matmul(x, y)
        r = self.activation_post_process(r)
        return r

class FXFloatFunctional(torch.nn.Module):
    """ module to replace FloatFunctional module before FX graph mode quantization,
    since activation_post_process will be inserted in top level module directly

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('FloatFunctional is not intended to use the ' + "'forward'. Please use the underlying operation")
    'Operation equivalent to ``torch.add(Tensor, Tensor)``'

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        r = torch.add(x, y)
        return r
    'Operation equivalent to ``torch.add(Tensor, float)``'

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        if False:
            return 10
        r = torch.add(x, y)
        return r
    'Operation equivalent to ``torch.mul(Tensor, Tensor)``'

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        r = torch.mul(x, y)
        return r
    'Operation equivalent to ``torch.mul(Tensor, float)``'

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        if False:
            while True:
                i = 10
        r = torch.mul(x, y)
        return r
    'Operation equivalent to ``torch.cat``'

    def cat(self, x: List[Tensor], dim: int=0) -> Tensor:
        if False:
            i = 10
            return i + 15
        r = torch.cat(x, dim=dim)
        return r
    'Operation equivalent to ``relu(torch.add(x,y))``'

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        if False:
            return 10
        r = torch.add(x, y)
        r = torch.nn.functional.relu(r)
        return r
    'Operation equivalent to ``torch.matmul(Tensor, Tensor)``'

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        r = torch.matmul(x, y)
        return r

class QFunctional(torch.nn.Module):
    """Wrapper class for quantized operations.

    The instance of this class can be used instead of the
    ``torch.ops.quantized`` prefix. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> q_add = QFunctional()
        >>> # xdoctest: +SKIP
        >>> a = torch.quantize_per_tensor(torch.tensor(3.0), 1.0, 0, torch.qint32)
        >>> b = torch.quantize_per_tensor(torch.tensor(4.0), 1.0, 0, torch.qint32)
        >>> q_add.add(a, b)  # Equivalent to ``torch.ops.quantized.add(a, b, 1.0, 0)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.scale = 1.0
        self.zero_point = 0
        self.activation_post_process = torch.nn.Identity()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if False:
            print('Hello World!')
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if False:
            for i in range(10):
                print('nop')
        self.scale = float(state_dict.pop(prefix + 'scale'))
        self.zero_point = int(state_dict.pop(prefix + 'zero_point'))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)

    def _get_name(self):
        if False:
            return 10
        return 'QFunctional'

    def extra_repr(self):
        if False:
            return 10
        return f'scale={self.scale}, zero_point={self.zero_point}'

    def forward(self, x):
        if False:
            print('Hello World!')
        raise RuntimeError('Functional is not intended to use the ' + "'forward'. Please use the underlying operation")
    'Operation equivalent to ``torch.ops.quantized.add``'

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        r = ops.quantized.add(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.ops.quantized.add(Tensor, float)``'

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        if False:
            return 10
        r = ops.quantized.add_scalar(x, y)
        return r
    'Operation equivalent to ``torch.ops.quantized.mul(Tensor, Tensor)``'

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        r = ops.quantized.mul(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.ops.quantized.mul(Tensor, float)``'

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        if False:
            print('Hello World!')
        r = ops.quantized.mul_scalar(x, y)
        return r
    'Operation equivalent to ``torch.ops.quantized.cat``'

    def cat(self, x: List[Tensor], dim: int=0) -> Tensor:
        if False:
            return 10
        r = ops.quantized.cat(x, scale=self.scale, zero_point=self.zero_point, dim=dim)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.ops.quantized.add_relu``'

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        r = ops.quantized.add_relu(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.ops.quantized.matmul(Tensor, Tensor)``'

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        if False:
            return 10
        r = ops.quantized.matmul(x, y, scale=self.scale, zero_point=self.zero_point)
        return r

    @classmethod
    def from_float(cls, mod):
        if False:
            return 10
        assert type(mod) == FloatFunctional, 'QFunctional.from_float expects an instance of FloatFunctional'
        (scale, zero_point) = mod.activation_post_process.calculate_qparams()
        new_mod = QFunctional()
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod