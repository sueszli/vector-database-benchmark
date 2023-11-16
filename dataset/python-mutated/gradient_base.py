"""The base interface for Aqua's gradient."""
from typing import Union
from qiskit.utils.deprecation import deprecate_func
from .circuit_gradients.circuit_gradient import CircuitGradient
from .derivative_base import DerivativeBase

class GradientBase(DerivativeBase):
    """Deprecated: Base class for first-order operator gradient.

    Convert an operator expression to the first-order gradient.
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, grad_method: Union[str, CircuitGradient]='param_shift', **kwargs):
        if False:
            return 10
        "\n        Args:\n            grad_method: The method used to compute the state/probability gradient. Can be either\n                         ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``.\n                         Ignored for gradients w.r.t observable parameters.\n            kwargs (dict): Optional parameters for a CircuitGradient\n\n        Raises:\n            ValueError: If method != ``fin_diff`` and ``epsilon`` is not None.\n        "
        super().__init__()
        if isinstance(grad_method, CircuitGradient):
            self._grad_method = grad_method
        elif grad_method == 'param_shift':
            from .circuit_gradients.param_shift import ParamShift
            self._grad_method = ParamShift(analytic=True)
        elif grad_method == 'fin_diff':
            from .circuit_gradients.param_shift import ParamShift
            epsilon = kwargs.get('epsilon', 1e-06)
            self._grad_method = ParamShift(analytic=False, epsilon=epsilon)
        elif grad_method == 'lin_comb':
            from .circuit_gradients.lin_comb import LinComb
            self._grad_method = LinComb()
        else:
            raise ValueError("Unrecognized input provided for `grad_method`. Please provide a CircuitGradient object or one of the pre-defined string arguments: {'param_shift', 'fin_diff', 'lin_comb'}. ")

    @property
    def grad_method(self) -> CircuitGradient:
        if False:
            i = 10
            return i + 15
        'Returns ``CircuitGradient``.\n\n        Returns:\n            ``CircuitGradient``.\n\n        '
        return self._grad_method