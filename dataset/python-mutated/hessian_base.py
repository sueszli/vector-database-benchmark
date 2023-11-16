"""The module to compute Hessians."""
from typing import Union
from qiskit.utils.deprecation import deprecate_func
from .circuit_gradients.circuit_gradient import CircuitGradient
from .derivative_base import DerivativeBase

class HessianBase(DerivativeBase):
    """Deprecated: Base class for the Hessian of an expected value."""

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, hess_method: Union[str, CircuitGradient]='param_shift', **kwargs):
        if False:
            while True:
                i = 10
        "\n        Args:\n            hess_method: The method used to compute the state/probability gradient. Can be either\n                         ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``.\n                         Ignored for gradients w.r.t observable parameters.\n            kwargs (dict): Optional parameters for a CircuitGradient\n\n        Raises:\n            ValueError: If method != ``fin_diff`` and ``epsilon`` is not None.\n        "
        super().__init__()
        if isinstance(hess_method, CircuitGradient):
            self._hess_method = hess_method
        elif hess_method == 'param_shift':
            from .circuit_gradients import ParamShift
            self._hess_method = ParamShift()
        elif hess_method == 'fin_diff':
            from .circuit_gradients import ParamShift
            epsilon = kwargs.get('epsilon', 1e-06)
            self._hess_method = ParamShift(analytic=False, epsilon=epsilon)
        elif hess_method == 'lin_comb':
            from .circuit_gradients import LinComb
            self._hess_method = LinComb()
        else:
            raise ValueError("Unrecognized input provided for `hess_method`. Please provide a CircuitGradient object or one of the pre-defined string arguments: {'param_shift', 'fin_diff', 'lin_comb'}. ")

    @property
    def hess_method(self) -> CircuitGradient:
        if False:
            return 10
        'Returns ``CircuitGradient``.\n\n        Returns:\n            ``CircuitGradient``.\n\n        '
        return self._hess_method