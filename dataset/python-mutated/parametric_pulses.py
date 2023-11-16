"""Parametric waveforms module. These are pulses which are described by a specified
parameterization.

If a backend supports parametric pulses, it will have the attribute
`backend.configuration().parametric_pulses`, which is a list of supported pulse shapes, such as
`['gaussian', 'gaussian_square', 'drag']`. A Pulse Schedule, using parametric pulses, which is
assembled for a backend which supports those pulses, will result in a Qobj which is dramatically
smaller than one which uses Waveforms.

This module can easily be extended to describe more pulse shapes. The new class should:
  - have a descriptive name
  - be a well known and/or well described formula (include the formula in the class docstring)
  - take some parameters (at least `duration`) and validate them, if necessary
  - implement a ``get_waveform`` method which returns a corresponding Waveform in the case that
    it is assembled for a backend which does not support it. Ends are zeroed to avoid steep jumps at
    pulse edges. By default, the ends are defined such that ``f(-1), f(duration+1) = 0``.

The new pulse must then be registered by the assembler in
`qiskit/qobj/converters/pulse_instruction.py:ParametricPulseShapes`
by following the existing pattern:

    class ParametricPulseShapes(Enum):
        gaussian = library.Gaussian
        ...
        new_supported_pulse_name = library.YourPulseWaveformClass
"""
from abc import abstractmethod
from typing import Any, Dict, Optional, Union
import math
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library import continuous
from qiskit.pulse.library.discrete import gaussian, gaussian_square, drag, constant
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform
from qiskit.utils.deprecation import deprecate_func

class ParametricPulse(Pulse):
    """The abstract superclass for parametric pulses.

    .. warning::

        This class is superseded by :class:`.SymbolicPulse` and will be deprecated
        and eventually removed in the future because of the poor flexibility
        for defining a new waveform type and serializing it through the :mod:`qiskit.qpy` framework.

    """

    @abstractmethod
    @deprecate_func(additional_msg='Instead, use SymbolicPulse because of QPY serialization support. See qiskit.pulse.library.symbolic_pulses for details.', since='0.22', package_name='qiskit-terra', pending=True)
    def __init__(self, duration: Union[int, ParameterExpression], name: Optional[str]=None, limit_amplitude: Optional[bool]=None):
        if False:
            print('Hello World!')
        'Create a parametric pulse and validate the input parameters.\n\n        Args:\n            duration: Pulse length in terms of the sampling period `dt`.\n            name: Display name for this pulse envelope.\n            limit_amplitude: If ``True``, then limit the amplitude of the\n                             waveform to 1. The default is ``True`` and the\n                             amplitude is constrained to 1.\n        '
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)
        self.validate_parameters()

    @abstractmethod
    def get_waveform(self) -> Waveform:
        if False:
            while True:
                i = 10
        'Return a Waveform with samples filled according to the formula that the pulse\n        represents and the parameter values it contains.\n        '
        raise NotImplementedError

    @abstractmethod
    def validate_parameters(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Validate parameters.\n\n        Raises:\n            PulseError: If the parameters passed are not valid.\n        '
        raise NotImplementedError

    def is_parameterized(self) -> bool:
        if False:
            while True:
                i = 10
        'Return True iff the instruction is parameterized.'
        return any((_is_parameterized(val) for val in self.parameters.values()))

    def __eq__(self, other: Pulse) -> bool:
        if False:
            return 10
        return super().__eq__(other) and self.parameters == other.parameters

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return hash(tuple((self.parameters[k] for k in sorted(self.parameters))))

class Gaussian(ParametricPulse):
    """A lifted and truncated pulse envelope shaped according to the Gaussian function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \\exp\\Bigl( -\\frac12 \\frac{{(x - \\text{duration}/2)}^2}{\\text{sigma}^2} \\Bigr)\\\\
        f(x) &= \\text{amp} \\times \\frac{f'(x) - f'(-1)}{1-f'(-1)}, \\quad 0 \\le x < \\text{duration}

    where :math:`f'(x)` is the gaussian waveform without lifting or amplitude scaling.

    This pulse would be more accurately named as ``LiftedGaussian``, however, for historical
    and practical DSP reasons it has the name ``Gaussian``.
    """

    @deprecate_func(additional_msg='Instead, use Gaussian from qiskit.pulse.library.symbolic_pulses because of QPY serialization support.', since='0.22', package_name='qiskit-terra', pending=True)
    def __init__(self, duration: Union[int, ParameterExpression], amp: Union[complex, ParameterExpression], sigma: Union[float, ParameterExpression], name: Optional[str]=None, limit_amplitude: Optional[bool]=None):
        if False:
            return 10
        'Initialize the gaussian pulse.\n\n        Args:\n            duration: Pulse length in terms of the sampling period `dt`.\n            amp: The amplitude of the Gaussian envelope.\n            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically\n                   in the class docstring.\n            name: Display name for this pulse envelope.\n            limit_amplitude: If ``True``, then limit the amplitude of the\n                             waveform to 1. The default is ``True`` and the\n                             amplitude is constrained to 1.\n        '
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._sigma = sigma
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        if False:
            while True:
                i = 10
        'The Gaussian amplitude.'
        return self._amp

    @property
    def sigma(self) -> Union[float, ParameterExpression]:
        if False:
            i = 10
            return i + 15
        'The Gaussian standard deviation of the pulse width.'
        return self._sigma

    def get_waveform(self) -> Waveform:
        if False:
            print('Hello World!')
        return gaussian(duration=self.duration, amp=self.amp, sigma=self.sigma, zero_ends=True)

    def validate_parameters(self) -> None:
        if False:
            i = 10
            return i + 15
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self._limit_amplitude:
            raise PulseError(f'The amplitude norm must be <= 1, found: {abs(self.amp)}' + 'This can be overruled by setting Pulse.limit_amplitude.')
        if not _is_parameterized(self.sigma) and self.sigma <= 0:
            raise PulseError('Sigma must be greater than 0.')

    @property
    def parameters(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {'duration': self.duration, 'amp': self.amp, 'sigma': self.sigma}

    def __repr__(self) -> str:
        if False:
            return 10
        return '{}(duration={}, amp={}, sigma={}{})'.format(self.__class__.__name__, self.duration, self.amp, self.sigma, f", name='{self.name}'" if self.name is not None else '')

class GaussianSquare(ParametricPulse):
    """A square pulse with a Gaussian shaped risefall on both sides lifted such that
    its first sample is zero.

    Either the ``risefall_sigma_ratio`` or ``width`` parameter has to be specified.

    If ``risefall_sigma_ratio`` is not None and ``width`` is None:

    .. math::

        \\text{risefall} &= \\text{risefall_sigma_ratio} \\times \\text{sigma}\\\\
        \\text{width} &= \\text{duration} - 2 \\times \\text{risefall}

    If ``width`` is not None and ``risefall_sigma_ratio`` is None:

    .. math:: \\text{risefall} = \\frac{\\text{duration} - \\text{width}}{2}

    In both cases, the lifted gaussian square pulse :math:`f'(x)` is defined as:

    .. math::

        f'(x) &= \\begin{cases}            \\exp\\biggl(-\\frac12 \\frac{(x - \\text{risefall})^2}{\\text{sigma}^2}\\biggr)                & x < \\text{risefall}\\\\
            1                & \\text{risefall} \\le x < \\text{risefall} + \\text{width}\\\\
            \\exp\\biggl(-\\frac12                    \\frac{{\\bigl(x - (\\text{risefall} + \\text{width})\\bigr)}^2}                          {\\text{sigma}^2}                    \\biggr)                & \\text{risefall} + \\text{width} \\le x        \\end{cases}\\\\
        f(x) &= \\text{amp} \\times \\frac{f'(x) - f'(-1)}{1-f'(-1)},            \\quad 0 \\le x < \\text{duration}

    where :math:`f'(x)` is the gaussian square waveform without lifting or amplitude scaling.

    This pulse would be more accurately named as ``LiftedGaussianSquare``, however, for historical
    and practical DSP reasons it has the name ``GaussianSquare``.
    """

    @deprecate_func(additional_msg='Instead, use GaussianSquare from qiskit.pulse.library.symbolic_pulses because of QPY serialization support.', since='0.22', package_name='qiskit-terra', pending=True)
    def __init__(self, duration: Union[int, ParameterExpression], amp: Union[complex, ParameterExpression], sigma: Union[float, ParameterExpression], width: Union[float, ParameterExpression]=None, risefall_sigma_ratio: Union[float, ParameterExpression]=None, name: Optional[str]=None, limit_amplitude: Optional[bool]=None):
        if False:
            i = 10
            return i + 15
        'Initialize the gaussian square pulse.\n\n        Args:\n            duration: Pulse length in terms of the sampling period `dt`.\n            amp: The amplitude of the Gaussian and of the square pulse.\n            sigma: A measure of how wide or narrow the Gaussian risefall is; see the class\n                   docstring for more details.\n            width: The duration of the embedded square pulse.\n            risefall_sigma_ratio: The ratio of each risefall duration to sigma.\n            name: Display name for this pulse envelope.\n            limit_amplitude: If ``True``, then limit the amplitude of the\n                             waveform to 1. The default is ``True`` and the\n                             amplitude is constrained to 1.\n\n        Raises:\n            PulseError: If the parameters passed are not valid.\n        '
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._sigma = sigma
        self._risefall_sigma_ratio = risefall_sigma_ratio
        self._width = width
        if self.width is not None and self.risefall_sigma_ratio is not None:
            raise PulseError('Either the pulse width or the risefall_sigma_ratio parameter can be specified but not both.')
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        if False:
            return 10
        'The Gaussian amplitude.'
        return self._amp

    @property
    def sigma(self) -> Union[float, ParameterExpression]:
        if False:
            print('Hello World!')
        'The Gaussian standard deviation of the pulse width.'
        return self._sigma

    @property
    def risefall_sigma_ratio(self) -> Union[float, ParameterExpression]:
        if False:
            i = 10
            return i + 15
        'The duration of each risefall in terms of sigma.'
        return self._risefall_sigma_ratio

    @property
    def width(self) -> Union[float, ParameterExpression]:
        if False:
            return 10
        'The width of the square portion of the pulse.'
        return self._width

    def get_waveform(self) -> Waveform:
        if False:
            return 10
        return gaussian_square(duration=self.duration, amp=self.amp, width=self.width, sigma=self.sigma, zero_ends=True)

    def validate_parameters(self) -> None:
        if False:
            while True:
                i = 10
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self._limit_amplitude:
            raise PulseError(f'The amplitude norm must be <= 1, found: {abs(self.amp)}' + 'This can be overruled by setting Pulse.limit_amplitude.')
        if not _is_parameterized(self.sigma) and self.sigma <= 0:
            raise PulseError('Sigma must be greater than 0.')
        if self.width is None and self.risefall_sigma_ratio is None:
            raise PulseError('Either the pulse width or the risefall_sigma_ratio parameter must be specified.')
        if self.width is not None:
            if not _is_parameterized(self.width) and self.width < 0:
                raise PulseError('The pulse width must be at least 0.')
            if not (_is_parameterized(self.width) or _is_parameterized(self.duration)) and self.width >= self.duration:
                raise PulseError('The pulse width must be less than its duration.')
            self._risefall_sigma_ratio = (self.duration - self.width) / (2.0 * self.sigma)
        else:
            if not _is_parameterized(self.risefall_sigma_ratio) and self.risefall_sigma_ratio <= 0:
                raise PulseError('The parameter risefall_sigma_ratio must be greater than 0.')
            if not (_is_parameterized(self.risefall_sigma_ratio) or _is_parameterized(self.duration) or _is_parameterized(self.sigma)) and self.risefall_sigma_ratio >= self.duration / (2.0 * self.sigma):
                raise PulseError('The parameter risefall_sigma_ratio must be less than duration/(2*sigma)={}.'.format(self.duration / (2.0 * self.sigma)))
            self._width = self.duration - 2.0 * self.risefall_sigma_ratio * self.sigma

    @property
    def parameters(self) -> Dict[str, Any]:
        if False:
            return 10
        return {'duration': self.duration, 'amp': self.amp, 'sigma': self.sigma, 'width': self.width}

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '{}(duration={}, amp={}, sigma={}, width={}{})'.format(self.__class__.__name__, self.duration, self.amp, self.sigma, self.width, f", name='{self.name}'" if self.name is not None else '')

class Drag(ParametricPulse):
    """The Derivative Removal by Adiabatic Gate (DRAG) pulse is a standard Gaussian pulse
    with an additional Gaussian derivative component and lifting applied.

    It is designed to reduce the frequency spectrum of a standard Gaussian pulse near
    the :math:`|1\\rangle\\leftrightarrow|2\\rangle` transition,
    reducing the chance of leakage to the :math:`|2\\rangle` state.

    .. math::

        g(x) &= \\exp\\Bigl(-\\frac12 \\frac{(x - \\text{duration}/2)^2}{\\text{sigma}^2}\\Bigr)\\\\
        g'(x) &= \\text{amp}\\times\\frac{g(x)-g(-1)}{1-g(-1)}\\\\
        f(x) &=  g'(x) \\times \\Bigl(1 + 1j \\times \\text{beta} \\times            \\Bigl(-\\frac{x - \\text{duration}/2}{\\text{sigma}^2}\\Bigr)  \\Bigr),
            \\quad 0 \\le x < \\text{duration}

    where :math:`g(x)` is a standard unlifted Gaussian waveform and
    :math:`g'(x)` is the lifted :class:`~qiskit.pulse.library.Gaussian` waveform.

    This pulse, defined by :math:`f(x)`, would be more accurately named as ``LiftedDrag``, however,
    for historical and practical DSP reasons it has the name ``Drag``.

    References:
        1. |citation1|_

        .. _citation1: https://link.aps.org/doi/10.1103/PhysRevA.83.012308

        .. |citation1| replace:: *Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
           Analytic control methods for high-fidelity unitary operations
           in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).*

        2. |citation2|_

        .. _citation2: https://link.aps.org/doi/10.1103/PhysRevLett.103.110501

        .. |citation2| replace:: *F. Motzoi, J. M. Gambetta, P. Rebentrost, and F. K. Wilhelm
           Phys. Rev. Lett. 103, 110501 – Published 8 September 2009.*
    """

    @deprecate_func(additional_msg='Instead, use Drag from qiskit.pulse.library.symbolic_pulses because of QPY serialization support.', since='0.22', package_name='qiskit-terra', pending=True)
    def __init__(self, duration: Union[int, ParameterExpression], amp: Union[complex, ParameterExpression], sigma: Union[float, ParameterExpression], beta: Union[float, ParameterExpression], name: Optional[str]=None, limit_amplitude: Optional[bool]=None):
        if False:
            i = 10
            return i + 15
        'Initialize the drag pulse.\n\n        Args:\n            duration: Pulse length in terms of the sampling period `dt`.\n            amp: The amplitude of the Drag envelope.\n            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically\n                   in the class docstring.\n            beta: The correction amplitude.\n            name: Display name for this pulse envelope.\n            limit_amplitude: If ``True``, then limit the amplitude of the\n                             waveform to 1. The default is ``True`` and the\n                             amplitude is constrained to 1.\n        '
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._sigma = sigma
        self._beta = beta
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        if False:
            while True:
                i = 10
        'The Gaussian amplitude.'
        return self._amp

    @property
    def sigma(self) -> Union[float, ParameterExpression]:
        if False:
            for i in range(10):
                print('nop')
        'The Gaussian standard deviation of the pulse width.'
        return self._sigma

    @property
    def beta(self) -> Union[float, ParameterExpression]:
        if False:
            i = 10
            return i + 15
        'The weighing factor for the Gaussian derivative component of the waveform.'
        return self._beta

    def get_waveform(self) -> Waveform:
        if False:
            print('Hello World!')
        return drag(duration=self.duration, amp=self.amp, sigma=self.sigma, beta=self.beta, zero_ends=True)

    def validate_parameters(self) -> None:
        if False:
            return 10
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self._limit_amplitude:
            raise PulseError(f'The amplitude norm must be <= 1, found: {abs(self.amp)}' + 'This can be overruled by setting Pulse.limit_amplitude.')
        if not _is_parameterized(self.sigma) and self.sigma <= 0:
            raise PulseError('Sigma must be greater than 0.')
        if not _is_parameterized(self.beta) and isinstance(self.beta, complex):
            raise PulseError('Beta must be real.')
        if not _is_parameterized(self.beta) and (not _is_parameterized(self.sigma)) and (np.abs(self.beta) > self.sigma) and self._limit_amplitude:
            argmax_x = self.duration / 2 - self.sigma / self.beta * math.sqrt(self.beta ** 2 - self.sigma ** 2)
            argmax_x = max(argmax_x, 0)
            max_val = continuous.drag(np.array(argmax_x), sigma=self.sigma, beta=self.beta, amp=self.amp, center=self.duration / 2)
            if abs(max_val) > 1.0:
                raise PulseError('Beta is too large; pulse amplitude norm exceeds 1.')

    @property
    def parameters(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return {'duration': self.duration, 'amp': self.amp, 'sigma': self.sigma, 'beta': self.beta}

    def __repr__(self) -> str:
        if False:
            return 10
        return '{}(duration={}, amp={}, sigma={}, beta={}{})'.format(self.__class__.__name__, self.duration, self.amp, self.sigma, self.beta, f", name='{self.name}'" if self.name is not None else '')

class Constant(ParametricPulse):
    """
    A simple constant pulse, with an amplitude value and a duration:

    .. math::

        f(x) = amp    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """

    @deprecate_func(additional_msg='Instead, use Constant from qiskit.pulse.library.symbolic_pulses because of QPY serialization support.', since='0.22', package_name='qiskit-terra', pending=True)
    def __init__(self, duration: Union[int, ParameterExpression], amp: Union[complex, ParameterExpression], name: Optional[str]=None, limit_amplitude: Optional[bool]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the constant-valued pulse.\n\n        Args:\n            duration: Pulse length in terms of the sampling period `dt`.\n            amp: The amplitude of the constant square pulse.\n            name: Display name for this pulse envelope.\n            limit_amplitude: If ``True``, then limit the amplitude of the\n                             waveform to 1. The default is ``True`` and the\n                             amplitude is constrained to 1.\n        '
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        if False:
            print('Hello World!')
        'The constant value amplitude.'
        return self._amp

    def get_waveform(self) -> Waveform:
        if False:
            return 10
        return constant(duration=self.duration, amp=self.amp)

    def validate_parameters(self) -> None:
        if False:
            i = 10
            return i + 15
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self._limit_amplitude:
            raise PulseError(f'The amplitude norm must be <= 1, found: {abs(self.amp)}' + 'This can be overruled by setting Pulse.limit_amplitude.')

    @property
    def parameters(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return {'duration': self.duration, 'amp': self.amp}

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '{}(duration={}, amp={}{})'.format(self.__class__.__name__, self.duration, self.amp, f", name='{self.name}'" if self.name is not None else '')

def _is_parameterized(value: Any) -> bool:
    if False:
        print('Hello World!')
    'Shorthand for a frequently checked predicate. ParameterExpressions cannot be\n    validated until they are numerically assigned.\n    '
    return isinstance(value, ParameterExpression)