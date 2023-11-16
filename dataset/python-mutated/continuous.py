"""Module for builtin continuous pulse functions."""
import functools
from typing import Union, Tuple, Optional
import numpy as np
from qiskit.pulse.exceptions import PulseError

def constant(times: np.ndarray, amp: complex) -> np.ndarray:
    if False:
        return 10
    'Continuous constant pulse.\n\n    Args:\n        times: Times to output pulse for.\n        amp: Complex pulse amplitude.\n    '
    return np.full(len(times), amp, dtype=np.complex128)

def zero(times: np.ndarray) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Continuous zero pulse.\n\n    Args:\n        times: Times to output pulse for.\n    '
    return constant(times, 0)

def square(times: np.ndarray, amp: complex, freq: float, phase: float=0) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Continuous square wave.\n\n    Args:\n        times: Times to output wave for.\n        amp: Pulse amplitude. Wave range is [-amp, amp].\n        freq: Pulse frequency. units of 1/dt.\n        phase: Pulse phase.\n    '
    x = times * freq + phase / np.pi
    return amp * (2 * (2 * np.floor(x) - np.floor(2 * x)) + 1).astype(np.complex128)

def sawtooth(times: np.ndarray, amp: complex, freq: float, phase: float=0) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Continuous sawtooth wave.\n\n    Args:\n        times: Times to output wave for.\n        amp: Pulse amplitude. Wave range is [-amp, amp].\n        freq: Pulse frequency. units of 1/dt.\n        phase: Pulse phase.\n    '
    x = times * freq + phase / np.pi
    return amp * 2 * (x - np.floor(1 / 2 + x)).astype(np.complex128)

def triangle(times: np.ndarray, amp: complex, freq: float, phase: float=0) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Continuous triangle wave.\n\n    Args:\n        times: Times to output wave for.\n        amp: Pulse amplitude. Wave range is [-amp, amp].\n        freq: Pulse frequency. units of 1/dt.\n        phase: Pulse phase.\n    '
    return amp * (-2 * np.abs(sawtooth(times, 1, freq, phase=(phase - np.pi / 2) / 2)) + 1).astype(np.complex128)

def cos(times: np.ndarray, amp: complex, freq: float, phase: float=0) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Continuous cosine wave.\n\n    Args:\n        times: Times to output wave for.\n        amp: Pulse amplitude.\n        freq: Pulse frequency, units of 1/dt.\n        phase: Pulse phase.\n    '
    return amp * np.cos(2 * np.pi * freq * times + phase).astype(np.complex128)

def sin(times: np.ndarray, amp: complex, freq: float, phase: float=0) -> np.ndarray:
    if False:
        print('Hello World!')
    'Continuous cosine wave.\n\n    Args:\n        times: Times to output wave for.\n        amp: Pulse amplitude.\n        freq: Pulse frequency, units of 1/dt.\n        phase: Pulse phase.\n    '
    return amp * np.sin(2 * np.pi * freq * times + phase).astype(np.complex128)

def _fix_gaussian_width(gaussian_samples, amp: float, center: float, sigma: float, zeroed_width: Optional[float]=None, rescale_amp: bool=False, ret_scale_factor: bool=False) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Enforce that the supplied gaussian pulse is zeroed at a specific width.\n\n    This is achieved by subtracting $\\Omega_g(center \\pm zeroed_width/2)$ from all samples.\n\n    amp: Pulse amplitude at `center`.\n    center: Center (mean) of pulse.\n    sigma: Standard deviation of pulse.\n    zeroed_width: Subtract baseline from gaussian pulses to make sure\n        $\\Omega_g(center \\pm zeroed_width/2)=0$ is satisfied. This is used to avoid\n        large discontinuities at the start of a gaussian pulse. If unsupplied,\n        defaults to $2*(center + 1)$ such that $\\Omega_g(-1)=0$ and $\\Omega_g(2*(center + 1))=0$.\n    rescale_amp: If True the pulse will be rescaled so that $\\Omega_g(center)=amp$.\n    ret_scale_factor: Return amplitude scale factor.\n    '
    if zeroed_width is None:
        zeroed_width = 2 * (center + 1)
    zero_offset = gaussian(np.array([zeroed_width / 2]), amp, 0, sigma)
    gaussian_samples -= zero_offset
    amp_scale_factor = 1.0
    if rescale_amp:
        amp_scale_factor = amp / (amp - zero_offset) if amp - zero_offset != 0 else 1.0
        gaussian_samples *= amp_scale_factor
    if ret_scale_factor:
        return (gaussian_samples, amp_scale_factor)
    return gaussian_samples

def gaussian(times: np.ndarray, amp: complex, center: float, sigma: float, zeroed_width: Optional[float]=None, rescale_amp: bool=False, ret_x: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if False:
        while True:
            i = 10
    'Continuous unnormalized gaussian pulse.\n\n    Integrated area under curve is $\\Omega_g(amp, sigma) = amp \\times np.sqrt(2\\pi \\sigma^2)$\n\n    Args:\n        times: Times to output pulse for.\n        amp: Pulse amplitude at `center`. If `zeroed_width` is set pulse amplitude at center\n            will be $amp-\\Omega_g(center \\pm zeroed_width/2)$ unless `rescale_amp` is set,\n            in which case all samples will be rescaled such that the center\n            amplitude will be `amp`.\n        center: Center (mean) of pulse.\n        sigma: Width (standard deviation) of pulse.\n        zeroed_width: Subtract baseline from gaussian pulses to make sure\n            $\\Omega_g(center \\pm zeroed_width/2)=0$ is satisfied. This is used to avoid\n            large discontinuities at the start of a gaussian pulse.\n        rescale_amp: If `zeroed_width` is not `None` and `rescale_amp=True` the pulse will\n            be rescaled so that $\\Omega_g(center)=amp$.\n        ret_x: Return centered and standard deviation normalized pulse location.\n               $x=(times-center)/sigma.\n    '
    times = np.asarray(times, dtype=np.complex128)
    x = (times - center) / sigma
    gauss = amp * np.exp(-x ** 2 / 2).astype(np.complex128)
    if zeroed_width is not None:
        gauss = _fix_gaussian_width(gauss, amp=amp, center=center, sigma=sigma, zeroed_width=zeroed_width, rescale_amp=rescale_amp)
    if ret_x:
        return (gauss, x)
    return gauss

def gaussian_deriv(times: np.ndarray, amp: complex, center: float, sigma: float, ret_gaussian: bool=False, zeroed_width: Optional[float]=None, rescale_amp: bool=False) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Continuous unnormalized gaussian derivative pulse.\n\n    Args:\n        times: Times to output pulse for.\n        amp: Pulse amplitude at `center`.\n        center: Center (mean) of pulse.\n        sigma: Width (standard deviation) of pulse.\n        ret_gaussian: Return gaussian with which derivative was taken with.\n        zeroed_width: Subtract baseline of pulse to make sure\n            $\\Omega_g(center \\pm zeroed_width/2)=0$ is satisfied. This is used to avoid\n            large discontinuities at the start of a pulse.\n        rescale_amp: If `zeroed_width` is not `None` and `rescale_amp=True` the pulse will\n            be rescaled so that $\\Omega_g(center)=amp$.\n    '
    (gauss, x) = gaussian(times, amp=amp, center=center, sigma=sigma, zeroed_width=zeroed_width, rescale_amp=rescale_amp, ret_x=True)
    gauss_deriv = -x / sigma * gauss
    if ret_gaussian:
        return (gauss_deriv, gauss)
    return gauss_deriv

def _fix_sech_width(sech_samples, amp: float, center: float, sigma: float, zeroed_width: Optional[float]=None, rescale_amp: bool=False, ret_scale_factor: bool=False) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Enforce that the supplied sech pulse is zeroed at a specific width.\n\n    This is achieved by subtracting $\\Omega_g(center \\pm zeroed_width/2)$ from all samples.\n\n    amp: Pulse amplitude at `center`.\n    center: Center (mean) of pulse.\n    sigma: Standard deviation of pulse.\n    zeroed_width: Subtract baseline from sech pulses to make sure\n        $\\Omega_g(center \\pm zeroed_width/2)=0$ is satisfied. This is used to avoid\n        large discontinuities at the start of a sech pulse. If unsupplied,\n        defaults to $2*(center + 1)$ such that $\\Omega_g(-1)=0$ and $\\Omega_g(2*(center + 1))=0$.\n    rescale_amp: If True the pulse will be rescaled so that $\\Omega_g(center)=amp$.\n    ret_scale_factor: Return amplitude scale factor.\n    '
    if zeroed_width is None:
        zeroed_width = 2 * (center + 1)
    zero_offset = sech(np.array([zeroed_width / 2]), amp, 0, sigma)
    sech_samples -= zero_offset
    amp_scale_factor = 1.0
    if rescale_amp:
        amp_scale_factor = amp / (amp - zero_offset) if amp - zero_offset != 0 else 1.0
        sech_samples *= amp_scale_factor
    if ret_scale_factor:
        return (sech_samples, amp_scale_factor)
    return sech_samples

def sech_fn(x, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Hyperbolic secant function'
    return 1.0 / np.cosh(x, *args, **kwargs)

def sech(times: np.ndarray, amp: complex, center: float, sigma: float, zeroed_width: Optional[float]=None, rescale_amp: bool=False, ret_x: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if False:
        while True:
            i = 10
    'Continuous unnormalized sech pulse.\n\n    Args:\n        times: Times to output pulse for.\n        amp: Pulse amplitude at `center`.\n        center: Center (mean) of pulse.\n        sigma: Width (standard deviation) of pulse.\n        zeroed_width: Subtract baseline from pulse to make sure\n            $\\Omega_g(center \\pm zeroed_width/2)=0$ is satisfied. This is used to avoid\n            large discontinuities at the start and end of the pulse.\n        rescale_amp: If `zeroed_width` is not `None` and `rescale_amp=True` the pulse will\n            be rescaled so that $\\Omega_g(center)=amp$.\n        ret_x: Return centered and standard deviation normalized pulse location.\n            $x=(times-center)/sigma$.\n    '
    times = np.asarray(times, dtype=np.complex128)
    x = (times - center) / sigma
    sech_out = amp * sech_fn(x).astype(np.complex128)
    if zeroed_width is not None:
        sech_out = _fix_sech_width(sech_out, amp=amp, center=center, sigma=sigma, zeroed_width=zeroed_width, rescale_amp=rescale_amp)
    if ret_x:
        return (sech_out, x)
    return sech_out

def sech_deriv(times: np.ndarray, amp: complex, center: float, sigma: float, ret_sech: bool=False) -> np.ndarray:
    if False:
        return 10
    'Continuous unnormalized sech derivative pulse.\n\n    Args:\n        times: Times to output pulse for.\n        amp: Pulse amplitude at `center`.\n        center: Center (mean) of pulse.\n        sigma: Width (standard deviation) of pulse.\n        ret_sech: Return sech with which derivative was taken with.\n    '
    (sech_out, x) = sech(times, amp=amp, center=center, sigma=sigma, ret_x=True)
    sech_out_deriv = -sech_out * np.tanh(x) / sigma
    if ret_sech:
        return (sech_out_deriv, sech_out)
    return sech_out_deriv

def gaussian_square(times: np.ndarray, amp: complex, center: float, square_width: float, sigma: float, zeroed_width: Optional[float]=None) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Continuous gaussian square pulse.\n\n    Args:\n        times: Times to output pulse for.\n        amp: Pulse amplitude.\n        center: Center of the square pulse component.\n        square_width: Width of the square pulse component.\n        sigma: Standard deviation of Gaussian rise/fall portion of the pulse.\n        zeroed_width: Subtract baseline of gaussian square pulse\n            to enforce $\\OmegaSquare(center \\pm zeroed_width/2)=0$.\n\n    Raises:\n        PulseError: if zeroed_width is not compatible with square_width.\n    '
    square_start = center - square_width / 2
    square_stop = center + square_width / 2
    if zeroed_width:
        if zeroed_width < square_width:
            raise PulseError('zeroed_width cannot be smaller than square_width.')
        gaussian_zeroed_width = zeroed_width - square_width
    else:
        gaussian_zeroed_width = None
    funclist = [functools.partial(gaussian, amp=amp, center=square_start, sigma=sigma, zeroed_width=gaussian_zeroed_width, rescale_amp=True), functools.partial(gaussian, amp=amp, center=square_stop, sigma=sigma, zeroed_width=gaussian_zeroed_width, rescale_amp=True), functools.partial(constant, amp=amp)]
    condlist = [times <= square_start, times >= square_stop]
    return np.piecewise(times.astype(np.complex128), condlist, funclist)

def drag(times: np.ndarray, amp: complex, center: float, sigma: float, beta: float, zeroed_width: Optional[float]=None, rescale_amp: bool=False) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Continuous Y-only correction DRAG pulse for standard nonlinear oscillator (SNO) [1].\n\n    [1] Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.\n        Analytic control methods for high-fidelity unitary operations\n        in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).\n\n    Args:\n        times: Times to output pulse for.\n        amp: Pulse amplitude at `center`.\n        center: Center (mean) of pulse.\n        sigma: Width (standard deviation) of pulse.\n        beta: Y correction amplitude. For the SNO this is $\\beta=-\\frac{\\lambda_1^2}{4\\Delta_2}$.\n            Where $\\lambds_1$ is the relative coupling strength between the first excited and second\n            excited states and $\\Delta_2$ is the detuning between the respective excited states.\n        zeroed_width: Subtract baseline of drag pulse to make sure\n            $\\Omega_g(center \\pm zeroed_width/2)=0$ is satisfied. This is used to avoid\n            large discontinuities at the start of a drag pulse.\n        rescale_amp: If `zeroed_width` is not `None` and `rescale_amp=True` the pulse will\n            be rescaled so that $\\Omega_g(center)=amp$.\n\n    '
    (gauss_deriv, gauss) = gaussian_deriv(times, amp=amp, center=center, sigma=sigma, ret_gaussian=True, zeroed_width=zeroed_width, rescale_amp=rescale_amp)
    return gauss + 1j * beta * gauss_deriv