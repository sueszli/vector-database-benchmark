from sympy.core.numbers import I, pi
from sympy.functions.elementary.exponential import exp, log
from sympy.polys.partfrac import apart
from sympy.core.symbol import Dummy
from sympy.external import import_module
from sympy.functions import arg, Abs
from sympy.integrals.laplace import _fast_inverse_laplace
from sympy.physics.control.lti import SISOLinearTimeInvariant
from sympy.plotting.series import LineOver1DRangeSeries
from sympy.polys.polytools import Poly
from sympy.printing.latex import latex
__all__ = ['pole_zero_numerical_data', 'pole_zero_plot', 'step_response_numerical_data', 'step_response_plot', 'impulse_response_numerical_data', 'impulse_response_plot', 'ramp_response_numerical_data', 'ramp_response_plot', 'bode_magnitude_numerical_data', 'bode_phase_numerical_data', 'bode_magnitude_plot', 'bode_phase_plot', 'bode_plot']
matplotlib = import_module('matplotlib', import_kwargs={'fromlist': ['pyplot']}, catch=(RuntimeError,))
numpy = import_module('numpy')
if matplotlib:
    plt = matplotlib.pyplot
if numpy:
    np = numpy

def _check_system(system):
    if False:
        i = 10
        return i + 15
    'Function to check whether the dynamical system passed for plots is\n    compatible or not.'
    if not isinstance(system, SISOLinearTimeInvariant):
        raise NotImplementedError('Only SISO LTI systems are currently supported.')
    sys = system.to_expr()
    len_free_symbols = len(sys.free_symbols)
    if len_free_symbols > 1:
        raise ValueError('Extra degree of freedom found. Make sure that there are no free symbols in the dynamical system other than the variable of Laplace transform.')
    if sys.has(exp):
        raise NotImplementedError('Time delay terms are not supported.')

def pole_zero_numerical_data(system):
    if False:
        return 10
    '\n    Returns the numerical data of poles and zeros of the system.\n    It is internally used by ``pole_zero_plot`` to get the data\n    for plotting poles and zeros. Users can use this data to further\n    analyse the dynamics of the system or plot using a different\n    backend/plotting-module.\n\n    Parameters\n    ==========\n\n    system : SISOLinearTimeInvariant\n        The system for which the pole-zero data is to be computed.\n\n    Returns\n    =======\n\n    tuple : (zeros, poles)\n        zeros = Zeros of the system. NumPy array of complex numbers.\n        poles = Poles of the system. NumPy array of complex numbers.\n\n    Raises\n    ======\n\n    NotImplementedError\n        When a SISO LTI system is not passed.\n\n        When time delay terms are present in the system.\n\n    ValueError\n        When more than one free symbol is present in the system.\n        The only variable in the transfer function should be\n        the variable of the Laplace transform.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import s\n    >>> from sympy.physics.control.lti import TransferFunction\n    >>> from sympy.physics.control.control_plots import pole_zero_numerical_data\n    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)\n    >>> pole_zero_numerical_data(tf1)   # doctest: +SKIP\n    ([-0.+1.j  0.-1.j], [-2. +0.j        -0.5+0.8660254j -0.5-0.8660254j -1. +0.j       ])\n\n    See Also\n    ========\n\n    pole_zero_plot\n\n    '
    _check_system(system)
    system = system.doit()
    num_poly = Poly(system.num, system.var).all_coeffs()
    den_poly = Poly(system.den, system.var).all_coeffs()
    num_poly = np.array(num_poly, dtype=np.complex128)
    den_poly = np.array(den_poly, dtype=np.complex128)
    zeros = np.roots(num_poly)
    poles = np.roots(den_poly)
    return (zeros, poles)

def pole_zero_plot(system, pole_color='blue', pole_markersize=10, zero_color='orange', zero_markersize=7, grid=True, show_axes=True, show=True, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns the Pole-Zero plot (also known as PZ Plot or PZ Map) of a system.\n\n    A Pole-Zero plot is a graphical representation of a system's poles and\n    zeros. It is plotted on a complex plane, with circular markers representing\n    the system's zeros and 'x' shaped markers representing the system's poles.\n\n    Parameters\n    ==========\n\n    system : SISOLinearTimeInvariant type systems\n        The system for which the pole-zero plot is to be computed.\n    pole_color : str, tuple, optional\n        The color of the pole points on the plot. Default color\n        is blue. The color can be provided as a matplotlib color string,\n        or a 3-tuple of floats each in the 0-1 range.\n    pole_markersize : Number, optional\n        The size of the markers used to mark the poles in the plot.\n        Default pole markersize is 10.\n    zero_color : str, tuple, optional\n        The color of the zero points on the plot. Default color\n        is orange. The color can be provided as a matplotlib color string,\n        or a 3-tuple of floats each in the 0-1 range.\n    zero_markersize : Number, optional\n        The size of the markers used to mark the zeros in the plot.\n        Default zero markersize is 7.\n    grid : boolean, optional\n        If ``True``, the plot will have a grid. Defaults to True.\n    show_axes : boolean, optional\n        If ``True``, the coordinate axes will be shown. Defaults to False.\n    show : boolean, optional\n        If ``True``, the plot will be displayed otherwise\n        the equivalent matplotlib ``plot`` object will be returned.\n        Defaults to True.\n\n    Examples\n    ========\n\n    .. plot::\n        :context: close-figs\n        :format: doctest\n        :include-source: True\n\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> from sympy.physics.control.control_plots import pole_zero_plot\n        >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)\n        >>> pole_zero_plot(tf1)   # doctest: +SKIP\n\n    See Also\n    ========\n\n    pole_zero_numerical_data\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Pole%E2%80%93zero_plot\n\n    "
    (zeros, poles) = pole_zero_numerical_data(system)
    zero_real = np.real(zeros)
    zero_imag = np.imag(zeros)
    pole_real = np.real(poles)
    pole_imag = np.imag(poles)
    plt.plot(pole_real, pole_imag, 'x', mfc='none', markersize=pole_markersize, color=pole_color)
    plt.plot(zero_real, zero_imag, 'o', markersize=zero_markersize, color=zero_color)
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.title(f'Poles and Zeros of ${latex(system)}$', pad=20)
    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return
    return plt

def step_response_numerical_data(system, prec=8, lower_limit=0, upper_limit=10, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the numerical values of the points in the step response plot\n    of a SISO continuous-time system. By default, adaptive sampling\n    is used. If the user wants to instead get an uniformly\n    sampled response, then ``adaptive`` kwarg should be passed ``False``\n    and ``n`` must be passed as additional kwargs.\n    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`\n    for more details.\n\n    Parameters\n    ==========\n\n    system : SISOLinearTimeInvariant\n        The system for which the unit step response data is to be computed.\n    prec : int, optional\n        The decimal point precision for the point coordinate values.\n        Defaults to 8.\n    lower_limit : Number, optional\n        The lower limit of the plot range. Defaults to 0.\n    upper_limit : Number, optional\n        The upper limit of the plot range. Defaults to 10.\n    kwargs :\n        Additional keyword arguments are passed to the underlying\n        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.\n\n    Returns\n    =======\n\n    tuple : (x, y)\n        x = Time-axis values of the points in the step response. NumPy array.\n        y = Amplitude-axis values of the points in the step response. NumPy array.\n\n    Raises\n    ======\n\n    NotImplementedError\n        When a SISO LTI system is not passed.\n\n        When time delay terms are present in the system.\n\n    ValueError\n        When more than one free symbol is present in the system.\n        The only variable in the transfer function should be\n        the variable of the Laplace transform.\n\n        When ``lower_limit`` parameter is less than 0.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import s\n    >>> from sympy.physics.control.lti import TransferFunction\n    >>> from sympy.physics.control.control_plots import step_response_numerical_data\n    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)\n    >>> step_response_numerical_data(tf1)   # doctest: +SKIP\n    ([0.0, 0.025413462339411542, 0.0484508722725343, ... , 9.670250533855183, 9.844291913708725, 10.0],\n    [0.0, 0.023844582399907256, 0.042894276802320226, ..., 6.828770759094287e-12, 6.456457160755703e-12])\n\n    See Also\n    ========\n\n    step_response_plot\n\n    '
    if lower_limit < 0:
        raise ValueError('Lower limit of time must be greater than or equal to zero.')
    _check_system(system)
    _x = Dummy('x')
    expr = system.to_expr() / system.var
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit), **kwargs).get_points()

def step_response_plot(system, color='b', prec=8, lower_limit=0, upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    if False:
        return 10
    '\n    Returns the unit step response of a continuous-time system. It is\n    the response of the system when the input signal is a step function.\n\n    Parameters\n    ==========\n\n    system : SISOLinearTimeInvariant type\n        The LTI SISO system for which the Step Response is to be computed.\n    color : str, tuple, optional\n        The color of the line. Default is Blue.\n    show : boolean, optional\n        If ``True``, the plot will be displayed otherwise\n        the equivalent matplotlib ``plot`` object will be returned.\n        Defaults to True.\n    lower_limit : Number, optional\n        The lower limit of the plot range. Defaults to 0.\n    upper_limit : Number, optional\n        The upper limit of the plot range. Defaults to 10.\n    prec : int, optional\n        The decimal point precision for the point coordinate values.\n        Defaults to 8.\n    show_axes : boolean, optional\n        If ``True``, the coordinate axes will be shown. Defaults to False.\n    grid : boolean, optional\n        If ``True``, the plot will have a grid. Defaults to True.\n\n    Examples\n    ========\n\n    .. plot::\n        :context: close-figs\n        :format: doctest\n        :include-source: True\n\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> from sympy.physics.control.control_plots import step_response_plot\n        >>> tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)\n        >>> step_response_plot(tf1)   # doctest: +SKIP\n\n    See Also\n    ========\n\n    impulse_response_plot, ramp_response_plot\n\n    References\n    ==========\n\n    .. [1] https://www.mathworks.com/help/control/ref/lti.step.html\n\n    '
    (x, y) = step_response_numerical_data(system, prec=prec, lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Unit Step Response of ${latex(system)}$', pad=20)
    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return
    return plt

def impulse_response_numerical_data(system, prec=8, lower_limit=0, upper_limit=10, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Returns the numerical values of the points in the impulse response plot\n    of a SISO continuous-time system. By default, adaptive sampling\n    is used. If the user wants to instead get an uniformly\n    sampled response, then ``adaptive`` kwarg should be passed ``False``\n    and ``n`` must be passed as additional kwargs.\n    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`\n    for more details.\n\n    Parameters\n    ==========\n\n    system : SISOLinearTimeInvariant\n        The system for which the impulse response data is to be computed.\n    prec : int, optional\n        The decimal point precision for the point coordinate values.\n        Defaults to 8.\n    lower_limit : Number, optional\n        The lower limit of the plot range. Defaults to 0.\n    upper_limit : Number, optional\n        The upper limit of the plot range. Defaults to 10.\n    kwargs :\n        Additional keyword arguments are passed to the underlying\n        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.\n\n    Returns\n    =======\n\n    tuple : (x, y)\n        x = Time-axis values of the points in the impulse response. NumPy array.\n        y = Amplitude-axis values of the points in the impulse response. NumPy array.\n\n    Raises\n    ======\n\n    NotImplementedError\n        When a SISO LTI system is not passed.\n\n        When time delay terms are present in the system.\n\n    ValueError\n        When more than one free symbol is present in the system.\n        The only variable in the transfer function should be\n        the variable of the Laplace transform.\n\n        When ``lower_limit`` parameter is less than 0.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import s\n    >>> from sympy.physics.control.lti import TransferFunction\n    >>> from sympy.physics.control.control_plots import impulse_response_numerical_data\n    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)\n    >>> impulse_response_numerical_data(tf1)   # doctest: +SKIP\n    ([0.0, 0.06616480200395854,... , 9.854500743565858, 10.0],\n    [0.9999999799999999, 0.7042848373025861,...,7.170748906965121e-13, -5.1901263495547205e-12])\n\n    See Also\n    ========\n\n    impulse_response_plot\n\n    '
    if lower_limit < 0:
        raise ValueError('Lower limit of time must be greater than or equal to zero.')
    _check_system(system)
    _x = Dummy('x')
    expr = system.to_expr()
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit), **kwargs).get_points()

def impulse_response_plot(system, color='b', prec=8, lower_limit=0, upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Returns the unit impulse response (Input is the Dirac-Delta Function) of a\n    continuous-time system.\n\n    Parameters\n    ==========\n\n    system : SISOLinearTimeInvariant type\n        The LTI SISO system for which the Impulse Response is to be computed.\n    color : str, tuple, optional\n        The color of the line. Default is Blue.\n    show : boolean, optional\n        If ``True``, the plot will be displayed otherwise\n        the equivalent matplotlib ``plot`` object will be returned.\n        Defaults to True.\n    lower_limit : Number, optional\n        The lower limit of the plot range. Defaults to 0.\n    upper_limit : Number, optional\n        The upper limit of the plot range. Defaults to 10.\n    prec : int, optional\n        The decimal point precision for the point coordinate values.\n        Defaults to 8.\n    show_axes : boolean, optional\n        If ``True``, the coordinate axes will be shown. Defaults to False.\n    grid : boolean, optional\n        If ``True``, the plot will have a grid. Defaults to True.\n\n    Examples\n    ========\n\n    .. plot::\n        :context: close-figs\n        :format: doctest\n        :include-source: True\n\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> from sympy.physics.control.control_plots import impulse_response_plot\n        >>> tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)\n        >>> impulse_response_plot(tf1)   # doctest: +SKIP\n\n    See Also\n    ========\n\n    step_response_plot, ramp_response_plot\n\n    References\n    ==========\n\n    .. [1] https://www.mathworks.com/help/control/ref/lti.impulse.html\n\n    '
    (x, y) = impulse_response_numerical_data(system, prec=prec, lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Impulse Response of ${latex(system)}$', pad=20)
    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return
    return plt

def ramp_response_numerical_data(system, slope=1, prec=8, lower_limit=0, upper_limit=10, **kwargs):
    if False:
        print('Hello World!')
    '\n    Returns the numerical values of the points in the ramp response plot\n    of a SISO continuous-time system. By default, adaptive sampling\n    is used. If the user wants to instead get an uniformly\n    sampled response, then ``adaptive`` kwarg should be passed ``False``\n    and ``n`` must be passed as additional kwargs.\n    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`\n    for more details.\n\n    Parameters\n    ==========\n\n    system : SISOLinearTimeInvariant\n        The system for which the ramp response data is to be computed.\n    slope : Number, optional\n        The slope of the input ramp function. Defaults to 1.\n    prec : int, optional\n        The decimal point precision for the point coordinate values.\n        Defaults to 8.\n    lower_limit : Number, optional\n        The lower limit of the plot range. Defaults to 0.\n    upper_limit : Number, optional\n        The upper limit of the plot range. Defaults to 10.\n    kwargs :\n        Additional keyword arguments are passed to the underlying\n        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.\n\n    Returns\n    =======\n\n    tuple : (x, y)\n        x = Time-axis values of the points in the ramp response plot. NumPy array.\n        y = Amplitude-axis values of the points in the ramp response plot. NumPy array.\n\n    Raises\n    ======\n\n    NotImplementedError\n        When a SISO LTI system is not passed.\n\n        When time delay terms are present in the system.\n\n    ValueError\n        When more than one free symbol is present in the system.\n        The only variable in the transfer function should be\n        the variable of the Laplace transform.\n\n        When ``lower_limit`` parameter is less than 0.\n\n        When ``slope`` is negative.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import s\n    >>> from sympy.physics.control.lti import TransferFunction\n    >>> from sympy.physics.control.control_plots import ramp_response_numerical_data\n    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)\n    >>> ramp_response_numerical_data(tf1)   # doctest: +SKIP\n    (([0.0, 0.12166980856813935,..., 9.861246379582118, 10.0],\n    [1.4504508011325967e-09, 0.006046440489058766,..., 0.12499999999568202, 0.12499999999661349]))\n\n    See Also\n    ========\n\n    ramp_response_plot\n\n    '
    if slope < 0:
        raise ValueError('Slope must be greater than or equal to zero.')
    if lower_limit < 0:
        raise ValueError('Lower limit of time must be greater than or equal to zero.')
    _check_system(system)
    _x = Dummy('x')
    expr = slope * system.to_expr() / system.var ** 2
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit), **kwargs).get_points()

def ramp_response_plot(system, slope=1, color='b', prec=8, lower_limit=0, upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the ramp response of a continuous-time system.\n\n    Ramp function is defined as the straight line\n    passing through origin ($f(x) = mx$). The slope of\n    the ramp function can be varied by the user and\n    the default value is 1.\n\n    Parameters\n    ==========\n\n    system : SISOLinearTimeInvariant type\n        The LTI SISO system for which the Ramp Response is to be computed.\n    slope : Number, optional\n        The slope of the input ramp function. Defaults to 1.\n    color : str, tuple, optional\n        The color of the line. Default is Blue.\n    show : boolean, optional\n        If ``True``, the plot will be displayed otherwise\n        the equivalent matplotlib ``plot`` object will be returned.\n        Defaults to True.\n    lower_limit : Number, optional\n        The lower limit of the plot range. Defaults to 0.\n    upper_limit : Number, optional\n        The upper limit of the plot range. Defaults to 10.\n    prec : int, optional\n        The decimal point precision for the point coordinate values.\n        Defaults to 8.\n    show_axes : boolean, optional\n        If ``True``, the coordinate axes will be shown. Defaults to False.\n    grid : boolean, optional\n        If ``True``, the plot will have a grid. Defaults to True.\n\n    Examples\n    ========\n\n    .. plot::\n        :context: close-figs\n        :format: doctest\n        :include-source: True\n\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> from sympy.physics.control.control_plots import ramp_response_plot\n        >>> tf1 = TransferFunction(s, (s+4)*(s+8), s)\n        >>> ramp_response_plot(tf1, upper_limit=2)   # doctest: +SKIP\n\n    See Also\n    ========\n\n    step_response_plot, impulse_response_plot\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Ramp_function\n\n    '
    (x, y) = ramp_response_numerical_data(system, slope=slope, prec=prec, lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Ramp Response of ${latex(system)}$ [Slope = {slope}]', pad=20)
    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return
    return plt

def bode_magnitude_numerical_data(system, initial_exp=-5, final_exp=5, freq_unit='rad/sec', **kwargs):
    if False:
        return 10
    "\n    Returns the numerical data of the Bode magnitude plot of the system.\n    It is internally used by ``bode_magnitude_plot`` to get the data\n    for plotting Bode magnitude plot. Users can use this data to further\n    analyse the dynamics of the system or plot using a different\n    backend/plotting-module.\n\n    Parameters\n    ==========\n\n    system : SISOLinearTimeInvariant\n        The system for which the data is to be computed.\n    initial_exp : Number, optional\n        The initial exponent of 10 of the semilog plot. Defaults to -5.\n    final_exp : Number, optional\n        The final exponent of 10 of the semilog plot. Defaults to 5.\n    freq_unit : string, optional\n        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'`` (Hertz) as frequency units.\n\n    Returns\n    =======\n\n    tuple : (x, y)\n        x = x-axis values of the Bode magnitude plot.\n        y = y-axis values of the Bode magnitude plot.\n\n    Raises\n    ======\n\n    NotImplementedError\n        When a SISO LTI system is not passed.\n\n        When time delay terms are present in the system.\n\n    ValueError\n        When more than one free symbol is present in the system.\n        The only variable in the transfer function should be\n        the variable of the Laplace transform.\n\n        When incorrect frequency units are given as input.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import s\n    >>> from sympy.physics.control.lti import TransferFunction\n    >>> from sympy.physics.control.control_plots import bode_magnitude_numerical_data\n    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)\n    >>> bode_magnitude_numerical_data(tf1)   # doctest: +SKIP\n    ([1e-05, 1.5148378120533502e-05,..., 68437.36188804005, 100000.0],\n    [-6.020599914256786, -6.0205999155219505,..., -193.4117304087953, -200.00000000260573])\n\n    See Also\n    ========\n\n    bode_magnitude_plot, bode_phase_numerical_data\n\n    "
    _check_system(system)
    expr = system.to_expr()
    freq_units = ('rad/sec', 'Hz')
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')
    _w = Dummy('w', real=True)
    if freq_unit == 'Hz':
        repl = I * _w * 2 * pi
    else:
        repl = I * _w
    w_expr = expr.subs({system.var: repl})
    mag = 20 * log(Abs(w_expr), 10)
    (x, y) = LineOver1DRangeSeries(mag, (_w, 10 ** initial_exp, 10 ** final_exp), xscale='log', **kwargs).get_points()
    return (x, y)

def bode_magnitude_plot(system, initial_exp=-5, final_exp=5, color='b', show_axes=False, grid=True, show=True, freq_unit='rad/sec', **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Returns the Bode magnitude plot of a continuous-time system.\n\n    See ``bode_plot`` for all the parameters.\n    '
    (x, y) = bode_magnitude_numerical_data(system, initial_exp=initial_exp, final_exp=final_exp, freq_unit=freq_unit)
    plt.plot(x, y, color=color, **kwargs)
    plt.xscale('log')
    plt.xlabel('Frequency (%s) [Log Scale]' % freq_unit)
    plt.ylabel('Magnitude (dB)')
    plt.title(f'Bode Plot (Magnitude) of ${latex(system)}$', pad=20)
    if grid:
        plt.grid(True)
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return
    return plt

def bode_phase_numerical_data(system, initial_exp=-5, final_exp=5, freq_unit='rad/sec', phase_unit='rad', phase_unwrap=True, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns the numerical data of the Bode phase plot of the system.\n    It is internally used by ``bode_phase_plot`` to get the data\n    for plotting Bode phase plot. Users can use this data to further\n    analyse the dynamics of the system or plot using a different\n    backend/plotting-module.\n\n    Parameters\n    ==========\n\n    system : SISOLinearTimeInvariant\n        The system for which the Bode phase plot data is to be computed.\n    initial_exp : Number, optional\n        The initial exponent of 10 of the semilog plot. Defaults to -5.\n    final_exp : Number, optional\n        The final exponent of 10 of the semilog plot. Defaults to 5.\n    freq_unit : string, optional\n        User can choose between ``'rad/sec'`` (radians/second) and '``'Hz'`` (Hertz) as frequency units.\n    phase_unit : string, optional\n        User can choose between ``'rad'`` (radians) and ``'deg'`` (degree) as phase units.\n    phase_unwrap : bool, optional\n        Set to ``True`` by default.\n\n    Returns\n    =======\n\n    tuple : (x, y)\n        x = x-axis values of the Bode phase plot.\n        y = y-axis values of the Bode phase plot.\n\n    Raises\n    ======\n\n    NotImplementedError\n        When a SISO LTI system is not passed.\n\n        When time delay terms are present in the system.\n\n    ValueError\n        When more than one free symbol is present in the system.\n        The only variable in the transfer function should be\n        the variable of the Laplace transform.\n\n        When incorrect frequency or phase units are given as input.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import s\n    >>> from sympy.physics.control.lti import TransferFunction\n    >>> from sympy.physics.control.control_plots import bode_phase_numerical_data\n    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)\n    >>> bode_phase_numerical_data(tf1)   # doctest: +SKIP\n    ([1e-05, 1.4472354033813751e-05, 2.035581932165858e-05,..., 47577.3248186011, 67884.09326036123, 100000.0],\n    [-2.5000000000291665e-05, -3.6180885085e-05, -5.08895483066e-05,...,-3.1415085799262523, -3.14155265358979])\n\n    See Also\n    ========\n\n    bode_magnitude_plot, bode_phase_numerical_data\n\n    "
    _check_system(system)
    expr = system.to_expr()
    freq_units = ('rad/sec', 'Hz')
    phase_units = ('rad', 'deg')
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')
    if phase_unit not in phase_units:
        raise ValueError('Only "rad" and "deg" are accepted phase units.')
    _w = Dummy('w', real=True)
    if freq_unit == 'Hz':
        repl = I * _w * 2 * pi
    else:
        repl = I * _w
    w_expr = expr.subs({system.var: repl})
    if phase_unit == 'deg':
        phase = arg(w_expr) * 180 / pi
    else:
        phase = arg(w_expr)
    (x, y) = LineOver1DRangeSeries(phase, (_w, 10 ** initial_exp, 10 ** final_exp), xscale='log', **kwargs).get_points()
    half = None
    if phase_unwrap:
        if phase_unit == 'rad':
            half = pi
        elif phase_unit == 'deg':
            half = 180
    if half:
        unit = 2 * half
        for i in range(1, len(y)):
            diff = y[i] - y[i - 1]
            if diff > half:
                y[i] = y[i] - unit
            elif diff < -half:
                y[i] = y[i] + unit
    return (x, y)

def bode_phase_plot(system, initial_exp=-5, final_exp=5, color='b', show_axes=False, grid=True, show=True, freq_unit='rad/sec', phase_unit='rad', phase_unwrap=True, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Returns the Bode phase plot of a continuous-time system.\n\n    See ``bode_plot`` for all the parameters.\n    '
    (x, y) = bode_phase_numerical_data(system, initial_exp=initial_exp, final_exp=final_exp, freq_unit=freq_unit, phase_unit=phase_unit, phase_unwrap=phase_unwrap)
    plt.plot(x, y, color=color, **kwargs)
    plt.xscale('log')
    plt.xlabel('Frequency (%s) [Log Scale]' % freq_unit)
    plt.ylabel('Phase (%s)' % phase_unit)
    plt.title(f'Bode Plot (Phase) of ${latex(system)}$', pad=20)
    if grid:
        plt.grid(True)
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return
    return plt

def bode_plot(system, initial_exp=-5, final_exp=5, grid=True, show_axes=False, show=True, freq_unit='rad/sec', phase_unit='rad', phase_unwrap=True, **kwargs):
    if False:
        return 10
    "\n    Returns the Bode phase and magnitude plots of a continuous-time system.\n\n    Parameters\n    ==========\n\n    system : SISOLinearTimeInvariant type\n        The LTI SISO system for which the Bode Plot is to be computed.\n    initial_exp : Number, optional\n        The initial exponent of 10 of the semilog plot. Defaults to -5.\n    final_exp : Number, optional\n        The final exponent of 10 of the semilog plot. Defaults to 5.\n    show : boolean, optional\n        If ``True``, the plot will be displayed otherwise\n        the equivalent matplotlib ``plot`` object will be returned.\n        Defaults to True.\n    prec : int, optional\n        The decimal point precision for the point coordinate values.\n        Defaults to 8.\n    grid : boolean, optional\n        If ``True``, the plot will have a grid. Defaults to True.\n    show_axes : boolean, optional\n        If ``True``, the coordinate axes will be shown. Defaults to False.\n    freq_unit : string, optional\n        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'`` (Hertz) as frequency units.\n    phase_unit : string, optional\n        User can choose between ``'rad'`` (radians) and ``'deg'`` (degree) as phase units.\n\n    Examples\n    ========\n\n    .. plot::\n        :context: close-figs\n        :format: doctest\n        :include-source: True\n\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> from sympy.physics.control.control_plots import bode_plot\n        >>> tf1 = TransferFunction(1*s**2 + 0.1*s + 7.5, 1*s**4 + 0.12*s**3 + 9*s**2, s)\n        >>> bode_plot(tf1, initial_exp=0.2, final_exp=0.7)   # doctest: +SKIP\n\n    See Also\n    ========\n\n    bode_magnitude_plot, bode_phase_plot\n\n    "
    plt.subplot(211)
    mag = bode_magnitude_plot(system, initial_exp=initial_exp, final_exp=final_exp, show=False, grid=grid, show_axes=show_axes, freq_unit=freq_unit, **kwargs)
    mag.title(f'Bode Plot of ${latex(system)}$', pad=20)
    mag.xlabel(None)
    plt.subplot(212)
    bode_phase_plot(system, initial_exp=initial_exp, final_exp=final_exp, show=False, grid=grid, show_axes=show_axes, freq_unit=freq_unit, phase_unit=phase_unit, phase_unwrap=phase_unwrap, **kwargs).title(None)
    if show:
        plt.show()
        return
    return plt