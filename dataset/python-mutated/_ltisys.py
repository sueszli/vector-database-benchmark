"""
ltisys -- a collection of classes and functions for modeling linear
time invariant systems.
"""
import copy
import cupy
from cupyx.scipy import linalg
from cupyx.scipy.interpolate import make_interp_spline
from cupyx.scipy.linalg import expm, block_diag
from cupyx.scipy.signal._lti_conversion import _atleast_2d_or_none, abcd_normalize
from cupyx.scipy.signal._iir_filter_conversions import normalize, tf2zpk, tf2ss, zpk2ss, ss2tf, ss2zpk, zpk2tf
from cupyx.scipy.signal._filter_design import freqz, freqz_zpk, freqs, freqs_zpk

class LinearTimeInvariant:

    def __new__(cls, *system, **kwargs):
        if False:
            print('Hello World!')
        "Create a new object, don't allow direct instances."
        if cls is LinearTimeInvariant:
            raise NotImplementedError('The LinearTimeInvariant class is not meant to be used directly, use `lti` or `dlti` instead.')
        return super().__new__(cls)

    def __init__(self):
        if False:
            return 10
        '\n        Initialize the `lti` baseclass.\n\n        The heavy lifting is done by the subclasses.\n        '
        super().__init__()
        self.inputs = None
        self.outputs = None
        self._dt = None

    @property
    def dt(self):
        if False:
            while True:
                i = 10
        'Return the sampling time of the system, `None` for `lti` systems.'
        return self._dt

    @property
    def _dt_dict(self):
        if False:
            return 10
        if self.dt is None:
            return {}
        else:
            return {'dt': self.dt}

    @property
    def zeros(self):
        if False:
            return 10
        'Zeros of the system.'
        return self.to_zpk().zeros

    @property
    def poles(self):
        if False:
            i = 10
            return i + 15
        'Poles of the system.'
        return self.to_zpk().poles

    def _as_ss(self):
        if False:
            return 10
        'Convert to `StateSpace` system, without copying.\n\n        Returns\n        -------\n        sys: StateSpace\n            The `StateSpace` system. If the class is already an instance of\n            `StateSpace` then this instance is returned.\n        '
        if isinstance(self, StateSpace):
            return self
        else:
            return self.to_ss()

    def _as_zpk(self):
        if False:
            while True:
                i = 10
        'Convert to `ZerosPolesGain` system, without copying.\n\n        Returns\n        -------\n        sys: ZerosPolesGain\n            The `ZerosPolesGain` system. If the class is already an instance of\n            `ZerosPolesGain` then this instance is returned.\n        '
        if isinstance(self, ZerosPolesGain):
            return self
        else:
            return self.to_zpk()

    def _as_tf(self):
        if False:
            i = 10
            return i + 15
        'Convert to `TransferFunction` system, without copying.\n\n        Returns\n        -------\n        sys: ZerosPolesGain\n            The `TransferFunction` system. If the class is already an instance\n            of `TransferFunction` then this instance is returned.\n        '
        if isinstance(self, TransferFunction):
            return self
        else:
            return self.to_tf()

class lti(LinearTimeInvariant):
    """
    Continuous-time linear time invariant system base class.

    Parameters
    ----------
    *system : arguments
        The `lti` class can be instantiated with either 2, 3 or 4 arguments.
        The following gives the number of arguments and the corresponding
        continuous-time subclass that is created:

            * 2: `TransferFunction`:  (numerator, denominator)
            * 3: `ZerosPolesGain`: (zeros, poles, gain)
            * 4: `StateSpace`:  (A, B, C, D)

        Each argument can be an array or a sequence.

    See Also
    --------
    scipy.signal.lti
    ZerosPolesGain, StateSpace, TransferFunction, dlti

    Notes
    -----
    `lti` instances do not exist directly. Instead, `lti` creates an instance
    of one of its subclasses: `StateSpace`, `TransferFunction` or
    `ZerosPolesGain`.

    If (numerator, denominator) is passed in for ``*system``, coefficients for
    both the numerator and denominator should be specified in descending
    exponent order (e.g., ``s^2 + 3s + 5`` would be represented as ``[1, 3,
    5]``).

    Changing the value of properties that are not directly part of the current
    system representation (such as the `zeros` of a `StateSpace` system) is
    very inefficient and may lead to numerical inaccuracies. It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.
    """

    def __new__(cls, *system):
        if False:
            i = 10
            return i + 15
        'Create an instance of the appropriate subclass.'
        if cls is lti:
            N = len(system)
            if N == 2:
                return TransferFunctionContinuous.__new__(TransferFunctionContinuous, *system)
            elif N == 3:
                return ZerosPolesGainContinuous.__new__(ZerosPolesGainContinuous, *system)
            elif N == 4:
                return StateSpaceContinuous.__new__(StateSpaceContinuous, *system)
            else:
                raise ValueError('`system` needs to be an instance of `lti` or have 2, 3 or 4 arguments.')
        return super().__new__(cls)

    def __init__(self, *system):
        if False:
            return 10
        '\n        Initialize the `lti` baseclass.\n\n        The heavy lifting is done by the subclasses.\n        '
        super().__init__(*system)

    def impulse(self, X0=None, T=None, N=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the impulse response of a continuous-time system.\n        See `impulse` for details.\n        '
        return impulse(self, X0=X0, T=T, N=N)

    def step(self, X0=None, T=None, N=None):
        if False:
            while True:
                i = 10
        '\n        Return the step response of a continuous-time system.\n        See `step` for details.\n        '
        return step(self, X0=X0, T=T, N=N)

    def output(self, U, T, X0=None):
        if False:
            return 10
        '\n        Return the response of a continuous-time system to input `U`.\n        See `lsim` for details.\n        '
        return lsim(self, U, T, X0=X0)

    def bode(self, w=None, n=100):
        if False:
            print('Hello World!')
        '\n        Calculate Bode magnitude and phase data of a continuous-time system.\n\n        Returns a 3-tuple containing arrays of frequencies [rad/s], magnitude\n        [dB] and phase [deg]. See `bode` for details.\n        '
        return bode(self, w=w, n=n)

    def freqresp(self, w=None, n=10000):
        if False:
            while True:
                i = 10
        '\n        Calculate the frequency response of a continuous-time system.\n\n        Returns a 2-tuple containing arrays of frequencies [rad/s] and\n        complex magnitude.\n        See `freqresp` for details.\n        '
        return freqresp(self, w=w, n=n)

    def to_discrete(self, dt, method='zoh', alpha=None):
        if False:
            while True:
                i = 10
        'Return a discretized version of the current system.\n\n        Parameters: See `cont2discrete` for details.\n\n        Returns\n        -------\n        sys: instance of `dlti`\n        '
        raise NotImplementedError('to_discrete is not implemented for this system class.')

class dlti(LinearTimeInvariant):
    """
    Discrete-time linear time invariant system base class.

    Parameters
    ----------
    *system: arguments
        The `dlti` class can be instantiated with either 2, 3 or 4 arguments.
        The following gives the number of arguments and the corresponding
        discrete-time subclass that is created:

            * 2: `TransferFunction`:  (numerator, denominator)
            * 3: `ZerosPolesGain`: (zeros, poles, gain)
            * 4: `StateSpace`:  (A, B, C, D)

        Each argument can be an array or a sequence.
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to ``True``
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.

    See Also
    --------
    scipy.signal.dlti
    ZerosPolesGain, StateSpace, TransferFunction, lti

    Notes
    -----
    `dlti` instances do not exist directly. Instead, `dlti` creates an instance
    of one of its subclasses: `StateSpace`, `TransferFunction` or
    `ZerosPolesGain`.

    Changing the value of properties that are not directly part of the current
    system representation (such as the `zeros` of a `StateSpace` system) is
    very inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.

    If (numerator, denominator) is passed in for ``*system``, coefficients for
    both the numerator and denominator should be specified in descending
    exponent order (e.g., ``z^2 + 3z + 5`` would be represented as ``[1, 3,
    5]``).
    """

    def __new__(cls, *system, **kwargs):
        if False:
            i = 10
            return i + 15
        'Create an instance of the appropriate subclass.'
        if cls is dlti:
            N = len(system)
            if N == 2:
                return TransferFunctionDiscrete.__new__(TransferFunctionDiscrete, *system, **kwargs)
            elif N == 3:
                return ZerosPolesGainDiscrete.__new__(ZerosPolesGainDiscrete, *system, **kwargs)
            elif N == 4:
                return StateSpaceDiscrete.__new__(StateSpaceDiscrete, *system, **kwargs)
            else:
                raise ValueError('`system` needs to be an instance of `dlti` or have 2, 3 or 4 arguments.')
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the `lti` baseclass.\n\n        The heavy lifting is done by the subclasses.\n        '
        dt = kwargs.pop('dt', True)
        super().__init__(*system, **kwargs)
        self.dt = dt

    @property
    def dt(self):
        if False:
            while True:
                i = 10
        'Return the sampling time of the system.'
        return self._dt

    @dt.setter
    def dt(self, dt):
        if False:
            return 10
        self._dt = dt

    def impulse(self, x0=None, t=None, n=None):
        if False:
            print('Hello World!')
        '\n        Return the impulse response of the discrete-time `dlti` system.\n        See `dimpulse` for details.\n        '
        return dimpulse(self, x0=x0, t=t, n=n)

    def step(self, x0=None, t=None, n=None):
        if False:
            while True:
                i = 10
        '\n        Return the step response of the discrete-time `dlti` system.\n        See `dstep` for details.\n        '
        return dstep(self, x0=x0, t=t, n=n)

    def output(self, u, t, x0=None):
        if False:
            print('Hello World!')
        '\n        Return the response of the discrete-time system to input `u`.\n        See `dlsim` for details.\n        '
        return dlsim(self, u, t, x0=x0)

    def bode(self, w=None, n=100):
        if False:
            return 10
        '\n        Calculate Bode magnitude and phase data of a discrete-time system.\n\n        Returns a 3-tuple containing arrays of frequencies [rad/s], magnitude\n        [dB] and phase [deg]. See `dbode` for details.\n        '
        return dbode(self, w=w, n=n)

    def freqresp(self, w=None, n=10000, whole=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculate the frequency response of a discrete-time system.\n\n        Returns a 2-tuple containing arrays of frequencies [rad/s] and\n        complex magnitude.\n        See `dfreqresp` for details.\n\n        '
        return dfreqresp(self, w=w, n=n, whole=whole)

class TransferFunction(LinearTimeInvariant):
    """Linear Time Invariant system class in transfer function form.

    Represents the system as the continuous-time transfer function
    :math:`H(s)=\\sum_{i=0}^N b[N-i] s^i / \\sum_{j=0}^M a[M-j] s^j` or the
    discrete-time transfer function
    :math:`H(z)=\\sum_{i=0}^N b[N-i] z^i / \\sum_{j=0}^M a[M-j] z^j`, where
    :math:`b` are elements of the numerator `num`, :math:`a` are elements of
    the denominator `den`, and ``N == len(b) - 1``, ``M == len(a) - 1``.
    `TransferFunction` systems inherit additional
    functionality from the `lti`, respectively the `dlti` classes, depending on
    which system representation is used.

    Parameters
    ----------
    *system: arguments
        The `TransferFunction` class can be instantiated with 1 or 2
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` or `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 2: array_like: (numerator, denominator)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `None`
        (continuous-time). Must be specified as a keyword argument, for
        example, ``dt=0.1``.

    See Also
    --------
    scipy.signal.TransferFunction
    ZerosPolesGain, StateSpace, lti, dlti
    tf2ss, tf2zpk, tf2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `TransferFunction` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    If (numerator, denominator) is passed in for ``*system``, coefficients
    for both the numerator and denominator should be specified in descending
    exponent order (e.g. ``s^2 + 3s + 5`` or ``z^2 + 3z + 5`` would be
    represented as ``[1, 3, 5]``)
    """

    def __new__(cls, *system, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Handle object conversion if input is an instance of lti.'
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            return system[0].to_tf()
        if cls is TransferFunction:
            if kwargs.get('dt') is None:
                return TransferFunctionContinuous.__new__(TransferFunctionContinuous, *system, **kwargs)
            else:
                return TransferFunctionDiscrete.__new__(TransferFunctionDiscrete, *system, **kwargs)
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the state space LTI system.'
        if isinstance(system[0], LinearTimeInvariant):
            return
        super().__init__(**kwargs)
        self._num = None
        self._den = None
        (self.num, self.den) = normalize(*system)

    def __repr__(self):
        if False:
            print('Hello World!')
        "Return representation of the system's transfer function"
        return '{}(\n{},\n{},\ndt: {}\n)'.format(self.__class__.__name__, repr(self.num), repr(self.den), repr(self.dt))

    @property
    def num(self):
        if False:
            return 10
        'Numerator of the `TransferFunction` system.'
        return self._num

    @num.setter
    def num(self, num):
        if False:
            for i in range(10):
                print('nop')
        self._num = cupy.atleast_1d(num)
        if len(self.num.shape) > 1:
            (self.outputs, self.inputs) = self.num.shape
        else:
            self.outputs = 1
            self.inputs = 1

    @property
    def den(self):
        if False:
            print('Hello World!')
        'Denominator of the `TransferFunction` system.'
        return self._den

    @den.setter
    def den(self, den):
        if False:
            i = 10
            return i + 15
        self._den = cupy.atleast_1d(den)

    def _copy(self, system):
        if False:
            while True:
                i = 10
        '\n        Copy the parameters of another `TransferFunction` object\n\n        Parameters\n        ----------\n        system : `TransferFunction`\n            The `StateSpace` system that is to be copied\n\n        '
        self.num = system.num
        self.den = system.den

    def to_tf(self):
        if False:
            return 10
        '\n        Return a copy of the current `TransferFunction` system.\n\n        Returns\n        -------\n        sys : instance of `TransferFunction`\n            The current system (copy)\n\n        '
        return copy.deepcopy(self)

    def to_zpk(self):
        if False:
            return 10
        '\n        Convert system representation to `ZerosPolesGain`.\n\n        Returns\n        -------\n        sys : instance of `ZerosPolesGain`\n            Zeros, poles, gain representation of the current system\n\n        '
        return ZerosPolesGain(*tf2zpk(self.num, self.den), **self._dt_dict)

    def to_ss(self):
        if False:
            while True:
                i = 10
        '\n        Convert system representation to `StateSpace`.\n\n        Returns\n        -------\n        sys : instance of `StateSpace`\n            State space model of the current system\n\n        '
        return StateSpace(*tf2ss(self.num, self.den), **self._dt_dict)

    @staticmethod
    def _z_to_zinv(num, den):
        if False:
            print('Hello World!')
        "Change a transfer function from the variable `z` to `z**-1`.\n\n        Parameters\n        ----------\n        num, den: 1d array_like\n            Sequences representing the coefficients of the numerator and\n            denominator polynomials, in order of descending degree of 'z'.\n            That is, ``5z**2 + 3z + 2`` is presented as ``[5, 3, 2]``.\n\n        Returns\n        -------\n        num, den: 1d array_like\n            Sequences representing the coefficients of the numerator and\n            denominator polynomials, in order of ascending degree of 'z**-1'.\n            That is, ``5 + 3 z**-1 + 2 z**-2`` is presented as ``[5, 3, 2]``.\n        "
        diff = len(num) - len(den)
        if diff > 0:
            den = cupy.hstack((cupy.zeros(diff), den))
        elif diff < 0:
            num = cupy.hstack((cupy.zeros(-diff), num))
        return (num, den)

    @staticmethod
    def _zinv_to_z(num, den):
        if False:
            i = 10
            return i + 15
        "Change a transfer function from the variable `z` to `z**-1`.\n\n        Parameters\n        ----------\n        num, den: 1d array_like\n            Sequences representing the coefficients of the numerator and\n            denominator polynomials, in order of ascending degree of 'z**-1'.\n            That is, ``5 + 3 z**-1 + 2 z**-2`` is presented as ``[5, 3, 2]``.\n\n        Returns\n        -------\n        num, den: 1d array_like\n            Sequences representing the coefficients of the numerator and\n            denominator polynomials, in order of descending degree of 'z'.\n            That is, ``5z**2 + 3z + 2`` is presented as ``[5, 3, 2]``.\n        "
        diff = len(num) - len(den)
        if diff > 0:
            den = cupy.hstack((den, cupy.zeros(diff)))
        elif diff < 0:
            num = cupy.hstack((num, cupy.zeros(-diff)))
        return (num, den)

class TransferFunctionContinuous(TransferFunction, lti):
    """
    Continuous-time Linear Time Invariant system in transfer function form.

    Represents the system as the transfer function
    :math:`H(s)=\\sum_{i=0}^N b[N-i] s^i / \\sum_{j=0}^M a[M-j] s^j`, where
    :math:`b` are elements of the numerator `num`, :math:`a` are elements of
    the denominator `den`, and ``N == len(b) - 1``, ``M == len(a) - 1``.
    Continuous-time `TransferFunction` systems inherit additional
    functionality from the `lti` class.

    Parameters
    ----------
    *system: arguments
        The `TransferFunction` class can be instantiated with 1 or 2
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 2: array_like: (numerator, denominator)

    See Also
    --------
    scipy.signal.TransferFunction
    ZerosPolesGain, StateSpace, lti
    tf2ss, tf2zpk, tf2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `TransferFunction` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    If (numerator, denominator) is passed in for ``*system``, coefficients
    for both the numerator and denominator should be specified in descending
    exponent order (e.g. ``s^2 + 3s + 5`` would be represented as
    ``[1, 3, 5]``)

    """

    def to_discrete(self, dt, method='zoh', alpha=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns the discretized `TransferFunction` system.\n\n        Parameters: See `cont2discrete` for details.\n\n        Returns\n        -------\n        sys: instance of `dlti` and `StateSpace`\n        '
        return TransferFunction(*cont2discrete((self.num, self.den), dt, method=method, alpha=alpha)[:-1], dt=dt)

class TransferFunctionDiscrete(TransferFunction, dlti):
    """
    Discrete-time Linear Time Invariant system in transfer function form.

    Represents the system as the transfer function
    :math:`H(z)=\\sum_{i=0}^N b[N-i] z^i / \\sum_{j=0}^M a[M-j] z^j`, where
    :math:`b` are elements of the numerator `num`, :math:`a` are elements of
    the denominator `den`, and ``N == len(b) - 1``, ``M == len(a) - 1``.
    Discrete-time `TransferFunction` systems inherit additional functionality
    from the `dlti` class.

    Parameters
    ----------
    *system: arguments
        The `TransferFunction` class can be instantiated with 1 or 2
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 2: array_like: (numerator, denominator)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `True`
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.

    See Also
    --------
    scipy.signal.TransferFunctionDiscrete
    ZerosPolesGain, StateSpace, dlti
    tf2ss, tf2zpk, tf2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `TransferFunction` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.

    If (numerator, denominator) is passed in for ``*system``, coefficients
    for both the numerator and denominator should be specified in descending
    exponent order (e.g., ``z^2 + 3z + 5`` would be represented as
    ``[1, 3, 5]``).
    """
    pass

class ZerosPolesGain(LinearTimeInvariant):
    """
    Linear Time Invariant system class in zeros, poles, gain form.

    Represents the system as the continuous- or discrete-time transfer function
    :math:`H(s)=k \\prod_i (s - z[i]) / \\prod_j (s - p[j])`, where :math:`k` is
    the `gain`, :math:`z` are the `zeros` and :math:`p` are the `poles`.
    `ZerosPolesGain` systems inherit additional functionality from the `lti`,
    respectively the `dlti` classes, depending on which system representation
    is used.

    Parameters
    ----------
    *system : arguments
        The `ZerosPolesGain` class can be instantiated with 1 or 3
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` or `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 3: array_like: (zeros, poles, gain)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `None`
        (continuous-time). Must be specified as a keyword argument, for
        example, ``dt=0.1``.


    See Also
    --------
    scipy.signal.ZerosPolesGain
    TransferFunction, StateSpace, lti, dlti
    zpk2ss, zpk2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `ZerosPolesGain` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.
    """

    def __new__(cls, *system, **kwargs):
        if False:
            print('Hello World!')
        'Handle object conversion if input is an instance of `lti`'
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            return system[0].to_zpk()
        if cls is ZerosPolesGain:
            if kwargs.get('dt') is None:
                return ZerosPolesGainContinuous.__new__(ZerosPolesGainContinuous, *system, **kwargs)
            else:
                return ZerosPolesGainDiscrete.__new__(ZerosPolesGainDiscrete, *system, **kwargs)
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        if False:
            while True:
                i = 10
        'Initialize the zeros, poles, gain system.'
        if isinstance(system[0], LinearTimeInvariant):
            return
        super().__init__(**kwargs)
        self._zeros = None
        self._poles = None
        self._gain = None
        (self.zeros, self.poles, self.gain) = system

    def __repr__(self):
        if False:
            return 10
        'Return representation of the `ZerosPolesGain` system.'
        return '{}(\n{},\n{},\n{},\ndt: {}\n)'.format(self.__class__.__name__, repr(self.zeros), repr(self.poles), repr(self.gain), repr(self.dt))

    @property
    def zeros(self):
        if False:
            i = 10
            return i + 15
        'Zeros of the `ZerosPolesGain` system.'
        return self._zeros

    @zeros.setter
    def zeros(self, zeros):
        if False:
            print('Hello World!')
        self._zeros = cupy.atleast_1d(zeros)
        if len(self.zeros.shape) > 1:
            (self.outputs, self.inputs) = self.zeros.shape
        else:
            self.outputs = 1
            self.inputs = 1

    @property
    def poles(self):
        if False:
            print('Hello World!')
        'Poles of the `ZerosPolesGain` system.'
        return self._poles

    @poles.setter
    def poles(self, poles):
        if False:
            return 10
        self._poles = cupy.atleast_1d(poles)

    @property
    def gain(self):
        if False:
            i = 10
            return i + 15
        'Gain of the `ZerosPolesGain` system.'
        return self._gain

    @gain.setter
    def gain(self, gain):
        if False:
            for i in range(10):
                print('nop')
        self._gain = gain

    def _copy(self, system):
        if False:
            i = 10
            return i + 15
        '\n        Copy the parameters of another `ZerosPolesGain` system.\n\n        Parameters\n        ----------\n        system : instance of `ZerosPolesGain`\n            The zeros, poles gain system that is to be copied\n\n        '
        self.poles = system.poles
        self.zeros = system.zeros
        self.gain = system.gain

    def to_tf(self):
        if False:
            while True:
                i = 10
        '\n        Convert system representation to `TransferFunction`.\n\n        Returns\n        -------\n        sys : instance of `TransferFunction`\n            Transfer function of the current system\n\n        '
        return TransferFunction(*zpk2tf(self.zeros, self.poles, self.gain), **self._dt_dict)

    def to_zpk(self):
        if False:
            i = 10
            return i + 15
        "\n        Return a copy of the current 'ZerosPolesGain' system.\n\n        Returns\n        -------\n        sys : instance of `ZerosPolesGain`\n            The current system (copy)\n\n        "
        return copy.deepcopy(self)

    def to_ss(self):
        if False:
            while True:
                i = 10
        '\n        Convert system representation to `StateSpace`.\n\n        Returns\n        -------\n        sys : instance of `StateSpace`\n            State space model of the current system\n\n        '
        return StateSpace(*zpk2ss(self.zeros, self.poles, self.gain), **self._dt_dict)

class ZerosPolesGainContinuous(ZerosPolesGain, lti):
    """
    Continuous-time Linear Time Invariant system in zeros, poles, gain form.

    Represents the system as the continuous time transfer function
    :math:`H(s)=k \\prod_i (s - z[i]) / \\prod_j (s - p[j])`, where :math:`k` is
    the `gain`, :math:`z` are the `zeros` and :math:`p` are the `poles`.
    Continuous-time `ZerosPolesGain` systems inherit additional functionality
    from the `lti` class.

    Parameters
    ----------
    *system : arguments
        The `ZerosPolesGain` class can be instantiated with 1 or 3
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 3: array_like: (zeros, poles, gain)

    See Also
    --------
    TransferFunction, StateSpace, lti
    zpk2ss, zpk2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `ZerosPolesGain` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    Examples
    --------
    Construct the transfer function
    :math:`H(s)=\\frac{5(s - 1)(s - 2)}{(s - 3)(s - 4)}`:

    >>> from scipy import signal

    >>> signal.ZerosPolesGain([1, 2], [3, 4], 5)
    ZerosPolesGainContinuous(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: None
    )

    """

    def to_discrete(self, dt, method='zoh', alpha=None):
        if False:
            return 10
        '\n        Returns the discretized `ZerosPolesGain` system.\n\n        Parameters: See `cont2discrete` for details.\n\n        Returns\n        -------\n        sys: instance of `dlti` and `ZerosPolesGain`\n        '
        return ZerosPolesGain(*cont2discrete((self.zeros, self.poles, self.gain), dt, method=method, alpha=alpha)[:-1], dt=dt)

class ZerosPolesGainDiscrete(ZerosPolesGain, dlti):
    """
    Discrete-time Linear Time Invariant system in zeros, poles, gain form.

    Represents the system as the discrete-time transfer function
    :math:`H(z)=k \\prod_i (z - q[i]) / \\prod_j (z - p[j])`, where :math:`k` is
    the `gain`, :math:`q` are the `zeros` and :math:`p` are the `poles`.
    Discrete-time `ZerosPolesGain` systems inherit additional functionality
    from the `dlti` class.

    Parameters
    ----------
    *system : arguments
        The `ZerosPolesGain` class can be instantiated with 1 or 3
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 3: array_like: (zeros, poles, gain)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `True`
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.

    See Also
    --------
    scipy.signal.ZerosPolesGainDiscrete
    TransferFunction, StateSpace, dlti
    zpk2ss, zpk2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `ZerosPolesGain` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.
    """
    pass

class StateSpace(LinearTimeInvariant):
    """
    Linear Time Invariant system in state-space form.

    Represents the system as the continuous-time, first order differential
    equation :math:`\\dot{x} = A x + B u` or the discrete-time difference
    equation :math:`x[k+1] = A x[k] + B u[k]`. `StateSpace` systems
    inherit additional functionality from the `lti`, respectively the `dlti`
    classes, depending on which system representation is used.

    Parameters
    ----------
    *system: arguments
        The `StateSpace` class can be instantiated with 1 or 4 arguments.
        The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` or `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 4: array_like: (A, B, C, D)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `None`
        (continuous-time). Must be specified as a keyword argument, for
        example, ``dt=0.1``.

    See Also
    --------
    scipy.signal.StateSpace
    TransferFunction, ZerosPolesGain, lti, dlti
    ss2zpk, ss2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `StateSpace` system representation (such as `zeros` or `poles`) is very
    inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.
    """
    __array_priority__ = 100.0
    __array_ufunc__ = None

    def __new__(cls, *system, **kwargs):
        if False:
            print('Hello World!')
        'Create new StateSpace object and settle inheritance.'
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            return system[0].to_ss()
        if cls is StateSpace:
            if kwargs.get('dt') is None:
                return StateSpaceContinuous.__new__(StateSpaceContinuous, *system, **kwargs)
            else:
                return StateSpaceDiscrete.__new__(StateSpaceDiscrete, *system, **kwargs)
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the state space lti/dlti system.'
        if isinstance(system[0], LinearTimeInvariant):
            return
        super().__init__(**kwargs)
        self._A = None
        self._B = None
        self._C = None
        self._D = None
        (self.A, self.B, self.C, self.D) = abcd_normalize(*system)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return representation of the `StateSpace` system.'
        return '{}(\n{},\n{},\n{},\n{},\ndt: {}\n)'.format(self.__class__.__name__, repr(self.A), repr(self.B), repr(self.C), repr(self.D), repr(self.dt))

    def _check_binop_other(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, (StateSpace, cupy.ndarray, float, complex, cupy.number, int))

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Post-multiply another system or a scalar\n\n        Handles multiplication of systems in the sense of a frequency domain\n        multiplication. That means, given two systems E1(s) and E2(s), their\n        multiplication, H(s) = E1(s) * E2(s), means that applying H(s) to U(s)\n        is equivalent to first applying E2(s), and then E1(s).\n\n        Notes\n        -----\n        For SISO systems the order of system application does not matter.\n        However, for MIMO systems, where the two systems are matrices, the\n        order above ensures standard Matrix multiplication rules apply.\n        '
        if not self._check_binop_other(other):
            return NotImplemented
        if isinstance(other, StateSpace):
            if type(other) is not type(self):
                return NotImplemented
            if self.dt != other.dt:
                raise TypeError('Cannot multiply systems with different `dt`.')
            n1 = self.A.shape[0]
            n2 = other.A.shape[0]
            a = cupy.vstack((cupy.hstack((self.A, self.B @ other.C)), cupy.hstack((cupy.zeros((n2, n1)), other.A))))
            b = cupy.vstack((self.B @ other.D, other.B))
            c = cupy.hstack((self.C, self.D @ other.C))
            d = self.D @ other.D
        else:
            a = self.A
            b = self.B @ other
            c = self.C
            d = self.D @ other
        common_dtype = cupy.result_type(a.dtype, b.dtype, c.dtype, d.dtype)
        return StateSpace(cupy.asarray(a, dtype=common_dtype), cupy.asarray(b, dtype=common_dtype), cupy.asarray(c, dtype=common_dtype), cupy.asarray(d, dtype=common_dtype), **self._dt_dict)

    def __rmul__(self, other):
        if False:
            return 10
        'Pre-multiply a scalar or matrix (but not StateSpace)'
        if not self._check_binop_other(other) or isinstance(other, StateSpace):
            return NotImplemented
        a = self.A
        b = self.B
        c = other @ self.C
        d = other @ self.D
        common_dtype = cupy.result_type(a.dtype, b.dtype, c.dtype, d.dtype)
        return StateSpace(cupy.asarray(a, dtype=common_dtype), cupy.asarray(b, dtype=common_dtype), cupy.asarray(c, dtype=common_dtype), cupy.asarray(d, dtype=common_dtype), **self._dt_dict)

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        'Negate the system (equivalent to pre-multiplying by -1).'
        return StateSpace(self.A, self.B, -self.C, -self.D, **self._dt_dict)

    def __add__(self, other):
        if False:
            print('Hello World!')
        '\n        Adds two systems in the sense of frequency domain addition.\n        '
        if not self._check_binop_other(other):
            return NotImplemented
        if isinstance(other, StateSpace):
            if type(other) is not type(self):
                raise TypeError('Cannot add {} and {}'.format(type(self), type(other)))
            if self.dt != other.dt:
                raise TypeError('Cannot add systems with different `dt`.')
            a = block_diag(self.A, other.A)
            b = cupy.vstack((self.B, other.B))
            c = cupy.hstack((self.C, other.C))
            d = self.D + other.D
        else:
            other = cupy.atleast_2d(other)
            if self.D.shape == other.shape:
                a = self.A
                b = self.B
                c = self.C
                d = self.D + other
            else:
                raise ValueError('Cannot add systems with incompatible dimensions ({} and {})'.format(self.D.shape, other.shape))
        common_dtype = cupy.result_type(a.dtype, b.dtype, c.dtype, d.dtype)
        return StateSpace(cupy.asarray(a, dtype=common_dtype), cupy.asarray(b, dtype=common_dtype), cupy.asarray(c, dtype=common_dtype), cupy.asarray(d, dtype=common_dtype), **self._dt_dict)

    def __sub__(self, other):
        if False:
            return 10
        if not self._check_binop_other(other):
            return NotImplemented
        return self.__add__(-other)

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not self._check_binop_other(other):
            return NotImplemented
        return self.__add__(other)

    def __rsub__(self, other):
        if False:
            print('Hello World!')
        if not self._check_binop_other(other):
            return NotImplemented
        return (-self).__add__(other)

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        '\n        Divide by a scalar\n        '
        if not self._check_binop_other(other) or isinstance(other, StateSpace):
            return NotImplemented
        if isinstance(other, cupy.ndarray) and other.ndim > 0:
            raise ValueError('Cannot divide StateSpace by non-scalar numpy arrays')
        return self.__mul__(1 / other)

    @property
    def A(self):
        if False:
            for i in range(10):
                print('nop')
        'State matrix of the `StateSpace` system.'
        return self._A

    @A.setter
    def A(self, A):
        if False:
            return 10
        self._A = _atleast_2d_or_none(A)

    @property
    def B(self):
        if False:
            i = 10
            return i + 15
        'Input matrix of the `StateSpace` system.'
        return self._B

    @B.setter
    def B(self, B):
        if False:
            return 10
        self._B = _atleast_2d_or_none(B)
        self.inputs = self.B.shape[-1]

    @property
    def C(self):
        if False:
            return 10
        'Output matrix of the `StateSpace` system.'
        return self._C

    @C.setter
    def C(self, C):
        if False:
            while True:
                i = 10
        self._C = _atleast_2d_or_none(C)
        self.outputs = self.C.shape[0]

    @property
    def D(self):
        if False:
            print('Hello World!')
        'Feedthrough matrix of the `StateSpace` system.'
        return self._D

    @D.setter
    def D(self, D):
        if False:
            while True:
                i = 10
        self._D = _atleast_2d_or_none(D)

    def _copy(self, system):
        if False:
            while True:
                i = 10
        '\n        Copy the parameters of another `StateSpace` system.\n\n        Parameters\n        ----------\n        system : instance of `StateSpace`\n            The state-space system that is to be copied\n\n        '
        self.A = system.A
        self.B = system.B
        self.C = system.C
        self.D = system.D

    def to_tf(self, **kwargs):
        if False:
            return 10
        '\n        Convert system representation to `TransferFunction`.\n\n        Parameters\n        ----------\n        kwargs : dict, optional\n            Additional keywords passed to `ss2zpk`\n\n        Returns\n        -------\n        sys : instance of `TransferFunction`\n            Transfer function of the current system\n\n        '
        return TransferFunction(*ss2tf(self._A, self._B, self._C, self._D, **kwargs), **self._dt_dict)

    def to_zpk(self, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Convert system representation to `ZerosPolesGain`.\n\n        Parameters\n        ----------\n        kwargs : dict, optional\n            Additional keywords passed to `ss2zpk`\n\n        Returns\n        -------\n        sys : instance of `ZerosPolesGain`\n            Zeros, poles, gain representation of the current system\n\n        '
        return ZerosPolesGain(*ss2zpk(self._A, self._B, self._C, self._D, **kwargs), **self._dt_dict)

    def to_ss(self):
        if False:
            print('Hello World!')
        '\n        Return a copy of the current `StateSpace` system.\n\n        Returns\n        -------\n        sys : instance of `StateSpace`\n            The current system (copy)\n\n        '
        return copy.deepcopy(self)

class StateSpaceContinuous(StateSpace, lti):
    """
    Continuous-time Linear Time Invariant system in state-space form.

    Represents the system as the continuous-time, first order differential
    equation :math:`\\dot{x} = A x + B u`.
    Continuous-time `StateSpace` systems inherit additional functionality
    from the `lti` class.

    Parameters
    ----------
    *system: arguments
        The `StateSpace` class can be instantiated with 1 or 3 arguments.
        The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 4: array_like: (A, B, C, D)

    See Also
    --------
    scipy.signal.StateSpaceContinuous
    TransferFunction, ZerosPolesGain, lti
    ss2zpk, ss2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `StateSpace` system representation (such as `zeros` or `poles`) is very
    inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.
    """

    def to_discrete(self, dt, method='zoh', alpha=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns the discretized `StateSpace` system.\n\n        Parameters: See `cont2discrete` for details.\n\n        Returns\n        -------\n        sys: instance of `dlti` and `StateSpace`\n        '
        return StateSpace(*cont2discrete((self.A, self.B, self.C, self.D), dt, method=method, alpha=alpha)[:-1], dt=dt)

class StateSpaceDiscrete(StateSpace, dlti):
    """
    Discrete-time Linear Time Invariant system in state-space form.

    Represents the system as the discrete-time difference equation
    :math:`x[k+1] = A x[k] + B u[k]`.
    `StateSpace` systems inherit additional functionality from the `dlti`
    class.

    Parameters
    ----------
    *system: arguments
        The `StateSpace` class can be instantiated with 1 or 3 arguments.
        The following gives the number of input arguments and their
        interpretation:

            * 1: `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 4: array_like: (A, B, C, D)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `True`
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.

    See Also
    --------
    scipy.signal.StateSpaceDiscrete
    TransferFunction, ZerosPolesGain, dlti
    ss2zpk, ss2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `StateSpace` system representation (such as `zeros` or `poles`) is very
    inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.
    """
    pass

def lsim(system, U, T, X0=None, interp=True):
    if False:
        return 10
    '\n    Simulate output of a continuous-time linear system.\n\n    Parameters\n    ----------\n    system : an instance of the LTI class or a tuple describing the system.\n        The following gives the number of elements in the tuple and\n        the interpretation:\n\n        * 1: (instance of `lti`)\n        * 2: (num, den)\n        * 3: (zeros, poles, gain)\n        * 4: (A, B, C, D)\n\n    U : array_like\n        An input array describing the input at each time `T`\n        (interpolation is assumed between given times).  If there are\n        multiple inputs, then each column of the rank-2 array\n        represents an input.  If U = 0 or None, a zero input is used.\n    T : array_like\n        The time steps at which the input is defined and at which the\n        output is desired.  Must be nonnegative, increasing, and equally spaced\n    X0 : array_like, optional\n        The initial conditions on the state vector (zero by default).\n    interp : bool, optional\n        Whether to use linear (True, the default) or zero-order-hold (False)\n        interpolation for the input array.\n\n    Returns\n    -------\n    T : 1D ndarray\n        Time values for the output.\n    yout : 1D ndarray\n        System response.\n    xout : ndarray\n        Time evolution of the state vector.\n\n    Notes\n    -----\n    If (num, den) is passed in for ``system``, coefficients for both the\n    numerator and denominator should be specified in descending exponent\n    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).\n\n    See Also\n    --------\n    scipy.signal.lsim\n\n    '
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError('lsim can only be used with continuous-time systems.')
    else:
        sys = lti(*system)._as_ss()
    T = cupy.atleast_1d(T)
    if len(T.shape) != 1:
        raise ValueError('T must be a rank-1 array.')
    (A, B, C, D) = map(cupy.asarray, (sys.A, sys.B, sys.C, sys.D))
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    n_steps = T.size
    if X0 is None:
        X0 = cupy.zeros(n_states, sys.A.dtype)
    xout = cupy.empty((n_steps, n_states), sys.A.dtype)
    if T[0] == 0:
        xout[0] = X0
    elif T[0] > 0:
        xout[0] = X0 @ expm(A.T * T[0])
    else:
        raise ValueError('Initial time must be nonnegative')
    no_input = U is None or (isinstance(U, (int, float)) and U == 0.0) or (not cupy.any(U))
    if n_steps == 1:
        yout = cupy.squeeze(xout @ C.T)
        if not no_input:
            yout += cupy.squeeze(U @ D.T)
        return (T, cupy.squeeze(yout), cupy.squeeze(xout))
    dt = T[1] - T[0]
    if not cupy.allclose(cupy.diff(T), dt):
        raise ValueError('Time steps are not equally spaced.')
    if no_input:
        expAT_dt = expm(A.T * dt)
        for i in range(1, n_steps):
            xout[i] = xout[i - 1] @ expAT_dt
        yout = cupy.squeeze(xout @ C.T)
        return (T, cupy.squeeze(yout), cupy.squeeze(xout))
    U = cupy.atleast_1d(U)
    if U.ndim == 1:
        U = U[:, None]
    if U.shape[0] != n_steps:
        raise ValueError('U must have the same number of rows as elements in T.')
    if U.shape[1] != n_inputs:
        raise ValueError('System does not define that many inputs.')
    if not interp:
        M = cupy.vstack([cupy.hstack([A * dt, B * dt]), cupy.zeros((n_inputs, n_states + n_inputs))])
        expMT = expm(M.T)
        Ad = expMT[:n_states, :n_states]
        Bd = expMT[n_states:, :n_states]
        for i in range(1, n_steps):
            xout[i] = xout[i - 1] @ Ad + U[i - 1] @ Bd
    else:
        Mlst = [cupy.hstack([A * dt, B * dt, cupy.zeros((n_states, n_inputs))]), cupy.hstack([cupy.zeros((n_inputs, n_states + n_inputs)), cupy.identity(n_inputs)]), cupy.zeros((n_inputs, n_states + 2 * n_inputs))]
        M = cupy.vstack(Mlst)
        expMT = expm(M.T)
        Ad = expMT[:n_states, :n_states]
        Bd1 = expMT[n_states + n_inputs:, :n_states]
        Bd0 = expMT[n_states:n_states + n_inputs, :n_states] - Bd1
        for i in range(1, n_steps):
            xout[i] = xout[i - 1] @ Ad + U[i - 1] @ Bd0 + U[i] @ Bd1
    yout = cupy.squeeze(xout @ C.T) + cupy.squeeze(U @ D.T)
    return (T, cupy.squeeze(yout), cupy.squeeze(xout))

def _default_response_times(A, n):
    if False:
        print('Hello World!')
    'Compute a reasonable set of time samples for the response time.\n\n    This function is used by `impulse`, `impulse2`, `step` and `step2`\n    to compute the response time when the `T` argument to the function\n    is None.\n\n    Parameters\n    ----------\n    A : array_like\n        The system matrix, which is square.\n    n : int\n        The number of time samples to generate.\n\n    Returns\n    -------\n    t : ndarray\n        The 1-D array of length `n` of time samples at which the response\n        is to be computed.\n\n    '
    import numpy as np
    vals = np.linalg.eigvals(A.get())
    vals = cupy.asarray(vals)
    r = cupy.min(cupy.abs(vals.real))
    if r == 0.0:
        r = 1.0
    tc = 1.0 / r
    t = cupy.linspace(0.0, 7 * tc, n)
    return t

def impulse(system, X0=None, T=None, N=None):
    if False:
        for i in range(10):
            print('nop')
    'Impulse response of continuous-time system.\n\n    Parameters\n    ----------\n    system : an instance of the LTI class or a tuple of array_like\n        describing the system.\n        The following gives the number of elements in the tuple and\n        the interpretation:\n\n            * 1 (instance of `lti`)\n            * 2 (num, den)\n            * 3 (zeros, poles, gain)\n            * 4 (A, B, C, D)\n\n    X0 : array_like, optional\n        Initial state-vector.  Defaults to zero.\n    T : array_like, optional\n        Time points.  Computed if not given.\n    N : int, optional\n        The number of time points to compute (if `T` is not given).\n\n    Returns\n    -------\n    T : ndarray\n        A 1-D array of time points.\n    yout : ndarray\n        A 1-D array containing the impulse response of the system (except for\n        singularities at zero).\n\n    Notes\n    -----\n    If (num, den) is passed in for ``system``, coefficients for both the\n    numerator and denominator should be specified in descending exponent\n    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).\n\n    See Also\n    --------\n    scipy.signal.impulse\n\n    '
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError('impulse can only be used with continuous-time systems.')
    else:
        sys = lti(*system)._as_ss()
    if X0 is None:
        X = cupy.squeeze(sys.B)
    else:
        X = cupy.squeeze(sys.B + X0)
    if N is None:
        N = 100
    if T is None:
        T = _default_response_times(sys.A, N)
    else:
        T = cupy.asarray(T)
    (_, h, _) = lsim(sys, 0.0, T, X, interp=False)
    return (T, h)

def step(system, X0=None, T=None, N=None):
    if False:
        for i in range(10):
            print('nop')
    'Step response of continuous-time system.\n\n    Parameters\n    ----------\n    system : an instance of the LTI class or a tuple of array_like\n        describing the system.\n        The following gives the number of elements in the tuple and\n        the interpretation:\n\n            * 1 (instance of `lti`)\n            * 2 (num, den)\n            * 3 (zeros, poles, gain)\n            * 4 (A, B, C, D)\n\n    X0 : array_like, optional\n        Initial state-vector (default is zero).\n    T : array_like, optional\n        Time points (computed if not given).\n    N : int, optional\n        Number of time points to compute if `T` is not given.\n\n    Returns\n    -------\n    T : 1D ndarray\n        Output time points.\n    yout : 1D ndarray\n        Step response of system.\n\n\n    Notes\n    -----\n    If (num, den) is passed in for ``system``, coefficients for both the\n    numerator and denominator should be specified in descending exponent\n    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).\n\n    See Also\n    --------\n    scipy.signal.step\n\n    '
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError('step can only be used with continuous-time systems.')
    else:
        sys = lti(*system)._as_ss()
    if N is None:
        N = 100
    if T is None:
        T = _default_response_times(sys.A, N)
    else:
        T = cupy.asarray(T)
    U = cupy.ones(T.shape, sys.A.dtype)
    vals = lsim(sys, U, T, X0=X0, interp=False)
    return (vals[0], vals[1])

def bode(system, w=None, n=100):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate Bode magnitude and phase data of a continuous-time system.\n\n    Parameters\n    ----------\n    system : an instance of the LTI class or a tuple describing the system.\n        The following gives the number of elements in the tuple and\n        the interpretation:\n\n            * 1 (instance of `lti`)\n            * 2 (num, den)\n            * 3 (zeros, poles, gain)\n            * 4 (A, B, C, D)\n\n    w : array_like, optional\n        Array of frequencies (in rad/s). Magnitude and phase data is calculated\n        for every value in this array. If not given a reasonable set will be\n        calculated.\n    n : int, optional\n        Number of frequency points to compute if `w` is not given. The `n`\n        frequencies are logarithmically spaced in an interval chosen to\n        include the influence of the poles and zeros of the system.\n\n    Returns\n    -------\n    w : 1D ndarray\n        Frequency array [rad/s]\n    mag : 1D ndarray\n        Magnitude array [dB]\n    phase : 1D ndarray\n        Phase array [deg]\n\n    See Also\n    --------\n    scipy.signal.bode\n\n    Notes\n    -----\n    If (num, den) is passed in for ``system``, coefficients for both the\n    numerator and denominator should be specified in descending exponent\n    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).\n\n    '
    (w, y) = freqresp(system, w=w, n=n)
    mag = 20.0 * cupy.log10(abs(y))
    phase = cupy.unwrap(cupy.arctan2(y.imag, y.real)) * 180.0 / cupy.pi
    return (w, mag, phase)

def freqresp(system, w=None, n=10000):
    if False:
        i = 10
        return i + 15
    'Calculate the frequency response of a continuous-time system.\n\n    Parameters\n    ----------\n    system : an instance of the `lti` class or a tuple describing the system.\n        The following gives the number of elements in the tuple and\n        the interpretation:\n\n            * 1 (instance of `lti`)\n            * 2 (num, den)\n            * 3 (zeros, poles, gain)\n            * 4 (A, B, C, D)\n\n    w : array_like, optional\n        Array of frequencies (in rad/s). Magnitude and phase data is\n        calculated for every value in this array. If not given, a reasonable\n        set will be calculated.\n    n : int, optional\n        Number of frequency points to compute if `w` is not given. The `n`\n        frequencies are logarithmically spaced in an interval chosen to\n        include the influence of the poles and zeros of the system.\n\n    Returns\n    -------\n    w : 1D ndarray\n        Frequency array [rad/s]\n    H : 1D ndarray\n        Array of complex magnitude values\n\n    Notes\n    -----\n    If (num, den) is passed in for ``system``, coefficients for both the\n    numerator and denominator should be specified in descending exponent\n    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).\n\n    See Also\n    --------\n    scipy.signal.freqresp\n\n    '
    if isinstance(system, lti):
        if isinstance(system, (TransferFunction, ZerosPolesGain)):
            sys = system
        else:
            sys = system._as_zpk()
    elif isinstance(system, dlti):
        raise AttributeError('freqresp can only be used with continuous-time systems.')
    else:
        sys = lti(*system)._as_zpk()
    if sys.inputs != 1 or sys.outputs != 1:
        raise ValueError('freqresp() requires a SISO (single input, single output) system.')
    if w is not None:
        worN = w
    else:
        worN = n
    if isinstance(sys, TransferFunction):
        (w, h) = freqs(sys.num.ravel(), sys.den, worN=worN)
    elif isinstance(sys, ZerosPolesGain):
        (w, h) = freqs_zpk(sys.zeros, sys.poles, sys.gain, worN=worN)
    return (w, h)

def dlsim(system, u, t=None, x0=None):
    if False:
        i = 10
        return i + 15
    '\n    Simulate output of a discrete-time linear system.\n\n    Parameters\n    ----------\n    system : tuple of array_like or instance of `dlti`\n        A tuple describing the system.\n        The following gives the number of elements in the tuple and\n        the interpretation:\n\n            * 1: (instance of `dlti`)\n            * 3: (num, den, dt)\n            * 4: (zeros, poles, gain, dt)\n            * 5: (A, B, C, D, dt)\n\n    u : array_like\n        An input array describing the input at each time `t` (interpolation is\n        assumed between given times).  If there are multiple inputs, then each\n        column of the rank-2 array represents an input.\n    t : array_like, optional\n        The time steps at which the input is defined.  If `t` is given, it\n        must be the same length as `u`, and the final value in `t` determines\n        the number of steps returned in the output.\n    x0 : array_like, optional\n        The initial conditions on the state vector (zero by default).\n\n    Returns\n    -------\n    tout : ndarray\n        Time values for the output, as a 1-D array.\n    yout : ndarray\n        System response, as a 1-D array.\n    xout : ndarray, optional\n        Time-evolution of the state-vector.  Only generated if the input is a\n        `StateSpace` system.\n\n    See Also\n    --------\n    scipy.signal.dlsim\n    lsim, dstep, dimpulse, cont2discrete\n    '
    if isinstance(system, lti):
        raise AttributeError('dlsim can only be used with discrete-time dlti systems.')
    elif not isinstance(system, dlti):
        system = dlti(*system[:-1], dt=system[-1])
    is_ss_input = isinstance(system, StateSpace)
    system = system._as_ss()
    u = cupy.atleast_1d(u)
    if u.ndim == 1:
        u = cupy.atleast_2d(u).T
    if t is None:
        out_samples = len(u)
        stoptime = (out_samples - 1) * system.dt
    else:
        stoptime = t[-1]
        out_samples = int(cupy.floor(stoptime / system.dt)) + 1
    xout = cupy.zeros((out_samples, system.A.shape[0]))
    yout = cupy.zeros((out_samples, system.C.shape[0]))
    tout = cupy.linspace(0.0, stoptime, num=out_samples)
    if x0 is None:
        xout[0, :] = cupy.zeros((system.A.shape[1],))
    else:
        xout[0, :] = cupy.asarray(x0)
    if t is None:
        u_dt = u
    else:
        if len(u.shape) == 1:
            u = u[:, None]
        u_dt = make_interp_spline(t, u, k=1)(tout)
    for i in range(0, out_samples - 1):
        xout[i + 1, :] = system.A @ xout[i, :] + system.B @ u_dt[i, :]
        yout[i, :] = system.C @ xout[i, :] + system.D @ u_dt[i, :]
    yout[out_samples - 1, :] = system.C @ xout[out_samples - 1, :] + system.D @ u_dt[out_samples - 1, :]
    if is_ss_input:
        return (tout, yout, xout)
    else:
        return (tout, yout)

def dimpulse(system, x0=None, t=None, n=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Impulse response of discrete-time system.\n\n    Parameters\n    ----------\n    system : tuple of array_like or instance of `dlti`\n        A tuple describing the system.\n        The following gives the number of elements in the tuple and\n        the interpretation:\n\n            * 1: (instance of `dlti`)\n            * 3: (num, den, dt)\n            * 4: (zeros, poles, gain, dt)\n            * 5: (A, B, C, D, dt)\n\n    x0 : array_like, optional\n        Initial state-vector.  Defaults to zero.\n    t : array_like, optional\n        Time points.  Computed if not given.\n    n : int, optional\n        The number of time points to compute (if `t` is not given).\n\n    Returns\n    -------\n    tout : ndarray\n        Time values for the output, as a 1-D array.\n    yout : tuple of ndarray\n        Impulse response of system.  Each element of the tuple represents\n        the output of the system based on an impulse in each input.\n\n    See Also\n    --------\n    scipy.signal.dimpulse\n    impulse, dstep, dlsim, cont2discrete\n    '
    if isinstance(system, dlti):
        system = system._as_ss()
    elif isinstance(system, lti):
        raise AttributeError('dimpulse can only be used with discrete-time dlti systems.')
    else:
        system = dlti(*system[:-1], dt=system[-1])._as_ss()
    if n is None:
        n = 100
    if t is None:
        t = cupy.linspace(0, n * system.dt, n, endpoint=False)
    else:
        t = cupy.asarray(t)
    yout = None
    for i in range(0, system.inputs):
        u = cupy.zeros((t.shape[0], system.inputs))
        u[0, i] = 1.0
        one_output = dlsim(system, u, t=t, x0=x0)
        if yout is None:
            yout = (one_output[1],)
        else:
            yout = yout + (one_output[1],)
        tout = one_output[0]
    return (tout, yout)

def dstep(system, x0=None, t=None, n=None):
    if False:
        while True:
            i = 10
    '\n    Step response of discrete-time system.\n\n    Parameters\n    ----------\n    system : tuple of array_like\n        A tuple describing the system.\n        The following gives the number of elements in the tuple and\n        the interpretation:\n\n            * 1: (instance of `dlti`)\n            * 3: (num, den, dt)\n            * 4: (zeros, poles, gain, dt)\n            * 5: (A, B, C, D, dt)\n\n    x0 : array_like, optional\n        Initial state-vector.  Defaults to zero.\n    t : array_like, optional\n        Time points.  Computed if not given.\n    n : int, optional\n        The number of time points to compute (if `t` is not given).\n\n    Returns\n    -------\n    tout : ndarray\n        Output time points, as a 1-D array.\n    yout : tuple of ndarray\n        Step response of system.  Each element of the tuple represents\n        the output of the system based on a step response to each input.\n\n    See Also\n    --------\n    scipy.signal.dlstep\n    step, dimpulse, dlsim, cont2discrete\n    '
    if isinstance(system, dlti):
        system = system._as_ss()
    elif isinstance(system, lti):
        raise AttributeError('dstep can only be used with discrete-time dlti systems.')
    else:
        system = dlti(*system[:-1], dt=system[-1])._as_ss()
    if n is None:
        n = 100
    if t is None:
        t = cupy.linspace(0, n * system.dt, n, endpoint=False)
    else:
        t = cupy.asarray(t)
    yout = None
    for i in range(0, system.inputs):
        u = cupy.zeros((t.shape[0], system.inputs))
        u[:, i] = cupy.ones((t.shape[0],))
        one_output = dlsim(system, u, t=t, x0=x0)
        if yout is None:
            yout = (one_output[1],)
        else:
            yout = yout + (one_output[1],)
        tout = one_output[0]
    return (tout, yout)

def dfreqresp(system, w=None, n=10000, whole=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Calculate the frequency response of a discrete-time system.\n\n    Parameters\n    ----------\n    system : an instance of the `dlti` class or a tuple describing the system.\n        The following gives the number of elements in the tuple and\n        the interpretation:\n\n            * 1 (instance of `dlti`)\n            * 2 (numerator, denominator, dt)\n            * 3 (zeros, poles, gain, dt)\n            * 4 (A, B, C, D, dt)\n\n    w : array_like, optional\n        Array of frequencies (in radians/sample). Magnitude and phase data is\n        calculated for every value in this array. If not given a reasonable\n        set will be calculated.\n    n : int, optional\n        Number of frequency points to compute if `w` is not given. The `n`\n        frequencies are logarithmically spaced in an interval chosen to\n        include the influence of the poles and zeros of the system.\n    whole : bool, optional\n        Normally, if 'w' is not given, frequencies are computed from 0 to the\n        Nyquist frequency, pi radians/sample (upper-half of unit-circle). If\n        `whole` is True, compute frequencies from 0 to 2*pi radians/sample.\n\n    Returns\n    -------\n    w : 1D ndarray\n        Frequency array [radians/sample]\n    H : 1D ndarray\n        Array of complex magnitude values\n\n    See Also\n    --------\n    scipy.signal.dfeqresp\n\n    Notes\n    -----\n    If (num, den) is passed in for ``system``, coefficients for both the\n    numerator and denominator should be specified in descending exponent\n    order (e.g. ``z^2 + 3z + 5`` would be represented as ``[1, 3, 5]``).\n    "
    if not isinstance(system, dlti):
        if isinstance(system, lti):
            raise AttributeError('dfreqresp can only be used with discrete-time systems.')
        system = dlti(*system[:-1], dt=system[-1])
    if isinstance(system, StateSpace):
        system = system._as_tf()
    if not isinstance(system, (TransferFunction, ZerosPolesGain)):
        raise ValueError('Unknown system type')
    if system.inputs != 1 or system.outputs != 1:
        raise ValueError('dfreqresp requires a SISO (single input, single output) system.')
    if w is not None:
        worN = w
    else:
        worN = n
    if isinstance(system, TransferFunction):
        (num, den) = TransferFunction._z_to_zinv(system.num.ravel(), system.den)
        (w, h) = freqz(num, den, worN=worN, whole=whole)
    elif isinstance(system, ZerosPolesGain):
        (w, h) = freqz_zpk(system.zeros, system.poles, system.gain, worN=worN, whole=whole)
    return (w, h)

def dbode(system, w=None, n=100):
    if False:
        return 10
    '\n    Calculate Bode magnitude and phase data of a discrete-time system.\n\n    Parameters\n    ----------\n    system : an instance of the LTI class or a tuple describing the system.\n        The following gives the number of elements in the tuple and\n        the interpretation:\n\n            * 1 (instance of `dlti`)\n            * 2 (num, den, dt)\n            * 3 (zeros, poles, gain, dt)\n            * 4 (A, B, C, D, dt)\n\n    w : array_like, optional\n        Array of frequencies (in radians/sample). Magnitude and phase data is\n        calculated for every value in this array. If not given a reasonable\n        set will be calculated.\n    n : int, optional\n        Number of frequency points to compute if `w` is not given. The `n`\n        frequencies are logarithmically spaced in an interval chosen to\n        include the influence of the poles and zeros of the system.\n\n    Returns\n    -------\n    w : 1D ndarray\n        Frequency array [rad/time_unit]\n    mag : 1D ndarray\n        Magnitude array [dB]\n    phase : 1D ndarray\n        Phase array [deg]\n\n    See Also\n    --------\n    scipy.signal.dbode\n\n    Notes\n    -----\n    If (num, den) is passed in for ``system``, coefficients for both the\n    numerator and denominator should be specified in descending exponent\n    order (e.g. ``z^2 + 3z + 5`` would be represented as ``[1, 3, 5]``).\n    '
    (w, y) = dfreqresp(system, w=w, n=n)
    if isinstance(system, dlti):
        dt = system.dt
    else:
        dt = system[-1]
    mag = 20.0 * cupy.log10(abs(y))
    phase = cupy.rad2deg(cupy.unwrap(cupy.angle(y)))
    return (w / dt, mag, phase)

def cont2discrete(system, dt, method='zoh', alpha=None):
    if False:
        i = 10
        return i + 15
    '\n    Transform a continuous to a discrete state-space system.\n\n    Parameters\n    ----------\n    system : a tuple describing the system or an instance of `lti`\n        The following gives the number of elements in the tuple and\n        the interpretation:\n\n            * 1: (instance of `lti`)\n            * 2: (num, den)\n            * 3: (zeros, poles, gain)\n            * 4: (A, B, C, D)\n\n    dt : float\n        The discretization time step.\n    method : str, optional\n        Which method to use:\n\n            * gbt: generalized bilinear transformation\n            * bilinear: Tustin\'s approximation ("gbt" with alpha=0.5)\n            * euler: Euler (or forward differencing) method\n              ("gbt" with alpha=0)\n            * backward_diff: Backwards differencing ("gbt" with alpha=1.0)\n            * zoh: zero-order hold (default)\n            * foh: first-order hold (*versionadded: 1.3.0*)\n            * impulse: equivalent impulse response (*versionadded: 1.3.0*)\n\n    alpha : float within [0, 1], optional\n        The generalized bilinear transformation weighting parameter, which\n        should only be specified with method="gbt", and is ignored otherwise\n\n    Returns\n    -------\n    sysd : tuple containing the discrete system\n        Based on the input type, the output will be of the form\n\n        * (num, den, dt)   for transfer function input\n        * (zeros, poles, gain, dt)   for zeros-poles-gain input\n        * (A, B, C, D, dt) for state-space system input\n\n    Notes\n    -----\n    By default, the routine uses a Zero-Order Hold (zoh) method to perform\n    the transformation. Alternatively, a generalized bilinear transformation\n    may be used, which includes the common Tustin\'s bilinear approximation,\n    an Euler\'s method technique, or a backwards differencing technique.\n\n    See Also\n    --------\n    scipy.signal.cont2discrete\n\n\n    '
    if len(system) == 1:
        return system.to_discrete()
    if len(system) == 2:
        sysd = cont2discrete(tf2ss(system[0], system[1]), dt, method=method, alpha=alpha)
        return ss2tf(sysd[0], sysd[1], sysd[2], sysd[3]) + (dt,)
    elif len(system) == 3:
        sysd = cont2discrete(zpk2ss(system[0], system[1], system[2]), dt, method=method, alpha=alpha)
        return ss2zpk(sysd[0], sysd[1], sysd[2], sysd[3]) + (dt,)
    elif len(system) == 4:
        (a, b, c, d) = system
    else:
        raise ValueError('First argument must either be a tuple of 2 (tf), 3 (zpk), or 4 (ss) arrays.')
    if method == 'gbt':
        if alpha is None:
            raise ValueError('Alpha parameter must be specified for the generalized bilinear transform (gbt) method')
        elif alpha < 0 or alpha > 1:
            raise ValueError('Alpha parameter must be within the interval [0,1] for the gbt method')
    if method == 'gbt':
        ima = cupy.eye(a.shape[0]) - alpha * dt * a
        rhs = cupy.eye(a.shape[0]) + (1.0 - alpha) * dt * a
        ad = cupy.linalg.solve(ima, rhs)
        bd = cupy.linalg.solve(ima, dt * b)
        cd = cupy.linalg.solve(ima.T, c.T)
        cd = cd.T
        dd = d + alpha * (c @ bd)
    elif method == 'bilinear' or method == 'tustin':
        return cont2discrete(system, dt, method='gbt', alpha=0.5)
    elif method == 'euler' or method == 'forward_diff':
        return cont2discrete(system, dt, method='gbt', alpha=0.0)
    elif method == 'backward_diff':
        return cont2discrete(system, dt, method='gbt', alpha=1.0)
    elif method == 'zoh':
        em_upper = cupy.hstack((a, b))
        em_lower = cupy.hstack((cupy.zeros((b.shape[1], a.shape[0])), cupy.zeros((b.shape[1], b.shape[1]))))
        em = cupy.vstack((em_upper, em_lower))
        ms = expm(dt * em)
        ms = ms[:a.shape[0], :]
        ad = ms[:, 0:a.shape[1]]
        bd = ms[:, a.shape[1]:]
        cd = c
        dd = d
    elif method == 'foh':
        n = a.shape[0]
        m = b.shape[1]
        em_upper = block_diag(cupy.hstack([a, b]) * dt, cupy.eye(m))
        em_lower = cupy.zeros((m, n + 2 * m))
        em = cupy.vstack([em_upper, em_lower])
        ms = linalg.expm(em)
        ms11 = ms[:n, 0:n]
        ms12 = ms[:n, n:n + m]
        ms13 = ms[:n, n + m:]
        ad = ms11
        bd = ms12 - ms13 + ms11 @ ms13
        cd = c
        dd = d + c @ ms13
    elif method == 'impulse':
        if not cupy.allclose(d, 0):
            raise ValueError('Impulse method is only applicableto strictly proper systems')
        ad = expm(a * dt)
        bd = ad @ b * dt
        cd = c
        dd = c @ b * dt
    else:
        raise ValueError("Unknown transformation method '%s'" % method)
    return (ad, bd, cd, dd, dt)