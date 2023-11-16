"""Machine limits for Float32 and Float64 and (long double) if available...

"""
__all__ = ['finfo', 'iinfo']
import warnings
from .._utils import set_module
from ._machar import MachAr
from . import numeric
from . import numerictypes as ntypes
from .numeric import array, inf, nan
from .umath import log10, exp2, nextafter, isnan

def _fr0(a):
    if False:
        i = 10
        return i + 15
    'fix rank-0 --> rank-1'
    if a.ndim == 0:
        a = a.copy()
        a.shape = (1,)
    return a

def _fr1(a):
    if False:
        print('Hello World!')
    'fix rank > 0 --> rank-0'
    if a.size == 1:
        a = a.copy()
        a.shape = ()
    return a

class MachArLike:
    """ Object to simulate MachAr instance """

    def __init__(self, ftype, *, eps, epsneg, huge, tiny, ibeta, smallest_subnormal=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.params = _MACHAR_PARAMS[ftype]
        self.ftype = ftype
        self.title = self.params['title']
        if not smallest_subnormal:
            self._smallest_subnormal = nextafter(self.ftype(0), self.ftype(1), dtype=self.ftype)
        else:
            self._smallest_subnormal = smallest_subnormal
        self.epsilon = self.eps = self._float_to_float(eps)
        self.epsneg = self._float_to_float(epsneg)
        self.xmax = self.huge = self._float_to_float(huge)
        self.xmin = self._float_to_float(tiny)
        self.smallest_normal = self.tiny = self._float_to_float(tiny)
        self.ibeta = self.params['itype'](ibeta)
        self.__dict__.update(kwargs)
        self.precision = int(-log10(self.eps))
        self.resolution = self._float_to_float(self._float_conv(10) ** (-self.precision))
        self._str_eps = self._float_to_str(self.eps)
        self._str_epsneg = self._float_to_str(self.epsneg)
        self._str_xmin = self._float_to_str(self.xmin)
        self._str_xmax = self._float_to_str(self.xmax)
        self._str_resolution = self._float_to_str(self.resolution)
        self._str_smallest_normal = self._float_to_str(self.xmin)

    @property
    def smallest_subnormal(self):
        if False:
            i = 10
            return i + 15
        'Return the value for the smallest subnormal.\n\n        Returns\n        -------\n        smallest_subnormal : float\n            value for the smallest subnormal.\n\n        Warns\n        -----\n        UserWarning\n            If the calculated value for the smallest subnormal is zero.\n        '
        value = self._smallest_subnormal
        if self.ftype(0) == value:
            warnings.warn('The value of the smallest subnormal for {} type is zero.'.format(self.ftype), UserWarning, stacklevel=2)
        return self._float_to_float(value)

    @property
    def _str_smallest_subnormal(self):
        if False:
            i = 10
            return i + 15
        'Return the string representation of the smallest subnormal.'
        return self._float_to_str(self.smallest_subnormal)

    def _float_to_float(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Converts float to float.\n\n        Parameters\n        ----------\n        value : float\n            value to be converted.\n        '
        return _fr1(self._float_conv(value))

    def _float_conv(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Converts float to conv.\n\n        Parameters\n        ----------\n        value : float\n            value to be converted.\n        '
        return array([value], self.ftype)

    def _float_to_str(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Converts float to str.\n\n        Parameters\n        ----------\n        value : float\n            value to be converted.\n        '
        return self.params['fmt'] % array(_fr0(value)[0], self.ftype)
_convert_to_float = {ntypes.csingle: ntypes.single, ntypes.complex128: ntypes.float64, ntypes.clongdouble: ntypes.longdouble}
_title_fmt = 'numpy {} precision floating point number'
_MACHAR_PARAMS = {ntypes.double: dict(itype=ntypes.int64, fmt='%24.16e', title=_title_fmt.format('double')), ntypes.single: dict(itype=ntypes.int32, fmt='%15.7e', title=_title_fmt.format('single')), ntypes.longdouble: dict(itype=ntypes.longlong, fmt='%s', title=_title_fmt.format('long double')), ntypes.half: dict(itype=ntypes.int16, fmt='%12.5e', title=_title_fmt.format('half'))}
_KNOWN_TYPES = {}

def _register_type(machar, bytepat):
    if False:
        while True:
            i = 10
    _KNOWN_TYPES[bytepat] = machar
_float_ma = {}

def _register_known_types():
    if False:
        return 10
    f16 = ntypes.float16
    float16_ma = MachArLike(f16, machep=-10, negep=-11, minexp=-14, maxexp=16, it=10, iexp=5, ibeta=2, irnd=5, ngrd=0, eps=exp2(f16(-10)), epsneg=exp2(f16(-11)), huge=f16(65504), tiny=f16(2 ** (-14)))
    _register_type(float16_ma, b'f\xae')
    _float_ma[16] = float16_ma
    f32 = ntypes.float32
    float32_ma = MachArLike(f32, machep=-23, negep=-24, minexp=-126, maxexp=128, it=23, iexp=8, ibeta=2, irnd=5, ngrd=0, eps=exp2(f32(-23)), epsneg=exp2(f32(-24)), huge=f32((1 - 2 ** (-24)) * 2 ** 128), tiny=exp2(f32(-126)))
    _register_type(float32_ma, b'\xcd\xcc\xcc\xbd')
    _float_ma[32] = float32_ma
    f64 = ntypes.float64
    epsneg_f64 = 2.0 ** (-53.0)
    tiny_f64 = 2.0 ** (-1022.0)
    float64_ma = MachArLike(f64, machep=-52, negep=-53, minexp=-1022, maxexp=1024, it=52, iexp=11, ibeta=2, irnd=5, ngrd=0, eps=2.0 ** (-52.0), epsneg=epsneg_f64, huge=(1.0 - epsneg_f64) / tiny_f64 * f64(4), tiny=tiny_f64)
    _register_type(float64_ma, b'\x9a\x99\x99\x99\x99\x99\xb9\xbf')
    _float_ma[64] = float64_ma
    ld = ntypes.longdouble
    epsneg_f128 = exp2(ld(-113))
    tiny_f128 = exp2(ld(-16382))
    with numeric.errstate(all='ignore'):
        huge_f128 = (ld(1) - epsneg_f128) / tiny_f128 * ld(4)
    float128_ma = MachArLike(ld, machep=-112, negep=-113, minexp=-16382, maxexp=16384, it=112, iexp=15, ibeta=2, irnd=5, ngrd=0, eps=exp2(ld(-112)), epsneg=epsneg_f128, huge=huge_f128, tiny=tiny_f128)
    _register_type(float128_ma, b'\x9a\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\xfb\xbf')
    _float_ma[128] = float128_ma
    epsneg_f80 = exp2(ld(-64))
    tiny_f80 = exp2(ld(-16382))
    with numeric.errstate(all='ignore'):
        huge_f80 = (ld(1) - epsneg_f80) / tiny_f80 * ld(4)
    float80_ma = MachArLike(ld, machep=-63, negep=-64, minexp=-16382, maxexp=16384, it=63, iexp=15, ibeta=2, irnd=5, ngrd=0, eps=exp2(ld(-63)), epsneg=epsneg_f80, huge=huge_f80, tiny=tiny_f80)
    _register_type(float80_ma, b'\xcd\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf')
    _float_ma[80] = float80_ma
    huge_dd = nextafter(ld(inf), ld(0), dtype=ld)
    smallest_normal_dd = nan
    smallest_subnormal_dd = ld(nextafter(0.0, 1.0))
    float_dd_ma = MachArLike(ld, machep=-105, negep=-106, minexp=-1022, maxexp=1024, it=105, iexp=11, ibeta=2, irnd=5, ngrd=0, eps=exp2(ld(-105)), epsneg=exp2(ld(-106)), huge=huge_dd, tiny=smallest_normal_dd, smallest_subnormal=smallest_subnormal_dd)
    _register_type(float_dd_ma, b'\x9a\x99\x99\x99\x99\x99Y<\x9a\x99\x99\x99\x99\x99\xb9\xbf')
    _register_type(float_dd_ma, b'\x9a\x99\x99\x99\x99\x99\xb9\xbf\x9a\x99\x99\x99\x99\x99Y<')
    _float_ma['dd'] = float_dd_ma

def _get_machar(ftype):
    if False:
        i = 10
        return i + 15
    ' Get MachAr instance or MachAr-like instance\n\n    Get parameters for floating point type, by first trying signatures of\n    various known floating point types, then, if none match, attempting to\n    identify parameters by analysis.\n\n    Parameters\n    ----------\n    ftype : class\n        Numpy floating point type class (e.g. ``np.float64``)\n\n    Returns\n    -------\n    ma_like : instance of :class:`MachAr` or :class:`MachArLike`\n        Object giving floating point parameters for `ftype`.\n\n    Warns\n    -----\n    UserWarning\n        If the binary signature of the float type is not in the dictionary of\n        known float types.\n    '
    params = _MACHAR_PARAMS.get(ftype)
    if params is None:
        raise ValueError(repr(ftype))
    key = ftype(-1.0) / ftype(10.0)
    key = key.view(key.dtype.newbyteorder('<')).tobytes()
    ma_like = None
    if ftype == ntypes.longdouble:
        ma_like = _KNOWN_TYPES.get(key[:10])
    if ma_like is None:
        ma_like = _KNOWN_TYPES.get(key)
    if ma_like is None and len(key) == 16:
        _kt = {k[:10]: v for (k, v) in _KNOWN_TYPES.items() if len(k) == 16}
        ma_like = _kt.get(key[:10])
    if ma_like is not None:
        return ma_like
    warnings.warn(f'Signature {key} for {ftype} does not match any known type: falling back to type probe function.\nThis warnings indicates broken support for the dtype!', UserWarning, stacklevel=2)
    return _discovered_machar(ftype)

def _discovered_machar(ftype):
    if False:
        return 10
    ' Create MachAr instance with found information on float types\n\n    TODO: MachAr should be retired completely ideally.  We currently only\n          ever use it system with broken longdouble (valgrind, WSL).\n    '
    params = _MACHAR_PARAMS[ftype]
    return MachAr(lambda v: array([v], ftype), lambda v: _fr0(v.astype(params['itype']))[0], lambda v: array(_fr0(v)[0], ftype), lambda v: params['fmt'] % array(_fr0(v)[0], ftype), params['title'])

@set_module('numpy')
class finfo:
    """
    finfo(dtype)

    Machine limits for floating point types.

    Attributes
    ----------
    bits : int
        The number of bits occupied by the type.
    dtype : dtype
        Returns the dtype for which `finfo` returns information. For complex
        input, the returned dtype is the associated ``float*`` dtype for its
        real and complex components.
    eps : float
        The difference between 1.0 and the next smallest representable float
        larger than 1.0. For example, for 64-bit binary floats in the IEEE-754
        standard, ``eps = 2**-52``, approximately 2.22e-16.
    epsneg : float
        The difference between 1.0 and the next smallest representable float
        less than 1.0. For example, for 64-bit binary floats in the IEEE-754
        standard, ``epsneg = 2**-53``, approximately 1.11e-16.
    iexp : int
        The number of bits in the exponent portion of the floating point
        representation.
    machep : int
        The exponent that yields `eps`.
    max : floating point number of the appropriate type
        The largest representable number.
    maxexp : int
        The smallest positive power of the base (2) that causes overflow.
    min : floating point number of the appropriate type
        The smallest representable number, typically ``-max``.
    minexp : int
        The most negative power of the base (2) consistent with there
        being no leading 0's in the mantissa.
    negep : int
        The exponent that yields `epsneg`.
    nexp : int
        The number of bits in the exponent including its sign and bias.
    nmant : int
        The number of bits in the mantissa.
    precision : int
        The approximate number of decimal digits to which this kind of
        float is precise.
    resolution : floating point number of the appropriate type
        The approximate decimal resolution of this type, i.e.,
        ``10**-precision``.
    tiny : float
        An alias for `smallest_normal`, kept for backwards compatibility.
    smallest_normal : float
        The smallest positive floating point number with 1 as leading bit in
        the mantissa following IEEE-754 (see Notes).
    smallest_subnormal : float
        The smallest positive floating point number with 0 as leading bit in
        the mantissa following IEEE-754.

    Parameters
    ----------
    dtype : float, dtype, or instance
        Kind of floating point or complex floating point
        data-type about which to get information.

    See Also
    --------
    iinfo : The equivalent for integer data types.
    spacing : The distance between a value and the nearest adjacent number
    nextafter : The next floating point value after x1 towards x2

    Notes
    -----
    For developers of NumPy: do not instantiate this at the module level.
    The initial calculation of these parameters is expensive and negatively
    impacts import times.  These objects are cached, so calling ``finfo()``
    repeatedly inside your functions is not a problem.

    Note that ``smallest_normal`` is not actually the smallest positive
    representable value in a NumPy floating point type. As in the IEEE-754
    standard [1]_, NumPy floating point types make use of subnormal numbers to
    fill the gap between 0 and ``smallest_normal``. However, subnormal numbers
    may have significantly reduced precision [2]_.

    This function can also be used for complex data types as well. If used,
    the output will be the same as the corresponding real float type
    (e.g. numpy.finfo(numpy.csingle) is the same as numpy.finfo(numpy.single)).
    However, the output is true for the real and imaginary components.

    References
    ----------
    .. [1] IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2008,
           pp.1-70, 2008, https://doi.org/10.1109/IEEESTD.2008.4610935
    .. [2] Wikipedia, "Denormal Numbers",
           https://en.wikipedia.org/wiki/Denormal_number

    Examples
    --------
    >>> np.finfo(np.float64).dtype
    dtype('float64')
    >>> np.finfo(np.complex64).dtype
    dtype('float32')

    """
    _finfo_cache = {}

    def __new__(cls, dtype):
        if False:
            return 10
        try:
            obj = cls._finfo_cache.get(dtype)
            if obj is not None:
                return obj
        except TypeError:
            pass
        if dtype is None:
            warnings.warn('finfo() dtype cannot be None. This behavior will raise an error in the future. (Deprecated in NumPy 1.25)', DeprecationWarning, stacklevel=2)
        try:
            dtype = numeric.dtype(dtype)
        except TypeError:
            dtype = numeric.dtype(type(dtype))
        obj = cls._finfo_cache.get(dtype)
        if obj is not None:
            return obj
        dtypes = [dtype]
        newdtype = ntypes.obj2sctype(dtype)
        if newdtype is not dtype:
            dtypes.append(newdtype)
            dtype = newdtype
        if not issubclass(dtype, numeric.inexact):
            raise ValueError('data type %r not inexact' % dtype)
        obj = cls._finfo_cache.get(dtype)
        if obj is not None:
            return obj
        if not issubclass(dtype, numeric.floating):
            newdtype = _convert_to_float[dtype]
            if newdtype is not dtype:
                dtypes.append(newdtype)
                dtype = newdtype
                obj = cls._finfo_cache.get(dtype, None)
                if obj is not None:
                    for dt in dtypes:
                        cls._finfo_cache[dt] = obj
                    return obj
        obj = object.__new__(cls)._init(dtype)
        for dt in dtypes:
            cls._finfo_cache[dt] = obj
        return obj

    def _init(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = numeric.dtype(dtype)
        machar = _get_machar(dtype)
        for word in ['precision', 'iexp', 'maxexp', 'minexp', 'negep', 'machep']:
            setattr(self, word, getattr(machar, word))
        for word in ['resolution', 'epsneg', 'smallest_subnormal']:
            setattr(self, word, getattr(machar, word).flat[0])
        self.bits = self.dtype.itemsize * 8
        self.max = machar.huge.flat[0]
        self.min = -self.max
        self.eps = machar.eps.flat[0]
        self.nexp = machar.iexp
        self.nmant = machar.it
        self._machar = machar
        self._str_tiny = machar._str_xmin.strip()
        self._str_max = machar._str_xmax.strip()
        self._str_epsneg = machar._str_epsneg.strip()
        self._str_eps = machar._str_eps.strip()
        self._str_resolution = machar._str_resolution.strip()
        self._str_smallest_normal = machar._str_smallest_normal.strip()
        self._str_smallest_subnormal = machar._str_smallest_subnormal.strip()
        return self

    def __str__(self):
        if False:
            i = 10
            return i + 15
        fmt = 'Machine parameters for %(dtype)s\n---------------------------------------------------------------\nprecision = %(precision)3s   resolution = %(_str_resolution)s\nmachep = %(machep)6s   eps =        %(_str_eps)s\nnegep =  %(negep)6s   epsneg =     %(_str_epsneg)s\nminexp = %(minexp)6s   tiny =       %(_str_tiny)s\nmaxexp = %(maxexp)6s   max =        %(_str_max)s\nnexp =   %(nexp)6s   min =        -max\nsmallest_normal = %(_str_smallest_normal)s   smallest_subnormal = %(_str_smallest_subnormal)s\n---------------------------------------------------------------\n'
        return fmt % self.__dict__

    def __repr__(self):
        if False:
            print('Hello World!')
        c = self.__class__.__name__
        d = self.__dict__.copy()
        d['klass'] = c
        return '%(klass)s(resolution=%(resolution)s, min=-%(_str_max)s, max=%(_str_max)s, dtype=%(dtype)s)' % d

    @property
    def smallest_normal(self):
        if False:
            return 10
        'Return the value for the smallest normal.\n\n        Returns\n        -------\n        smallest_normal : float\n            Value for the smallest normal.\n\n        Warns\n        -----\n        UserWarning\n            If the calculated value for the smallest normal is requested for\n            double-double.\n        '
        if isnan(self._machar.smallest_normal.flat[0]):
            warnings.warn('The value of smallest normal is undefined for double double', UserWarning, stacklevel=2)
        return self._machar.smallest_normal.flat[0]

    @property
    def tiny(self):
        if False:
            while True:
                i = 10
        'Return the value for tiny, alias of smallest_normal.\n\n        Returns\n        -------\n        tiny : float\n            Value for the smallest normal, alias of smallest_normal.\n\n        Warns\n        -----\n        UserWarning\n            If the calculated value for the smallest normal is requested for\n            double-double.\n        '
        return self.smallest_normal

@set_module('numpy')
class iinfo:
    """
    iinfo(type)

    Machine limits for integer types.

    Attributes
    ----------
    bits : int
        The number of bits occupied by the type.
    dtype : dtype
        Returns the dtype for which `iinfo` returns information.
    min : int
        The smallest integer expressible by the type.
    max : int
        The largest integer expressible by the type.

    Parameters
    ----------
    int_type : integer type, dtype, or instance
        The kind of integer data type to get information about.

    See Also
    --------
    finfo : The equivalent for floating point data types.

    Examples
    --------
    With types:

    >>> ii16 = np.iinfo(np.int16)
    >>> ii16.min
    -32768
    >>> ii16.max
    32767
    >>> ii32 = np.iinfo(np.int32)
    >>> ii32.min
    -2147483648
    >>> ii32.max
    2147483647

    With instances:

    >>> ii32 = np.iinfo(np.int32(10))
    >>> ii32.min
    -2147483648
    >>> ii32.max
    2147483647

    """
    _min_vals = {}
    _max_vals = {}

    def __init__(self, int_type):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.dtype = numeric.dtype(int_type)
        except TypeError:
            self.dtype = numeric.dtype(type(int_type))
        self.kind = self.dtype.kind
        self.bits = self.dtype.itemsize * 8
        self.key = '%s%d' % (self.kind, self.bits)
        if self.kind not in 'iu':
            raise ValueError('Invalid integer data type %r.' % (self.kind,))

    @property
    def min(self):
        if False:
            return 10
        'Minimum value of given dtype.'
        if self.kind == 'u':
            return 0
        else:
            try:
                val = iinfo._min_vals[self.key]
            except KeyError:
                val = int(-(1 << self.bits - 1))
                iinfo._min_vals[self.key] = val
            return val

    @property
    def max(self):
        if False:
            while True:
                i = 10
        'Maximum value of given dtype.'
        try:
            val = iinfo._max_vals[self.key]
        except KeyError:
            if self.kind == 'u':
                val = int((1 << self.bits) - 1)
            else:
                val = int((1 << self.bits - 1) - 1)
            iinfo._max_vals[self.key] = val
        return val

    def __str__(self):
        if False:
            print('Hello World!')
        'String representation.'
        fmt = 'Machine parameters for %(dtype)s\n---------------------------------------------------------------\nmin = %(min)s\nmax = %(max)s\n---------------------------------------------------------------\n'
        return fmt % {'dtype': self.dtype, 'min': self.min, 'max': self.max}

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '%s(min=%s, max=%s, dtype=%s)' % (self.__class__.__name__, self.min, self.max, self.dtype)