from __future__ import absolute_import
import numpy
import chainer
from chainer import _backend
from chainer.backends import _cpu
from chainer.configuration import config
_ideep_version = None
_error = None
try:
    import ideep4py as ideep
    from ideep4py import mdarray
    _ideep_version = 2 if hasattr(ideep, '__version__') else 1
except ImportError as e:
    _error = e
    _ideep_version = None

    class mdarray(object):
        pass

class Intel64Device(_backend.Device):
    """Device for Intel64 (Intel Architecture) backend with iDeep"""
    xp = numpy
    name = '@intel64'
    supported_array_types = (numpy.ndarray, mdarray)
    __hash__ = _backend.Device.__hash__

    def __init__(self):
        if False:
            i = 10
            return i + 15
        check_ideep_available()
        super(Intel64Device, self).__init__()

    @staticmethod
    def from_array(array):
        if False:
            print('Hello World!')
        if isinstance(array, mdarray):
            return Intel64Device()
        return None

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, Intel64Device)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<{}>'.format(self.__class__.__name__)

    def send_array(self, array):
        if False:
            print('Hello World!')
        if isinstance(array, ideep.mdarray):
            return array
        if not isinstance(array, numpy.ndarray):
            array = _cpu._to_cpu(array)
        if isinstance(array, numpy.ndarray) and array.ndim in (1, 2, 4) and (0 not in array.shape):
            array = ideep.array(array, itype=ideep.wgt_array)
        return array

    def is_array_supported(self, array):
        if False:
            print('Hello World!')
        return isinstance(array, (numpy.ndarray, mdarray))
_SHOULD_USE_IDEEP = {'==always': {'always': True, 'auto': False, 'never': False}, '>=auto': {'always': True, 'auto': True, 'never': False}}

def is_ideep_available():
    if False:
        return 10
    'Returns if iDeep is available.\n\n    Returns:\n        bool: ``True`` if the supported version of iDeep is installed.\n    '
    return _ideep_version is not None and _ideep_version == 2

def check_ideep_available():
    if False:
        i = 10
        return i + 15
    'Checks if iDeep is available.\n\n    When iDeep is correctly set up, nothing happens.\n    Otherwise it raises ``RuntimeError``.\n    '
    if _ideep_version is None:
        msg = str(_error)
        if 'cannot open shared object file' in msg:
            msg += '\n\nEnsure iDeep requirements are satisfied: https://github.com/intel/ideep'
        raise RuntimeError('iDeep is not available.\nReason: {}: {}'.format(type(_error).__name__, msg))
    elif _ideep_version != 2:
        raise RuntimeError('iDeep is not available.\nReason: Unsupported iDeep version ({})'.format(_ideep_version))

def should_use_ideep(level):
    if False:
        print('Hello World!')
    "Determines if we should use iDeep.\n\n    This function checks ``chainer.config.use_ideep`` and availability\n    of ``ideep4py`` package.\n\n    Args:\n        level (str): iDeep use level. It must be either ``'==always'`` or\n            ``'>=auto'``. ``'==always'`` indicates that the ``use_ideep``\n            config must be ``'always'`` to use iDeep.\n\n    Returns:\n        bool: ``True`` if the caller should use iDeep.\n\n    "
    if not is_ideep_available():
        return False
    if level not in _SHOULD_USE_IDEEP:
        raise ValueError('invalid iDeep use level: %s (must be either of "==always" or ">=auto")' % repr(level))
    flags = _SHOULD_USE_IDEEP[level]
    use_ideep = config.use_ideep
    if use_ideep not in flags:
        raise ValueError('invalid use_ideep configuration: %s (must be either of "always", "auto", or "never")' % repr(use_ideep))
    return flags[use_ideep]

def inputs_all_ready(inputs, supported_ndim=(2, 4)):
    if False:
        print('Hello World!')
    'Checks if input arrays are supported for an iDeep primitive.\n\n    Before calling an iDeep primitive (e.g., ``ideep4py.linear.Forward``), you\n    need to make sure that all input arrays are ready for the primitive by\n    calling this function.\n    Information to be checked includes array types, dimesions and data types.\n    The function checks ``inputs`` info and ``supported_ndim``.\n\n    Inputs to be tested can be any of ``Variable``, ``numpy.ndarray`` or\n    ``ideep4py.mdarray``. However, all inputs to iDeep primitives must be\n    ``ideep4py.mdarray``. Callers of iDeep primitives are responsible of\n    converting all inputs to ``ideep4py.mdarray``.\n\n    Args:\n        inputs (sequence of arrays or variables):\n            Inputs to be checked.\n        supported_ndim (tuple of ints):\n            Supported ndim values for the iDeep primitive.\n\n    Returns:\n        bool: ``True`` if all conditions meet.\n\n    '

    def _is_supported_array_type(a):
        if False:
            return 10
        return isinstance(a, ideep.mdarray) or ideep.check_type([a])
    if not is_ideep_available():
        return False
    inputs = [x.data if isinstance(x, chainer.variable.Variable) else x for x in inputs]
    return ideep.check_ndim(inputs, supported_ndim) and all([_is_supported_array_type(a) for a in inputs])