"""
A module that provides functions for manipulating bit masks and data quality
(DQ) arrays.

"""
import numbers
import warnings
from collections import OrderedDict
import numpy as np
__all__ = ['bitfield_to_boolean_mask', 'interpret_bit_flags', 'BitFlagNameMap', 'extend_bit_flag_map', 'InvalidBitFlag']
_ENABLE_BITFLAG_CACHING = True
_SUPPORTED_FLAGS = int(np.bitwise_not(0, dtype='uint64', casting='unsafe'))

def _is_bit_flag(n):
    if False:
        i = 10
        return i + 15
    '\n    Verifies if the input number is a bit flag (i.e., an integer number that is\n    an integer power of 2).\n\n    Parameters\n    ----------\n    n : int\n        A positive integer number. Non-positive integers are considered not to\n        be "flags".\n\n    Returns\n    -------\n    bool\n        ``True`` if input ``n`` is a bit flag and ``False`` if it is not.\n\n    '
    if n < 1:
        return False
    return bin(n).count('1') == 1

def _is_int(n):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(n, numbers.Integral) and (not isinstance(n, bool)) or (isinstance(n, np.generic) and np.issubdtype(n, np.integer))

class InvalidBitFlag(ValueError):
    """Indicates that a value is not an integer that is a power of 2."""

class BitFlag(int):
    """Bit flags: integer values that are powers of 2."""

    def __new__(cls, val, doc=None):
        if False:
            i = 10
            return i + 15
        if isinstance(val, tuple):
            if doc is not None:
                raise ValueError("Flag's doc string cannot be provided twice.")
            (val, doc) = val
        if not (_is_int(val) and _is_bit_flag(val)):
            raise InvalidBitFlag(f"Value '{val}' is not a valid bit flag: bit flag value must be an integral power of two.")
        s = int.__new__(cls, val)
        if doc is not None:
            s.__doc__ = doc
        return s

class BitFlagNameMeta(type):

    def __new__(mcls, name, bases, members):
        if False:
            print('Hello World!')
        for (k, v) in members.items():
            if not k.startswith('_'):
                v = BitFlag(v)
        attr = [k for k in members.keys() if not k.startswith('_')]
        attrl = list(map(str.lower, attr))
        if _ENABLE_BITFLAG_CACHING:
            cache = OrderedDict()
        for b in bases:
            for (k, v) in b.__dict__.items():
                if k.startswith('_'):
                    continue
                kl = k.lower()
                if kl in attrl:
                    idx = attrl.index(kl)
                    raise AttributeError(f"Bit flag '{attr[idx]:s}' was already defined.")
                if _ENABLE_BITFLAG_CACHING:
                    cache[kl] = v
        members = {k: v if k.startswith('_') else BitFlag(v) for (k, v) in members.items()}
        if _ENABLE_BITFLAG_CACHING:
            cache.update({k.lower(): v for (k, v) in members.items() if not k.startswith('_')})
            members = {'_locked': True, '__version__': '', **members, '_cache': cache}
        else:
            members = {'_locked': True, '__version__': '', **members}
        return super().__new__(mcls, name, bases, members)

    def __setattr__(cls, name, val):
        if False:
            print('Hello World!')
        if name == '_locked':
            return super().__setattr__(name, True)
        else:
            if name == '__version__':
                if cls._locked:
                    raise AttributeError('Version cannot be modified.')
                return super().__setattr__(name, val)
            err_msg = f'Bit flags are read-only. Unable to reassign attribute {name}'
            if cls._locked:
                raise AttributeError(err_msg)
        namel = name.lower()
        if _ENABLE_BITFLAG_CACHING:
            if not namel.startswith('_') and namel in cls._cache:
                raise AttributeError(err_msg)
        else:
            for b in cls.__bases__:
                if not namel.startswith('_') and namel in list(map(str.lower, b.__dict__)):
                    raise AttributeError(err_msg)
            if namel in list(map(str.lower, cls.__dict__)):
                raise AttributeError(err_msg)
        val = BitFlag(val)
        if _ENABLE_BITFLAG_CACHING and (not namel.startswith('_')):
            cls._cache[namel] = val
        return super().__setattr__(name, val)

    def __getattr__(cls, name):
        if False:
            i = 10
            return i + 15
        if _ENABLE_BITFLAG_CACHING:
            flagnames = cls._cache
        else:
            flagnames = {k.lower(): v for (k, v) in cls.__dict__.items()}
            flagnames.update({k.lower(): v for b in cls.__bases__ for (k, v) in b.__dict__.items()})
        try:
            return flagnames[name.lower()]
        except KeyError:
            raise AttributeError(f"Flag '{name}' not defined")

    def __getitem__(cls, key):
        if False:
            while True:
                i = 10
        return cls.__getattr__(key)

    def __add__(cls, items):
        if False:
            while True:
                i = 10
        if not isinstance(items, dict):
            if not isinstance(items[0], (tuple, list)):
                items = [items]
            items = dict(items)
        return extend_bit_flag_map(cls.__name__ + '_' + '_'.join(list(items)), cls, **items)

    def __iadd__(cls, other):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError("Unary '+' is not supported. Use binary operator instead.")

    def __delattr__(cls, name):
        if False:
            for i in range(10):
                print('nop')
        raise AttributeError(f'{cls.__name__}: cannot delete {cls.mro()[-2].__name__} member.')

    def __delitem__(cls, name):
        if False:
            i = 10
            return i + 15
        raise AttributeError(f'{cls.__name__}: cannot delete {cls.mro()[-2].__name__} member.')

    def __repr__(cls):
        if False:
            for i in range(10):
                print('nop')
        return f"<{cls.mro()[-2].__name__} '{cls.__name__}'>"

class BitFlagNameMap(metaclass=BitFlagNameMeta):
    """
    A base class for bit flag name maps used to describe data quality (DQ)
    flags of images by provinding a mapping from a mnemonic flag name to a flag
    value.

    Mapping for a specific instrument should subclass this class.
    Subclasses should define flags as class attributes with integer values
    that are powers of 2. Each bit flag may also contain a string
    comment following the flag value.

    Examples
    --------
        >>> from astropy.nddata.bitmask import BitFlagNameMap
        >>> class ST_DQ(BitFlagNameMap):
        ...     __version__ = '1.0.0'  # optional
        ...     CR = 1, 'Cosmic Ray'
        ...     CLOUDY = 4  # no docstring comment
        ...     RAINY = 8, 'Dome closed'
        ...
        >>> class ST_CAM1_DQ(ST_DQ):
        ...     HOT = 16
        ...     DEAD = 32

    """

def extend_bit_flag_map(cls_name, base_cls=BitFlagNameMap, **kwargs):
    if False:
        while True:
            i = 10
    "\n    A convenience function for creating bit flags maps by subclassing an\n    existing map and adding additional flags supplied as keyword arguments.\n\n    Parameters\n    ----------\n    cls_name : str\n        Class name of the bit flag map to be created.\n\n    base_cls : BitFlagNameMap, optional\n        Base class for the new bit flag map.\n\n    **kwargs : int\n        Each supplied keyword argument will be used to define bit flag\n        names in the new map. In addition to bit flag names, ``__version__`` is\n        allowed to indicate the version of the newly created map.\n\n    Examples\n    --------\n        >>> from astropy.nddata.bitmask import extend_bit_flag_map\n        >>> ST_DQ = extend_bit_flag_map('ST_DQ', __version__='1.0.0', CR=1, CLOUDY=4, RAINY=8)\n        >>> ST_CAM1_DQ = extend_bit_flag_map('ST_CAM1_DQ', ST_DQ, HOT=16, DEAD=32)\n        >>> ST_CAM1_DQ['HOT']  # <-- Access flags as dictionary keys\n        16\n        >>> ST_CAM1_DQ.HOT  # <-- Access flags as class attributes\n        16\n\n    "
    new_cls = BitFlagNameMeta.__new__(BitFlagNameMeta, cls_name, (base_cls,), {'_locked': False})
    for (k, v) in kwargs.items():
        try:
            setattr(new_cls, k, v)
        except AttributeError as e:
            if new_cls[k] != int(v):
                raise e
    new_cls._locked = True
    return new_cls

def interpret_bit_flags(bit_flags, flip_bits=None, flag_name_map=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts input bit flags to a single integer value (bit mask) or `None`.\n\n    When input is a list of flags (either a Python list of integer flags or a\n    string of comma-, ``\'|\'``-, or ``\'+\'``-separated list of flags),\n    the returned bit mask is obtained by summing input flags.\n\n    .. note::\n        In order to flip the bits of the returned bit mask,\n        for input of `str` type, prepend \'~\' to the input string. \'~\' must\n        be prepended to the *entire string* and not to each bit flag! For\n        input that is already a bit mask or a Python list of bit flags, set\n        ``flip_bits`` for `True` in order to flip the bits of the returned\n        bit mask.\n\n    Parameters\n    ----------\n    bit_flags : int, str, list, None\n        An integer bit mask or flag, `None`, a string of comma-, ``\'|\'``- or\n        ``\'+\'``-separated list of integer bit flags or mnemonic flag names,\n        or a Python list of integer bit flags. If ``bit_flags`` is a `str`\n        and if it is prepended with \'~\', then the output bit mask will have\n        its bits flipped (compared to simple sum of input flags).\n        For input ``bit_flags`` that is already a bit mask or a Python list\n        of bit flags, bit-flipping can be controlled through ``flip_bits``\n        parameter.\n\n        .. note::\n            When ``bit_flags`` is a list of flag names, the ``flag_name_map``\n            parameter must be provided.\n\n        .. note::\n            Only one flag separator is supported at a time. ``bit_flags``\n            string should not mix ``\',\'``, ``\'+\'``, and ``\'|\'`` separators.\n\n    flip_bits : bool, None\n        Indicates whether or not to flip the bits of the returned bit mask\n        obtained from input bit flags. This parameter must be set to `None`\n        when input ``bit_flags`` is either `None` or a Python list of flags.\n\n    flag_name_map : BitFlagNameMap\n         A `BitFlagNameMap` object that provides mapping from mnemonic\n         bit flag names to integer bit values in order to translate mnemonic\n         flags to numeric values when ``bit_flags`` that are comma- or\n         \'+\'-separated list of menmonic bit flag names.\n\n    Returns\n    -------\n    bitmask : int or None\n        Returns an integer bit mask formed from the input bit value or `None`\n        if input ``bit_flags`` parameter is `None` or an empty string.\n        If input string value was prepended with \'~\' (or ``flip_bits`` was set\n        to `True`), then returned value will have its bits flipped\n        (inverse mask).\n\n    Examples\n    --------\n        >>> from astropy.nddata.bitmask import interpret_bit_flags, extend_bit_flag_map\n        >>> ST_DQ = extend_bit_flag_map(\'ST_DQ\', CR=1, CLOUDY=4, RAINY=8, HOT=16, DEAD=32)\n        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags(28))\n        \'0000000000011100\'\n        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags(\'4,8,16\'))\n        \'0000000000011100\'\n        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags(\'CLOUDY,RAINY,HOT\', flag_name_map=ST_DQ))\n        \'0000000000011100\'\n        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags(\'~4,8,16\'))\n        \'1111111111100011\'\n        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags(\'~(4+8+16)\'))\n        \'1111111111100011\'\n        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags(\'~(CLOUDY+RAINY+HOT)\',\n        ... flag_name_map=ST_DQ))\n        \'1111111111100011\'\n        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags([4, 8, 16]))\n        \'0000000000011100\'\n        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags([4, 8, 16], flip_bits=True))\n        \'1111111111100011\'\n\n    '
    has_flip_bits = flip_bits is not None
    flip_bits = bool(flip_bits)
    allow_non_flags = False
    if _is_int(bit_flags):
        return ~int(bit_flags) if flip_bits else int(bit_flags)
    elif bit_flags is None:
        if has_flip_bits:
            raise TypeError("Keyword argument 'flip_bits' must be set to 'None' when input 'bit_flags' is None.")
        return None
    elif isinstance(bit_flags, str):
        if has_flip_bits:
            raise TypeError("Keyword argument 'flip_bits' is not permitted for comma-separated string lists of bit flags. Prepend '~' to the string to indicate bit-flipping.")
        bit_flags = str(bit_flags).strip()
        if bit_flags.upper() in ['', 'NONE', 'INDEF']:
            return None
        bitflip_pos = bit_flags.find('~')
        if bitflip_pos == 0:
            flip_bits = True
            bit_flags = bit_flags[1:].lstrip()
        else:
            if bitflip_pos > 0:
                raise ValueError('Bitwise-NOT must precede bit flag list.')
            flip_bits = False
        while True:
            nlpar = bit_flags.count('(')
            nrpar = bit_flags.count(')')
            if nlpar == 0 and nrpar == 0:
                break
            if nlpar != nrpar:
                raise ValueError('Unbalanced parentheses in bit flag list.')
            lpar_pos = bit_flags.find('(')
            rpar_pos = bit_flags.rfind(')')
            if lpar_pos > 0 or rpar_pos < len(bit_flags) - 1:
                raise ValueError('Incorrect syntax (incorrect use of parenthesis) in bit flag list.')
            bit_flags = bit_flags[1:-1].strip()
        if sum((k in bit_flags for k in '+,|')) > 1:
            raise ValueError("Only one type of bit flag separator may be used in one expression. Allowed separators are: '+', '|', or ','.")
        if ',' in bit_flags:
            bit_flags = bit_flags.split(',')
        elif '+' in bit_flags:
            bit_flags = bit_flags.split('+')
        elif '|' in bit_flags:
            bit_flags = bit_flags.split('|')
        else:
            if bit_flags == '':
                raise ValueError('Empty bit flag lists not allowed when either bitwise-NOT or parenthesis are present.')
            bit_flags = [bit_flags]
        if flag_name_map is not None:
            try:
                int(bit_flags[0])
            except ValueError:
                bit_flags = [flag_name_map[f] for f in bit_flags]
        allow_non_flags = len(bit_flags) == 1
    elif hasattr(bit_flags, '__iter__'):
        if not all((_is_int(flag) for flag in bit_flags)):
            if flag_name_map is not None and all((isinstance(flag, str) for flag in bit_flags)):
                bit_flags = [flag_name_map[f] for f in bit_flags]
            else:
                raise TypeError("Every bit flag in a list must be either an integer flag value or a 'str' flag name.")
    else:
        raise TypeError("Unsupported type for argument 'bit_flags'.")
    bitset = set(map(int, bit_flags))
    if len(bitset) != len(bit_flags):
        warnings.warn('Duplicate bit flags will be ignored')
    bitmask = 0
    for v in bitset:
        if not _is_bit_flag(v) and (not allow_non_flags):
            raise ValueError(f'Input list contains invalid (not powers of two) bit flag: {v}')
        bitmask += v
    if flip_bits:
        bitmask = ~bitmask
    return bitmask

def bitfield_to_boolean_mask(bitfield, ignore_flags=0, flip_bits=None, good_mask_value=False, dtype=np.bool_, flag_name_map=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    bitfield_to_boolean_mask(bitfield, ignore_flags=None, flip_bits=None, good_mask_value=False, dtype=numpy.bool_)\n    Converts an array of bit fields to a boolean (or integer) mask array\n    according to a bit mask constructed from the supplied bit flags (see\n    ``ignore_flags`` parameter).\n\n    This function is particularly useful to convert data quality arrays to\n    boolean masks with selective filtering of DQ flags.\n\n    Parameters\n    ----------\n    bitfield : ndarray\n        An array of bit flags. By default, values different from zero are\n        interpreted as "bad" values and values equal to zero are considered\n        as "good" values. However, see ``ignore_flags`` parameter on how to\n        selectively ignore some bits in the ``bitfield`` array data.\n\n    ignore_flags : int, str, list, None (default = 0)\n        An integer bit mask, `None`, a Python list of bit flags, a comma-,\n        or ``\'|\'``-separated, ``\'+\'``-separated string list of integer\n        bit flags or mnemonic flag names that indicate what bits in the input\n        ``bitfield`` should be *ignored* (i.e., zeroed), or `None`.\n\n        .. note::\n            When ``bit_flags`` is a list of flag names, the ``flag_name_map``\n            parameter must be provided.\n\n        | Setting ``ignore_flags`` to `None` effectively will make\n          `bitfield_to_boolean_mask` interpret all ``bitfield`` elements\n          as "good" regardless of their value.\n\n        | When ``ignore_flags`` argument is an integer bit mask, it will be\n          combined using bitwise-NOT and bitwise-AND with each element of the\n          input ``bitfield`` array (``~ignore_flags & bitfield``). If the\n          resultant bitfield element is non-zero, that element will be\n          interpreted as a "bad" in the output boolean mask and it will be\n          interpreted as "good" otherwise. ``flip_bits`` parameter may be used\n          to flip the bits (``bitwise-NOT``) of the bit mask thus effectively\n          changing the meaning of the ``ignore_flags`` parameter from "ignore"\n          to "use only" these flags.\n\n        .. note::\n\n            Setting ``ignore_flags`` to 0 effectively will assume that all\n            non-zero elements in the input ``bitfield`` array are to be\n            interpreted as "bad".\n\n        | When ``ignore_flags`` argument is a Python list of integer bit\n          flags, these flags are added together to create an integer bit mask.\n          Each item in the list must be a flag, i.e., an integer that is an\n          integer power of 2. In order to flip the bits of the resultant\n          bit mask, use ``flip_bits`` parameter.\n\n        | Alternatively, ``ignore_flags`` may be a string of comma- or\n          ``\'+\'``(or ``\'|\'``)-separated list of integer bit flags that should\n          be added (bitwise OR) together to create an integer bit mask.\n          For example, both ``\'4,8\'``, ``\'4|8\'``, and ``\'4+8\'`` are equivalent\n          and indicate that bit flags 4 and 8 in the input ``bitfield``\n          array should be ignored when generating boolean mask.\n\n        .. note::\n\n            ``\'None\'``, ``\'INDEF\'``, and empty (or all white space) strings\n            are special values of string ``ignore_flags`` that are\n            interpreted as `None`.\n\n        .. note::\n\n            Each item in the list must be a flag, i.e., an integer that is an\n            integer power of 2. In addition, for convenience, an arbitrary\n            **single** integer is allowed and it will be interpreted as an\n            integer bit mask. For example, instead of ``\'4,8\'`` one could\n            simply provide string ``\'12\'``.\n\n        .. note::\n            Only one flag separator is supported at a time. ``ignore_flags``\n            string should not mix ``\',\'``, ``\'+\'``, and ``\'|\'`` separators.\n\n        .. note::\n\n            When ``ignore_flags`` is a `str` and when it is prepended with\n            \'~\', then the meaning of ``ignore_flags`` parameters will be\n            reversed: now it will be interpreted as a list of bit flags to be\n            *used* (or *not ignored*) when deciding which elements of the\n            input ``bitfield`` array are "bad". Following this convention,\n            an ``ignore_flags`` string value of ``\'~0\'`` would be equivalent\n            to setting ``ignore_flags=None``.\n\n        .. warning::\n\n            Because prepending \'~\' to a string ``ignore_flags`` is equivalent\n            to setting ``flip_bits`` to `True`, ``flip_bits`` cannot be used\n            with string ``ignore_flags`` and it must be set to `None`.\n\n    flip_bits : bool, None (default = None)\n        Specifies whether or not to invert the bits of the bit mask either\n        supplied directly through ``ignore_flags`` parameter or built from the\n        bit flags passed through ``ignore_flags`` (only when bit flags are\n        passed as Python lists of integer bit flags). Occasionally, it may be\n        useful to *consider only specific bit flags* in the ``bitfield``\n        array when creating a boolean mask as opposed to *ignoring* specific\n        bit flags as ``ignore_flags`` behaves by default. This can be achieved\n        by inverting/flipping the bits of the bit mask created from\n        ``ignore_flags`` flags which effectively changes the meaning of the\n        ``ignore_flags`` parameter from "ignore" to "use only" these flags.\n        Setting ``flip_bits`` to `None` means that no bit flipping will be\n        performed. Bit flipping for string lists of bit flags must be\n        specified by prepending \'~\' to string bit flag lists\n        (see documentation for ``ignore_flags`` for more details).\n\n        .. warning::\n            This parameter can be set to either `True` or `False` **ONLY** when\n            ``ignore_flags`` is either an integer bit mask or a Python\n            list of integer bit flags. When ``ignore_flags`` is either\n            `None` or a string list of flags, ``flip_bits`` **MUST** be set\n            to `None`.\n\n    good_mask_value : int, bool (default = False)\n        This parameter is used to derive the values that will be assigned to\n        the elements in the output boolean mask array that correspond to the\n        "good" bit fields (that are 0 after zeroing bits specified by\n        ``ignore_flags``) in the input ``bitfield`` array. When\n        ``good_mask_value`` is non-zero or ``numpy.True_`` then values in the\n        output boolean mask array corresponding to "good" bit fields in\n        ``bitfield`` will be ``numpy.True_`` (if ``dtype`` is ``numpy.bool_``)\n        or 1 (if ``dtype`` is of numerical type) and values of corresponding\n        to "bad" flags will be ``numpy.False_`` (or 0). When\n        ``good_mask_value`` is zero or ``numpy.False_`` then the values\n        in the output boolean mask array corresponding to "good" bit fields\n        in ``bitfield`` will be ``numpy.False_`` (if ``dtype`` is\n        ``numpy.bool_``) or 0 (if ``dtype`` is of numerical type) and values\n        of corresponding to "bad" flags will be ``numpy.True_`` (or 1).\n\n    dtype : data-type (default = ``numpy.bool_``)\n        The desired data-type for the output binary mask array.\n\n    flag_name_map : BitFlagNameMap\n         A `BitFlagNameMap` object that provides mapping from mnemonic\n         bit flag names to integer bit values in order to translate mnemonic\n         flags to numeric values when ``bit_flags`` that are comma- or\n         \'+\'-separated list of menmonic bit flag names.\n\n    Returns\n    -------\n    mask : ndarray\n        Returns an array of the same dimensionality as the input ``bitfield``\n        array whose elements can have two possible values,\n        e.g., ``numpy.True_`` or ``numpy.False_`` (or 1 or 0 for integer\n        ``dtype``) according to values of to the input ``bitfield`` elements,\n        ``ignore_flags`` parameter, and the ``good_mask_value`` parameter.\n\n    Examples\n    --------\n        >>> from astropy.nddata import bitmask\n        >>> import numpy as np\n        >>> dqarr = np.asarray([[0, 0, 1, 2, 0, 8, 12, 0],\n        ...                     [10, 4, 0, 0, 0, 16, 6, 0]])\n        >>> flag_map = bitmask.extend_bit_flag_map(\n        ...     \'ST_DQ\', CR=2, CLOUDY=4, RAINY=8, HOT=16, DEAD=32\n        ... )\n        >>> bitmask.bitfield_to_boolean_mask(dqarr, ignore_flags=0,\n        ...                                  dtype=int)\n        array([[0, 0, 1, 1, 0, 1, 1, 0],\n               [1, 1, 0, 0, 0, 1, 1, 0]])\n        >>> bitmask.bitfield_to_boolean_mask(dqarr, ignore_flags=0,\n        ...                                  dtype=bool)\n        array([[False, False,  True,  True, False,  True,  True, False],\n               [ True,  True, False, False, False,  True,  True, False]]...)\n        >>> bitmask.bitfield_to_boolean_mask(dqarr, ignore_flags=6,\n        ...                                  good_mask_value=0, dtype=int)\n        array([[0, 0, 1, 0, 0, 1, 1, 0],\n               [1, 0, 0, 0, 0, 1, 0, 0]])\n        >>> bitmask.bitfield_to_boolean_mask(dqarr, ignore_flags=~6,\n        ...                                  good_mask_value=0, dtype=int)\n        array([[0, 0, 0, 1, 0, 0, 1, 0],\n               [1, 1, 0, 0, 0, 0, 1, 0]])\n        >>> bitmask.bitfield_to_boolean_mask(dqarr, ignore_flags=6, dtype=int,\n        ...                                  flip_bits=True, good_mask_value=0)\n        array([[0, 0, 0, 1, 0, 0, 1, 0],\n               [1, 1, 0, 0, 0, 0, 1, 0]])\n        >>> bitmask.bitfield_to_boolean_mask(dqarr, ignore_flags=\'~(2+4)\',\n        ...                                  good_mask_value=0, dtype=int)\n        array([[0, 0, 0, 1, 0, 0, 1, 0],\n               [1, 1, 0, 0, 0, 0, 1, 0]])\n        >>> bitmask.bitfield_to_boolean_mask(dqarr, ignore_flags=[2, 4],\n        ...                                  flip_bits=True, good_mask_value=0,\n        ...                                  dtype=int)\n        array([[0, 0, 0, 1, 0, 0, 1, 0],\n               [1, 1, 0, 0, 0, 0, 1, 0]])\n        >>> bitmask.bitfield_to_boolean_mask(dqarr, ignore_flags=\'~(CR,CLOUDY)\',\n        ...                                  good_mask_value=0, dtype=int,\n        ...                                  flag_name_map=flag_map)\n        array([[0, 0, 0, 1, 0, 0, 1, 0],\n               [1, 1, 0, 0, 0, 0, 1, 0]])\n        >>> bitmask.bitfield_to_boolean_mask(dqarr, ignore_flags=\'~(CR+CLOUDY)\',\n        ...                                  good_mask_value=0, dtype=int,\n        ...                                  flag_name_map=flag_map)\n        array([[0, 0, 0, 1, 0, 0, 1, 0],\n               [1, 1, 0, 0, 0, 0, 1, 0]])\n\n    '
    bitfield = np.asarray(bitfield)
    if not np.issubdtype(bitfield.dtype, np.integer):
        raise TypeError('Input bitfield array must be of integer type.')
    ignore_mask = interpret_bit_flags(ignore_flags, flip_bits=flip_bits, flag_name_map=flag_name_map)
    if ignore_mask is None:
        if good_mask_value:
            mask = np.ones_like(bitfield, dtype=dtype)
        else:
            mask = np.zeros_like(bitfield, dtype=dtype)
        return mask
    ignore_mask = ignore_mask & _SUPPORTED_FLAGS
    ignore_mask = np.bitwise_not(ignore_mask, dtype=bitfield.dtype.type, casting='unsafe')
    mask = np.empty_like(bitfield, dtype=np.bool_)
    np.bitwise_and(bitfield, ignore_mask, out=mask, casting='unsafe')
    if good_mask_value:
        np.logical_not(mask, out=mask)
    return mask.astype(dtype=dtype, subok=False, copy=False)