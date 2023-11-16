"""
A place for code to be called from the implementation of np.dtype

String handling is much easier to do correctly in python.
"""
import numpy as np
_kind_to_stem = {'u': 'uint', 'i': 'int', 'c': 'complex', 'f': 'float', 'b': 'bool', 'V': 'void', 'O': 'object', 'M': 'datetime', 'm': 'timedelta', 'S': 'bytes', 'U': 'str'}

def _kind_name(dtype):
    if False:
        for i in range(10):
            print('nop')
    try:
        return _kind_to_stem[dtype.kind]
    except KeyError as e:
        raise RuntimeError('internal dtype error, unknown kind {!r}'.format(dtype.kind)) from None

def __str__(dtype):
    if False:
        print('Hello World!')
    if dtype.fields is not None:
        return _struct_str(dtype, include_align=True)
    elif dtype.subdtype:
        return _subarray_str(dtype)
    elif issubclass(dtype.type, np.flexible) or not dtype.isnative:
        return dtype.str
    else:
        return dtype.name

def __repr__(dtype):
    if False:
        print('Hello World!')
    arg_str = _construction_repr(dtype, include_align=False)
    if dtype.isalignedstruct:
        arg_str = arg_str + ', align=True'
    return 'dtype({})'.format(arg_str)

def _unpack_field(dtype, offset, title=None):
    if False:
        print('Hello World!')
    '\n    Helper function to normalize the items in dtype.fields.\n\n    Call as:\n\n    dtype, offset, title = _unpack_field(*dtype.fields[name])\n    '
    return (dtype, offset, title)

def _isunsized(dtype):
    if False:
        print('Hello World!')
    return dtype.itemsize == 0

def _construction_repr(dtype, include_align=False, short=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Creates a string repr of the dtype, excluding the 'dtype()' part\n    surrounding the object. This object may be a string, a list, or\n    a dict depending on the nature of the dtype. This\n    is the object passed as the first parameter to the dtype\n    constructor, and if no additional constructor parameters are\n    given, will reproduce the exact memory layout.\n\n    Parameters\n    ----------\n    short : bool\n        If true, this creates a shorter repr using 'kind' and 'itemsize',\n        instead of the longer type name.\n\n    include_align : bool\n        If true, this includes the 'align=True' parameter\n        inside the struct dtype construction dict when needed. Use this flag\n        if you want a proper repr string without the 'dtype()' part around it.\n\n        If false, this does not preserve the\n        'align=True' parameter or sticky NPY_ALIGNED_STRUCT flag for\n        struct arrays like the regular repr does, because the 'align'\n        flag is not part of first dtype constructor parameter. This\n        mode is intended for a full 'repr', where the 'align=True' is\n        provided as the second parameter.\n    "
    if dtype.fields is not None:
        return _struct_str(dtype, include_align=include_align)
    elif dtype.subdtype:
        return _subarray_str(dtype)
    else:
        return _scalar_str(dtype, short=short)

def _scalar_str(dtype, short):
    if False:
        return 10
    byteorder = _byte_order_str(dtype)
    if dtype.type == np.bool_:
        if short:
            return "'?'"
        else:
            return "'bool'"
    elif dtype.type == np.object_:
        return "'O'"
    elif dtype.type == np.bytes_:
        if _isunsized(dtype):
            return "'S'"
        else:
            return "'S%d'" % dtype.itemsize
    elif dtype.type == np.str_:
        if _isunsized(dtype):
            return "'%sU'" % byteorder
        else:
            return "'%sU%d'" % (byteorder, dtype.itemsize / 4)
    elif not type(dtype)._legacy:
        return f"'{byteorder}{type(dtype).__name__}{dtype.itemsize * 8}'"
    elif issubclass(dtype.type, np.void):
        if _isunsized(dtype):
            return "'V'"
        else:
            return "'V%d'" % dtype.itemsize
    elif dtype.type == np.datetime64:
        return "'%sM8%s'" % (byteorder, _datetime_metadata_str(dtype))
    elif dtype.type == np.timedelta64:
        return "'%sm8%s'" % (byteorder, _datetime_metadata_str(dtype))
    elif np.issubdtype(dtype, np.number):
        if short or dtype.byteorder not in ('=', '|'):
            return "'%s%c%d'" % (byteorder, dtype.kind, dtype.itemsize)
        else:
            return "'%s%d'" % (_kind_name(dtype), 8 * dtype.itemsize)
    elif dtype.isbuiltin == 2:
        return dtype.type.__name__
    else:
        raise RuntimeError('Internal error: NumPy dtype unrecognized type number')

def _byte_order_str(dtype):
    if False:
        i = 10
        return i + 15
    " Normalize byteorder to '<' or '>' "
    swapped = np.dtype(int).newbyteorder('S')
    native = swapped.newbyteorder('S')
    byteorder = dtype.byteorder
    if byteorder == '=':
        return native.byteorder
    if byteorder == 'S':
        return swapped.byteorder
    elif byteorder == '|':
        return ''
    else:
        return byteorder

def _datetime_metadata_str(dtype):
    if False:
        for i in range(10):
            print('nop')
    (unit, count) = np.datetime_data(dtype)
    if unit == 'generic':
        return ''
    elif count == 1:
        return '[{}]'.format(unit)
    else:
        return '[{}{}]'.format(count, unit)

def _struct_dict_str(dtype, includealignedflag):
    if False:
        for i in range(10):
            print('nop')
    names = dtype.names
    fld_dtypes = []
    offsets = []
    titles = []
    for name in names:
        (fld_dtype, offset, title) = _unpack_field(*dtype.fields[name])
        fld_dtypes.append(fld_dtype)
        offsets.append(offset)
        titles.append(title)
    if np._core.arrayprint._get_legacy_print_mode() <= 121:
        colon = ':'
        fieldsep = ','
    else:
        colon = ': '
        fieldsep = ', '
    ret = "{'names'%s[" % colon
    ret += fieldsep.join((repr(name) for name in names))
    ret += "], 'formats'%s[" % colon
    ret += fieldsep.join((_construction_repr(fld_dtype, short=True) for fld_dtype in fld_dtypes))
    ret += "], 'offsets'%s[" % colon
    ret += fieldsep.join(('%d' % offset for offset in offsets))
    if any((title is not None for title in titles)):
        ret += "], 'titles'%s[" % colon
        ret += fieldsep.join((repr(title) for title in titles))
    ret += "], 'itemsize'%s%d" % (colon, dtype.itemsize)
    if includealignedflag and dtype.isalignedstruct:
        ret += ", 'aligned'%sTrue}" % colon
    else:
        ret += '}'
    return ret

def _aligned_offset(offset, alignment):
    if False:
        for i in range(10):
            print('nop')
    return -(-offset // alignment) * alignment

def _is_packed(dtype):
    if False:
        print('Hello World!')
    "\n    Checks whether the structured data type in 'dtype'\n    has a simple layout, where all the fields are in order,\n    and follow each other with no alignment padding.\n\n    When this returns true, the dtype can be reconstructed\n    from a list of the field names and dtypes with no additional\n    dtype parameters.\n\n    Duplicates the C `is_dtype_struct_simple_unaligned_layout` function.\n    "
    align = dtype.isalignedstruct
    max_alignment = 1
    total_offset = 0
    for name in dtype.names:
        (fld_dtype, fld_offset, title) = _unpack_field(*dtype.fields[name])
        if align:
            total_offset = _aligned_offset(total_offset, fld_dtype.alignment)
            max_alignment = max(max_alignment, fld_dtype.alignment)
        if fld_offset != total_offset:
            return False
        total_offset += fld_dtype.itemsize
    if align:
        total_offset = _aligned_offset(total_offset, max_alignment)
    if total_offset != dtype.itemsize:
        return False
    return True

def _struct_list_str(dtype):
    if False:
        print('Hello World!')
    items = []
    for name in dtype.names:
        (fld_dtype, fld_offset, title) = _unpack_field(*dtype.fields[name])
        item = '('
        if title is not None:
            item += '({!r}, {!r}), '.format(title, name)
        else:
            item += '{!r}, '.format(name)
        if fld_dtype.subdtype is not None:
            (base, shape) = fld_dtype.subdtype
            item += '{}, {}'.format(_construction_repr(base, short=True), shape)
        else:
            item += _construction_repr(fld_dtype, short=True)
        item += ')'
        items.append(item)
    return '[' + ', '.join(items) + ']'

def _struct_str(dtype, include_align):
    if False:
        print('Hello World!')
    if not (include_align and dtype.isalignedstruct) and _is_packed(dtype):
        sub = _struct_list_str(dtype)
    else:
        sub = _struct_dict_str(dtype, include_align)
    if dtype.type != np.void:
        return '({t.__module__}.{t.__name__}, {f})'.format(t=dtype.type, f=sub)
    else:
        return sub

def _subarray_str(dtype):
    if False:
        while True:
            i = 10
    (base, shape) = dtype.subdtype
    return '({}, {})'.format(_construction_repr(base, short=True), shape)

def _name_includes_bit_suffix(dtype):
    if False:
        i = 10
        return i + 15
    if dtype.type == np.object_:
        return False
    elif dtype.type == np.bool_:
        return False
    elif dtype.type is None:
        return True
    elif np.issubdtype(dtype, np.flexible) and _isunsized(dtype):
        return False
    else:
        return True

def _name_get(dtype):
    if False:
        while True:
            i = 10
    if dtype.isbuiltin == 2:
        return dtype.type.__name__
    if not type(dtype)._legacy:
        name = type(dtype).__name__
    elif issubclass(dtype.type, np.void):
        name = dtype.type.__name__
    else:
        name = _kind_name(dtype)
    if _name_includes_bit_suffix(dtype):
        name += '{}'.format(dtype.itemsize * 8)
    if dtype.type in (np.datetime64, np.timedelta64):
        name += _datetime_metadata_str(dtype)
    return name