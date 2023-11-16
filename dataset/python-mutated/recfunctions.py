"""
Collection of utilities to manipulate structured arrays.

Most of these functions were initially implemented by John Hunter for
matplotlib.  They have been rewritten and extended for convenience.

"""
import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy._core.overrides import array_function_dispatch
from numpy._core.records import recarray
from numpy.lib._iotools import _is_string_like
_check_fill_value = np.ma.core._check_fill_value
__all__ = ['append_fields', 'apply_along_fields', 'assign_fields_by_name', 'drop_fields', 'find_duplicates', 'flatten_descr', 'get_fieldstructure', 'get_names', 'get_names_flat', 'join_by', 'merge_arrays', 'rec_append_fields', 'rec_drop_fields', 'rec_join', 'recursive_fill_fields', 'rename_fields', 'repack_fields', 'require_fields', 'stack_arrays', 'structured_to_unstructured', 'unstructured_to_structured']

def _recursive_fill_fields_dispatcher(input, output):
    if False:
        i = 10
        return i + 15
    return (input, output)

@array_function_dispatch(_recursive_fill_fields_dispatcher)
def recursive_fill_fields(input, output):
    if False:
        i = 10
        return i + 15
    "\n    Fills fields from output with fields from input,\n    with support for nested structures.\n\n    Parameters\n    ----------\n    input : ndarray\n        Input array.\n    output : ndarray\n        Output array.\n\n    Notes\n    -----\n    * `output` should be at least the same size as `input`\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> a = np.array([(1, 10.), (2, 20.)], dtype=[('A', np.int64), ('B', np.float64)])\n    >>> b = np.zeros((3,), dtype=a.dtype)\n    >>> rfn.recursive_fill_fields(a, b)\n    array([(1, 10.), (2, 20.), (0,  0.)], dtype=[('A', '<i8'), ('B', '<f8')])\n\n    "
    newdtype = output.dtype
    for field in newdtype.names:
        try:
            current = input[field]
        except ValueError:
            continue
        if current.dtype.names is not None:
            recursive_fill_fields(current, output[field])
        else:
            output[field][:len(current)] = current
    return output

def _get_fieldspec(dtype):
    if False:
        return 10
    "\n    Produce a list of name/dtype pairs corresponding to the dtype fields\n\n    Similar to dtype.descr, but the second item of each tuple is a dtype, not a\n    string. As a result, this handles subarray dtypes\n\n    Can be passed to the dtype constructor to reconstruct the dtype, noting that\n    this (deliberately) discards field offsets.\n\n    Examples\n    --------\n    >>> dt = np.dtype([(('a', 'A'), np.int64), ('b', np.double, 3)])\n    >>> dt.descr\n    [(('a', 'A'), '<i8'), ('b', '<f8', (3,))]\n    >>> _get_fieldspec(dt)\n    [(('a', 'A'), dtype('int64')), ('b', dtype(('<f8', (3,))))]\n\n    "
    if dtype.names is None:
        return [('', dtype)]
    else:
        fields = ((name, dtype.fields[name]) for name in dtype.names)
        return [(name if len(f) == 2 else (f[2], name), f[0]) for (name, f) in fields]

def get_names(adtype):
    if False:
        while True:
            i = 10
    "\n    Returns the field names of the input datatype as a tuple. Input datatype\n    must have fields otherwise error is raised.\n\n    Parameters\n    ----------\n    adtype : dtype\n        Input datatype\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> rfn.get_names(np.empty((1,), dtype=[('A', int)]).dtype)\n    ('A',)\n    >>> rfn.get_names(np.empty((1,), dtype=[('A',int), ('B', float)]).dtype)\n    ('A', 'B')\n    >>> adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])\n    >>> rfn.get_names(adtype)\n    ('a', ('b', ('ba', 'bb')))\n    "
    listnames = []
    names = adtype.names
    for name in names:
        current = adtype[name]
        if current.names is not None:
            listnames.append((name, tuple(get_names(current))))
        else:
            listnames.append(name)
    return tuple(listnames)

def get_names_flat(adtype):
    if False:
        return 10
    "\n    Returns the field names of the input datatype as a tuple. Input datatype\n    must have fields otherwise error is raised.\n    Nested structure are flattened beforehand.\n\n    Parameters\n    ----------\n    adtype : dtype\n        Input datatype\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> rfn.get_names_flat(np.empty((1,), dtype=[('A', int)]).dtype) is None\n    False\n    >>> rfn.get_names_flat(np.empty((1,), dtype=[('A',int), ('B', str)]).dtype)\n    ('A', 'B')\n    >>> adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])\n    >>> rfn.get_names_flat(adtype)\n    ('a', 'b', 'ba', 'bb')\n    "
    listnames = []
    names = adtype.names
    for name in names:
        listnames.append(name)
        current = adtype[name]
        if current.names is not None:
            listnames.extend(get_names_flat(current))
    return tuple(listnames)

def flatten_descr(ndtype):
    if False:
        i = 10
        return i + 15
    "\n    Flatten a structured data-type description.\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> ndtype = np.dtype([('a', '<i4'), ('b', [('ba', '<f8'), ('bb', '<i4')])])\n    >>> rfn.flatten_descr(ndtype)\n    (('a', dtype('int32')), ('ba', dtype('float64')), ('bb', dtype('int32')))\n\n    "
    names = ndtype.names
    if names is None:
        return (('', ndtype),)
    else:
        descr = []
        for field in names:
            (typ, _) = ndtype.fields[field]
            if typ.names is not None:
                descr.extend(flatten_descr(typ))
            else:
                descr.append((field, typ))
        return tuple(descr)

def _zip_dtype(seqarrays, flatten=False):
    if False:
        return 10
    newdtype = []
    if flatten:
        for a in seqarrays:
            newdtype.extend(flatten_descr(a.dtype))
    else:
        for a in seqarrays:
            current = a.dtype
            if current.names is not None and len(current.names) == 1:
                newdtype.extend(_get_fieldspec(current))
            else:
                newdtype.append(('', current))
    return np.dtype(newdtype)

def _zip_descr(seqarrays, flatten=False):
    if False:
        return 10
    '\n    Combine the dtype description of a series of arrays.\n\n    Parameters\n    ----------\n    seqarrays : sequence of arrays\n        Sequence of arrays\n    flatten : {boolean}, optional\n        Whether to collapse nested descriptions.\n    '
    return _zip_dtype(seqarrays, flatten=flatten).descr

def get_fieldstructure(adtype, lastname=None, parents=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a dictionary with fields indexing lists of their parent fields.\n\n    This function is used to simplify access to fields nested in other fields.\n\n    Parameters\n    ----------\n    adtype : np.dtype\n        Input datatype\n    lastname : optional\n        Last processed field name (used internally during recursion).\n    parents : dictionary\n        Dictionary of parent fields (used interbally during recursion).\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> ndtype =  np.dtype([('A', int),\n    ...                     ('B', [('BA', int),\n    ...                            ('BB', [('BBA', int), ('BBB', int)])])])\n    >>> rfn.get_fieldstructure(ndtype)\n    ... # XXX: possible regression, order of BBA and BBB is swapped\n    {'A': [], 'B': [], 'BA': ['B'], 'BB': ['B'], 'BBA': ['B', 'BB'], 'BBB': ['B', 'BB']}\n\n    "
    if parents is None:
        parents = {}
    names = adtype.names
    for name in names:
        current = adtype[name]
        if current.names is not None:
            if lastname:
                parents[name] = [lastname]
            else:
                parents[name] = []
            parents.update(get_fieldstructure(current, name, parents))
        else:
            lastparent = [_ for _ in parents.get(lastname, []) or []]
            if lastparent:
                lastparent.append(lastname)
            elif lastname:
                lastparent = [lastname]
            parents[name] = lastparent or []
    return parents

def _izip_fields_flat(iterable):
    if False:
        i = 10
        return i + 15
    '\n    Returns an iterator of concatenated fields from a sequence of arrays,\n    collapsing any nested structure.\n\n    '
    for element in iterable:
        if isinstance(element, np.void):
            yield from _izip_fields_flat(tuple(element))
        else:
            yield element

def _izip_fields(iterable):
    if False:
        print('Hello World!')
    '\n    Returns an iterator of concatenated fields from a sequence of arrays.\n\n    '
    for element in iterable:
        if hasattr(element, '__iter__') and (not isinstance(element, str)):
            yield from _izip_fields(element)
        elif isinstance(element, np.void) and len(tuple(element)) == 1:
            yield from _izip_fields(element)
        else:
            yield element

def _izip_records(seqarrays, fill_value=None, flatten=True):
    if False:
        return 10
    '\n    Returns an iterator of concatenated items from a sequence of arrays.\n\n    Parameters\n    ----------\n    seqarrays : sequence of arrays\n        Sequence of arrays.\n    fill_value : {None, integer}\n        Value used to pad shorter iterables.\n    flatten : {True, False},\n        Whether to\n    '
    if flatten:
        zipfunc = _izip_fields_flat
    else:
        zipfunc = _izip_fields
    for tup in itertools.zip_longest(*seqarrays, fillvalue=fill_value):
        yield tuple(zipfunc(tup))

def _fix_output(output, usemask=True, asrecarray=False):
    if False:
        print('Hello World!')
    '\n    Private function: return a recarray, a ndarray, a MaskedArray\n    or a MaskedRecords depending on the input parameters\n    '
    if not isinstance(output, MaskedArray):
        usemask = False
    if usemask:
        if asrecarray:
            output = output.view(MaskedRecords)
    else:
        output = ma.filled(output)
        if asrecarray:
            output = output.view(recarray)
    return output

def _fix_defaults(output, defaults=None):
    if False:
        i = 10
        return i + 15
    '\n    Update the fill_value and masked data of `output`\n    from the default given in a dictionary defaults.\n    '
    names = output.dtype.names
    (data, mask, fill_value) = (output.data, output.mask, output.fill_value)
    for (k, v) in (defaults or {}).items():
        if k in names:
            fill_value[k] = v
            data[k][mask[k]] = v
    return output

def _merge_arrays_dispatcher(seqarrays, fill_value=None, flatten=None, usemask=None, asrecarray=None):
    if False:
        for i in range(10):
            print('nop')
    return seqarrays

@array_function_dispatch(_merge_arrays_dispatcher)
def merge_arrays(seqarrays, fill_value=-1, flatten=False, usemask=False, asrecarray=False):
    if False:
        return 10
    "\n    Merge arrays field by field.\n\n    Parameters\n    ----------\n    seqarrays : sequence of ndarrays\n        Sequence of arrays\n    fill_value : {float}, optional\n        Filling value used to pad missing data on the shorter arrays.\n    flatten : {False, True}, optional\n        Whether to collapse nested fields.\n    usemask : {False, True}, optional\n        Whether to return a masked array or not.\n    asrecarray : {False, True}, optional\n        Whether to return a recarray (MaskedRecords) or not.\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> rfn.merge_arrays((np.array([1, 2]), np.array([10., 20., 30.])))\n    array([( 1, 10.), ( 2, 20.), (-1, 30.)],\n          dtype=[('f0', '<i8'), ('f1', '<f8')])\n\n    >>> rfn.merge_arrays((np.array([1, 2], dtype=np.int64),\n    ...         np.array([10., 20., 30.])), usemask=False)\n     array([(1, 10.0), (2, 20.0), (-1, 30.0)],\n             dtype=[('f0', '<i8'), ('f1', '<f8')])\n    >>> rfn.merge_arrays((np.array([1, 2]).view([('a', np.int64)]),\n    ...               np.array([10., 20., 30.])),\n    ...              usemask=False, asrecarray=True)\n    rec.array([( 1, 10.), ( 2, 20.), (-1, 30.)],\n              dtype=[('a', '<i8'), ('f1', '<f8')])\n\n    Notes\n    -----\n    * Without a mask, the missing value will be filled with something,\n      depending on what its corresponding type:\n\n      * ``-1``      for integers\n      * ``-1.0``    for floating point numbers\n      * ``'-'``     for characters\n      * ``'-1'``    for strings\n      * ``True``    for boolean values\n    * XXX: I just obtained these values empirically\n    "
    if len(seqarrays) == 1:
        seqarrays = np.asanyarray(seqarrays[0])
    if isinstance(seqarrays, (ndarray, np.void)):
        seqdtype = seqarrays.dtype
        if seqdtype.names is None:
            seqdtype = np.dtype([('', seqdtype)])
        if not flatten or _zip_dtype((seqarrays,), flatten=True) == seqdtype:
            seqarrays = seqarrays.ravel()
            if usemask:
                if asrecarray:
                    seqtype = MaskedRecords
                else:
                    seqtype = MaskedArray
            elif asrecarray:
                seqtype = recarray
            else:
                seqtype = ndarray
            return seqarrays.view(dtype=seqdtype, type=seqtype)
        else:
            seqarrays = (seqarrays,)
    else:
        seqarrays = [np.asanyarray(_m) for _m in seqarrays]
    sizes = tuple((a.size for a in seqarrays))
    maxlength = max(sizes)
    newdtype = _zip_dtype(seqarrays, flatten=flatten)
    seqdata = []
    seqmask = []
    if usemask:
        for (a, n) in zip(seqarrays, sizes):
            nbmissing = maxlength - n
            data = a.ravel().__array__()
            mask = ma.getmaskarray(a).ravel()
            if nbmissing:
                fval = _check_fill_value(fill_value, a.dtype)
                if isinstance(fval, (ndarray, np.void)):
                    if len(fval.dtype) == 1:
                        fval = fval.item()[0]
                        fmsk = True
                    else:
                        fval = np.array(fval, dtype=a.dtype, ndmin=1)
                        fmsk = np.ones((1,), dtype=mask.dtype)
            else:
                fval = None
                fmsk = True
            seqdata.append(itertools.chain(data, [fval] * nbmissing))
            seqmask.append(itertools.chain(mask, [fmsk] * nbmissing))
        data = tuple(_izip_records(seqdata, flatten=flatten))
        output = ma.array(np.fromiter(data, dtype=newdtype, count=maxlength), mask=list(_izip_records(seqmask, flatten=flatten)))
        if asrecarray:
            output = output.view(MaskedRecords)
    else:
        for (a, n) in zip(seqarrays, sizes):
            nbmissing = maxlength - n
            data = a.ravel().__array__()
            if nbmissing:
                fval = _check_fill_value(fill_value, a.dtype)
                if isinstance(fval, (ndarray, np.void)):
                    if len(fval.dtype) == 1:
                        fval = fval.item()[0]
                    else:
                        fval = np.array(fval, dtype=a.dtype, ndmin=1)
            else:
                fval = None
            seqdata.append(itertools.chain(data, [fval] * nbmissing))
        output = np.fromiter(tuple(_izip_records(seqdata, flatten=flatten)), dtype=newdtype, count=maxlength)
        if asrecarray:
            output = output.view(recarray)
    return output

def _drop_fields_dispatcher(base, drop_names, usemask=None, asrecarray=None):
    if False:
        print('Hello World!')
    return (base,)

@array_function_dispatch(_drop_fields_dispatcher)
def drop_fields(base, drop_names, usemask=True, asrecarray=False):
    if False:
        print('Hello World!')
    "\n    Return a new array with fields in `drop_names` dropped.\n\n    Nested fields are supported.\n\n    .. versionchanged:: 1.18.0\n        `drop_fields` returns an array with 0 fields if all fields are dropped,\n        rather than returning ``None`` as it did previously.\n\n    Parameters\n    ----------\n    base : array\n        Input array\n    drop_names : string or sequence\n        String or sequence of strings corresponding to the names of the\n        fields to drop.\n    usemask : {False, True}, optional\n        Whether to return a masked array or not.\n    asrecarray : string or sequence, optional\n        Whether to return a recarray or a mrecarray (`asrecarray=True`) or\n        a plain ndarray or masked array with flexible dtype. The default\n        is False.\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> a = np.array([(1, (2, 3.0)), (4, (5, 6.0))],\n    ...   dtype=[('a', np.int64), ('b', [('ba', np.double), ('bb', np.int64)])])\n    >>> rfn.drop_fields(a, 'a')\n    array([((2., 3),), ((5., 6),)],\n          dtype=[('b', [('ba', '<f8'), ('bb', '<i8')])])\n    >>> rfn.drop_fields(a, 'ba')\n    array([(1, (3,)), (4, (6,))], dtype=[('a', '<i8'), ('b', [('bb', '<i8')])])\n    >>> rfn.drop_fields(a, ['ba', 'bb'])\n    array([(1,), (4,)], dtype=[('a', '<i8')])\n    "
    if _is_string_like(drop_names):
        drop_names = [drop_names]
    else:
        drop_names = set(drop_names)

    def _drop_descr(ndtype, drop_names):
        if False:
            i = 10
            return i + 15
        names = ndtype.names
        newdtype = []
        for name in names:
            current = ndtype[name]
            if name in drop_names:
                continue
            if current.names is not None:
                descr = _drop_descr(current, drop_names)
                if descr:
                    newdtype.append((name, descr))
            else:
                newdtype.append((name, current))
        return newdtype
    newdtype = _drop_descr(base.dtype, drop_names)
    output = np.empty(base.shape, dtype=newdtype)
    output = recursive_fill_fields(base, output)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)

def _keep_fields(base, keep_names, usemask=True, asrecarray=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a new array keeping only the fields in `keep_names`,\n    and preserving the order of those fields.\n\n    Parameters\n    ----------\n    base : array\n        Input array\n    keep_names : string or sequence\n        String or sequence of strings corresponding to the names of the\n        fields to keep. Order of the names will be preserved.\n    usemask : {False, True}, optional\n        Whether to return a masked array or not.\n    asrecarray : string or sequence, optional\n        Whether to return a recarray or a mrecarray (`asrecarray=True`) or\n        a plain ndarray or masked array with flexible dtype. The default\n        is False.\n    '
    newdtype = [(n, base.dtype[n]) for n in keep_names]
    output = np.empty(base.shape, dtype=newdtype)
    output = recursive_fill_fields(base, output)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)

def _rec_drop_fields_dispatcher(base, drop_names):
    if False:
        i = 10
        return i + 15
    return (base,)

@array_function_dispatch(_rec_drop_fields_dispatcher)
def rec_drop_fields(base, drop_names):
    if False:
        i = 10
        return i + 15
    '\n    Returns a new numpy.recarray with fields in `drop_names` dropped.\n    '
    return drop_fields(base, drop_names, usemask=False, asrecarray=True)

def _rename_fields_dispatcher(base, namemapper):
    if False:
        for i in range(10):
            print('nop')
    return (base,)

@array_function_dispatch(_rename_fields_dispatcher)
def rename_fields(base, namemapper):
    if False:
        for i in range(10):
            print('nop')
    "\n    Rename the fields from a flexible-datatype ndarray or recarray.\n\n    Nested fields are supported.\n\n    Parameters\n    ----------\n    base : ndarray\n        Input array whose fields must be modified.\n    namemapper : dictionary\n        Dictionary mapping old field names to their new version.\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> a = np.array([(1, (2, [3.0, 30.])), (4, (5, [6.0, 60.]))],\n    ...   dtype=[('a', int),('b', [('ba', float), ('bb', (float, 2))])])\n    >>> rfn.rename_fields(a, {'a':'A', 'bb':'BB'})\n    array([(1, (2., [ 3., 30.])), (4, (5., [ 6., 60.]))],\n          dtype=[('A', '<i8'), ('b', [('ba', '<f8'), ('BB', '<f8', (2,))])])\n\n    "

    def _recursive_rename_fields(ndtype, namemapper):
        if False:
            for i in range(10):
                print('nop')
        newdtype = []
        for name in ndtype.names:
            newname = namemapper.get(name, name)
            current = ndtype[name]
            if current.names is not None:
                newdtype.append((newname, _recursive_rename_fields(current, namemapper)))
            else:
                newdtype.append((newname, current))
        return newdtype
    newdtype = _recursive_rename_fields(base.dtype, namemapper)
    return base.view(newdtype)

def _append_fields_dispatcher(base, names, data, dtypes=None, fill_value=None, usemask=None, asrecarray=None):
    if False:
        for i in range(10):
            print('nop')
    yield base
    yield from data

@array_function_dispatch(_append_fields_dispatcher)
def append_fields(base, names, data, dtypes=None, fill_value=-1, usemask=True, asrecarray=False):
    if False:
        return 10
    '\n    Add new fields to an existing array.\n\n    The names of the fields are given with the `names` arguments,\n    the corresponding values with the `data` arguments.\n    If a single field is appended, `names`, `data` and `dtypes` do not have\n    to be lists but just values.\n\n    Parameters\n    ----------\n    base : array\n        Input array to extend.\n    names : string, sequence\n        String or sequence of strings corresponding to the names\n        of the new fields.\n    data : array or sequence of arrays\n        Array or sequence of arrays storing the fields to add to the base.\n    dtypes : sequence of datatypes, optional\n        Datatype or sequence of datatypes.\n        If None, the datatypes are estimated from the `data`.\n    fill_value : {float}, optional\n        Filling value used to pad missing data on the shorter arrays.\n    usemask : {False, True}, optional\n        Whether to return a masked array or not.\n    asrecarray : {False, True}, optional\n        Whether to return a recarray (MaskedRecords) or not.\n\n    '
    if isinstance(names, (tuple, list)):
        if len(names) != len(data):
            msg = 'The number of arrays does not match the number of names'
            raise ValueError(msg)
    elif isinstance(names, str):
        names = [names]
        data = [data]
    if dtypes is None:
        data = [np.array(a, copy=False, subok=True) for a in data]
        data = [a.view([(name, a.dtype)]) for (name, a) in zip(names, data)]
    else:
        if not isinstance(dtypes, (tuple, list)):
            dtypes = [dtypes]
        if len(data) != len(dtypes):
            if len(dtypes) == 1:
                dtypes = dtypes * len(data)
            else:
                msg = 'The dtypes argument must be None, a dtype, or a list.'
                raise ValueError(msg)
        data = [np.array(a, copy=False, subok=True, dtype=d).view([(n, d)]) for (a, n, d) in zip(data, names, dtypes)]
    base = merge_arrays(base, usemask=usemask, fill_value=fill_value)
    if len(data) > 1:
        data = merge_arrays(data, flatten=True, usemask=usemask, fill_value=fill_value)
    else:
        data = data.pop()
    output = ma.masked_all(max(len(base), len(data)), dtype=_get_fieldspec(base.dtype) + _get_fieldspec(data.dtype))
    output = recursive_fill_fields(base, output)
    output = recursive_fill_fields(data, output)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)

def _rec_append_fields_dispatcher(base, names, data, dtypes=None):
    if False:
        print('Hello World!')
    yield base
    yield from data

@array_function_dispatch(_rec_append_fields_dispatcher)
def rec_append_fields(base, names, data, dtypes=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add new fields to an existing array.\n\n    The names of the fields are given with the `names` arguments,\n    the corresponding values with the `data` arguments.\n    If a single field is appended, `names`, `data` and `dtypes` do not have\n    to be lists but just values.\n\n    Parameters\n    ----------\n    base : array\n        Input array to extend.\n    names : string, sequence\n        String or sequence of strings corresponding to the names\n        of the new fields.\n    data : array or sequence of arrays\n        Array or sequence of arrays storing the fields to add to the base.\n    dtypes : sequence of datatypes, optional\n        Datatype or sequence of datatypes.\n        If None, the datatypes are estimated from the `data`.\n\n    See Also\n    --------\n    append_fields\n\n    Returns\n    -------\n    appended_array : np.recarray\n    '
    return append_fields(base, names, data=data, dtypes=dtypes, asrecarray=True, usemask=False)

def _repack_fields_dispatcher(a, align=None, recurse=None):
    if False:
        return 10
    return (a,)

@array_function_dispatch(_repack_fields_dispatcher)
def repack_fields(a, align=False, recurse=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Re-pack the fields of a structured array or dtype in memory.\n\n    The memory layout of structured datatypes allows fields at arbitrary\n    byte offsets. This means the fields can be separated by padding bytes,\n    their offsets can be non-monotonically increasing, and they can overlap.\n\n    This method removes any overlaps and reorders the fields in memory so they\n    have increasing byte offsets, and adds or removes padding bytes depending\n    on the `align` option, which behaves like the `align` option to\n    `numpy.dtype`.\n\n    If `align=False`, this method produces a "packed" memory layout in which\n    each field starts at the byte the previous field ended, and any padding\n    bytes are removed.\n\n    If `align=True`, this methods produces an "aligned" memory layout in which\n    each field\'s offset is a multiple of its alignment, and the total itemsize\n    is a multiple of the largest alignment, by adding padding bytes as needed.\n\n    Parameters\n    ----------\n    a : ndarray or dtype\n       array or dtype for which to repack the fields.\n    align : boolean\n       If true, use an "aligned" memory layout, otherwise use a "packed" layout.\n    recurse : boolean\n       If True, also repack nested structures.\n\n    Returns\n    -------\n    repacked : ndarray or dtype\n       Copy of `a` with fields repacked, or `a` itself if no repacking was\n       needed.\n\n    Examples\n    --------\n\n    >>> from numpy.lib import recfunctions as rfn\n    >>> def print_offsets(d):\n    ...     print("offsets:", [d.fields[name][1] for name in d.names])\n    ...     print("itemsize:", d.itemsize)\n    ...\n    >>> dt = np.dtype(\'u1, <i8, <f8\', align=True)\n    >>> dt\n    dtype({\'names\': [\'f0\', \'f1\', \'f2\'], \'formats\': [\'u1\', \'<i8\', \'<f8\'], \'offsets\': [0, 8, 16], \'itemsize\': 24}, align=True)\n    >>> print_offsets(dt)\n    offsets: [0, 8, 16]\n    itemsize: 24\n    >>> packed_dt = rfn.repack_fields(dt)\n    >>> packed_dt\n    dtype([(\'f0\', \'u1\'), (\'f1\', \'<i8\'), (\'f2\', \'<f8\')])\n    >>> print_offsets(packed_dt)\n    offsets: [0, 1, 9]\n    itemsize: 17\n\n    '
    if not isinstance(a, np.dtype):
        dt = repack_fields(a.dtype, align=align, recurse=recurse)
        return a.astype(dt, copy=False)
    if a.names is None:
        return a
    fieldinfo = []
    for name in a.names:
        tup = a.fields[name]
        if recurse:
            fmt = repack_fields(tup[0], align=align, recurse=True)
        else:
            fmt = tup[0]
        if len(tup) == 3:
            name = (tup[2], name)
        fieldinfo.append((name, fmt))
    dt = np.dtype(fieldinfo, align=align)
    return np.dtype((a.type, dt))

def _get_fields_and_offsets(dt, offset=0):
    if False:
        return 10
    '\n    Returns a flat list of (dtype, count, offset) tuples of all the\n    scalar fields in the dtype "dt", including nested fields, in left\n    to right order.\n    '

    def count_elem(dt):
        if False:
            i = 10
            return i + 15
        count = 1
        while dt.shape != ():
            for size in dt.shape:
                count *= size
            dt = dt.base
        return (dt, count)
    fields = []
    for name in dt.names:
        field = dt.fields[name]
        (f_dt, f_offset) = (field[0], field[1])
        (f_dt, n) = count_elem(f_dt)
        if f_dt.names is None:
            fields.append((np.dtype((f_dt, (n,))), n, f_offset + offset))
        else:
            subfields = _get_fields_and_offsets(f_dt, f_offset + offset)
            size = f_dt.itemsize
            for i in range(n):
                if i == 0:
                    fields.extend(subfields)
                else:
                    fields.extend([(d, c, o + i * size) for (d, c, o) in subfields])
    return fields

def _common_stride(offsets, counts, itemsize):
    if False:
        i = 10
        return i + 15
    '\n    Returns the stride between the fields, or None if the stride is not\n    constant. The values in "counts" designate the lengths of\n    subarrays. Subarrays are treated as many contiguous fields, with\n    always positive stride.\n    '
    if len(offsets) <= 1:
        return itemsize
    negative = offsets[1] < offsets[0]
    if negative:
        it = zip(reversed(offsets), reversed(counts))
    else:
        it = zip(offsets, counts)
    prev_offset = None
    stride = None
    for (offset, count) in it:
        if count != 1:
            if negative:
                return None
            if stride is None:
                stride = itemsize
            if stride != itemsize:
                return None
            end_offset = offset + (count - 1) * itemsize
        else:
            end_offset = offset
        if prev_offset is not None:
            new_stride = offset - prev_offset
            if stride is None:
                stride = new_stride
            if stride != new_stride:
                return None
        prev_offset = end_offset
    if negative:
        return -stride
    return stride

def _structured_to_unstructured_dispatcher(arr, dtype=None, copy=None, casting=None):
    if False:
        i = 10
        return i + 15
    return (arr,)

@array_function_dispatch(_structured_to_unstructured_dispatcher)
def structured_to_unstructured(arr, dtype=None, copy=False, casting='unsafe'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Converts an n-D structured array into an (n+1)-D unstructured array.\n\n    The new array will have a new last dimension equal in size to the\n    number of field-elements of the input array. If not supplied, the output\n    datatype is determined from the numpy type promotion rules applied to all\n    the field datatypes.\n\n    Nested fields, as well as each element of any subarray fields, all count\n    as a single field-elements.\n\n    Parameters\n    ----------\n    arr : ndarray\n       Structured array or dtype to convert. Cannot contain object datatype.\n    dtype : dtype, optional\n       The dtype of the output unstructured array.\n    copy : bool, optional\n        If true, always return a copy. If false, a view is returned if\n        possible, such as when the `dtype` and strides of the fields are\n        suitable and the array subtype is one of `numpy.ndarray`,\n        `numpy.recarray` or `numpy.memmap`.\n\n        .. versionchanged:: 1.25.0\n            A view can now be returned if the fields are separated by a\n            uniform stride.\n\n    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional\n        See casting argument of `numpy.ndarray.astype`. Controls what kind of\n        data casting may occur.\n\n    Returns\n    -------\n    unstructured : ndarray\n       Unstructured array with one more dimension.\n\n    Examples\n    --------\n\n    >>> from numpy.lib import recfunctions as rfn\n    >>> a = np.zeros(4, dtype=[('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])\n    >>> a\n    array([(0, (0., 0), [0., 0.]), (0, (0., 0), [0., 0.]),\n           (0, (0., 0), [0., 0.]), (0, (0., 0), [0., 0.])],\n          dtype=[('a', '<i4'), ('b', [('f0', '<f4'), ('f1', '<u2')]), ('c', '<f4', (2,))])\n    >>> rfn.structured_to_unstructured(a)\n    array([[0., 0., 0., 0., 0.],\n           [0., 0., 0., 0., 0.],\n           [0., 0., 0., 0., 0.],\n           [0., 0., 0., 0., 0.]])\n\n    >>> b = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],\n    ...              dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])\n    >>> np.mean(rfn.structured_to_unstructured(b[['x', 'z']]), axis=-1)\n    array([ 3. ,  5.5,  9. , 11. ])\n\n    "
    if arr.dtype.names is None:
        raise ValueError('arr must be a structured array')
    fields = _get_fields_and_offsets(arr.dtype)
    n_fields = len(fields)
    if n_fields == 0 and dtype is None:
        raise ValueError('arr has no fields. Unable to guess dtype')
    elif n_fields == 0:
        raise NotImplementedError('arr with no fields is not supported')
    (dts, counts, offsets) = zip(*fields)
    names = ['f{}'.format(n) for n in range(n_fields)]
    if dtype is None:
        out_dtype = np.result_type(*[dt.base for dt in dts])
    else:
        out_dtype = np.dtype(dtype)
    flattened_fields = np.dtype({'names': names, 'formats': dts, 'offsets': offsets, 'itemsize': arr.dtype.itemsize})
    arr = arr.view(flattened_fields)
    can_view = type(arr) in (np.ndarray, np.recarray, np.memmap)
    if not copy and can_view and all((dt.base == out_dtype for dt in dts)):
        common_stride = _common_stride(offsets, counts, out_dtype.itemsize)
        if common_stride is not None:
            wrap = arr.__array_wrap__
            new_shape = arr.shape + (sum(counts), out_dtype.itemsize)
            new_strides = arr.strides + (abs(common_stride), 1)
            arr = arr[..., np.newaxis].view(np.uint8)
            arr = arr[..., min(offsets):]
            arr = np.lib.stride_tricks.as_strided(arr, new_shape, new_strides, subok=True)
            arr = arr.view(out_dtype)[..., 0]
            if common_stride < 0:
                arr = arr[..., ::-1]
            if type(arr) is not type(wrap.__self__):
                arr = wrap(arr)
            return arr
    packed_fields = np.dtype({'names': names, 'formats': [(out_dtype, dt.shape) for dt in dts]})
    arr = arr.astype(packed_fields, copy=copy, casting=casting)
    return arr.view((out_dtype, (sum(counts),)))

def _unstructured_to_structured_dispatcher(arr, dtype=None, names=None, align=None, copy=None, casting=None):
    if False:
        i = 10
        return i + 15
    return (arr,)

@array_function_dispatch(_unstructured_to_structured_dispatcher)
def unstructured_to_structured(arr, dtype=None, names=None, align=False, copy=False, casting='unsafe'):
    if False:
        print('Hello World!')
    "\n    Converts an n-D unstructured array into an (n-1)-D structured array.\n\n    The last dimension of the input array is converted into a structure, with\n    number of field-elements equal to the size of the last dimension of the\n    input array. By default all output fields have the input array's dtype, but\n    an output structured dtype with an equal number of fields-elements can be\n    supplied instead.\n\n    Nested fields, as well as each element of any subarray fields, all count\n    towards the number of field-elements.\n\n    Parameters\n    ----------\n    arr : ndarray\n       Unstructured array or dtype to convert.\n    dtype : dtype, optional\n       The structured dtype of the output array\n    names : list of strings, optional\n       If dtype is not supplied, this specifies the field names for the output\n       dtype, in order. The field dtypes will be the same as the input array.\n    align : boolean, optional\n       Whether to create an aligned memory layout.\n    copy : bool, optional\n        See copy argument to `numpy.ndarray.astype`. If true, always return a\n        copy. If false, and `dtype` requirements are satisfied, a view is\n        returned.\n    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional\n        See casting argument of `numpy.ndarray.astype`. Controls what kind of\n        data casting may occur.\n\n    Returns\n    -------\n    structured : ndarray\n       Structured array with fewer dimensions.\n\n    Examples\n    --------\n\n    >>> from numpy.lib import recfunctions as rfn\n    >>> dt = np.dtype([('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])\n    >>> a = np.arange(20).reshape((4,5))\n    >>> a\n    array([[ 0,  1,  2,  3,  4],\n           [ 5,  6,  7,  8,  9],\n           [10, 11, 12, 13, 14],\n           [15, 16, 17, 18, 19]])\n    >>> rfn.unstructured_to_structured(a, dt)\n    array([( 0, ( 1.,  2), [ 3.,  4.]), ( 5, ( 6.,  7), [ 8.,  9.]),\n           (10, (11., 12), [13., 14.]), (15, (16., 17), [18., 19.])],\n          dtype=[('a', '<i4'), ('b', [('f0', '<f4'), ('f1', '<u2')]), ('c', '<f4', (2,))])\n\n    "
    if arr.shape == ():
        raise ValueError('arr must have at least one dimension')
    n_elem = arr.shape[-1]
    if n_elem == 0:
        raise NotImplementedError('last axis with size 0 is not supported')
    if dtype is None:
        if names is None:
            names = ['f{}'.format(n) for n in range(n_elem)]
        out_dtype = np.dtype([(n, arr.dtype) for n in names], align=align)
        fields = _get_fields_and_offsets(out_dtype)
        (dts, counts, offsets) = zip(*fields)
    else:
        if names is not None:
            raise ValueError("don't supply both dtype and names")
        dtype = np.dtype(dtype)
        fields = _get_fields_and_offsets(dtype)
        if len(fields) == 0:
            (dts, counts, offsets) = ([], [], [])
        else:
            (dts, counts, offsets) = zip(*fields)
        if n_elem != sum(counts):
            raise ValueError('The length of the last dimension of arr must be equal to the number of fields in dtype')
        out_dtype = dtype
        if align and (not out_dtype.isalignedstruct):
            raise ValueError('align was True but dtype is not aligned')
    names = ['f{}'.format(n) for n in range(len(fields))]
    packed_fields = np.dtype({'names': names, 'formats': [(arr.dtype, dt.shape) for dt in dts]})
    arr = np.ascontiguousarray(arr).view(packed_fields)
    flattened_fields = np.dtype({'names': names, 'formats': dts, 'offsets': offsets, 'itemsize': out_dtype.itemsize})
    arr = arr.astype(flattened_fields, copy=copy, casting=casting)
    return arr.view(out_dtype)[..., 0]

def _apply_along_fields_dispatcher(func, arr):
    if False:
        while True:
            i = 10
    return (arr,)

@array_function_dispatch(_apply_along_fields_dispatcher)
def apply_along_fields(func, arr):
    if False:
        print('Hello World!')
    '\n    Apply function \'func\' as a reduction across fields of a structured array.\n\n    This is similar to `numpy.apply_along_axis`, but treats the fields of a\n    structured array as an extra axis. The fields are all first cast to a\n    common type following the type-promotion rules from `numpy.result_type`\n    applied to the field\'s dtypes.\n\n    Parameters\n    ----------\n    func : function\n       Function to apply on the "field" dimension. This function must\n       support an `axis` argument, like `numpy.mean`, `numpy.sum`, etc.\n    arr : ndarray\n       Structured array for which to apply func.\n\n    Returns\n    -------\n    out : ndarray\n       Result of the recution operation\n\n    Examples\n    --------\n\n    >>> from numpy.lib import recfunctions as rfn\n    >>> b = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],\n    ...              dtype=[(\'x\', \'i4\'), (\'y\', \'f4\'), (\'z\', \'f8\')])\n    >>> rfn.apply_along_fields(np.mean, b)\n    array([ 2.66666667,  5.33333333,  8.66666667, 11.        ])\n    >>> rfn.apply_along_fields(np.mean, b[[\'x\', \'z\']])\n    array([ 3. ,  5.5,  9. , 11. ])\n\n    '
    if arr.dtype.names is None:
        raise ValueError('arr must be a structured array')
    uarr = structured_to_unstructured(arr)
    return func(uarr, axis=-1)

def _assign_fields_by_name_dispatcher(dst, src, zero_unassigned=None):
    if False:
        while True:
            i = 10
    return (dst, src)

@array_function_dispatch(_assign_fields_by_name_dispatcher)
def assign_fields_by_name(dst, src, zero_unassigned=True):
    if False:
        while True:
            i = 10
    '\n    Assigns values from one structured array to another by field name.\n\n    Normally in numpy >= 1.14, assignment of one structured array to another\n    copies fields "by position", meaning that the first field from the src is\n    copied to the first field of the dst, and so on, regardless of field name.\n\n    This function instead copies "by field name", such that fields in the dst\n    are assigned from the identically named field in the src. This applies\n    recursively for nested structures. This is how structure assignment worked\n    in numpy >= 1.6 to <= 1.13.\n\n    Parameters\n    ----------\n    dst : ndarray\n    src : ndarray\n        The source and destination arrays during assignment.\n    zero_unassigned : bool, optional\n        If True, fields in the dst for which there was no matching\n        field in the src are filled with the value 0 (zero). This\n        was the behavior of numpy <= 1.13. If False, those fields\n        are not modified.\n    '
    if dst.dtype.names is None:
        dst[...] = src
        return
    for name in dst.dtype.names:
        if name not in src.dtype.names:
            if zero_unassigned:
                dst[name] = 0
        else:
            assign_fields_by_name(dst[name], src[name], zero_unassigned)

def _require_fields_dispatcher(array, required_dtype):
    if False:
        i = 10
        return i + 15
    return (array,)

@array_function_dispatch(_require_fields_dispatcher)
def require_fields(array, required_dtype):
    if False:
        for i in range(10):
            print('nop')
    '\n    Casts a structured array to a new dtype using assignment by field-name.\n\n    This function assigns from the old to the new array by name, so the\n    value of a field in the output array is the value of the field with the\n    same name in the source array. This has the effect of creating a new\n    ndarray containing only the fields "required" by the required_dtype.\n\n    If a field name in the required_dtype does not exist in the\n    input array, that field is created and set to 0 in the output array.\n\n    Parameters\n    ----------\n    a : ndarray\n       array to cast\n    required_dtype : dtype\n       datatype for output array\n\n    Returns\n    -------\n    out : ndarray\n        array with the new dtype, with field values copied from the fields in\n        the input array with the same name\n\n    Examples\n    --------\n\n    >>> from numpy.lib import recfunctions as rfn\n    >>> a = np.ones(4, dtype=[(\'a\', \'i4\'), (\'b\', \'f8\'), (\'c\', \'u1\')])\n    >>> rfn.require_fields(a, [(\'b\', \'f4\'), (\'c\', \'u1\')])\n    array([(1., 1), (1., 1), (1., 1), (1., 1)],\n      dtype=[(\'b\', \'<f4\'), (\'c\', \'u1\')])\n    >>> rfn.require_fields(a, [(\'b\', \'f4\'), (\'newf\', \'u1\')])\n    array([(1., 0), (1., 0), (1., 0), (1., 0)],\n      dtype=[(\'b\', \'<f4\'), (\'newf\', \'u1\')])\n\n    '
    out = np.empty(array.shape, dtype=required_dtype)
    assign_fields_by_name(out, array)
    return out

def _stack_arrays_dispatcher(arrays, defaults=None, usemask=None, asrecarray=None, autoconvert=None):
    if False:
        i = 10
        return i + 15
    return arrays

@array_function_dispatch(_stack_arrays_dispatcher)
def stack_arrays(arrays, defaults=None, usemask=True, asrecarray=False, autoconvert=False):
    if False:
        while True:
            i = 10
    "\n    Superposes arrays fields by fields\n\n    Parameters\n    ----------\n    arrays : array or sequence\n        Sequence of input arrays.\n    defaults : dictionary, optional\n        Dictionary mapping field names to the corresponding default values.\n    usemask : {True, False}, optional\n        Whether to return a MaskedArray (or MaskedRecords is\n        `asrecarray==True`) or a ndarray.\n    asrecarray : {False, True}, optional\n        Whether to return a recarray (or MaskedRecords if `usemask==True`)\n        or just a flexible-type ndarray.\n    autoconvert : {False, True}, optional\n        Whether automatically cast the type of the field to the maximum.\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> x = np.array([1, 2,])\n    >>> rfn.stack_arrays(x) is x\n    True\n    >>> z = np.array([('A', 1), ('B', 2)], dtype=[('A', '|S3'), ('B', float)])\n    >>> zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],\n    ...   dtype=[('A', '|S3'), ('B', np.double), ('C', np.double)])\n    >>> test = rfn.stack_arrays((z,zz))\n    >>> test\n    masked_array(data=[(b'A', 1.0, --), (b'B', 2.0, --), (b'a', 10.0, 100.0),\n                       (b'b', 20.0, 200.0), (b'c', 30.0, 300.0)],\n                 mask=[(False, False,  True), (False, False,  True),\n                       (False, False, False), (False, False, False),\n                       (False, False, False)],\n           fill_value=(b'N/A', 1e+20, 1e+20),\n                dtype=[('A', 'S3'), ('B', '<f8'), ('C', '<f8')])\n\n    "
    if isinstance(arrays, ndarray):
        return arrays
    elif len(arrays) == 1:
        return arrays[0]
    seqarrays = [np.asanyarray(a).ravel() for a in arrays]
    nrecords = [len(a) for a in seqarrays]
    ndtype = [a.dtype for a in seqarrays]
    fldnames = [d.names for d in ndtype]
    dtype_l = ndtype[0]
    newdescr = _get_fieldspec(dtype_l)
    names = [n for (n, d) in newdescr]
    for dtype_n in ndtype[1:]:
        for (fname, fdtype) in _get_fieldspec(dtype_n):
            if fname not in names:
                newdescr.append((fname, fdtype))
                names.append(fname)
            else:
                nameidx = names.index(fname)
                (_, cdtype) = newdescr[nameidx]
                if autoconvert:
                    newdescr[nameidx] = (fname, max(fdtype, cdtype))
                elif fdtype != cdtype:
                    raise TypeError("Incompatible type '%s' <> '%s'" % (cdtype, fdtype))
    if len(newdescr) == 1:
        output = ma.concatenate(seqarrays)
    else:
        output = ma.masked_all((np.sum(nrecords),), newdescr)
        offset = np.cumsum(np.r_[0, nrecords])
        seen = []
        for (a, n, i, j) in zip(seqarrays, fldnames, offset[:-1], offset[1:]):
            names = a.dtype.names
            if names is None:
                output['f%i' % len(seen)][i:j] = a
            else:
                for name in n:
                    output[name][i:j] = a[name]
                    if name not in seen:
                        seen.append(name)
    return _fix_output(_fix_defaults(output, defaults), usemask=usemask, asrecarray=asrecarray)

def _find_duplicates_dispatcher(a, key=None, ignoremask=None, return_index=None):
    if False:
        return 10
    return (a,)

@array_function_dispatch(_find_duplicates_dispatcher)
def find_duplicates(a, key=None, ignoremask=True, return_index=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Find the duplicates in a structured array along a given key\n\n    Parameters\n    ----------\n    a : array-like\n        Input array\n    key : {string, None}, optional\n        Name of the fields along which to check the duplicates.\n        If None, the search is performed by records\n    ignoremask : {True, False}, optional\n        Whether masked data should be discarded or considered as duplicates.\n    return_index : {False, True}, optional\n        Whether to return the indices of the duplicated values.\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> ndtype = [('a', int)]\n    >>> a = np.ma.array([1, 1, 1, 2, 2, 3, 3],\n    ...         mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)\n    >>> rfn.find_duplicates(a, ignoremask=True, return_index=True)\n    (masked_array(data=[(1,), (1,), (2,), (2,)],\n                 mask=[(False,), (False,), (False,), (False,)],\n           fill_value=(999999,),\n                dtype=[('a', '<i8')]), array([0, 1, 3, 4]))\n    "
    a = np.asanyarray(a).ravel()
    fields = get_fieldstructure(a.dtype)
    base = a
    if key:
        for f in fields[key]:
            base = base[f]
        base = base[key]
    sortidx = base.argsort()
    sortedbase = base[sortidx]
    sorteddata = sortedbase.filled()
    flag = sorteddata[:-1] == sorteddata[1:]
    if ignoremask:
        sortedmask = sortedbase.recordmask
        flag[sortedmask[1:]] = False
    flag = np.concatenate(([False], flag))
    flag[:-1] = flag[:-1] + flag[1:]
    duplicates = a[sortidx][flag]
    if return_index:
        return (duplicates, sortidx[flag])
    else:
        return duplicates

def _join_by_dispatcher(key, r1, r2, jointype=None, r1postfix=None, r2postfix=None, defaults=None, usemask=None, asrecarray=None):
    if False:
        for i in range(10):
            print('nop')
    return (r1, r2)

@array_function_dispatch(_join_by_dispatcher)
def join_by(key, r1, r2, jointype='inner', r1postfix='1', r2postfix='2', defaults=None, usemask=True, asrecarray=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Join arrays `r1` and `r2` on key `key`.\n\n    The key should be either a string or a sequence of string corresponding\n    to the fields used to join the array.  An exception is raised if the\n    `key` field cannot be found in the two input arrays.  Neither `r1` nor\n    `r2` should have any duplicates along `key`: the presence of duplicates\n    will make the output quite unreliable. Note that duplicates are not\n    looked for by the algorithm.\n\n    Parameters\n    ----------\n    key : {string, sequence}\n        A string or a sequence of strings corresponding to the fields used\n        for comparison.\n    r1, r2 : arrays\n        Structured arrays.\n    jointype : {'inner', 'outer', 'leftouter'}, optional\n        If 'inner', returns the elements common to both r1 and r2.\n        If 'outer', returns the common elements as well as the elements of\n        r1 not in r2 and the elements of not in r2.\n        If 'leftouter', returns the common elements and the elements of r1\n        not in r2.\n    r1postfix : string, optional\n        String appended to the names of the fields of r1 that are present\n        in r2 but absent of the key.\n    r2postfix : string, optional\n        String appended to the names of the fields of r2 that are present\n        in r1 but absent of the key.\n    defaults : {dictionary}, optional\n        Dictionary mapping field names to the corresponding default values.\n    usemask : {True, False}, optional\n        Whether to return a MaskedArray (or MaskedRecords is\n        `asrecarray==True`) or a ndarray.\n    asrecarray : {False, True}, optional\n        Whether to return a recarray (or MaskedRecords if `usemask==True`)\n        or just a flexible-type ndarray.\n\n    Notes\n    -----\n    * The output is sorted along the key.\n    * A temporary array is formed by dropping the fields not in the key for\n      the two arrays and concatenating the result. This array is then\n      sorted, and the common entries selected. The output is constructed by\n      filling the fields with the selected entries. Matching is not\n      preserved if there are some duplicates...\n\n    "
    if jointype not in ('inner', 'outer', 'leftouter'):
        raise ValueError("The 'jointype' argument should be in 'inner', 'outer' or 'leftouter' (got '%s' instead)" % jointype)
    if isinstance(key, str):
        key = (key,)
    if len(set(key)) != len(key):
        dup = next((x for (n, x) in enumerate(key) if x in key[n + 1:]))
        raise ValueError('duplicate join key %r' % dup)
    for name in key:
        if name not in r1.dtype.names:
            raise ValueError('r1 does not have key field %r' % name)
        if name not in r2.dtype.names:
            raise ValueError('r2 does not have key field %r' % name)
    r1 = r1.ravel()
    r2 = r2.ravel()
    nb1 = len(r1)
    (r1names, r2names) = (r1.dtype.names, r2.dtype.names)
    collisions = (set(r1names) & set(r2names)) - set(key)
    if collisions and (not (r1postfix or r2postfix)):
        msg = 'r1 and r2 contain common names, r1postfix and r2postfix '
        msg += "can't both be empty"
        raise ValueError(msg)
    key1 = [n for n in r1names if n in key]
    r1k = _keep_fields(r1, key1)
    r2k = _keep_fields(r2, key1)
    aux = ma.concatenate((r1k, r2k))
    idx_sort = aux.argsort(order=key)
    aux = aux[idx_sort]
    flag_in = ma.concatenate(([False], aux[1:] == aux[:-1]))
    flag_in[:-1] = flag_in[1:] + flag_in[:-1]
    idx_in = idx_sort[flag_in]
    idx_1 = idx_in[idx_in < nb1]
    idx_2 = idx_in[idx_in >= nb1] - nb1
    (r1cmn, r2cmn) = (len(idx_1), len(idx_2))
    if jointype == 'inner':
        (r1spc, r2spc) = (0, 0)
    elif jointype == 'outer':
        idx_out = idx_sort[~flag_in]
        idx_1 = np.concatenate((idx_1, idx_out[idx_out < nb1]))
        idx_2 = np.concatenate((idx_2, idx_out[idx_out >= nb1] - nb1))
        (r1spc, r2spc) = (len(idx_1) - r1cmn, len(idx_2) - r2cmn)
    elif jointype == 'leftouter':
        idx_out = idx_sort[~flag_in]
        idx_1 = np.concatenate((idx_1, idx_out[idx_out < nb1]))
        (r1spc, r2spc) = (len(idx_1) - r1cmn, 0)
    (s1, s2) = (r1[idx_1], r2[idx_2])
    ndtype = _get_fieldspec(r1k.dtype)
    for (fname, fdtype) in _get_fieldspec(r1.dtype):
        if fname not in key:
            ndtype.append((fname, fdtype))
    for (fname, fdtype) in _get_fieldspec(r2.dtype):
        names = list((name for (name, dtype) in ndtype))
        try:
            nameidx = names.index(fname)
        except ValueError:
            ndtype.append((fname, fdtype))
        else:
            (_, cdtype) = ndtype[nameidx]
            if fname in key:
                ndtype[nameidx] = (fname, max(fdtype, cdtype))
            else:
                ndtype[nameidx:nameidx + 1] = [(fname + r1postfix, cdtype), (fname + r2postfix, fdtype)]
    ndtype = np.dtype(ndtype)
    cmn = max(r1cmn, r2cmn)
    output = ma.masked_all((cmn + r1spc + r2spc,), dtype=ndtype)
    names = output.dtype.names
    for f in r1names:
        selected = s1[f]
        if f not in names or (f in r2names and (not r2postfix) and (f not in key)):
            f += r1postfix
        current = output[f]
        current[:r1cmn] = selected[:r1cmn]
        if jointype in ('outer', 'leftouter'):
            current[cmn:cmn + r1spc] = selected[r1cmn:]
    for f in r2names:
        selected = s2[f]
        if f not in names or (f in r1names and (not r1postfix) and (f not in key)):
            f += r2postfix
        current = output[f]
        current[:r2cmn] = selected[:r2cmn]
        if jointype == 'outer' and r2spc:
            current[-r2spc:] = selected[r2cmn:]
    output.sort(order=key)
    kwargs = dict(usemask=usemask, asrecarray=asrecarray)
    return _fix_output(_fix_defaults(output, defaults), **kwargs)

def _rec_join_dispatcher(key, r1, r2, jointype=None, r1postfix=None, r2postfix=None, defaults=None):
    if False:
        for i in range(10):
            print('nop')
    return (r1, r2)

@array_function_dispatch(_rec_join_dispatcher)
def rec_join(key, r1, r2, jointype='inner', r1postfix='1', r2postfix='2', defaults=None):
    if False:
        while True:
            i = 10
    '\n    Join arrays `r1` and `r2` on keys.\n    Alternative to join_by, that always returns a np.recarray.\n\n    See Also\n    --------\n    join_by : equivalent function\n    '
    kwargs = dict(jointype=jointype, r1postfix=r1postfix, r2postfix=r2postfix, defaults=defaults, usemask=False, asrecarray=True)
    return join_by(key, r1, r2, **kwargs)