import os
import sys
import textwrap
import types
import re
import warnings
import functools
import platform
from numpy._core import ndarray
from numpy._utils import set_module
import numpy as np
__all__ = ['get_include', 'info', 'show_runtime']

@set_module('numpy')
def show_runtime():
    if False:
        i = 10
        return i + 15
    '\n    Print information about various resources in the system\n    including available intrinsic support and BLAS/LAPACK library\n    in use\n\n    .. versionadded:: 1.24.0\n\n    See Also\n    --------\n    show_config : Show libraries in the system on which NumPy was built.\n\n    Notes\n    -----\n    1. Information is derived with the help of `threadpoolctl <https://pypi.org/project/threadpoolctl/>`_\n       library if available.\n    2. SIMD related information is derived from ``__cpu_features__``,\n       ``__cpu_baseline__`` and ``__cpu_dispatch__``\n\n    '
    from numpy._core._multiarray_umath import __cpu_features__, __cpu_baseline__, __cpu_dispatch__
    from pprint import pprint
    config_found = [{'numpy_version': np.__version__, 'python': sys.version, 'uname': platform.uname()}]
    (features_found, features_not_found) = ([], [])
    for feature in __cpu_dispatch__:
        if __cpu_features__[feature]:
            features_found.append(feature)
        else:
            features_not_found.append(feature)
    config_found.append({'simd_extensions': {'baseline': __cpu_baseline__, 'found': features_found, 'not_found': features_not_found}})
    try:
        from threadpoolctl import threadpool_info
        config_found.extend(threadpool_info())
    except ImportError:
        print('WARNING: `threadpoolctl` not found in system! Install it by `pip install threadpoolctl`. Once installed, try `np.show_runtime` again for more detailed build information')
    pprint(config_found)

@set_module('numpy')
def get_include():
    if False:
        while True:
            i = 10
    "\n    Return the directory that contains the NumPy \\*.h header files.\n\n    Extension modules that need to compile against NumPy should use this\n    function to locate the appropriate include directory.\n\n    Notes\n    -----\n    When using ``distutils``, for example in ``setup.py``::\n\n        import numpy as np\n        ...\n        Extension('extension_name', ...\n                include_dirs=[np.get_include()])\n        ...\n\n    "
    import numpy
    if numpy.show_config is None:
        d = os.path.join(os.path.dirname(numpy.__file__), '_core', 'include')
    else:
        import numpy._core as _core
        d = os.path.join(os.path.dirname(_core.__file__), 'include')
    return d

class _Deprecate:
    """
    Decorator class to deprecate old functions.

    Refer to `deprecate` for details.

    See Also
    --------
    deprecate

    """

    def __init__(self, old_name=None, new_name=None, message=None):
        if False:
            i = 10
            return i + 15
        self.old_name = old_name
        self.new_name = new_name
        self.message = message

    def __call__(self, func, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Decorator call.  Refer to ``decorate``.\n\n        '
        old_name = self.old_name
        new_name = self.new_name
        message = self.message
        if old_name is None:
            old_name = func.__name__
        if new_name is None:
            depdoc = '`%s` is deprecated!' % old_name
        else:
            depdoc = '`%s` is deprecated, use `%s` instead!' % (old_name, new_name)
        if message is not None:
            depdoc += '\n' + message

        @functools.wraps(func)
        def newfunc(*args, **kwds):
            if False:
                for i in range(10):
                    print('nop')
            warnings.warn(depdoc, DeprecationWarning, stacklevel=2)
            return func(*args, **kwds)
        newfunc.__name__ = old_name
        doc = func.__doc__
        if doc is None:
            doc = depdoc
        else:
            lines = doc.expandtabs().split('\n')
            indent = _get_indent(lines[1:])
            if lines[0].lstrip():
                doc = indent * ' ' + doc
            else:
                skip = len(lines[0]) + 1
                for line in lines[1:]:
                    if len(line) > indent:
                        break
                    skip += len(line) + 1
                doc = doc[skip:]
            depdoc = textwrap.indent(depdoc, ' ' * indent)
            doc = '\n\n'.join([depdoc, doc])
        newfunc.__doc__ = doc
        return newfunc

def _get_indent(lines):
    if False:
        i = 10
        return i + 15
    '\n    Determines the leading whitespace that could be removed from all the lines.\n    '
    indent = sys.maxsize
    for line in lines:
        content = len(line.lstrip())
        if content:
            indent = min(indent, len(line) - content)
    if indent == sys.maxsize:
        indent = 0
    return indent

def deprecate(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Issues a DeprecationWarning, adds warning to `old_name`'s\n    docstring, rebinds ``old_name.__name__`` and returns the new\n    function object.\n\n    This function may also be used as a decorator.\n\n    .. deprecated:: 2.0\n        Use `~warnings.warn` with :exc:`DeprecationWarning` instead.\n\n    Parameters\n    ----------\n    func : function\n        The function to be deprecated.\n    old_name : str, optional\n        The name of the function to be deprecated. Default is None, in\n        which case the name of `func` is used.\n    new_name : str, optional\n        The new name for the function. Default is None, in which case the\n        deprecation message is that `old_name` is deprecated. If given, the\n        deprecation message is that `old_name` is deprecated and `new_name`\n        should be used instead.\n    message : str, optional\n        Additional explanation of the deprecation.  Displayed in the\n        docstring after the warning.\n\n    Returns\n    -------\n    old_func : function\n        The deprecated function.\n\n    Examples\n    --------\n    Note that ``olduint`` returns a value after printing Deprecation\n    Warning:\n\n    >>> olduint = np.lib.utils.deprecate(np.uint)\n    DeprecationWarning: `uint64` is deprecated! # may vary\n    >>> olduint(6)\n    6\n\n    "
    warnings.warn('`deprecate` is deprecated, use `warn` with `DeprecationWarning` instead. (deprecated in NumPy 2.0)', DeprecationWarning, stacklevel=2)
    if args:
        fn = args[0]
        args = args[1:]
        return _Deprecate(*args, **kwargs)(fn)
    else:
        return _Deprecate(*args, **kwargs)

def deprecate_with_doc(msg):
    if False:
        return 10
    "\n    Deprecates a function and includes the deprecation in its docstring.\n\n    .. deprecated:: 2.0\n        Use `~warnings.warn` with :exc:`DeprecationWarning` instead.\n\n    This function is used as a decorator. It returns an object that can be\n    used to issue a DeprecationWarning, by passing the to-be decorated\n    function as argument, this adds warning to the to-be decorated function's\n    docstring and returns the new function object.\n\n    See Also\n    --------\n    deprecate : Decorate a function such that it issues a `DeprecationWarning`\n\n    Parameters\n    ----------\n    msg : str\n        Additional explanation of the deprecation. Displayed in the\n        docstring after the warning.\n\n    Returns\n    -------\n    obj : object\n\n    "
    warnings.warn('`deprecate` is deprecated, use `warn` with `DeprecationWarning` instead. (deprecated in NumPy 2.0)', DeprecationWarning, stacklevel=2)
    return _Deprecate(message=msg)

def _split_line(name, arguments, width):
    if False:
        for i in range(10):
            print('nop')
    firstwidth = len(name)
    k = firstwidth
    newstr = name
    sepstr = ', '
    arglist = arguments.split(sepstr)
    for argument in arglist:
        if k == firstwidth:
            addstr = ''
        else:
            addstr = sepstr
        k = k + len(argument) + len(addstr)
        if k > width:
            k = firstwidth + 1 + len(argument)
            newstr = newstr + ',\n' + ' ' * (firstwidth + 2) + argument
        else:
            newstr = newstr + addstr + argument
    return newstr
_namedict = None
_dictlist = None

def _makenamedict(module='numpy'):
    if False:
        i = 10
        return i + 15
    module = __import__(module, globals(), locals(), [])
    thedict = {module.__name__: module.__dict__}
    dictlist = [module.__name__]
    totraverse = [module.__dict__]
    while True:
        if len(totraverse) == 0:
            break
        thisdict = totraverse.pop(0)
        for x in thisdict.keys():
            if isinstance(thisdict[x], types.ModuleType):
                modname = thisdict[x].__name__
                if modname not in dictlist:
                    moddict = thisdict[x].__dict__
                    dictlist.append(modname)
                    totraverse.append(moddict)
                    thedict[modname] = moddict
    return (thedict, dictlist)

def _info(obj, output=None):
    if False:
        print('Hello World!')
    'Provide information about ndarray obj.\n\n    Parameters\n    ----------\n    obj : ndarray\n        Must be ndarray, not checked.\n    output\n        Where printed output goes.\n\n    Notes\n    -----\n    Copied over from the numarray module prior to its removal.\n    Adapted somewhat as only numpy is an option now.\n\n    Called by info.\n\n    '
    extra = ''
    tic = ''
    bp = lambda x: x
    cls = getattr(obj, '__class__', type(obj))
    nm = getattr(cls, '__name__', cls)
    strides = obj.strides
    endian = obj.dtype.byteorder
    if output is None:
        output = sys.stdout
    print('class: ', nm, file=output)
    print('shape: ', obj.shape, file=output)
    print('strides: ', strides, file=output)
    print('itemsize: ', obj.itemsize, file=output)
    print('aligned: ', bp(obj.flags.aligned), file=output)
    print('contiguous: ', bp(obj.flags.contiguous), file=output)
    print('fortran: ', obj.flags.fortran, file=output)
    print('data pointer: %s%s' % (hex(obj.ctypes._as_parameter_.value), extra), file=output)
    print('byteorder: ', end=' ', file=output)
    if endian in ['|', '=']:
        print('%s%s%s' % (tic, sys.byteorder, tic), file=output)
        byteswap = False
    elif endian == '>':
        print('%sbig%s' % (tic, tic), file=output)
        byteswap = sys.byteorder != 'big'
    else:
        print('%slittle%s' % (tic, tic), file=output)
        byteswap = sys.byteorder != 'little'
    print('byteswap: ', bp(byteswap), file=output)
    print('type: %s' % obj.dtype, file=output)

@set_module('numpy')
def info(object=None, maxwidth=76, output=None, toplevel='numpy'):
    if False:
        i = 10
        return i + 15
    "\n    Get help information for an array, function, class, or module.\n\n    Parameters\n    ----------\n    object : object or str, optional\n        Input object or name to get information about. If `object` is\n        an `ndarray` instance, information about the array is printed.\n        If `object` is a numpy object, its docstring is given. If it is\n        a string, available modules are searched for matching objects.\n        If None, information about `info` itself is returned.\n    maxwidth : int, optional\n        Printing width.\n    output : file like object, optional\n        File like object that the output is written to, default is\n        ``None``, in which case ``sys.stdout`` will be used.\n        The object has to be opened in 'w' or 'a' mode.\n    toplevel : str, optional\n        Start search at this level.\n\n    Notes\n    -----\n    When used interactively with an object, ``np.info(obj)`` is equivalent\n    to ``help(obj)`` on the Python prompt or ``obj?`` on the IPython\n    prompt.\n\n    Examples\n    --------\n    >>> np.info(np.polyval) # doctest: +SKIP\n       polyval(p, x)\n         Evaluate the polynomial p at x.\n         ...\n\n    When using a string for `object` it is possible to get multiple results.\n\n    >>> np.info('fft') # doctest: +SKIP\n         *** Found in numpy ***\n    Core FFT routines\n    ...\n         *** Found in numpy.fft ***\n     fft(a, n=None, axis=-1)\n    ...\n         *** Repeat reference found in numpy.fft.fftpack ***\n         *** Total of 3 references found. ***\n\n    When the argument is an array, information about the array is printed.\n\n    >>> a = np.array([[1 + 2j, 3, -4], [-5j, 6, 0]], dtype=np.complex64)\n    >>> np.info(a)\n    class:  ndarray\n    shape:  (2, 3)\n    strides:  (24, 8)\n    itemsize:  8\n    aligned:  True\n    contiguous:  True\n    fortran:  False\n    data pointer: 0x562b6e0d2860  # may vary\n    byteorder:  little\n    byteswap:  False\n    type: complex64\n\n    "
    global _namedict, _dictlist
    import pydoc
    import inspect
    if hasattr(object, '_ppimport_importer') or hasattr(object, '_ppimport_module'):
        object = object._ppimport_module
    elif hasattr(object, '_ppimport_attr'):
        object = object._ppimport_attr
    if output is None:
        output = sys.stdout
    if object is None:
        info(info)
    elif isinstance(object, ndarray):
        _info(object, output=output)
    elif isinstance(object, str):
        if _namedict is None:
            (_namedict, _dictlist) = _makenamedict(toplevel)
        numfound = 0
        objlist = []
        for namestr in _dictlist:
            try:
                obj = _namedict[namestr][object]
                if id(obj) in objlist:
                    print('\n     *** Repeat reference found in %s *** ' % namestr, file=output)
                else:
                    objlist.append(id(obj))
                    print('     *** Found in %s ***' % namestr, file=output)
                    info(obj)
                    print('-' * maxwidth, file=output)
                numfound += 1
            except KeyError:
                pass
        if numfound == 0:
            print('Help for %s not found.' % object, file=output)
        else:
            print('\n     *** Total of %d references found. ***' % numfound, file=output)
    elif inspect.isfunction(object) or inspect.ismethod(object):
        name = object.__name__
        try:
            arguments = str(inspect.signature(object))
        except Exception:
            arguments = '()'
        if len(name + arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments
        print(' ' + argstr + '\n', file=output)
        print(inspect.getdoc(object), file=output)
    elif inspect.isclass(object):
        name = object.__name__
        try:
            arguments = str(inspect.signature(object))
        except Exception:
            arguments = '()'
        if len(name + arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments
        print(' ' + argstr + '\n', file=output)
        doc1 = inspect.getdoc(object)
        if doc1 is None:
            if hasattr(object, '__init__'):
                print(inspect.getdoc(object.__init__), file=output)
        else:
            print(inspect.getdoc(object), file=output)
        methods = pydoc.allmethods(object)
        public_methods = [meth for meth in methods if meth[0] != '_']
        if public_methods:
            print('\n\nMethods:\n', file=output)
            for meth in public_methods:
                thisobj = getattr(object, meth, None)
                if thisobj is not None:
                    (methstr, other) = pydoc.splitdoc(inspect.getdoc(thisobj) or 'None')
                print('  %s  --  %s' % (meth, methstr), file=output)
    elif hasattr(object, '__doc__'):
        print(inspect.getdoc(object), file=output)

def safe_eval(source):
    if False:
        for i in range(10):
            print('nop')
    '\n    Protected string evaluation.\n\n    .. deprecated:: 2.0\n        Use `ast.literal_eval` instead.\n\n    Evaluate a string containing a Python literal expression without\n    allowing the execution of arbitrary non-literal code.\n\n    .. warning::\n\n        This function is identical to :py:meth:`ast.literal_eval` and\n        has the same security implications.  It may not always be safe\n        to evaluate large input strings.\n\n    Parameters\n    ----------\n    source : str\n        The string to evaluate.\n\n    Returns\n    -------\n    obj : object\n       The result of evaluating `source`.\n\n    Raises\n    ------\n    SyntaxError\n        If the code has invalid Python syntax, or if it contains\n        non-literal code.\n\n    Examples\n    --------\n    >>> np.safe_eval(\'1\')\n    1\n    >>> np.safe_eval(\'[1, 2, 3]\')\n    [1, 2, 3]\n    >>> np.safe_eval(\'{"foo": ("bar", 10.0)}\')\n    {\'foo\': (\'bar\', 10.0)}\n\n    >>> np.safe_eval(\'import os\')\n    Traceback (most recent call last):\n      ...\n    SyntaxError: invalid syntax\n\n    >>> np.safe_eval(\'open("/home/user/.ssh/id_dsa").read()\')\n    Traceback (most recent call last):\n      ...\n    ValueError: malformed node or string: <_ast.Call object at 0x...>\n\n    '
    warnings.warn('`safe_eval` is deprecated. Use `ast.literal_eval` instead. Be aware of security implications, such as memory exhaustion based attacks (deprecated in NumPy 2.0)', DeprecationWarning, stacklevel=2)
    import ast
    return ast.literal_eval(source)

def _median_nancheck(data, result, axis):
    if False:
        i = 10
        return i + 15
    '\n    Utility function to check median result from data for NaN values at the end\n    and return NaN in that case. Input result can also be a MaskedArray.\n\n    Parameters\n    ----------\n    data : array\n        Sorted input data to median function\n    result : Array or MaskedArray\n        Result of median function.\n    axis : int\n        Axis along which the median was computed.\n\n    Returns\n    -------\n    result : scalar or ndarray\n        Median or NaN in axes which contained NaN in the input.  If the input\n        was an array, NaN will be inserted in-place.  If a scalar, either the\n        input itself or a scalar NaN.\n    '
    if data.size == 0:
        return result
    potential_nans = data.take(-1, axis=axis)
    n = np.isnan(potential_nans)
    if np.ma.isMaskedArray(n):
        n = n.filled(False)
    if not n.any():
        return result
    if isinstance(result, np.generic):
        return potential_nans
    np.copyto(result, potential_nans, where=n)
    return result

def _opt_info():
    if False:
        return 10
    '\n    Returns a string containing the CPU features supported\n    by the current build.\n\n    The format of the string can be explained as follows:\n        - Dispatched features supported by the running machine end with `*`.\n        - Dispatched features not supported by the running machine\n          end with `?`.\n        - Remaining features represent the baseline.\n\n    Returns:\n        str: A formatted string indicating the supported CPU features.\n    '
    from numpy._core._multiarray_umath import __cpu_features__, __cpu_baseline__, __cpu_dispatch__
    if len(__cpu_baseline__) == 0 and len(__cpu_dispatch__) == 0:
        return ''
    enabled_features = ' '.join(__cpu_baseline__)
    for feature in __cpu_dispatch__:
        if __cpu_features__[feature]:
            enabled_features += f' {feature}*'
        else:
            enabled_features += f' {feature}?'
    return enabled_features

def drop_metadata(dtype, /):
    if False:
        return 10
    '\n    Returns the dtype unchanged if it contained no metadata or a copy of the\n    dtype if it (or any of its structure dtypes) contained metadata.\n\n    This utility is used by `np.save` and `np.savez` to drop metadata before\n    saving.\n\n    .. note::\n\n        Due to its limitation this function may move to a more appropriate\n        home or change in the future and is considered semi-public API only.\n\n    .. warning::\n\n        This function does not preserve more strange things like record dtypes\n        and user dtypes may simply return the wrong thing.  If you need to be\n        sure about the latter, check the result with:\n        ``np.can_cast(new_dtype, dtype, casting="no")``.\n\n    '
    if dtype.fields is not None:
        found_metadata = dtype.metadata is not None
        names = []
        formats = []
        offsets = []
        titles = []
        for (name, field) in dtype.fields.items():
            field_dt = drop_metadata(field[0])
            if field_dt is not field[0]:
                found_metadata = True
            names.append(name)
            formats.append(field_dt)
            offsets.append(field[1])
            titles.append(None if len(field) < 3 else field[2])
        if not found_metadata:
            return dtype
        structure = dict(names=names, formats=formats, offsets=offsets, titles=titles, itemsize=dtype.itemsize)
        return np.dtype(structure, align=dtype.isalignedstruct)
    elif dtype.subdtype is not None:
        (subdtype, shape) = dtype.subdtype
        new_subdtype = drop_metadata(subdtype)
        if dtype.metadata is None and new_subdtype is subdtype:
            return dtype
        return np.dtype((new_subdtype, shape))
    else:
        if dtype.metadata is None:
            return dtype
        return np.dtype(dtype.str)