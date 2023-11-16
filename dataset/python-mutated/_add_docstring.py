"""A module for creating docstrings for sphinx ``data`` domains."""
import re
import textwrap
from ._array_like import NDArray
_docstrings_list = []

def add_newdoc(name: str, value: str, doc: str) -> None:
    if False:
        i = 10
        return i + 15
    'Append ``_docstrings_list`` with a docstring for `name`.\n\n    Parameters\n    ----------\n    name : str\n        The name of the object.\n    value : str\n        A string-representation of the object.\n    doc : str\n        The docstring of the object.\n\n    '
    _docstrings_list.append((name, value, doc))

def _parse_docstrings() -> str:
    if False:
        print('Hello World!')
    'Convert all docstrings in ``_docstrings_list`` into a single\n    sphinx-legible text block.\n\n    '
    type_list_ret = []
    for (name, value, doc) in _docstrings_list:
        s = textwrap.dedent(doc).replace('\n', '\n    ')
        lines = s.split('\n')
        new_lines = []
        indent = ''
        for line in lines:
            m = re.match('^(\\s+)[-=]+\\s*$', line)
            if m and new_lines:
                prev = textwrap.dedent(new_lines.pop())
                if prev == 'Examples':
                    indent = ''
                    new_lines.append(f'{m.group(1)}.. rubric:: {prev}')
                else:
                    indent = 4 * ' '
                    new_lines.append(f'{m.group(1)}.. admonition:: {prev}')
                new_lines.append('')
            else:
                new_lines.append(f'{indent}{line}')
        s = '\n'.join(new_lines)
        s_block = f'.. data:: {name}\n    :value: {value}\n    {s}'
        type_list_ret.append(s_block)
    return '\n'.join(type_list_ret)
add_newdoc('ArrayLike', 'typing.Union[...]', '\n    A `~typing.Union` representing objects that can be coerced\n    into an `~numpy.ndarray`.\n\n    Among others this includes the likes of:\n\n    * Scalars.\n    * (Nested) sequences.\n    * Objects implementing the `~class.__array__` protocol.\n\n    .. versionadded:: 1.20\n\n    See Also\n    --------\n    :term:`array_like`:\n        Any scalar or sequence that can be interpreted as an ndarray.\n\n    Examples\n    --------\n    .. code-block:: python\n\n        >>> import numpy as np\n        >>> import numpy.typing as npt\n\n        >>> def as_array(a: npt.ArrayLike) -> np.ndarray:\n        ...     return np.array(a)\n\n    ')
add_newdoc('DTypeLike', 'typing.Union[...]', '\n    A `~typing.Union` representing objects that can be coerced\n    into a `~numpy.dtype`.\n\n    Among others this includes the likes of:\n\n    * :class:`type` objects.\n    * Character codes or the names of :class:`type` objects.\n    * Objects with the ``.dtype`` attribute.\n\n    .. versionadded:: 1.20\n\n    See Also\n    --------\n    :ref:`Specifying and constructing data types <arrays.dtypes.constructing>`\n        A comprehensive overview of all objects that can be coerced\n        into data types.\n\n    Examples\n    --------\n    .. code-block:: python\n\n        >>> import numpy as np\n        >>> import numpy.typing as npt\n\n        >>> def as_dtype(d: npt.DTypeLike) -> np.dtype:\n        ...     return np.dtype(d)\n\n    ')
add_newdoc('NDArray', repr(NDArray), '\n    A `np.ndarray[Any, np.dtype[+ScalarType]] <numpy.ndarray>` type alias \n    :term:`generic <generic type>` w.r.t. its `dtype.type <numpy.dtype.type>`.\n\n    Can be used during runtime for typing arrays with a given dtype\n    and unspecified shape.\n\n    .. versionadded:: 1.21\n\n    Examples\n    --------\n    .. code-block:: python\n\n        >>> import numpy as np\n        >>> import numpy.typing as npt\n\n        >>> print(npt.NDArray)\n        numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]]\n\n        >>> print(npt.NDArray[np.float64])\n        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]\n\n        >>> NDArrayInt = npt.NDArray[np.int_]\n        >>> a: NDArrayInt = np.arange(10)\n\n        >>> def func(a: npt.ArrayLike) -> npt.NDArray[Any]:\n        ...     return np.array(a)\n\n    ')
_docstrings = _parse_docstrings()