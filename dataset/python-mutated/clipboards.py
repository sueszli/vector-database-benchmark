""" io on the clipboard """
from __future__ import annotations
from io import StringIO
from typing import TYPE_CHECKING
import warnings
from pandas._libs import lib
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.generic import ABCDataFrame
from pandas import get_option, option_context
if TYPE_CHECKING:
    from pandas._typing import DtypeBackend

def read_clipboard(sep: str='\\s+', dtype_backend: DtypeBackend | lib.NoDefault=lib.no_default, **kwargs):
    if False:
        print('Hello World!')
    '\n    Read text from clipboard and pass to :func:`~pandas.read_csv`.\n\n    Parses clipboard contents similar to how CSV files are parsed\n    using :func:`~pandas.read_csv`.\n\n    Parameters\n    ----------\n    sep : str, default \'\\\\s+\'\n        A string or regex delimiter. The default of ``\'\\\\s+\'`` denotes\n        one or more whitespace characters.\n\n    dtype_backend : {\'numpy_nullable\', \'pyarrow\'}, default \'numpy_nullable\'\n        Back-end data type applied to the resultant :class:`DataFrame`\n        (still experimental). Behaviour is as follows:\n\n        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`\n          (default).\n        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`\n          DataFrame.\n\n        .. versionadded:: 2.0\n\n    **kwargs\n        See :func:`~pandas.read_csv` for the full argument list.\n\n    Returns\n    -------\n    DataFrame\n        A parsed :class:`~pandas.DataFrame` object.\n\n    See Also\n    --------\n    DataFrame.to_clipboard : Copy object to the system clipboard.\n    read_csv : Read a comma-separated values (csv) file into DataFrame.\n    read_fwf : Read a table of fixed-width formatted lines into DataFrame.\n\n    Examples\n    --------\n    >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=[\'A\', \'B\', \'C\'])\n    >>> df.to_clipboard()  # doctest: +SKIP\n    >>> pd.read_clipboard()  # doctest: +SKIP\n         A  B  C\n    0    1  2  3\n    1    4  5  6\n    '
    encoding = kwargs.pop('encoding', 'utf-8')
    if encoding is not None and encoding.lower().replace('-', '') != 'utf8':
        raise NotImplementedError('reading from clipboard only supports utf-8 encoding')
    check_dtype_backend(dtype_backend)
    from pandas.io.clipboard import clipboard_get
    from pandas.io.parsers import read_csv
    text = clipboard_get()
    try:
        text = text.decode(kwargs.get('encoding') or get_option('display.encoding'))
    except AttributeError:
        pass
    lines = text[:10000].split('\n')[:-1][:10]
    counts = {x.lstrip(' ').count('\t') for x in lines}
    if len(lines) > 1 and len(counts) == 1 and (counts.pop() != 0):
        sep = '\t'
        index_length = len(lines[0]) - len(lines[0].lstrip(' \t'))
        if index_length != 0:
            kwargs.setdefault('index_col', list(range(index_length)))
    if sep is None and kwargs.get('delim_whitespace') is None:
        sep = '\\s+'
    if len(sep) > 1 and kwargs.get('engine') is None:
        kwargs['engine'] = 'python'
    elif len(sep) > 1 and kwargs.get('engine') == 'c':
        warnings.warn('read_clipboard with regex separator does not work properly with c engine.', stacklevel=find_stack_level())
    return read_csv(StringIO(text), sep=sep, dtype_backend=dtype_backend, **kwargs)

def to_clipboard(obj, excel: bool | None=True, sep: str | None=None, **kwargs) -> None:
    if False:
        return 10
    '\n    Attempt to write text representation of object to the system clipboard\n    The clipboard can be then pasted into Excel for example.\n\n    Parameters\n    ----------\n    obj : the object to write to the clipboard\n    excel : bool, defaults to True\n            if True, use the provided separator, writing in a csv\n            format for allowing easy pasting into excel.\n            if False, write a string representation of the object\n            to the clipboard\n    sep : optional, defaults to tab\n    other keywords are passed to to_csv\n\n    Notes\n    -----\n    Requirements for your platform\n      - Linux: xclip, or xsel (with PyQt4 modules)\n      - Windows:\n      - OS X:\n    '
    encoding = kwargs.pop('encoding', 'utf-8')
    if encoding is not None and encoding.lower().replace('-', '') != 'utf8':
        raise ValueError('clipboard only supports utf-8 encoding')
    from pandas.io.clipboard import clipboard_set
    if excel is None:
        excel = True
    if excel:
        try:
            if sep is None:
                sep = '\t'
            buf = StringIO()
            obj.to_csv(buf, sep=sep, encoding='utf-8', **kwargs)
            text = buf.getvalue()
            clipboard_set(text)
            return
        except TypeError:
            warnings.warn('to_clipboard in excel mode requires a single character separator.', stacklevel=find_stack_level())
    elif sep is not None:
        warnings.warn('to_clipboard with excel=False ignores the sep argument.', stacklevel=find_stack_level())
    if isinstance(obj, ABCDataFrame):
        with option_context('display.max_colwidth', None):
            objstr = obj.to_string(**kwargs)
    else:
        objstr = str(obj)
    clipboard_set(objstr)