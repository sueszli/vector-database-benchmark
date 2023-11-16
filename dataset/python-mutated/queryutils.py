import ast
import datetime
from typing import Any, Dict
import numpy as np
from numba import cuda
import cudf
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column import column_empty
from cudf.utils import applyutils
from cudf.utils._numba import _CUDFNumbaConfig
from cudf.utils.dtypes import BOOL_TYPES, DATETIME_TYPES, NUMERIC_TYPES, TIMEDELTA_TYPES
ENVREF_PREFIX = '__CUDF_ENVREF__'
SUPPORTED_QUERY_TYPES = {np.dtype(dt) for dt in NUMERIC_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES | BOOL_TYPES}

class QuerySyntaxError(ValueError):
    pass

class _NameExtractor(ast.NodeVisitor):

    def __init__(self):
        if False:
            print('Hello World!')
        self.colnames = set()
        self.refnames = set()

    def visit_Name(self, node):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(node.ctx, ast.Load):
            raise QuerySyntaxError('assignment is not allowed')
        name = node.id
        chosen = self.refnames if name.startswith(ENVREF_PREFIX) else self.colnames
        chosen.add(name)

def query_parser(text):
    if False:
        i = 10
        return i + 15
    "The query expression parser.\n\n    See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html\n\n    * names with '@' prefix are global reference.\n    * other names must be column names of the dataframe.\n\n    Parameters\n    ----------\n    text: str\n        The query string\n\n    Returns\n    -------\n    info: a `dict` of the parsed info\n    "
    text = text.replace('@', ENVREF_PREFIX)
    tree = ast.parse(text)
    _check_error(tree)
    [expr] = tree.body
    extractor = _NameExtractor()
    extractor.visit(expr)
    colnames = sorted(extractor.colnames)
    refnames = sorted(extractor.refnames)
    info = {'source': text, 'args': colnames + refnames, 'colnames': colnames, 'refnames': refnames}
    return info

def query_builder(info, funcid):
    if False:
        print('Hello World!')
    'Function builder for the query expression\n\n    Parameters\n    ----------\n    info: dict\n        From the `query_parser()`\n    funcid: str\n        The name for the function being generated\n\n    Returns\n    -------\n    func: a python function of the query\n    '
    args = info['args']
    def_line = 'def {funcid}({args}):'.format(funcid=funcid, args=', '.join(args))
    lines = [def_line, '    return {}'.format(info['source'])]
    source = '\n'.join(lines)
    glbs = {}
    exec(source, glbs)
    return glbs[funcid]

def _check_error(tree):
    if False:
        print('Hello World!')
    if not isinstance(tree, ast.Module):
        raise QuerySyntaxError('top level should be of ast.Module')
    if len(tree.body) != 1:
        raise QuerySyntaxError('too many expressions')
_cache: Dict[Any, Any] = {}

def query_compile(expr):
    if False:
        i = 10
        return i + 15
    'Compile the query expression.\n\n    This generates a CUDA Kernel for the query expression.  The kernel is\n    cached for reuse.  All variable names, including both references to\n    columns and references to variables in the calling environment, in the\n    expression are passed as argument to the kernel. Thus, the kernel is\n    reusable on any dataframe and in any environment.\n\n    Parameters\n    ----------\n    expr : str\n        The boolean expression\n\n    Returns\n    -------\n    compiled: dict\n        key "kernel" is the cuda kernel for the query.\n        key "args" is a sequence of name of the arguments.\n    '
    funcid = f'queryexpr_{hash(expr) + 2 ** 63:x}'
    compiled = _cache.get(funcid)
    if compiled is None:
        info = query_parser(expr)
        fn = query_builder(info, funcid)
        args = info['args']
        devicefn = cuda.jit(device=True)(fn)
        kernelid = f'kernel_{funcid}'
        kernel = _wrap_query_expr(kernelid, devicefn, args)
        compiled = info.copy()
        compiled['kernel'] = kernel
        _cache[funcid] = compiled
    return compiled
_kernel_source = '\n@cuda.jit\ndef {kernelname}(out, {args}):\n    idx = cuda.grid(1)\n    if idx < out.size:\n        out[idx] = queryfn({indiced_args})\n'

def _wrap_query_expr(name, fn, args):
    if False:
        i = 10
        return i + 15
    'Wrap the query expression in a cuda kernel.'

    def _add_idx(arg):
        if False:
            while True:
                i = 10
        if arg.startswith(ENVREF_PREFIX):
            return arg
        else:
            return f'{arg}[idx]'

    def _add_prefix(arg):
        if False:
            print('Hello World!')
        return f'_args_{arg}'
    glbls = {'queryfn': fn, 'cuda': cuda}
    kernargs = map(_add_prefix, args)
    indiced_args = map(_add_prefix, map(_add_idx, args))
    src = _kernel_source.format(kernelname=name, args=', '.join(kernargs), indiced_args=', '.join(indiced_args))
    exec(src, glbls)
    kernel = glbls[name]
    return kernel

@acquire_spill_lock()
def query_execute(df, expr, callenv):
    if False:
        return 10
    "Compile & execute the query expression\n\n    Note: the expression is compiled and cached for future reuse.\n\n    Parameters\n    ----------\n    df : DataFrame\n    expr : str\n        boolean expression\n    callenv : dict\n        Contains keys 'local_dict', 'locals' and 'globals' which are all dict.\n        They represent the arg, local and global dictionaries of the caller.\n    "
    compiled = query_compile(expr)
    columns = compiled['colnames']
    colarrays = [cudf.core.dataframe.extract_col(df, col) for col in columns]
    if any((col.dtype not in SUPPORTED_QUERY_TYPES for col in colarrays)):
        raise TypeError('query only supports numeric, datetime, timedelta, or bool dtypes.')
    colarrays = [col.data_array_view(mode='read') for col in colarrays]
    kernel = compiled['kernel']
    envargs = []
    envdict = callenv['globals'].copy()
    envdict.update(callenv['locals'])
    envdict.update(callenv['local_dict'])
    for name in compiled['refnames']:
        name = name[len(ENVREF_PREFIX):]
        try:
            val = envdict[name]
            if isinstance(val, datetime.datetime):
                val = np.datetime64(val)
        except KeyError:
            msg = '{!r} not defined in the calling environment'
            raise NameError(msg.format(name))
        else:
            envargs.append(val)
    nrows = len(df)
    out = column_empty(nrows, dtype=np.bool_)
    args = [out] + colarrays + envargs
    with _CUDFNumbaConfig():
        kernel.forall(nrows)(*args)
    out_mask = applyutils.make_aggregate_nullmask(df, columns=columns)
    return out.set_mask(out_mask).fillna(False)