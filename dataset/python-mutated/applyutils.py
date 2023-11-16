import functools
from typing import Any, Dict
import cupy as cp
from numba import cuda
from numba.core.utils import pysignature
import cudf
from cudf import _lib as libcudf
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column import column
from cudf.utils import utils
from cudf.utils._numba import _CUDFNumbaConfig
from cudf.utils.docutils import docfmt_partial
_doc_applyparams = "\ndf : DataFrame\n    The source dataframe.\nfunc : function\n    The transformation function that will be executed on the CUDA GPU.\nincols: list or dict\n    A list of names of input columns that match the function arguments.\n    Or, a dictionary mapping input column names to their corresponding\n    function arguments such as {'col1': 'arg1'}.\noutcols: dict\n    A dictionary of output column names and their dtype.\nkwargs: dict\n    name-value of extra arguments.  These values are passed\n    directly into the function.\npessimistic_nulls : bool\n    Whether or not apply_rows output should be null when any corresponding\n    input is null. If False, all outputs will be non-null, but will be the\n    result of applying func against the underlying column data, which\n    may be garbage.\n"
_doc_applychunkparams = '\nchunks : int or Series-like\n    If it is an ``int``, it is the chunksize.\n    If it is an array, it contains integer offset for the start of each chunk.\n    The span of a chunk for chunk i-th is ``data[chunks[i] : chunks[i + 1]]``\n    for any ``i + 1 < chunks.size``; or, ``data[chunks[i]:]`` for the\n    ``i == len(chunks) - 1``.\ntpb : int; optional\n    The threads-per-block for the underlying kernel.\n    If not specified (Default), uses Numba ``.forall(...)`` built-in to query\n    the CUDA Driver API to determine optimal kernel launch configuration.\n    Specify 1 to emulate serial execution for each chunk.  It is a good\n    starting point but inefficient.\n    Its maximum possible value is limited by the available CUDA GPU resources.\nblkct : int; optional\n    The number of blocks for the underlying kernel.\n    If not specified (Default) and ``tpb`` is not specified (Default), uses\n    Numba ``.forall(...)`` built-in to query the CUDA Driver API to determine\n    optimal kernel launch configuration.\n    If not specified (Default) and ``tpb`` is specified, uses ``chunks`` as the\n    number of blocks.\n'
doc_apply = docfmt_partial(params=_doc_applyparams)
doc_applychunks = docfmt_partial(params=_doc_applyparams, params_chunks=_doc_applychunkparams)

@doc_apply()
def apply_rows(df, func, incols, outcols, kwargs, pessimistic_nulls, cache_key):
    if False:
        while True:
            i = 10
    'Row-wise transformation\n\n    Parameters\n    ----------\n    {params}\n    '
    applyrows = ApplyRowsCompiler(func, incols, outcols, kwargs, pessimistic_nulls, cache_key=cache_key)
    return applyrows.run(df)

@doc_applychunks()
def apply_chunks(df, func, incols, outcols, kwargs, pessimistic_nulls, chunks, blkct=None, tpb=None):
    if False:
        while True:
            i = 10
    'Chunk-wise transformation\n\n    Parameters\n    ----------\n    {params}\n    {params_chunks}\n    '
    applychunks = ApplyChunksCompiler(func, incols, outcols, kwargs, pessimistic_nulls, cache_key=None)
    return applychunks.run(df, chunks=chunks, tpb=tpb)

@acquire_spill_lock()
def make_aggregate_nullmask(df, columns=None, op='__and__'):
    if False:
        for i in range(10):
            print('nop')
    out_mask = None
    for k in columns or df._data:
        col = cudf.core.dataframe.extract_col(df, k)
        if not col.nullable:
            continue
        nullmask = column.as_column(df[k]._column.nullmask)
        if out_mask is None:
            out_mask = column.as_column(nullmask.copy(), dtype=utils.mask_dtype)
        else:
            out_mask = libcudf.binaryop.binaryop(nullmask, out_mask, op, out_mask.dtype)
    return out_mask

class ApplyKernelCompilerBase:

    def __init__(self, func, incols, outcols, kwargs, pessimistic_nulls, cache_key):
        if False:
            i = 10
            return i + 15
        sig = pysignature(func)
        self.sig = sig
        self.incols = incols
        self.outcols = outcols
        self.kwargs = kwargs
        self.pessimistic_nulls = pessimistic_nulls
        self.cache_key = cache_key
        self.kernel = self.compile(func, sig.parameters.keys(), kwargs.keys())

    @acquire_spill_lock()
    def run(self, df, **launch_params):
        if False:
            print('Hello World!')
        if isinstance(self.incols, dict):
            inputs = {v: df[k]._column.data_array_view(mode='read') for (k, v) in self.incols.items()}
        else:
            inputs = {k: df[k]._column.data_array_view(mode='read') for k in self.incols}
        outputs = {}
        for (k, dt) in self.outcols.items():
            outputs[k] = column.column_empty(len(df), dt, False).data_array_view(mode='write')
        args = {}
        for dct in [inputs, outputs, self.kwargs]:
            args.update(dct)
        bound = self.sig.bind(**args)
        self.launch_kernel(df, bound.args, **launch_params)
        if self.pessimistic_nulls:
            out_mask = make_aggregate_nullmask(df, columns=self.incols)
        else:
            out_mask = None
        outdf = df.copy()
        for k in sorted(self.outcols):
            outdf[k] = cudf.Series(outputs[k], index=outdf.index, nan_as_null=False)
            if out_mask is not None:
                outdf._data[k] = outdf[k]._column.set_mask(out_mask.data_array_view(mode='write'))
        return outdf

class ApplyRowsCompiler(ApplyKernelCompilerBase):

    def compile(self, func, argnames, extra_argnames):
        if False:
            for i in range(10):
                print('nop')
        kernel = _load_cache_or_make_row_wise_kernel(self.cache_key, func, argnames, extra_argnames)
        return kernel

    def launch_kernel(self, df, args):
        if False:
            for i in range(10):
                print('nop')
        with _CUDFNumbaConfig():
            self.kernel.forall(len(df))(*args)

class ApplyChunksCompiler(ApplyKernelCompilerBase):

    def compile(self, func, argnames, extra_argnames):
        if False:
            for i in range(10):
                print('nop')
        kernel = _load_cache_or_make_chunk_wise_kernel(func, argnames, extra_argnames)
        return kernel

    def launch_kernel(self, df, args, chunks, blkct=None, tpb=None):
        if False:
            i = 10
            return i + 15
        chunks = self.normalize_chunks(len(df), chunks)
        if blkct is None and tpb is None:
            with _CUDFNumbaConfig():
                self.kernel.forall(len(df))(len(df), chunks, *args)
        else:
            assert tpb is not None
            if blkct is None:
                blkct = chunks.size
            with _CUDFNumbaConfig():
                self.kernel[blkct, tpb](len(df), chunks, *args)

    def normalize_chunks(self, size, chunks):
        if False:
            i = 10
            return i + 15
        if isinstance(chunks, int):
            return cuda.as_cuda_array(cp.arange(start=0, stop=size, step=chunks)).view('int64')
        else:
            return cuda.as_cuda_array(cp.asarray(chunks)).view('int64')

def _make_row_wise_kernel(func, argnames, extras):
    if False:
        while True:
            i = 10
    '\n    Make a kernel that does a stride loop over the input rows.\n\n    Each thread is responsible for a row in each iteration.\n    Several iteration may be needed to handling a large number of rows.\n\n    The resulting kernel can be used with any 1D grid size and 1D block size.\n    '
    argnames = list(map(_mangle_user, argnames))
    extras = list(map(_mangle_user, extras))
    source = '\ndef row_wise_kernel({args}):\n{body}\n'
    args = ', '.join(argnames)
    body = []
    body.append('tid = cuda.grid(1)')
    body.append('ntid = cuda.gridsize(1)')
    for a in argnames:
        if a not in extras:
            start = 'tid'
            stop = ''
            stride = 'ntid'
            srcidx = '{a} = {a}[{start}:{stop}:{stride}]'
            body.append(srcidx.format(a=a, start=start, stop=stop, stride=stride))
    body.append(f'inner({args})')
    indented = ['{}{}'.format(' ' * 4, ln) for ln in body]
    concrete = source.format(args=args, body='\n'.join(indented))
    glbs = {'inner': cuda.jit(device=True)(func), 'cuda': cuda}
    exec(concrete, glbs)
    kernel = cuda.jit(glbs['row_wise_kernel'])
    return kernel

def _make_chunk_wise_kernel(func, argnames, extras):
    if False:
        while True:
            i = 10
    '\n    Make a kernel that does a stride loop over the input chunks.\n\n    Each block is responsible for a chunk in each iteration.\n    Several iteration may be needed to handling a large number of chunks.\n\n    The user function *func* will have all threads in the block for its\n    computation.\n\n    The resulting kernel can be used with any 1D grid size and 1D block size.\n    '
    argnames = list(map(_mangle_user, argnames))
    extras = list(map(_mangle_user, extras))
    source = '\ndef chunk_wise_kernel(nrows, chunks, {args}):\n{body}\n'
    args = ', '.join(argnames)
    body = []
    body.append('blkid = cuda.blockIdx.x')
    body.append('nblkid = cuda.gridDim.x')
    body.append('tid = cuda.threadIdx.x')
    body.append('ntid = cuda.blockDim.x')
    body.append('for curblk in range(blkid, chunks.size, nblkid):')
    indent = ' ' * 4
    body.append(indent + 'start = chunks[curblk]')
    body.append(indent + 'stop = chunks[curblk + 1]' + ' if curblk + 1 < chunks.size else nrows')
    slicedargs = {}
    for a in argnames:
        if a not in extras:
            slicedargs[a] = f'{a}[start:stop]'
        else:
            slicedargs[a] = str(a)
    body.append('{}inner({})'.format(indent, ', '.join((slicedargs[k] for k in argnames))))
    indented = ['{}{}'.format(' ' * 4, ln) for ln in body]
    concrete = source.format(args=args, body='\n'.join(indented))
    glbs = {'inner': cuda.jit(device=True)(func), 'cuda': cuda}
    exec(concrete, glbs)
    kernel = cuda.jit(glbs['chunk_wise_kernel'])
    return kernel
_cache: Dict[Any, Any] = dict()

@functools.wraps(_make_row_wise_kernel)
def _load_cache_or_make_row_wise_kernel(cache_key, func, *args, **kwargs):
    if False:
        return 10
    'Caching version of ``_make_row_wise_kernel``.'
    if cache_key is None:
        cache_key = func
    try:
        out = _cache[cache_key]
        return out
    except KeyError:
        kernel = _make_row_wise_kernel(func, *args, **kwargs)
        _cache[cache_key] = kernel
        return kernel

@functools.wraps(_make_chunk_wise_kernel)
def _load_cache_or_make_chunk_wise_kernel(func, *args, **kwargs):
    if False:
        return 10
    'Caching version of ``_make_row_wise_kernel``.'
    try:
        return _cache[func]
    except KeyError:
        kernel = _make_chunk_wise_kernel(func, *args, **kwargs)
        _cache[func] = kernel
        return kernel

def _mangle_user(name):
    if False:
        return 10
    'Mangle user variable name'
    return f'__user_{name}'