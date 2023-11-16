import sys
__author__ = 'github.com/casperdcl'
__all__ = ['tqdm_pandas']

def tqdm_pandas(tclass, **tqdm_kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Registers the given `tqdm` instance with\n    `pandas.core.groupby.DataFrameGroupBy.progress_apply`.\n    '
    from tqdm import TqdmDeprecationWarning
    if isinstance(tclass, type) or getattr(tclass, '__name__', '').startswith('tqdm_'):
        TqdmDeprecationWarning('Please use `tqdm.pandas(...)` instead of `tqdm_pandas(tqdm, ...)`.', fp_write=getattr(tqdm_kwargs.get('file', None), 'write', sys.stderr.write))
        tclass.pandas(**tqdm_kwargs)
    else:
        TqdmDeprecationWarning('Please use `tqdm.pandas(...)` instead of `tqdm_pandas(tqdm(...))`.', fp_write=getattr(tclass.fp, 'write', sys.stderr.write))
        type(tclass).pandas(deprecated_t=tclass)