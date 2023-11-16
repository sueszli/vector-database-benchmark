from __future__ import annotations
import copyreg
import warnings
import pandas as pd
from packaging.version import Version
try:
    import pyarrow as pa
    pa_version = pa.__version__
    if Version(pa_version) < Version('14.0.1'):
        try:
            import pyarrow_hotfix
            warnings.warn(f'Minimal version of pyarrow will soon be increased to 14.0.1. You are using {pa_version}. Please consider upgrading.', FutureWarning)
        except ImportError:
            warnings.warn(f'You are using pyarrow version {pa_version} which is known to be insecure. See https://www.cve.org/CVERecord?id=CVE-2023-47248 for further details. Please upgrade to pyarrow>=14.0.1 or install pyarrow-hotfix to patch your current version.')
except ImportError:
    pa = None
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200

def rebuild_arrowextensionarray(type_, chunks):
    if False:
        while True:
            i = 10
    array = pa.chunked_array(chunks)
    return type_(array)

def reduce_arrowextensionarray(x):
    if False:
        while True:
            i = 10
    return (rebuild_arrowextensionarray, (type(x), x._data.combine_chunks()))
if pa is not None and (not PANDAS_GE_200):
    if PANDAS_GE_150:
        for type_ in [pd.arrays.ArrowExtensionArray, pd.arrays.ArrowStringArray]:
            copyreg.dispatch_table[type_] = reduce_arrowextensionarray
    else:
        copyreg.dispatch_table[pd.arrays.ArrowStringArray] = reduce_arrowextensionarray