"""Compatibility library."""
from typing import List
'pandas'
try:
    from pandas import DataFrame as pd_DataFrame
    from pandas import Series as pd_Series
    from pandas import concat
    try:
        from pandas import CategoricalDtype as pd_CategoricalDtype
    except ImportError:
        from pandas.api.types import CategoricalDtype as pd_CategoricalDtype
    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False

    class pd_Series:
        """Dummy class for pandas.Series."""

        def __init__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            pass

    class pd_DataFrame:
        """Dummy class for pandas.DataFrame."""

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            pass

    class pd_CategoricalDtype:
        """Dummy class for pandas.CategoricalDtype."""

        def __init__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            pass
    concat = None
'numpy'
try:
    from numpy.random import Generator as np_random_Generator
except ImportError:

    class np_random_Generator:
        """Dummy class for np.random.Generator."""

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            pass
'matplotlib'
try:
    import matplotlib
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False
'graphviz'
try:
    import graphviz
    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False
'datatable'
try:
    import datatable
    if hasattr(datatable, 'Frame'):
        dt_DataTable = datatable.Frame
    else:
        dt_DataTable = datatable.DataTable
    DATATABLE_INSTALLED = True
except ImportError:
    DATATABLE_INSTALLED = False

    class dt_DataTable:
        """Dummy class for datatable.DataTable."""

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            pass
'sklearn'
try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.utils.multiclass import check_classification_targets
    from sklearn.utils.validation import assert_all_finite, check_array, check_X_y
    try:
        from sklearn.exceptions import NotFittedError
        from sklearn.model_selection import BaseCrossValidator, GroupKFold, StratifiedKFold
    except ImportError:
        from sklearn.cross_validation import BaseCrossValidator, GroupKFold, StratifiedKFold
        from sklearn.utils.validation import NotFittedError
    try:
        from sklearn.utils.validation import _check_sample_weight
    except ImportError:
        from sklearn.utils.validation import check_consistent_length

        def _check_sample_weight(sample_weight, X, dtype=None):
            if False:
                return 10
            check_consistent_length(sample_weight, X)
            return sample_weight
    SKLEARN_INSTALLED = True
    _LGBMBaseCrossValidator = BaseCrossValidator
    _LGBMModelBase = BaseEstimator
    _LGBMRegressorBase = RegressorMixin
    _LGBMClassifierBase = ClassifierMixin
    _LGBMLabelEncoder = LabelEncoder
    LGBMNotFittedError = NotFittedError
    _LGBMStratifiedKFold = StratifiedKFold
    _LGBMGroupKFold = GroupKFold
    _LGBMCheckXY = check_X_y
    _LGBMCheckArray = check_array
    _LGBMCheckSampleWeight = _check_sample_weight
    _LGBMAssertAllFinite = assert_all_finite
    _LGBMCheckClassificationTargets = check_classification_targets
    _LGBMComputeSampleWeight = compute_sample_weight
except ImportError:
    SKLEARN_INSTALLED = False

    class _LGBMModelBase:
        """Dummy class for sklearn.base.BaseEstimator."""
        pass

    class _LGBMClassifierBase:
        """Dummy class for sklearn.base.ClassifierMixin."""
        pass

    class _LGBMRegressorBase:
        """Dummy class for sklearn.base.RegressorMixin."""
        pass
    _LGBMBaseCrossValidator = None
    _LGBMLabelEncoder = None
    LGBMNotFittedError = ValueError
    _LGBMStratifiedKFold = None
    _LGBMGroupKFold = None
    _LGBMCheckXY = None
    _LGBMCheckArray = None
    _LGBMCheckSampleWeight = None
    _LGBMAssertAllFinite = None
    _LGBMCheckClassificationTargets = None
    _LGBMComputeSampleWeight = None
'dask'
try:
    from dask import delayed
    from dask.array import Array as dask_Array
    from dask.array import from_delayed as dask_array_from_delayed
    from dask.bag import from_delayed as dask_bag_from_delayed
    from dask.dataframe import DataFrame as dask_DataFrame
    from dask.dataframe import Series as dask_Series
    from dask.distributed import Client, Future, default_client, wait
    DASK_INSTALLED = True
except ImportError:
    DASK_INSTALLED = False
    dask_array_from_delayed = None
    dask_bag_from_delayed = None
    delayed = None
    default_client = None
    wait = None

    class Client:
        """Dummy class for dask.distributed.Client."""

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            pass

    class Future:
        """Dummy class for dask.distributed.Future."""

        def __init__(self, *args, **kwargs):
            if False:
                return 10
            pass

    class dask_Array:
        """Dummy class for dask.array.Array."""

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            pass

    class dask_DataFrame:
        """Dummy class for dask.dataframe.DataFrame."""

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            pass

    class dask_Series:
        """Dummy class for dask.dataframe.Series."""

        def __init__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            pass
'pyarrow'
try:
    import pyarrow.compute as pa_compute
    from pyarrow import Array as pa_Array
    from pyarrow import ChunkedArray as pa_ChunkedArray
    from pyarrow import Table as pa_Table
    from pyarrow.cffi import ffi as arrow_cffi
    from pyarrow.types import is_floating as arrow_is_floating
    from pyarrow.types import is_integer as arrow_is_integer
    PYARROW_INSTALLED = True
except ImportError:
    PYARROW_INSTALLED = False

    class pa_Array:
        """Dummy class for pa.Array."""

        def __init__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            pass

    class pa_ChunkedArray:
        """Dummy class for pa.ChunkedArray."""

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            pass

    class pa_Table:
        """Dummy class for pa.Table."""

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            pass

    class arrow_cffi:
        """Dummy class for pyarrow.cffi.ffi."""
        CData = None
        addressof = None
        cast = None
        new = None

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            pass

    class pa_compute:
        """Dummy class for pyarrow.compute."""
        all = None
        equal = None
    arrow_is_integer = None
    arrow_is_floating = None
'cpu_count()'
try:
    from joblib import cpu_count

    def _LGBMCpuCount(only_physical_cores: bool=True) -> int:
        if False:
            for i in range(10):
                print('nop')
        return cpu_count(only_physical_cores=only_physical_cores)
except ImportError:
    try:
        from psutil import cpu_count

        def _LGBMCpuCount(only_physical_cores: bool=True) -> int:
            if False:
                i = 10
                return i + 15
            return cpu_count(logical=not only_physical_cores) or 1
    except ImportError:
        from multiprocessing import cpu_count

        def _LGBMCpuCount(only_physical_cores: bool=True) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return cpu_count()
__all__: List[str] = []