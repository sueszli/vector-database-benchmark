from __future__ import annotations
import datetime
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, NoReturn, overload
import numpy as np
import pandas as pd
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import alignment, computation, dtypes, indexing, ops, utils
from xarray.core._aggregations import DataArrayAggregations
from xarray.core.accessor_dt import CombinedDatetimelikeAccessor
from xarray.core.accessor_str import StringAccessor
from xarray.core.alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from xarray.core.arithmetic import DataArrayArithmetic
from xarray.core.common import AbstractArray, DataWithCoords, get_chunksizes
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import Coordinates, DataArrayCoordinates, assert_coordinate_consistent, create_coords_with_default_indexes
from xarray.core.dataset import Dataset
from xarray.core.formatting import format_item
from xarray.core.indexes import Index, Indexes, PandasMultiIndex, filter_indexes_from_coords, isel_indexes
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import PANDAS_TYPES, MergeError
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import DaCompatible, T_DataArray, T_DataArrayOrSet
from xarray.core.utils import Default, HybridMappingProxy, ReprObject, _default, either_dict_or_kwargs
from xarray.core.variable import IndexVariable, Variable, as_compatible_data, as_variable
from xarray.plot.accessor import DataArrayPlotAccessor
from xarray.plot.utils import _get_units_from_attrs
from xarray.util.deprecation_helpers import _deprecate_positional_args
if TYPE_CHECKING:
    from typing import TypeVar, Union
    from numpy.typing import ArrayLike
    try:
        from dask.dataframe import DataFrame as DaskDataFrame
    except ImportError:
        DaskDataFrame = None
    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None
    try:
        from iris.cube import Cube as iris_Cube
    except ImportError:
        iris_Cube = None
    from xarray.backends import ZarrStore
    from xarray.backends.api import T_NetcdfEngine, T_NetcdfTypes
    from xarray.core.groupby import DataArrayGroupBy
    from xarray.core.parallelcompat import ChunkManagerEntrypoint
    from xarray.core.resample import DataArrayResample
    from xarray.core.rolling import DataArrayCoarsen, DataArrayRolling
    from xarray.core.types import CoarsenBoundaryOptions, DatetimeLike, DatetimeUnitOptions, Dims, ErrorOptions, ErrorOptionsWithWarn, InterpOptions, PadModeOptions, PadReflectOptions, QuantileMethods, QueryEngineOptions, QueryParserOptions, ReindexMethodOptions, Self, SideOptions, T_Chunks, T_Xarray
    from xarray.core.weighted import DataArrayWeighted
    T_XarrayOther = TypeVar('T_XarrayOther', bound=Union['DataArray', Dataset])

def _check_coords_dims(shape, coords, dims):
    if False:
        return 10
    sizes = dict(zip(dims, shape))
    for (k, v) in coords.items():
        if any((d not in dims for d in v.dims)):
            raise ValueError(f'coordinate {k} has dimensions {v.dims}, but these are not a subset of the DataArray dimensions {dims}')
        for (d, s) in v.sizes.items():
            if s != sizes[d]:
                raise ValueError(f'conflicting sizes for dimension {d!r}: length {sizes[d]} on the data but length {s} on coordinate {k!r}')

def _infer_coords_and_dims(shape, coords, dims) -> tuple[Mapping[Hashable, Any], tuple[Hashable, ...]]:
    if False:
        for i in range(10):
            print('nop')
    'All the logic for creating a new DataArray'
    if coords is not None and (not utils.is_dict_like(coords)) and (len(coords) != len(shape)):
        raise ValueError(f'coords is not dict-like, but it has {len(coords)} items, which does not match the {len(shape)} dimensions of the data')
    if isinstance(dims, str):
        dims = (dims,)
    if dims is None:
        dims = [f'dim_{n}' for n in range(len(shape))]
        if coords is not None and len(coords) == len(shape):
            if utils.is_dict_like(coords):
                dims = list(coords.keys())
            else:
                for (n, (dim, coord)) in enumerate(zip(dims, coords)):
                    coord = as_variable(coord, name=dims[n]).to_index_variable()
                    dims[n] = coord.name
        dims = tuple(dims)
    elif len(dims) != len(shape):
        raise ValueError(f'different number of dimensions on data and dims: {len(shape)} vs {len(dims)}')
    else:
        for d in dims:
            if not isinstance(d, str):
                raise TypeError(f'dimension {d} is not a string')
    new_coords: Mapping[Hashable, Any]
    if isinstance(coords, Coordinates):
        new_coords = coords
    else:
        new_coords = {}
        if utils.is_dict_like(coords):
            for (k, v) in coords.items():
                new_coords[k] = as_variable(v, name=k)
        elif coords is not None:
            for (dim, coord) in zip(dims, coords):
                var = as_variable(coord, name=dim)
                var.dims = (dim,)
                new_coords[dim] = var.to_index_variable()
    _check_coords_dims(shape, new_coords, dims)
    return (new_coords, dims)

def _check_data_shape(data, coords, dims):
    if False:
        i = 10
        return i + 15
    if data is dtypes.NA:
        data = np.nan
    if coords is not None and utils.is_scalar(data, include_0d=False):
        if utils.is_dict_like(coords):
            if dims is None:
                return data
            else:
                data_shape = tuple((as_variable(coords[k], k).size if k in coords.keys() else 1 for k in dims))
        else:
            data_shape = tuple((as_variable(coord, 'foo').size for coord in coords))
        data = np.full(data_shape, data)
    return data

class _LocIndexer(Generic[T_DataArray]):
    __slots__ = ('data_array',)

    def __init__(self, data_array: T_DataArray):
        if False:
            for i in range(10):
                print('nop')
        self.data_array = data_array

    def __getitem__(self, key) -> T_DataArray:
        if False:
            i = 10
            return i + 15
        if not utils.is_dict_like(key):
            labels = indexing.expanded_indexer(key, self.data_array.ndim)
            key = dict(zip(self.data_array.dims, labels))
        return self.data_array.sel(key)

    def __setitem__(self, key, value) -> None:
        if False:
            return 10
        if not utils.is_dict_like(key):
            labels = indexing.expanded_indexer(key, self.data_array.ndim)
            key = dict(zip(self.data_array.dims, labels))
        dim_indexers = map_index_queries(self.data_array, key).dim_indexers
        self.data_array[dim_indexers] = value
_THIS_ARRAY = ReprObject('<this-array>')

class DataArray(AbstractArray, DataWithCoords, DataArrayArithmetic, DataArrayAggregations):
    """N-dimensional array with labeled coordinates and dimensions.

    DataArray provides a wrapper around numpy ndarrays that uses
    labeled dimensions and coordinates to support metadata aware
    operations. The API is similar to that for the pandas Series or
    DataFrame, but DataArray objects can have any number of dimensions,
    and their contents have fixed data types.

    Additional features over raw numpy arrays:

    - Apply operations over dimensions by name: ``x.sum('time')``.
    - Select or assign values by integer location (like numpy):
      ``x[:10]`` or by label (like pandas): ``x.loc['2014-01-01']`` or
      ``x.sel(time='2014-01-01')``.
    - Mathematical operations (e.g., ``x - y``) vectorize across
      multiple dimensions (known in numpy as "broadcasting") based on
      dimension names, regardless of their original order.
    - Keep track of arbitrary metadata in the form of a Python
      dictionary: ``x.attrs``
    - Convert to a pandas Series: ``x.to_series()``.

    Getting items from or doing mathematical operations with a
    DataArray always returns another DataArray.

    Parameters
    ----------
    data : array_like
        Values for this array. Must be an ``numpy.ndarray``, ndarray
        like, or castable to an ``ndarray``. If a self-described xarray
        or pandas object, attempts are made to use this array's
        metadata to fill in other unspecified arguments. A view of the
        array's data is used instead of a copy if possible.
    coords : sequence or dict of array_like or :py:class:`~xarray.Coordinates`, optional
        Coordinates (tick labels) to use for indexing along each
        dimension. The following notations are accepted:

        - mapping {dimension name: array-like}
        - sequence of tuples that are valid arguments for
          ``xarray.Variable()``
          - (dims, data)
          - (dims, data, attrs)
          - (dims, data, attrs, encoding)

        Additionally, it is possible to define a coord whose name
        does not match the dimension name, or a coord based on multiple
        dimensions, with one of the following notations:

        - mapping {coord name: DataArray}
        - mapping {coord name: Variable}
        - mapping {coord name: (dimension name, array-like)}
        - mapping {coord name: (tuple of dimension names, array-like)}

        Alternatively, a :py:class:`~xarray.Coordinates` object may be used in
        order to explicitly pass indexes (e.g., a multi-index or any custom
        Xarray index) or to bypass the creation of a default index for any
        :term:`Dimension coordinate` included in that object.
    dims : Hashable or sequence of Hashable, optional
        Name(s) of the data dimension(s). Must be either a Hashable
        (only for 1D data) or a sequence of Hashables with length equal
        to the number of dimensions. If this argument is omitted,
        dimension names are taken from ``coords`` (if possible) and
        otherwise default to ``['dim_0', ... 'dim_n']``.
    name : str or None, optional
        Name of this array.
    attrs : dict_like or None, optional
        Attributes to assign to the new instance. By default, an empty
        attribute dictionary is initialized.
    indexes : py:class:`~xarray.Indexes` or dict-like, optional
        For internal use only. For passing indexes objects to the
        new DataArray, use the ``coords`` argument instead with a
        :py:class:`~xarray.Coordinate` object (both coordinate variables
        and indexes will be extracted from the latter).

    Examples
    --------
    Create data:

    >>> np.random.seed(0)
    >>> temperature = 15 + 8 * np.random.randn(2, 2, 3)
    >>> lon = [[-99.83, -99.32], [-99.79, -99.23]]
    >>> lat = [[42.25, 42.21], [42.63, 42.59]]
    >>> time = pd.date_range("2014-09-06", periods=3)
    >>> reference_time = pd.Timestamp("2014-09-05")

    Initialize a dataarray with multiple dimensions:

    >>> da = xr.DataArray(
    ...     data=temperature,
    ...     dims=["x", "y", "time"],
    ...     coords=dict(
    ...         lon=(["x", "y"], lon),
    ...         lat=(["x", "y"], lat),
    ...         time=time,
    ...         reference_time=reference_time,
    ...     ),
    ...     attrs=dict(
    ...         description="Ambient temperature.",
    ...         units="degC",
    ...     ),
    ... )
    >>> da
    <xarray.DataArray (x: 2, y: 2, time: 3)>
    array([[[29.11241877, 18.20125767, 22.82990387],
            [32.92714559, 29.94046392,  7.18177696]],
    <BLANKLINE>
           [[22.60070734, 13.78914233, 14.17424919],
            [18.28478802, 16.15234857, 26.63418806]]])
    Coordinates:
        lon             (x, y) float64 -99.83 -99.32 -99.79 -99.23
        lat             (x, y) float64 42.25 42.21 42.63 42.59
      * time            (time) datetime64[ns] 2014-09-06 2014-09-07 2014-09-08
        reference_time  datetime64[ns] 2014-09-05
    Dimensions without coordinates: x, y
    Attributes:
        description:  Ambient temperature.
        units:        degC

    Find out where the coldest temperature was:

    >>> da.isel(da.argmin(...))
    <xarray.DataArray ()>
    array(7.18177696)
    Coordinates:
        lon             float64 -99.32
        lat             float64 42.21
        time            datetime64[ns] 2014-09-08
        reference_time  datetime64[ns] 2014-09-05
    Attributes:
        description:  Ambient temperature.
        units:        degC
    """
    _cache: dict[str, Any]
    _coords: dict[Any, Variable]
    _close: Callable[[], None] | None
    _indexes: dict[Hashable, Index]
    _name: Hashable | None
    _variable: Variable
    __slots__ = ('_cache', '_coords', '_close', '_indexes', '_name', '_variable', '__weakref__')
    dt = utils.UncachedAccessor(CombinedDatetimelikeAccessor['DataArray'])

    def __init__(self, data: Any=dtypes.NA, coords: Sequence[Sequence[Any] | pd.Index | DataArray] | Mapping[Any, Any] | None=None, dims: Hashable | Sequence[Hashable] | None=None, name: Hashable | None=None, attrs: Mapping | None=None, indexes: Mapping[Any, Index] | None=None, fastpath: bool=False) -> None:
        if False:
            return 10
        if fastpath:
            variable = data
            assert dims is None
            assert attrs is None
            assert indexes is not None
        else:
            if indexes is not None:
                raise ValueError('Explicitly passing indexes via the `indexes` argument is not supported when `fastpath=False`. Use the `coords` argument instead.')
            if coords is None:
                if isinstance(data, DataArray):
                    coords = data.coords
                elif isinstance(data, pd.Series):
                    coords = [data.index]
                elif isinstance(data, pd.DataFrame):
                    coords = [data.index, data.columns]
                elif isinstance(data, (pd.Index, IndexVariable)):
                    coords = [data]
            if dims is None:
                dims = getattr(data, 'dims', getattr(coords, 'dims', None))
            if name is None:
                name = getattr(data, 'name', None)
            if attrs is None and (not isinstance(data, PANDAS_TYPES)):
                attrs = getattr(data, 'attrs', None)
            data = _check_data_shape(data, coords, dims)
            data = as_compatible_data(data)
            (coords, dims) = _infer_coords_and_dims(data.shape, coords, dims)
            variable = Variable(dims, data, attrs, fastpath=True)
            if not isinstance(coords, Coordinates):
                coords = create_coords_with_default_indexes(coords)
            indexes = dict(coords.xindexes)
            coords = {k: v.copy() for (k, v) in coords.variables.items()}
        self._variable = variable
        assert isinstance(coords, dict)
        self._coords = coords
        self._name = name
        self._indexes = indexes
        self._close = None

    @classmethod
    def _construct_direct(cls, variable: Variable, coords: dict[Any, Variable], name: Hashable, indexes: dict[Hashable, Index]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Shortcut around __init__ for internal use when we want to skip\n        costly validation\n        '
        obj = object.__new__(cls)
        obj._variable = variable
        obj._coords = coords
        obj._name = name
        obj._indexes = indexes
        obj._close = None
        return obj

    def _replace(self, variable: Variable | None=None, coords=None, name: Hashable | None | Default=_default, indexes=None) -> Self:
        if False:
            print('Hello World!')
        if variable is None:
            variable = self.variable
        if coords is None:
            coords = self._coords
        if indexes is None:
            indexes = self._indexes
        if name is _default:
            name = self.name
        return type(self)(variable, coords, name=name, indexes=indexes, fastpath=True)

    def _replace_maybe_drop_dims(self, variable: Variable, name: Hashable | None | Default=_default) -> Self:
        if False:
            return 10
        if variable.dims == self.dims and variable.shape == self.shape:
            coords = self._coords.copy()
            indexes = self._indexes
        elif variable.dims == self.dims:
            new_sizes = dict(zip(self.dims, variable.shape))
            coords = {k: v for (k, v) in self._coords.items() if v.shape == tuple((new_sizes[d] for d in v.dims))}
            indexes = filter_indexes_from_coords(self._indexes, set(coords))
        else:
            allowed_dims = set(variable.dims)
            coords = {k: v for (k, v) in self._coords.items() if set(v.dims) <= allowed_dims}
            indexes = filter_indexes_from_coords(self._indexes, set(coords))
        return self._replace(variable, coords, name, indexes=indexes)

    def _overwrite_indexes(self, indexes: Mapping[Any, Index], variables: Mapping[Any, Variable] | None=None, drop_coords: list[Hashable] | None=None, rename_dims: Mapping[Any, Any] | None=None) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Maybe replace indexes and their corresponding coordinates.'
        if not indexes:
            return self
        if variables is None:
            variables = {}
        if drop_coords is None:
            drop_coords = []
        new_variable = self.variable.copy()
        new_coords = self._coords.copy()
        new_indexes = dict(self._indexes)
        for name in indexes:
            new_coords[name] = variables[name]
            new_indexes[name] = indexes[name]
        for name in drop_coords:
            new_coords.pop(name)
            new_indexes.pop(name)
        if rename_dims:
            new_variable.dims = tuple((rename_dims.get(d, d) for d in new_variable.dims))
        return self._replace(variable=new_variable, coords=new_coords, indexes=new_indexes)

    def _to_temp_dataset(self) -> Dataset:
        if False:
            for i in range(10):
                print('nop')
        return self._to_dataset_whole(name=_THIS_ARRAY, shallow_copy=False)

    def _from_temp_dataset(self, dataset: Dataset, name: Hashable | None | Default=_default) -> Self:
        if False:
            i = 10
            return i + 15
        variable = dataset._variables.pop(_THIS_ARRAY)
        coords = dataset._variables
        indexes = dataset._indexes
        return self._replace(variable, coords, name, indexes=indexes)

    def _to_dataset_split(self, dim: Hashable) -> Dataset:
        if False:
            return 10
        "splits dataarray along dimension 'dim'"

        def subset(dim, label):
            if False:
                for i in range(10):
                    print('nop')
            array = self.loc[{dim: label}]
            array.attrs = {}
            return as_variable(array)
        variables_from_split = {label: subset(dim, label) for label in self.get_index(dim)}
        coord_names = set(self._coords) - {dim}
        ambiguous_vars = set(variables_from_split) & coord_names
        if ambiguous_vars:
            rename_msg_fmt = ', '.join([f'{v}=...' for v in sorted(ambiguous_vars)])
            raise ValueError(f'Splitting along the dimension {dim!r} would produce the variables {tuple(sorted(ambiguous_vars))} which are also existing coordinate variables. Use DataArray.rename({rename_msg_fmt}) or DataArray.assign_coords({dim}=...) to resolve this ambiguity.')
        variables = variables_from_split | {k: v for (k, v) in self._coords.items() if k != dim}
        indexes = filter_indexes_from_coords(self._indexes, coord_names)
        dataset = Dataset._construct_direct(variables, coord_names, indexes=indexes, attrs=self.attrs)
        return dataset

    def _to_dataset_whole(self, name: Hashable=None, shallow_copy: bool=True) -> Dataset:
        if False:
            print('Hello World!')
        if name is None:
            name = self.name
        if name is None:
            raise ValueError('unable to convert unnamed DataArray to a Dataset without providing an explicit name')
        if name in self.coords:
            raise ValueError('cannot create a Dataset from a DataArray with the same name as one of its coordinates')
        variables = self._coords.copy()
        variables[name] = self.variable
        if shallow_copy:
            for k in variables:
                variables[k] = variables[k].copy(deep=False)
        indexes = self._indexes
        coord_names = set(self._coords)
        return Dataset._construct_direct(variables, coord_names, indexes=indexes)

    def to_dataset(self, dim: Hashable=None, *, name: Hashable=None, promote_attrs: bool=False) -> Dataset:
        if False:
            i = 10
            return i + 15
        "Convert a DataArray to a Dataset.\n\n        Parameters\n        ----------\n        dim : Hashable, optional\n            Name of the dimension on this array along which to split this array\n            into separate variables. If not provided, this array is converted\n            into a Dataset of one variable.\n        name : Hashable, optional\n            Name to substitute for this array's name. Only valid if ``dim`` is\n            not provided.\n        promote_attrs : bool, default: False\n            Set to True to shallow copy attrs of DataArray to returned Dataset.\n\n        Returns\n        -------\n        dataset : Dataset\n        "
        if dim is not None and dim not in self.dims:
            raise TypeError(f'{dim} is not a dim. If supplying a ``name``, pass as a kwarg.')
        if dim is not None:
            if name is not None:
                raise TypeError('cannot supply both dim and name arguments')
            result = self._to_dataset_split(dim)
        else:
            result = self._to_dataset_whole(name)
        if promote_attrs:
            result.attrs = dict(self.attrs)
        return result

    @property
    def name(self) -> Hashable | None:
        if False:
            i = 10
            return i + 15
        'The name of this array.'
        return self._name

    @name.setter
    def name(self, value: Hashable | None) -> None:
        if False:
            i = 10
            return i + 15
        self._name = value

    @property
    def variable(self) -> Variable:
        if False:
            while True:
                i = 10
        'Low level interface to the Variable object for this DataArray.'
        return self._variable

    @property
    def dtype(self) -> np.dtype:
        if False:
            i = 10
            return i + 15
        '\n        Data-type of the array’s elements.\n\n        See Also\n        --------\n        ndarray.dtype\n        numpy.dtype\n        '
        return self.variable.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        if False:
            i = 10
            return i + 15
        '\n        Tuple of array dimensions.\n\n        See Also\n        --------\n        numpy.ndarray.shape\n        '
        return self.variable.shape

    @property
    def size(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Number of elements in the array.\n\n        Equal to ``np.prod(a.shape)``, i.e., the product of the array’s dimensions.\n\n        See Also\n        --------\n        numpy.ndarray.size\n        '
        return self.variable.size

    @property
    def nbytes(self) -> int:
        if False:
            return 10
        "\n        Total bytes consumed by the elements of this DataArray's data.\n\n        If the underlying data array does not include ``nbytes``, estimates\n        the bytes consumed based on the ``size`` and ``dtype``.\n        "
        return self.variable.nbytes

    @property
    def ndim(self) -> int:
        if False:
            return 10
        '\n        Number of array dimensions.\n\n        See Also\n        --------\n        numpy.ndarray.ndim\n        '
        return self.variable.ndim

    def __len__(self) -> int:
        if False:
            return 10
        return len(self.variable)

    @property
    def data(self) -> Any:
        if False:
            while True:
                i = 10
        "\n        The DataArray's data as an array. The underlying array type\n        (e.g. dask, sparse, pint) is preserved.\n\n        See Also\n        --------\n        DataArray.to_numpy\n        DataArray.as_numpy\n        DataArray.values\n        "
        return self.variable.data

    @data.setter
    def data(self, value: Any) -> None:
        if False:
            print('Hello World!')
        self.variable.data = value

    @property
    def values(self) -> np.ndarray:
        if False:
            return 10
        "\n        The array's data as a numpy.ndarray.\n\n        If the array's data is not a numpy.ndarray this will attempt to convert\n        it naively using np.array(), which will raise an error if the array\n        type does not support coercion like this (e.g. cupy).\n        "
        return self.variable.values

    @values.setter
    def values(self, value: Any) -> None:
        if False:
            print('Hello World!')
        self.variable.values = value

    def to_numpy(self) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Coerces wrapped data to numpy and returns a numpy.ndarray.\n\n        See Also\n        --------\n        DataArray.as_numpy : Same but returns the surrounding DataArray instead.\n        Dataset.as_numpy\n        DataArray.values\n        DataArray.data\n        '
        return self.variable.to_numpy()

    def as_numpy(self) -> Self:
        if False:
            i = 10
            return i + 15
        '\n        Coerces wrapped data and coordinates into numpy arrays, returning a DataArray.\n\n        See Also\n        --------\n        DataArray.to_numpy : Same but returns only the data as a numpy.ndarray object.\n        Dataset.as_numpy : Converts all variables in a Dataset.\n        DataArray.values\n        DataArray.data\n        '
        coords = {k: v.as_numpy() for (k, v) in self._coords.items()}
        return self._replace(self.variable.as_numpy(), coords, indexes=self._indexes)

    @property
    def _in_memory(self) -> bool:
        if False:
            while True:
                i = 10
        return self.variable._in_memory

    def _to_index(self) -> pd.Index:
        if False:
            i = 10
            return i + 15
        return self.variable._to_index()

    def to_index(self) -> pd.Index:
        if False:
            i = 10
            return i + 15
        'Convert this variable to a pandas.Index. Only possible for 1D\n        arrays.\n        '
        return self.variable.to_index()

    @property
    def dims(self) -> tuple[Hashable, ...]:
        if False:
            print('Hello World!')
        'Tuple of dimension names associated with this array.\n\n        Note that the type of this property is inconsistent with\n        `Dataset.dims`.  See `Dataset.sizes` and `DataArray.sizes` for\n        consistently named properties.\n\n        See Also\n        --------\n        DataArray.sizes\n        Dataset.dims\n        '
        return self.variable.dims

    @dims.setter
    def dims(self, value: Any) -> NoReturn:
        if False:
            while True:
                i = 10
        raise AttributeError('you cannot assign dims on a DataArray. Use .rename() or .swap_dims() instead.')

    def _item_key_to_dict(self, key: Any) -> Mapping[Hashable, Any]:
        if False:
            print('Hello World!')
        if utils.is_dict_like(key):
            return key
        key = indexing.expanded_indexer(key, self.ndim)
        return dict(zip(self.dims, key))

    def _getitem_coord(self, key: Any) -> Self:
        if False:
            i = 10
            return i + 15
        from xarray.core.dataset import _get_virtual_variable
        try:
            var = self._coords[key]
        except KeyError:
            dim_sizes = dict(zip(self.dims, self.shape))
            (_, key, var) = _get_virtual_variable(self._coords, key, dim_sizes)
        return self._replace_maybe_drop_dims(var, name=key)

    def __getitem__(self, key: Any) -> Self:
        if False:
            return 10
        if isinstance(key, str):
            return self._getitem_coord(key)
        else:
            return self.isel(indexers=self._item_key_to_dict(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(key, str):
            self.coords[key] = value
        else:
            obj = self[key]
            if isinstance(value, DataArray):
                assert_coordinate_consistent(value, obj.coords.variables)
                value = value.variable
            key = {k: v.variable if isinstance(v, DataArray) else v for (k, v) in self._item_key_to_dict(key).items()}
            self.variable[key] = value

    def __delitem__(self, key: Any) -> None:
        if False:
            while True:
                i = 10
        del self.coords[key]

    @property
    def _attr_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        if False:
            print('Hello World!')
        'Places to look-up items for attribute-style access'
        yield from self._item_sources
        yield self.attrs

    @property
    def _item_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        if False:
            print('Hello World!')
        'Places to look-up items for key-completion'
        yield HybridMappingProxy(keys=self._coords, mapping=self.coords)
        yield HybridMappingProxy(keys=self.dims, mapping={})

    def __contains__(self, key: Any) -> bool:
        if False:
            i = 10
            return i + 15
        return key in self.data

    @property
    def loc(self) -> _LocIndexer:
        if False:
            return 10
        'Attribute for location based indexing like pandas.'
        return _LocIndexer(self)

    @property
    def attrs(self) -> dict[Any, Any]:
        if False:
            i = 10
            return i + 15
        'Dictionary storing arbitrary metadata with this array.'
        return self.variable.attrs

    @attrs.setter
    def attrs(self, value: Mapping[Any, Any]) -> None:
        if False:
            print('Hello World!')
        self.variable.attrs = dict(value)

    @property
    def encoding(self) -> dict[Any, Any]:
        if False:
            return 10
        'Dictionary of format-specific settings for how this array should be\n        serialized.'
        return self.variable.encoding

    @encoding.setter
    def encoding(self, value: Mapping[Any, Any]) -> None:
        if False:
            while True:
                i = 10
        self.variable.encoding = dict(value)

    def reset_encoding(self) -> Self:
        if False:
            return 10
        warnings.warn('reset_encoding is deprecated since 2023.11, use `drop_encoding` instead')
        return self.drop_encoding()

    def drop_encoding(self) -> Self:
        if False:
            while True:
                i = 10
        'Return a new DataArray without encoding on the array or any attached\n        coords.'
        ds = self._to_temp_dataset().drop_encoding()
        return self._from_temp_dataset(ds)

    @property
    def indexes(self) -> Indexes:
        if False:
            for i in range(10):
                print('nop')
        'Mapping of pandas.Index objects used for label based indexing.\n\n        Raises an error if this Dataset has indexes that cannot be coerced\n        to pandas.Index objects.\n\n        See Also\n        --------\n        DataArray.xindexes\n\n        '
        return self.xindexes.to_pandas_indexes()

    @property
    def xindexes(self) -> Indexes:
        if False:
            i = 10
            return i + 15
        'Mapping of :py:class:`~xarray.indexes.Index` objects\n        used for label based indexing.\n        '
        return Indexes(self._indexes, {k: self._coords[k] for k in self._indexes})

    @property
    def coords(self) -> DataArrayCoordinates:
        if False:
            return 10
        'Mapping of :py:class:`~xarray.DataArray` objects corresponding to\n        coordinate variables.\n\n        See Also\n        --------\n        Coordinates\n        '
        return DataArrayCoordinates(self)

    @overload
    def reset_coords(self, names: Dims=None, *, drop: Literal[False]=False) -> Dataset:
        if False:
            return 10
        ...

    @overload
    def reset_coords(self, names: Dims=None, *, drop: Literal[True]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        ...

    @_deprecate_positional_args('v2023.10.0')
    def reset_coords(self, names: Dims=None, *, drop: bool=False) -> Self | Dataset:
        if False:
            while True:
                i = 10
        'Given names of coordinates, reset them to become variables.\n\n        Parameters\n        ----------\n        names : str, Iterable of Hashable or None, optional\n            Name(s) of non-index coordinates in this dataset to reset into\n            variables. By default, all non-index coordinates are reset.\n        drop : bool, default: False\n            If True, remove coordinates instead of converting them into\n            variables.\n\n        Returns\n        -------\n        Dataset, or DataArray if ``drop == True``\n\n        Examples\n        --------\n        >>> temperature = np.arange(25).reshape(5, 5)\n        >>> pressure = np.arange(50, 75).reshape(5, 5)\n        >>> da = xr.DataArray(\n        ...     data=temperature,\n        ...     dims=["x", "y"],\n        ...     coords=dict(\n        ...         lon=("x", np.arange(10, 15)),\n        ...         lat=("y", np.arange(20, 25)),\n        ...         Pressure=(["x", "y"], pressure),\n        ...     ),\n        ...     name="Temperature",\n        ... )\n        >>> da\n        <xarray.DataArray \'Temperature\' (x: 5, y: 5)>\n        array([[ 0,  1,  2,  3,  4],\n               [ 5,  6,  7,  8,  9],\n               [10, 11, 12, 13, 14],\n               [15, 16, 17, 18, 19],\n               [20, 21, 22, 23, 24]])\n        Coordinates:\n            lon       (x) int64 10 11 12 13 14\n            lat       (y) int64 20 21 22 23 24\n            Pressure  (x, y) int64 50 51 52 53 54 55 56 57 ... 67 68 69 70 71 72 73 74\n        Dimensions without coordinates: x, y\n\n        Return Dataset with target coordinate as a data variable rather than a coordinate variable:\n\n        >>> da.reset_coords(names="Pressure")\n        <xarray.Dataset>\n        Dimensions:      (x: 5, y: 5)\n        Coordinates:\n            lon          (x) int64 10 11 12 13 14\n            lat          (y) int64 20 21 22 23 24\n        Dimensions without coordinates: x, y\n        Data variables:\n            Pressure     (x, y) int64 50 51 52 53 54 55 56 57 ... 68 69 70 71 72 73 74\n            Temperature  (x, y) int64 0 1 2 3 4 5 6 7 8 9 ... 16 17 18 19 20 21 22 23 24\n\n        Return DataArray without targeted coordinate:\n\n        >>> da.reset_coords(names="Pressure", drop=True)\n        <xarray.DataArray \'Temperature\' (x: 5, y: 5)>\n        array([[ 0,  1,  2,  3,  4],\n               [ 5,  6,  7,  8,  9],\n               [10, 11, 12, 13, 14],\n               [15, 16, 17, 18, 19],\n               [20, 21, 22, 23, 24]])\n        Coordinates:\n            lon      (x) int64 10 11 12 13 14\n            lat      (y) int64 20 21 22 23 24\n        Dimensions without coordinates: x, y\n        '
        if names is None:
            names = set(self.coords) - set(self._indexes)
        dataset = self.coords.to_dataset().reset_coords(names, drop)
        if drop:
            return self._replace(coords=dataset._variables)
        if self.name is None:
            raise ValueError('cannot reset_coords with drop=False on an unnamed DataArrray')
        dataset[self.name] = self.variable
        return dataset

    def __dask_tokenize__(self):
        if False:
            return 10
        from dask.base import normalize_token
        return normalize_token((type(self), self._variable, self._coords, self._name))

    def __dask_graph__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._to_temp_dataset().__dask_graph__()

    def __dask_keys__(self):
        if False:
            while True:
                i = 10
        return self._to_temp_dataset().__dask_keys__()

    def __dask_layers__(self):
        if False:
            print('Hello World!')
        return self._to_temp_dataset().__dask_layers__()

    @property
    def __dask_optimize__(self):
        if False:
            while True:
                i = 10
        return self._to_temp_dataset().__dask_optimize__

    @property
    def __dask_scheduler__(self):
        if False:
            return 10
        return self._to_temp_dataset().__dask_scheduler__

    def __dask_postcompute__(self):
        if False:
            return 10
        (func, args) = self._to_temp_dataset().__dask_postcompute__()
        return (self._dask_finalize, (self.name, func) + args)

    def __dask_postpersist__(self):
        if False:
            while True:
                i = 10
        (func, args) = self._to_temp_dataset().__dask_postpersist__()
        return (self._dask_finalize, (self.name, func) + args)

    @classmethod
    def _dask_finalize(cls, results, name, func, *args, **kwargs) -> Self:
        if False:
            i = 10
            return i + 15
        ds = func(results, *args, **kwargs)
        variable = ds._variables.pop(_THIS_ARRAY)
        coords = ds._variables
        indexes = ds._indexes
        return cls(variable, coords, name=name, indexes=indexes, fastpath=True)

    def load(self, **kwargs) -> Self:
        if False:
            print('Hello World!')
        "Manually trigger loading of this array's data from disk or a\n        remote source into memory and return this array.\n\n        Normally, it should not be necessary to call this method in user code,\n        because all xarray functions should either work on deferred data or\n        load data automatically. However, this method can be necessary when\n        working with many file objects on disk.\n\n        Parameters\n        ----------\n        **kwargs : dict\n            Additional keyword arguments passed on to ``dask.compute``.\n\n        See Also\n        --------\n        dask.compute\n        "
        ds = self._to_temp_dataset().load(**kwargs)
        new = self._from_temp_dataset(ds)
        self._variable = new._variable
        self._coords = new._coords
        return self

    def compute(self, **kwargs) -> Self:
        if False:
            return 10
        "Manually trigger loading of this array's data from disk or a\n        remote source into memory and return a new array. The original is\n        left unaltered.\n\n        Normally, it should not be necessary to call this method in user code,\n        because all xarray functions should either work on deferred data or\n        load data automatically. However, this method can be necessary when\n        working with many file objects on disk.\n\n        Parameters\n        ----------\n        **kwargs : dict\n            Additional keyword arguments passed on to ``dask.compute``.\n\n        See Also\n        --------\n        dask.compute\n        "
        new = self.copy(deep=False)
        return new.load(**kwargs)

    def persist(self, **kwargs) -> Self:
        if False:
            print('Hello World!')
        'Trigger computation in constituent dask arrays\n\n        This keeps them as dask arrays but encourages them to keep data in\n        memory.  This is particularly useful when on a distributed machine.\n        When on a single machine consider using ``.compute()`` instead.\n\n        Parameters\n        ----------\n        **kwargs : dict\n            Additional keyword arguments passed on to ``dask.persist``.\n\n        See Also\n        --------\n        dask.persist\n        '
        ds = self._to_temp_dataset().persist(**kwargs)
        return self._from_temp_dataset(ds)

    def copy(self, deep: bool=True, data: Any=None) -> Self:
        if False:
            return 10
        'Returns a copy of this array.\n\n        If `deep=True`, a deep copy is made of the data array.\n        Otherwise, a shallow copy is made, and the returned data array\'s\n        values are a new view of this data array\'s values.\n\n        Use `data` to create a new object with the same structure as\n        original but entirely new data.\n\n        Parameters\n        ----------\n        deep : bool, optional\n            Whether the data array and its coordinates are loaded into memory\n            and copied onto the new object. Default is True.\n        data : array_like, optional\n            Data to use in the new object. Must have same shape as original.\n            When `data` is used, `deep` is ignored for all data variables,\n            and only used for coords.\n\n        Returns\n        -------\n        copy : DataArray\n            New object with dimensions, attributes, coordinates, name,\n            encoding, and optionally data copied from original.\n\n        Examples\n        --------\n        Shallow versus deep copy\n\n        >>> array = xr.DataArray([1, 2, 3], dims="x", coords={"x": ["a", "b", "c"]})\n        >>> array.copy()\n        <xarray.DataArray (x: 3)>\n        array([1, 2, 3])\n        Coordinates:\n          * x        (x) <U1 \'a\' \'b\' \'c\'\n        >>> array_0 = array.copy(deep=False)\n        >>> array_0[0] = 7\n        >>> array_0\n        <xarray.DataArray (x: 3)>\n        array([7, 2, 3])\n        Coordinates:\n          * x        (x) <U1 \'a\' \'b\' \'c\'\n        >>> array\n        <xarray.DataArray (x: 3)>\n        array([7, 2, 3])\n        Coordinates:\n          * x        (x) <U1 \'a\' \'b\' \'c\'\n\n        Changing the data using the ``data`` argument maintains the\n        structure of the original object, but with the new data. Original\n        object is unaffected.\n\n        >>> array.copy(data=[0.1, 0.2, 0.3])\n        <xarray.DataArray (x: 3)>\n        array([0.1, 0.2, 0.3])\n        Coordinates:\n          * x        (x) <U1 \'a\' \'b\' \'c\'\n        >>> array\n        <xarray.DataArray (x: 3)>\n        array([7, 2, 3])\n        Coordinates:\n          * x        (x) <U1 \'a\' \'b\' \'c\'\n\n        See Also\n        --------\n        pandas.DataFrame.copy\n        '
        return self._copy(deep=deep, data=data)

    def _copy(self, deep: bool=True, data: Any=None, memo: dict[int, Any] | None=None) -> Self:
        if False:
            return 10
        variable = self.variable._copy(deep=deep, data=data, memo=memo)
        (indexes, index_vars) = self.xindexes.copy_indexes(deep=deep)
        coords = {}
        for (k, v) in self._coords.items():
            if k in index_vars:
                coords[k] = index_vars[k]
            else:
                coords[k] = v._copy(deep=deep, memo=memo)
        return self._replace(variable, coords, indexes=indexes)

    def __copy__(self) -> Self:
        if False:
            i = 10
            return i + 15
        return self._copy(deep=False)

    def __deepcopy__(self, memo: dict[int, Any] | None=None) -> Self:
        if False:
            for i in range(10):
                print('nop')
        return self._copy(deep=True, memo=memo)
    __hash__ = None

    @property
    def chunks(self) -> tuple[tuple[int, ...], ...] | None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Tuple of block lengths for this dataarray's data, in order of dimensions, or None if\n        the underlying data is not a dask array.\n\n        See Also\n        --------\n        DataArray.chunk\n        DataArray.chunksizes\n        xarray.unify_chunks\n        "
        return self.variable.chunks

    @property
    def chunksizes(self) -> Mapping[Any, tuple[int, ...]]:
        if False:
            return 10
        "\n        Mapping from dimension names to block lengths for this dataarray's data, or None if\n        the underlying data is not a dask array.\n        Cannot be modified directly, but can be modified by calling .chunk().\n\n        Differs from DataArray.chunks because it returns a mapping of dimensions to chunk shapes\n        instead of a tuple of chunk shapes.\n\n        See Also\n        --------\n        DataArray.chunk\n        DataArray.chunks\n        xarray.unify_chunks\n        "
        all_variables = [self.variable] + [c.variable for c in self.coords.values()]
        return get_chunksizes(all_variables)

    @_deprecate_positional_args('v2023.10.0')
    def chunk(self, chunks: T_Chunks={}, *, name_prefix: str='xarray-', token: str | None=None, lock: bool=False, inline_array: bool=False, chunked_array_type: str | ChunkManagerEntrypoint | None=None, from_array_kwargs=None, **chunks_kwargs: Any) -> Self:
        if False:
            return 10
        'Coerce this array\'s data into a dask arrays with the given chunks.\n\n        If this variable is a non-dask array, it will be converted to dask\n        array. If it\'s a dask array, it will be rechunked to the given chunk\n        sizes.\n\n        If neither chunks is not provided for one or more dimensions, chunk\n        sizes along that dimension will not be updated; non-dask arrays will be\n        converted into dask arrays with a single block.\n\n        Parameters\n        ----------\n        chunks : int, "auto", tuple of int or mapping of Hashable to int, optional\n            Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, ``(5, 5)`` or\n            ``{"x": 5, "y": 5}``.\n        name_prefix : str, optional\n            Prefix for the name of the new dask array.\n        token : str, optional\n            Token uniquely identifying this array.\n        lock : bool, default: False\n            Passed on to :py:func:`dask.array.from_array`, if the array is not\n            already as dask array.\n        inline_array: bool, default: False\n            Passed on to :py:func:`dask.array.from_array`, if the array is not\n            already as dask array.\n        chunked_array_type: str, optional\n            Which chunked array type to coerce the underlying data array to.\n            Defaults to \'dask\' if installed, else whatever is registered via the `ChunkManagerEntryPoint` system.\n            Experimental API that should not be relied upon.\n        from_array_kwargs: dict, optional\n            Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create\n            chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.\n            For example, with dask as the default chunked array type, this method would pass additional kwargs\n            to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.\n        **chunks_kwargs : {dim: chunks, ...}, optional\n            The keyword arguments form of ``chunks``.\n            One of chunks or chunks_kwargs must be provided.\n\n        Returns\n        -------\n        chunked : xarray.DataArray\n\n        See Also\n        --------\n        DataArray.chunks\n        DataArray.chunksizes\n        xarray.unify_chunks\n        dask.array.from_array\n        '
        if chunks is None:
            warnings.warn("None value for 'chunks' is deprecated. It will raise an error in the future. Use instead '{}'", category=FutureWarning)
            chunks = {}
        if isinstance(chunks, (float, str, int)):
            chunks = dict.fromkeys(self.dims, chunks)
        elif isinstance(chunks, (tuple, list)):
            utils.emit_user_level_warning('Supplying chunks as dimension-order tuples is deprecated. It will raise an error in the future. Instead use a dict with dimension names as keys.', category=DeprecationWarning)
            chunks = dict(zip(self.dims, chunks))
        else:
            chunks = either_dict_or_kwargs(chunks, chunks_kwargs, 'chunk')
        ds = self._to_temp_dataset().chunk(chunks, name_prefix=name_prefix, token=token, lock=lock, inline_array=inline_array, chunked_array_type=chunked_array_type, from_array_kwargs=from_array_kwargs)
        return self._from_temp_dataset(ds)

    def isel(self, indexers: Mapping[Any, Any] | None=None, drop: bool=False, missing_dims: ErrorOptionsWithWarn='raise', **indexers_kwargs: Any) -> Self:
        if False:
            while True:
                i = 10
        'Return a new DataArray whose data is given by selecting indexes\n        along the specified dimension(s).\n\n        Parameters\n        ----------\n        indexers : dict, optional\n            A dict with keys matching dimensions and values given\n            by integers, slice objects or arrays.\n            indexer can be a integer, slice, array-like or DataArray.\n            If DataArrays are passed as indexers, xarray-style indexing will be\n            carried out. See :ref:`indexing` for the details.\n            One of indexers or indexers_kwargs must be provided.\n        drop : bool, default: False\n            If ``drop=True``, drop coordinates variables indexed by integers\n            instead of making them scalar.\n        missing_dims : {"raise", "warn", "ignore"}, default: "raise"\n            What to do if dimensions that should be selected from are not present in the\n            DataArray:\n            - "raise": raise an exception\n            - "warn": raise a warning, and ignore the missing dimensions\n            - "ignore": ignore the missing dimensions\n        **indexers_kwargs : {dim: indexer, ...}, optional\n            The keyword arguments form of ``indexers``.\n\n        Returns\n        -------\n        indexed : xarray.DataArray\n\n        See Also\n        --------\n        Dataset.isel\n        DataArray.sel\n\n        :doc:`xarray-tutorial:intermediate/indexing/indexing`\n            Tutorial material on indexing with Xarray objects\n\n        :doc:`xarray-tutorial:fundamentals/02.1_indexing_Basic`\n            Tutorial material on basics of indexing\n\n        Examples\n        --------\n        >>> da = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))\n        >>> da\n        <xarray.DataArray (x: 5, y: 5)>\n        array([[ 0,  1,  2,  3,  4],\n               [ 5,  6,  7,  8,  9],\n               [10, 11, 12, 13, 14],\n               [15, 16, 17, 18, 19],\n               [20, 21, 22, 23, 24]])\n        Dimensions without coordinates: x, y\n\n        >>> tgt_x = xr.DataArray(np.arange(0, 5), dims="points")\n        >>> tgt_y = xr.DataArray(np.arange(0, 5), dims="points")\n        >>> da = da.isel(x=tgt_x, y=tgt_y)\n        >>> da\n        <xarray.DataArray (points: 5)>\n        array([ 0,  6, 12, 18, 24])\n        Dimensions without coordinates: points\n        '
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'isel')
        if any((is_fancy_indexer(idx) for idx in indexers.values())):
            ds = self._to_temp_dataset()._isel_fancy(indexers, drop=drop, missing_dims=missing_dims)
            return self._from_temp_dataset(ds)
        variable = self._variable.isel(indexers, missing_dims=missing_dims)
        (indexes, index_variables) = isel_indexes(self.xindexes, indexers)
        coords = {}
        for (coord_name, coord_value) in self._coords.items():
            if coord_name in index_variables:
                coord_value = index_variables[coord_name]
            else:
                coord_indexers = {k: v for (k, v) in indexers.items() if k in coord_value.dims}
                if coord_indexers:
                    coord_value = coord_value.isel(coord_indexers)
                    if drop and coord_value.ndim == 0:
                        continue
            coords[coord_name] = coord_value
        return self._replace(variable=variable, coords=coords, indexes=indexes)

    def sel(self, indexers: Mapping[Any, Any] | None=None, method: str | None=None, tolerance=None, drop: bool=False, **indexers_kwargs: Any) -> Self:
        if False:
            return 10
        'Return a new DataArray whose data is given by selecting index\n        labels along the specified dimension(s).\n\n        In contrast to `DataArray.isel`, indexers for this method should use\n        labels instead of integers.\n\n        Under the hood, this method is powered by using pandas\'s powerful Index\n        objects. This makes label based indexing essentially just as fast as\n        using integer indexing.\n\n        It also means this method uses pandas\'s (well documented) logic for\n        indexing. This means you can use string shortcuts for datetime indexes\n        (e.g., \'2000-01\' to select all values in January 2000). It also means\n        that slices are treated as inclusive of both the start and stop values,\n        unlike normal Python indexing.\n\n        .. warning::\n\n          Do not try to assign values when using any of the indexing methods\n          ``isel`` or ``sel``::\n\n            da = xr.DataArray([0, 1, 2, 3], dims=[\'x\'])\n            # DO NOT do this\n            da.isel(x=[0, 1, 2])[1] = -1\n\n          Assigning values with the chained indexing using ``.sel`` or\n          ``.isel`` fails silently.\n\n        Parameters\n        ----------\n        indexers : dict, optional\n            A dict with keys matching dimensions and values given\n            by scalars, slices or arrays of tick labels. For dimensions with\n            multi-index, the indexer may also be a dict-like object with keys\n            matching index level names.\n            If DataArrays are passed as indexers, xarray-style indexing will be\n            carried out. See :ref:`indexing` for the details.\n            One of indexers or indexers_kwargs must be provided.\n        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional\n            Method to use for inexact matches:\n\n            - None (default): only exact matches\n            - pad / ffill: propagate last valid index value forward\n            - backfill / bfill: propagate next valid index value backward\n            - nearest: use nearest valid index value\n\n        tolerance : optional\n            Maximum distance between original and new labels for inexact\n            matches. The values of the index at the matching locations must\n            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.\n        drop : bool, optional\n            If ``drop=True``, drop coordinates variables in `indexers` instead\n            of making them scalar.\n        **indexers_kwargs : {dim: indexer, ...}, optional\n            The keyword arguments form of ``indexers``.\n            One of indexers or indexers_kwargs must be provided.\n\n        Returns\n        -------\n        obj : DataArray\n            A new DataArray with the same contents as this DataArray, except the\n            data and each dimension is indexed by the appropriate indexers.\n            If indexer DataArrays have coordinates that do not conflict with\n            this object, then these coordinates will be attached.\n            In general, each array\'s data will be a view of the array\'s data\n            in this DataArray, unless vectorized indexing was triggered by using\n            an array indexer, in which case the data will be a copy.\n\n        See Also\n        --------\n        Dataset.sel\n        DataArray.isel\n\n        :doc:`xarray-tutorial:intermediate/indexing/indexing`\n            Tutorial material on indexing with Xarray objects\n\n        :doc:`xarray-tutorial:fundamentals/02.1_indexing_Basic`\n            Tutorial material on basics of indexing\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     np.arange(25).reshape(5, 5),\n        ...     coords={"x": np.arange(5), "y": np.arange(5)},\n        ...     dims=("x", "y"),\n        ... )\n        >>> da\n        <xarray.DataArray (x: 5, y: 5)>\n        array([[ 0,  1,  2,  3,  4],\n               [ 5,  6,  7,  8,  9],\n               [10, 11, 12, 13, 14],\n               [15, 16, 17, 18, 19],\n               [20, 21, 22, 23, 24]])\n        Coordinates:\n          * x        (x) int64 0 1 2 3 4\n          * y        (y) int64 0 1 2 3 4\n\n        >>> tgt_x = xr.DataArray(np.linspace(0, 4, num=5), dims="points")\n        >>> tgt_y = xr.DataArray(np.linspace(0, 4, num=5), dims="points")\n        >>> da = da.sel(x=tgt_x, y=tgt_y, method="nearest")\n        >>> da\n        <xarray.DataArray (points: 5)>\n        array([ 0,  6, 12, 18, 24])\n        Coordinates:\n            x        (points) int64 0 1 2 3 4\n            y        (points) int64 0 1 2 3 4\n        Dimensions without coordinates: points\n        '
        ds = self._to_temp_dataset().sel(indexers=indexers, drop=drop, method=method, tolerance=tolerance, **indexers_kwargs)
        return self._from_temp_dataset(ds)

    def head(self, indexers: Mapping[Any, int] | int | None=None, **indexers_kwargs: Any) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Return a new DataArray whose data is given by the the first `n`\n        values along the specified dimension(s). Default `n` = 5\n\n        See Also\n        --------\n        Dataset.head\n        DataArray.tail\n        DataArray.thin\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     np.arange(25).reshape(5, 5),\n        ...     dims=("x", "y"),\n        ... )\n        >>> da\n        <xarray.DataArray (x: 5, y: 5)>\n        array([[ 0,  1,  2,  3,  4],\n               [ 5,  6,  7,  8,  9],\n               [10, 11, 12, 13, 14],\n               [15, 16, 17, 18, 19],\n               [20, 21, 22, 23, 24]])\n        Dimensions without coordinates: x, y\n\n        >>> da.head(x=1)\n        <xarray.DataArray (x: 1, y: 5)>\n        array([[0, 1, 2, 3, 4]])\n        Dimensions without coordinates: x, y\n\n        >>> da.head({"x": 2, "y": 2})\n        <xarray.DataArray (x: 2, y: 2)>\n        array([[0, 1],\n               [5, 6]])\n        Dimensions without coordinates: x, y\n        '
        ds = self._to_temp_dataset().head(indexers, **indexers_kwargs)
        return self._from_temp_dataset(ds)

    def tail(self, indexers: Mapping[Any, int] | int | None=None, **indexers_kwargs: Any) -> Self:
        if False:
            while True:
                i = 10
        'Return a new DataArray whose data is given by the the last `n`\n        values along the specified dimension(s). Default `n` = 5\n\n        See Also\n        --------\n        Dataset.tail\n        DataArray.head\n        DataArray.thin\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     np.arange(25).reshape(5, 5),\n        ...     dims=("x", "y"),\n        ... )\n        >>> da\n        <xarray.DataArray (x: 5, y: 5)>\n        array([[ 0,  1,  2,  3,  4],\n               [ 5,  6,  7,  8,  9],\n               [10, 11, 12, 13, 14],\n               [15, 16, 17, 18, 19],\n               [20, 21, 22, 23, 24]])\n        Dimensions without coordinates: x, y\n\n        >>> da.tail(y=1)\n        <xarray.DataArray (x: 5, y: 1)>\n        array([[ 4],\n               [ 9],\n               [14],\n               [19],\n               [24]])\n        Dimensions without coordinates: x, y\n\n        >>> da.tail({"x": 2, "y": 2})\n        <xarray.DataArray (x: 2, y: 2)>\n        array([[18, 19],\n               [23, 24]])\n        Dimensions without coordinates: x, y\n        '
        ds = self._to_temp_dataset().tail(indexers, **indexers_kwargs)
        return self._from_temp_dataset(ds)

    def thin(self, indexers: Mapping[Any, int] | int | None=None, **indexers_kwargs: Any) -> Self:
        if False:
            print('Hello World!')
        'Return a new DataArray whose data is given by each `n` value\n        along the specified dimension(s).\n\n        Examples\n        --------\n        >>> x_arr = np.arange(0, 26)\n        >>> x_arr\n        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n               17, 18, 19, 20, 21, 22, 23, 24, 25])\n        >>> x = xr.DataArray(\n        ...     np.reshape(x_arr, (2, 13)),\n        ...     dims=("x", "y"),\n        ...     coords={"x": [0, 1], "y": np.arange(0, 13)},\n        ... )\n        >>> x\n        <xarray.DataArray (x: 2, y: 13)>\n        array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],\n               [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]])\n        Coordinates:\n          * x        (x) int64 0 1\n          * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 11 12\n\n        >>>\n        >>> x.thin(3)\n        <xarray.DataArray (x: 1, y: 5)>\n        array([[ 0,  3,  6,  9, 12]])\n        Coordinates:\n          * x        (x) int64 0\n          * y        (y) int64 0 3 6 9 12\n        >>> x.thin({"x": 2, "y": 5})\n        <xarray.DataArray (x: 1, y: 3)>\n        array([[ 0,  5, 10]])\n        Coordinates:\n          * x        (x) int64 0\n          * y        (y) int64 0 5 10\n\n        See Also\n        --------\n        Dataset.thin\n        DataArray.head\n        DataArray.tail\n        '
        ds = self._to_temp_dataset().thin(indexers, **indexers_kwargs)
        return self._from_temp_dataset(ds)

    @_deprecate_positional_args('v2023.10.0')
    def broadcast_like(self, other: T_DataArrayOrSet, *, exclude: Iterable[Hashable] | None=None) -> Self:
        if False:
            while True:
                i = 10
        'Broadcast this DataArray against another Dataset or DataArray.\n\n        This is equivalent to xr.broadcast(other, self)[1]\n\n        xarray objects are broadcast against each other in arithmetic\n        operations, so this method is not be necessary for most uses.\n\n        If no change is needed, the input data is returned to the output\n        without being copied.\n\n        If new coords are added by the broadcast, their values are\n        NaN filled.\n\n        Parameters\n        ----------\n        other : Dataset or DataArray\n            Object against which to broadcast this array.\n        exclude : iterable of Hashable, optional\n            Dimensions that must not be broadcasted\n\n        Returns\n        -------\n        new_da : DataArray\n            The caller broadcasted against ``other``.\n\n        Examples\n        --------\n        >>> arr1 = xr.DataArray(\n        ...     np.random.randn(2, 3),\n        ...     dims=("x", "y"),\n        ...     coords={"x": ["a", "b"], "y": ["a", "b", "c"]},\n        ... )\n        >>> arr2 = xr.DataArray(\n        ...     np.random.randn(3, 2),\n        ...     dims=("x", "y"),\n        ...     coords={"x": ["a", "b", "c"], "y": ["a", "b"]},\n        ... )\n        >>> arr1\n        <xarray.DataArray (x: 2, y: 3)>\n        array([[ 1.76405235,  0.40015721,  0.97873798],\n               [ 2.2408932 ,  1.86755799, -0.97727788]])\n        Coordinates:\n          * x        (x) <U1 \'a\' \'b\'\n          * y        (y) <U1 \'a\' \'b\' \'c\'\n        >>> arr2\n        <xarray.DataArray (x: 3, y: 2)>\n        array([[ 0.95008842, -0.15135721],\n               [-0.10321885,  0.4105985 ],\n               [ 0.14404357,  1.45427351]])\n        Coordinates:\n          * x        (x) <U1 \'a\' \'b\' \'c\'\n          * y        (y) <U1 \'a\' \'b\'\n        >>> arr1.broadcast_like(arr2)\n        <xarray.DataArray (x: 3, y: 3)>\n        array([[ 1.76405235,  0.40015721,  0.97873798],\n               [ 2.2408932 ,  1.86755799, -0.97727788],\n               [        nan,         nan,         nan]])\n        Coordinates:\n          * x        (x) <U1 \'a\' \'b\' \'c\'\n          * y        (y) <U1 \'a\' \'b\' \'c\'\n        '
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)
        args = align(other, self, join='outer', copy=False, exclude=exclude)
        (dims_map, common_coords) = _get_broadcast_dims_map_common_coords(args, exclude)
        return _broadcast_helper(args[1], exclude, dims_map, common_coords)

    def _reindex_callback(self, aligner: alignment.Aligner, dim_pos_indexers: dict[Hashable, Any], variables: dict[Hashable, Variable], indexes: dict[Hashable, Index], fill_value: Any, exclude_dims: frozenset[Hashable], exclude_vars: frozenset[Hashable]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Callback called from ``Aligner`` to create a new reindexed DataArray.'
        if isinstance(fill_value, dict):
            fill_value = fill_value.copy()
            sentinel = object()
            value = fill_value.pop(self.name, sentinel)
            if value is not sentinel:
                fill_value[_THIS_ARRAY] = value
        ds = self._to_temp_dataset()
        reindexed = ds._reindex_callback(aligner, dim_pos_indexers, variables, indexes, fill_value, exclude_dims, exclude_vars)
        da = self._from_temp_dataset(reindexed)
        da.encoding = self.encoding
        return da

    @_deprecate_positional_args('v2023.10.0')
    def reindex_like(self, other: T_DataArrayOrSet, *, method: ReindexMethodOptions=None, tolerance: int | float | Iterable[int | float] | None=None, copy: bool=True, fill_value=dtypes.NA) -> Self:
        if False:
            i = 10
            return i + 15
        '\n        Conform this object onto the indexes of another object, for indexes which the\n        objects share. Missing values are filled with ``fill_value``. The default fill\n        value is NaN.\n\n        Parameters\n        ----------\n        other : Dataset or DataArray\n            Object with an \'indexes\' attribute giving a mapping from dimension\n            names to pandas.Index objects, which provides coordinates upon\n            which to index the variables in this dataset. The indexes on this\n            other object need not be the same as the indexes on this\n            dataset. Any mis-matched index values will be filled in with\n            NaN, and any mis-matched dimension names will simply be ignored.\n        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional\n            Method to use for filling index values from other not found on this\n            data array:\n\n            - None (default): don\'t fill gaps\n            - pad / ffill: propagate last valid index value forward\n            - backfill / bfill: propagate next valid index value backward\n            - nearest: use nearest valid index value\n\n        tolerance : optional\n            Maximum distance between original and new labels for inexact\n            matches. The values of the index at the matching locations must\n            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.\n            Tolerance may be a scalar value, which applies the same tolerance\n            to all values, or list-like, which applies variable tolerance per\n            element. List-like must be the same size as the index and its dtype\n            must exactly match the index’s type.\n        copy : bool, default: True\n            If ``copy=True``, data in the return value is always copied. If\n            ``copy=False`` and reindexing is unnecessary, or can be performed\n            with only slice operations, then the output may share memory with\n            the input. In either case, a new xarray object is always returned.\n        fill_value : scalar or dict-like, optional\n            Value to use for newly missing values. If a dict-like, maps\n            variable names (including coordinates) to fill values. Use this\n            data array\'s name to refer to the data array\'s values.\n\n        Returns\n        -------\n        reindexed : DataArray\n            Another dataset array, with this array\'s data but coordinates from\n            the other object.\n\n        Examples\n        --------\n        >>> data = np.arange(12).reshape(4, 3)\n        >>> da1 = xr.DataArray(\n        ...     data=data,\n        ...     dims=["x", "y"],\n        ...     coords={"x": [10, 20, 30, 40], "y": [70, 80, 90]},\n        ... )\n        >>> da1\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 0,  1,  2],\n               [ 3,  4,  5],\n               [ 6,  7,  8],\n               [ 9, 10, 11]])\n        Coordinates:\n          * x        (x) int64 10 20 30 40\n          * y        (y) int64 70 80 90\n        >>> da2 = xr.DataArray(\n        ...     data=data,\n        ...     dims=["x", "y"],\n        ...     coords={"x": [40, 30, 20, 10], "y": [90, 80, 70]},\n        ... )\n        >>> da2\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 0,  1,  2],\n               [ 3,  4,  5],\n               [ 6,  7,  8],\n               [ 9, 10, 11]])\n        Coordinates:\n          * x        (x) int64 40 30 20 10\n          * y        (y) int64 90 80 70\n\n        Reindexing with both DataArrays having the same coordinates set, but in different order:\n\n        >>> da1.reindex_like(da2)\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[11, 10,  9],\n               [ 8,  7,  6],\n               [ 5,  4,  3],\n               [ 2,  1,  0]])\n        Coordinates:\n          * x        (x) int64 40 30 20 10\n          * y        (y) int64 90 80 70\n\n        Reindexing with the other array having additional coordinates:\n\n        >>> da3 = xr.DataArray(\n        ...     data=data,\n        ...     dims=["x", "y"],\n        ...     coords={"x": [20, 10, 29, 39], "y": [70, 80, 90]},\n        ... )\n        >>> da1.reindex_like(da3)\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 3.,  4.,  5.],\n               [ 0.,  1.,  2.],\n               [nan, nan, nan],\n               [nan, nan, nan]])\n        Coordinates:\n          * x        (x) int64 20 10 29 39\n          * y        (y) int64 70 80 90\n\n        Filling missing values with the previous valid index with respect to the coordinates\' value:\n\n        >>> da1.reindex_like(da3, method="ffill")\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[3, 4, 5],\n               [0, 1, 2],\n               [3, 4, 5],\n               [6, 7, 8]])\n        Coordinates:\n          * x        (x) int64 20 10 29 39\n          * y        (y) int64 70 80 90\n\n        Filling missing values while tolerating specified error for inexact matches:\n\n        >>> da1.reindex_like(da3, method="ffill", tolerance=5)\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 3.,  4.,  5.],\n               [ 0.,  1.,  2.],\n               [nan, nan, nan],\n               [nan, nan, nan]])\n        Coordinates:\n          * x        (x) int64 20 10 29 39\n          * y        (y) int64 70 80 90\n\n        Filling missing values with manually specified values:\n\n        >>> da1.reindex_like(da3, fill_value=19)\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 3,  4,  5],\n               [ 0,  1,  2],\n               [19, 19, 19],\n               [19, 19, 19]])\n        Coordinates:\n          * x        (x) int64 20 10 29 39\n          * y        (y) int64 70 80 90\n\n        Note that unlike ``broadcast_like``, ``reindex_like`` doesn\'t create new dimensions:\n\n        >>> da1.sel(x=20)\n        <xarray.DataArray (y: 3)>\n        array([3, 4, 5])\n        Coordinates:\n            x        int64 20\n          * y        (y) int64 70 80 90\n\n        ...so ``b`` in not added here:\n\n        >>> da1.sel(x=20).reindex_like(da1)\n        <xarray.DataArray (y: 3)>\n        array([3, 4, 5])\n        Coordinates:\n            x        int64 20\n          * y        (y) int64 70 80 90\n\n        See Also\n        --------\n        DataArray.reindex\n        DataArray.broadcast_like\n        align\n        '
        return alignment.reindex_like(self, other=other, method=method, tolerance=tolerance, copy=copy, fill_value=fill_value)

    @_deprecate_positional_args('v2023.10.0')
    def reindex(self, indexers: Mapping[Any, Any] | None=None, *, method: ReindexMethodOptions=None, tolerance: float | Iterable[float] | None=None, copy: bool=True, fill_value=dtypes.NA, **indexers_kwargs: Any) -> Self:
        if False:
            while True:
                i = 10
        'Conform this object onto the indexes of another object, filling in\n        missing values with ``fill_value``. The default fill value is NaN.\n\n        Parameters\n        ----------\n        indexers : dict, optional\n            Dictionary with keys given by dimension names and values given by\n            arrays of coordinates tick labels. Any mis-matched coordinate\n            values will be filled in with NaN, and any mis-matched dimension\n            names will simply be ignored.\n            One of indexers or indexers_kwargs must be provided.\n        copy : bool, optional\n            If ``copy=True``, data in the return value is always copied. If\n            ``copy=False`` and reindexing is unnecessary, or can be performed\n            with only slice operations, then the output may share memory with\n            the input. In either case, a new xarray object is always returned.\n        method : {None, \'nearest\', \'pad\'/\'ffill\', \'backfill\'/\'bfill\'}, optional\n            Method to use for filling index values in ``indexers`` not found on\n            this data array:\n\n            - None (default): don\'t fill gaps\n            - pad / ffill: propagate last valid index value forward\n            - backfill / bfill: propagate next valid index value backward\n            - nearest: use nearest valid index value\n\n        tolerance : float | Iterable[float] | None, default: None\n            Maximum distance between original and new labels for inexact\n            matches. The values of the index at the matching locations must\n            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.\n            Tolerance may be a scalar value, which applies the same tolerance\n            to all values, or list-like, which applies variable tolerance per\n            element. List-like must be the same size as the index and its dtype\n            must exactly match the index’s type.\n        fill_value : scalar or dict-like, optional\n            Value to use for newly missing values. If a dict-like, maps\n            variable names (including coordinates) to fill values. Use this\n            data array\'s name to refer to the data array\'s values.\n        **indexers_kwargs : {dim: indexer, ...}, optional\n            The keyword arguments form of ``indexers``.\n            One of indexers or indexers_kwargs must be provided.\n\n        Returns\n        -------\n        reindexed : DataArray\n            Another dataset array, with this array\'s data but replaced\n            coordinates.\n\n        Examples\n        --------\n        Reverse latitude:\n\n        >>> da = xr.DataArray(\n        ...     np.arange(4),\n        ...     coords=[np.array([90, 89, 88, 87])],\n        ...     dims="lat",\n        ... )\n        >>> da\n        <xarray.DataArray (lat: 4)>\n        array([0, 1, 2, 3])\n        Coordinates:\n          * lat      (lat) int64 90 89 88 87\n        >>> da.reindex(lat=da.lat[::-1])\n        <xarray.DataArray (lat: 4)>\n        array([3, 2, 1, 0])\n        Coordinates:\n          * lat      (lat) int64 87 88 89 90\n\n        See Also\n        --------\n        DataArray.reindex_like\n        align\n        '
        indexers = utils.either_dict_or_kwargs(indexers, indexers_kwargs, 'reindex')
        return alignment.reindex(self, indexers=indexers, method=method, tolerance=tolerance, copy=copy, fill_value=fill_value)

    def interp(self, coords: Mapping[Any, Any] | None=None, method: InterpOptions='linear', assume_sorted: bool=False, kwargs: Mapping[str, Any] | None=None, **coords_kwargs: Any) -> Self:
        if False:
            print('Hello World!')
        'Interpolate a DataArray onto new coordinates\n\n        Performs univariate or multivariate interpolation of a DataArray onto\n        new coordinates using scipy\'s interpolation routines. If interpolating\n        along an existing dimension, :py:class:`scipy.interpolate.interp1d` is\n        called. When interpolating along multiple existing dimensions, an\n        attempt is made to decompose the interpolation into multiple\n        1-dimensional interpolations. If this is possible,\n        :py:class:`scipy.interpolate.interp1d` is called. Otherwise,\n        :py:func:`scipy.interpolate.interpn` is called.\n\n        Parameters\n        ----------\n        coords : dict, optional\n            Mapping from dimension names to the new coordinates.\n            New coordinate can be a scalar, array-like or DataArray.\n            If DataArrays are passed as new coordinates, their dimensions are\n            used for the broadcasting. Missing values are skipped.\n        method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"}, default: "linear"\n            The method used to interpolate. The method should be supported by\n            the scipy interpolator:\n\n            - ``interp1d``: {"linear", "nearest", "zero", "slinear",\n              "quadratic", "cubic", "polynomial"}\n            - ``interpn``: {"linear", "nearest"}\n\n            If ``"polynomial"`` is passed, the ``order`` keyword argument must\n            also be provided.\n        assume_sorted : bool, default: False\n            If False, values of x can be in any order and they are sorted\n            first. If True, x has to be an array of monotonically increasing\n            values.\n        kwargs : dict-like or None, default: None\n            Additional keyword arguments passed to scipy\'s interpolator. Valid\n            options and their behavior depend whether ``interp1d`` or\n            ``interpn`` is used.\n        **coords_kwargs : {dim: coordinate, ...}, optional\n            The keyword arguments form of ``coords``.\n            One of coords or coords_kwargs must be provided.\n\n        Returns\n        -------\n        interpolated : DataArray\n            New dataarray on the new coordinates.\n\n        Notes\n        -----\n        scipy is required.\n\n        See Also\n        --------\n        scipy.interpolate.interp1d\n        scipy.interpolate.interpn\n\n        :doc:`xarray-tutorial:fundamentals/02.2_manipulating_dimensions`\n            Tutorial material on manipulating data resolution using :py:func:`~xarray.DataArray.interp`\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     data=[[1, 4, 2, 9], [2, 7, 6, np.nan], [6, np.nan, 5, 8]],\n        ...     dims=("x", "y"),\n        ...     coords={"x": [0, 1, 2], "y": [10, 12, 14, 16]},\n        ... )\n        >>> da\n        <xarray.DataArray (x: 3, y: 4)>\n        array([[ 1.,  4.,  2.,  9.],\n               [ 2.,  7.,  6., nan],\n               [ 6., nan,  5.,  8.]])\n        Coordinates:\n          * x        (x) int64 0 1 2\n          * y        (y) int64 10 12 14 16\n\n        1D linear interpolation (the default):\n\n        >>> da.interp(x=[0, 0.75, 1.25, 1.75])\n        <xarray.DataArray (x: 4, y: 4)>\n        array([[1.  , 4.  , 2.  ,  nan],\n               [1.75, 6.25, 5.  ,  nan],\n               [3.  ,  nan, 5.75,  nan],\n               [5.  ,  nan, 5.25,  nan]])\n        Coordinates:\n          * y        (y) int64 10 12 14 16\n          * x        (x) float64 0.0 0.75 1.25 1.75\n\n        1D nearest interpolation:\n\n        >>> da.interp(x=[0, 0.75, 1.25, 1.75], method="nearest")\n        <xarray.DataArray (x: 4, y: 4)>\n        array([[ 1.,  4.,  2.,  9.],\n               [ 2.,  7.,  6., nan],\n               [ 2.,  7.,  6., nan],\n               [ 6., nan,  5.,  8.]])\n        Coordinates:\n          * y        (y) int64 10 12 14 16\n          * x        (x) float64 0.0 0.75 1.25 1.75\n\n        1D linear extrapolation:\n\n        >>> da.interp(\n        ...     x=[1, 1.5, 2.5, 3.5],\n        ...     method="linear",\n        ...     kwargs={"fill_value": "extrapolate"},\n        ... )\n        <xarray.DataArray (x: 4, y: 4)>\n        array([[ 2. ,  7. ,  6. ,  nan],\n               [ 4. ,  nan,  5.5,  nan],\n               [ 8. ,  nan,  4.5,  nan],\n               [12. ,  nan,  3.5,  nan]])\n        Coordinates:\n          * y        (y) int64 10 12 14 16\n          * x        (x) float64 1.0 1.5 2.5 3.5\n\n        2D linear interpolation:\n\n        >>> da.interp(x=[0, 0.75, 1.25, 1.75], y=[11, 13, 15], method="linear")\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[2.5  , 3.   ,   nan],\n               [4.   , 5.625,   nan],\n               [  nan,   nan,   nan],\n               [  nan,   nan,   nan]])\n        Coordinates:\n          * x        (x) float64 0.0 0.75 1.25 1.75\n          * y        (y) int64 11 13 15\n        '
        if self.dtype.kind not in 'uifc':
            raise TypeError(f'interp only works for a numeric type array. Given {self.dtype}.')
        ds = self._to_temp_dataset().interp(coords, method=method, kwargs=kwargs, assume_sorted=assume_sorted, **coords_kwargs)
        return self._from_temp_dataset(ds)

    def interp_like(self, other: T_Xarray, method: InterpOptions='linear', assume_sorted: bool=False, kwargs: Mapping[str, Any] | None=None) -> Self:
        if False:
            print('Hello World!')
        'Interpolate this object onto the coordinates of another object,\n        filling out of range values with NaN.\n\n        If interpolating along a single existing dimension,\n        :py:class:`scipy.interpolate.interp1d` is called. When interpolating\n        along multiple existing dimensions, an attempt is made to decompose the\n        interpolation into multiple 1-dimensional interpolations. If this is\n        possible, :py:class:`scipy.interpolate.interp1d` is called. Otherwise,\n        :py:func:`scipy.interpolate.interpn` is called.\n\n        Parameters\n        ----------\n        other : Dataset or DataArray\n            Object with an \'indexes\' attribute giving a mapping from dimension\n            names to an 1d array-like, which provides coordinates upon\n            which to index the variables in this dataset. Missing values are skipped.\n        method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"}, default: "linear"\n            The method used to interpolate. The method should be supported by\n            the scipy interpolator:\n\n            - {"linear", "nearest", "zero", "slinear", "quadratic", "cubic",\n              "polynomial"} when ``interp1d`` is called.\n            - {"linear", "nearest"} when ``interpn`` is called.\n\n            If ``"polynomial"`` is passed, the ``order`` keyword argument must\n            also be provided.\n        assume_sorted : bool, default: False\n            If False, values of coordinates that are interpolated over can be\n            in any order and they are sorted first. If True, interpolated\n            coordinates are assumed to be an array of monotonically increasing\n            values.\n        kwargs : dict, optional\n            Additional keyword passed to scipy\'s interpolator.\n\n        Returns\n        -------\n        interpolated : DataArray\n            Another dataarray by interpolating this dataarray\'s data along the\n            coordinates of the other object.\n\n        Examples\n        --------\n        >>> data = np.arange(12).reshape(4, 3)\n        >>> da1 = xr.DataArray(\n        ...     data=data,\n        ...     dims=["x", "y"],\n        ...     coords={"x": [10, 20, 30, 40], "y": [70, 80, 90]},\n        ... )\n        >>> da1\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 0,  1,  2],\n               [ 3,  4,  5],\n               [ 6,  7,  8],\n               [ 9, 10, 11]])\n        Coordinates:\n          * x        (x) int64 10 20 30 40\n          * y        (y) int64 70 80 90\n        >>> da2 = xr.DataArray(\n        ...     data=data,\n        ...     dims=["x", "y"],\n        ...     coords={"x": [10, 20, 29, 39], "y": [70, 80, 90]},\n        ... )\n        >>> da2\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 0,  1,  2],\n               [ 3,  4,  5],\n               [ 6,  7,  8],\n               [ 9, 10, 11]])\n        Coordinates:\n          * x        (x) int64 10 20 29 39\n          * y        (y) int64 70 80 90\n\n        Interpolate the values in the coordinates of the other DataArray with respect to the source\'s values:\n\n        >>> da2.interp_like(da1)\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[0. , 1. , 2. ],\n               [3. , 4. , 5. ],\n               [6.3, 7.3, 8.3],\n               [nan, nan, nan]])\n        Coordinates:\n          * x        (x) int64 10 20 30 40\n          * y        (y) int64 70 80 90\n\n        Could also extrapolate missing values:\n\n        >>> da2.interp_like(da1, kwargs={"fill_value": "extrapolate"})\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 0. ,  1. ,  2. ],\n               [ 3. ,  4. ,  5. ],\n               [ 6.3,  7.3,  8.3],\n               [ 9.3, 10.3, 11.3]])\n        Coordinates:\n          * x        (x) int64 10 20 30 40\n          * y        (y) int64 70 80 90\n\n        Notes\n        -----\n        scipy is required.\n        If the dataarray has object-type coordinates, reindex is used for these\n        coordinates instead of the interpolation.\n\n        See Also\n        --------\n        DataArray.interp\n        DataArray.reindex_like\n        '
        if self.dtype.kind not in 'uifc':
            raise TypeError(f'interp only works for a numeric type array. Given {self.dtype}.')
        ds = self._to_temp_dataset().interp_like(other, method=method, kwargs=kwargs, assume_sorted=assume_sorted)
        return self._from_temp_dataset(ds)

    def rename(self, new_name_or_name_dict: Hashable | Mapping[Any, Hashable] | None=None, **names: Hashable) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Returns a new DataArray with renamed coordinates, dimensions or a new name.\n\n        Parameters\n        ----------\n        new_name_or_name_dict : str or dict-like, optional\n            If the argument is dict-like, it used as a mapping from old\n            names to new names for coordinates or dimensions. Otherwise,\n            use the argument as the new name for this array.\n        **names : Hashable, optional\n            The keyword arguments form of a mapping from old names to\n            new names for coordinates or dimensions.\n            One of new_name_or_name_dict or names must be provided.\n\n        Returns\n        -------\n        renamed : DataArray\n            Renamed array or array with renamed coordinates.\n\n        See Also\n        --------\n        Dataset.rename\n        DataArray.swap_dims\n        '
        if new_name_or_name_dict is None and (not names):
            return self._replace(name=None)
        if utils.is_dict_like(new_name_or_name_dict) or new_name_or_name_dict is None:
            name_dict = either_dict_or_kwargs(new_name_or_name_dict, names, 'rename')
            dataset = self._to_temp_dataset()._rename(name_dict)
            return self._from_temp_dataset(dataset)
        if utils.hashable(new_name_or_name_dict) and names:
            dataset = self._to_temp_dataset()._rename(names)
            dataarray = self._from_temp_dataset(dataset)
            return dataarray._replace(name=new_name_or_name_dict)
        return self._replace(name=new_name_or_name_dict)

    def swap_dims(self, dims_dict: Mapping[Any, Hashable] | None=None, **dims_kwargs) -> Self:
        if False:
            return 10
        'Returns a new DataArray with swapped dimensions.\n\n        Parameters\n        ----------\n        dims_dict : dict-like\n            Dictionary whose keys are current dimension names and whose values\n            are new names.\n        **dims_kwargs : {existing_dim: new_dim, ...}, optional\n            The keyword arguments form of ``dims_dict``.\n            One of dims_dict or dims_kwargs must be provided.\n\n        Returns\n        -------\n        swapped : DataArray\n            DataArray with swapped dimensions.\n\n        Examples\n        --------\n        >>> arr = xr.DataArray(\n        ...     data=[0, 1],\n        ...     dims="x",\n        ...     coords={"x": ["a", "b"], "y": ("x", [0, 1])},\n        ... )\n        >>> arr\n        <xarray.DataArray (x: 2)>\n        array([0, 1])\n        Coordinates:\n          * x        (x) <U1 \'a\' \'b\'\n            y        (x) int64 0 1\n\n        >>> arr.swap_dims({"x": "y"})\n        <xarray.DataArray (y: 2)>\n        array([0, 1])\n        Coordinates:\n            x        (y) <U1 \'a\' \'b\'\n          * y        (y) int64 0 1\n\n        >>> arr.swap_dims({"x": "z"})\n        <xarray.DataArray (z: 2)>\n        array([0, 1])\n        Coordinates:\n            x        (z) <U1 \'a\' \'b\'\n            y        (z) int64 0 1\n        Dimensions without coordinates: z\n\n        See Also\n        --------\n        DataArray.rename\n        Dataset.swap_dims\n        '
        dims_dict = either_dict_or_kwargs(dims_dict, dims_kwargs, 'swap_dims')
        ds = self._to_temp_dataset().swap_dims(dims_dict)
        return self._from_temp_dataset(ds)

    def expand_dims(self, dim: None | Hashable | Sequence[Hashable] | Mapping[Any, Any]=None, axis: None | int | Sequence[int]=None, **dim_kwargs: Any) -> Self:
        if False:
            return 10
        'Return a new object with an additional axis (or axes) inserted at\n        the corresponding position in the array shape. The new object is a\n        view into the underlying array, not a copy.\n\n        If dim is already a scalar coordinate, it will be promoted to a 1D\n        coordinate consisting of a single value.\n\n        Parameters\n        ----------\n        dim : Hashable, sequence of Hashable, dict, or None, optional\n            Dimensions to include on the new variable.\n            If provided as str or sequence of str, then dimensions are inserted\n            with length 1. If provided as a dict, then the keys are the new\n            dimensions and the values are either integers (giving the length of\n            the new dimensions) or sequence/ndarray (giving the coordinates of\n            the new dimensions).\n        axis : int, sequence of int, or None, default: None\n            Axis position(s) where new axis is to be inserted (position(s) on\n            the result array). If a sequence of integers is passed,\n            multiple axes are inserted. In this case, dim arguments should be\n            same length list. If axis=None is passed, all the axes will be\n            inserted to the start of the result array.\n        **dim_kwargs : int or sequence or ndarray\n            The keywords are arbitrary dimensions being inserted and the values\n            are either the lengths of the new dims (if int is given), or their\n            coordinates. Note, this is an alternative to passing a dict to the\n            dim kwarg and will only be used if dim is None.\n\n        Returns\n        -------\n        expanded : DataArray\n            This object, but with additional dimension(s).\n\n        See Also\n        --------\n        Dataset.expand_dims\n\n        Examples\n        --------\n        >>> da = xr.DataArray(np.arange(5), dims=("x"))\n        >>> da\n        <xarray.DataArray (x: 5)>\n        array([0, 1, 2, 3, 4])\n        Dimensions without coordinates: x\n\n        Add new dimension of length 2:\n\n        >>> da.expand_dims(dim={"y": 2})\n        <xarray.DataArray (y: 2, x: 5)>\n        array([[0, 1, 2, 3, 4],\n               [0, 1, 2, 3, 4]])\n        Dimensions without coordinates: y, x\n\n        >>> da.expand_dims(dim={"y": 2}, axis=1)\n        <xarray.DataArray (x: 5, y: 2)>\n        array([[0, 0],\n               [1, 1],\n               [2, 2],\n               [3, 3],\n               [4, 4]])\n        Dimensions without coordinates: x, y\n\n        Add a new dimension with coordinates from array:\n\n        >>> da.expand_dims(dim={"y": np.arange(5)}, axis=0)\n        <xarray.DataArray (y: 5, x: 5)>\n        array([[0, 1, 2, 3, 4],\n               [0, 1, 2, 3, 4],\n               [0, 1, 2, 3, 4],\n               [0, 1, 2, 3, 4],\n               [0, 1, 2, 3, 4]])\n        Coordinates:\n          * y        (y) int64 0 1 2 3 4\n        Dimensions without coordinates: x\n        '
        if isinstance(dim, int):
            raise TypeError('dim should be Hashable or sequence/mapping of Hashables')
        elif isinstance(dim, Sequence) and (not isinstance(dim, str)):
            if len(dim) != len(set(dim)):
                raise ValueError('dims should not contain duplicate values.')
            dim = dict.fromkeys(dim, 1)
        elif dim is not None and (not isinstance(dim, Mapping)):
            dim = {dim: 1}
        dim = either_dict_or_kwargs(dim, dim_kwargs, 'expand_dims')
        ds = self._to_temp_dataset().expand_dims(dim, axis)
        return self._from_temp_dataset(ds)

    def set_index(self, indexes: Mapping[Any, Hashable | Sequence[Hashable]] | None=None, append: bool=False, **indexes_kwargs: Hashable | Sequence[Hashable]) -> Self:
        if False:
            i = 10
            return i + 15
        'Set DataArray (multi-)indexes using one or more existing\n        coordinates.\n\n        This legacy method is limited to pandas (multi-)indexes and\n        1-dimensional "dimension" coordinates. See\n        :py:meth:`~DataArray.set_xindex` for setting a pandas or a custom\n        Xarray-compatible index from one or more arbitrary coordinates.\n\n        Parameters\n        ----------\n        indexes : {dim: index, ...}\n            Mapping from names matching dimensions and values given\n            by (lists of) the names of existing coordinates or variables to set\n            as new (multi-)index.\n        append : bool, default: False\n            If True, append the supplied index(es) to the existing index(es).\n            Otherwise replace the existing index(es).\n        **indexes_kwargs : optional\n            The keyword arguments form of ``indexes``.\n            One of indexes or indexes_kwargs must be provided.\n\n        Returns\n        -------\n        obj : DataArray\n            Another DataArray, with this data but replaced coordinates.\n\n        Examples\n        --------\n        >>> arr = xr.DataArray(\n        ...     data=np.ones((2, 3)),\n        ...     dims=["x", "y"],\n        ...     coords={"x": range(2), "y": range(3), "a": ("x", [3, 4])},\n        ... )\n        >>> arr\n        <xarray.DataArray (x: 2, y: 3)>\n        array([[1., 1., 1.],\n               [1., 1., 1.]])\n        Coordinates:\n          * x        (x) int64 0 1\n          * y        (y) int64 0 1 2\n            a        (x) int64 3 4\n        >>> arr.set_index(x="a")\n        <xarray.DataArray (x: 2, y: 3)>\n        array([[1., 1., 1.],\n               [1., 1., 1.]])\n        Coordinates:\n          * x        (x) int64 3 4\n          * y        (y) int64 0 1 2\n\n        See Also\n        --------\n        DataArray.reset_index\n        DataArray.set_xindex\n        '
        ds = self._to_temp_dataset().set_index(indexes, append=append, **indexes_kwargs)
        return self._from_temp_dataset(ds)

    def reset_index(self, dims_or_levels: Hashable | Sequence[Hashable], drop: bool=False) -> Self:
        if False:
            while True:
                i = 10
        'Reset the specified index(es) or multi-index level(s).\n\n        This legacy method is specific to pandas (multi-)indexes and\n        1-dimensional "dimension" coordinates. See the more generic\n        :py:meth:`~DataArray.drop_indexes` and :py:meth:`~DataArray.set_xindex`\n        method to respectively drop and set pandas or custom indexes for\n        arbitrary coordinates.\n\n        Parameters\n        ----------\n        dims_or_levels : Hashable or sequence of Hashable\n            Name(s) of the dimension(s) and/or multi-index level(s) that will\n            be reset.\n        drop : bool, default: False\n            If True, remove the specified indexes and/or multi-index levels\n            instead of extracting them as new coordinates (default: False).\n\n        Returns\n        -------\n        obj : DataArray\n            Another dataarray, with this dataarray\'s data but replaced\n            coordinates.\n\n        See Also\n        --------\n        DataArray.set_index\n        DataArray.set_xindex\n        DataArray.drop_indexes\n        '
        ds = self._to_temp_dataset().reset_index(dims_or_levels, drop=drop)
        return self._from_temp_dataset(ds)

    def set_xindex(self, coord_names: str | Sequence[Hashable], index_cls: type[Index] | None=None, **options) -> Self:
        if False:
            return 10
        "Set a new, Xarray-compatible index from one or more existing\n        coordinate(s).\n\n        Parameters\n        ----------\n        coord_names : str or list\n            Name(s) of the coordinate(s) used to build the index.\n            If several names are given, their order matters.\n        index_cls : subclass of :class:`~xarray.indexes.Index`\n            The type of index to create. By default, try setting\n            a pandas (multi-)index from the supplied coordinates.\n        **options\n            Options passed to the index constructor.\n\n        Returns\n        -------\n        obj : DataArray\n            Another dataarray, with this dataarray's data and with a new index.\n\n        "
        ds = self._to_temp_dataset().set_xindex(coord_names, index_cls, **options)
        return self._from_temp_dataset(ds)

    def reorder_levels(self, dim_order: Mapping[Any, Sequence[int | Hashable]] | None=None, **dim_order_kwargs: Sequence[int | Hashable]) -> Self:
        if False:
            return 10
        "Rearrange index levels using input order.\n\n        Parameters\n        ----------\n        dim_order dict-like of Hashable to int or Hashable: optional\n            Mapping from names matching dimensions and values given\n            by lists representing new level orders. Every given dimension\n            must have a multi-index.\n        **dim_order_kwargs : optional\n            The keyword arguments form of ``dim_order``.\n            One of dim_order or dim_order_kwargs must be provided.\n\n        Returns\n        -------\n        obj : DataArray\n            Another dataarray, with this dataarray's data but replaced\n            coordinates.\n        "
        ds = self._to_temp_dataset().reorder_levels(dim_order, **dim_order_kwargs)
        return self._from_temp_dataset(ds)

    def stack(self, dimensions: Mapping[Any, Sequence[Hashable]] | None=None, create_index: bool | None=True, index_cls: type[Index]=PandasMultiIndex, **dimensions_kwargs: Sequence[Hashable]) -> Self:
        if False:
            while True:
                i = 10
        '\n        Stack any number of existing dimensions into a single new dimension.\n\n        New dimensions will be added at the end, and the corresponding\n        coordinate variables will be combined into a MultiIndex.\n\n        Parameters\n        ----------\n        dimensions : mapping of Hashable to sequence of Hashable\n            Mapping of the form `new_name=(dim1, dim2, ...)`.\n            Names of new dimensions, and the existing dimensions that they\n            replace. An ellipsis (`...`) will be replaced by all unlisted dimensions.\n            Passing a list containing an ellipsis (`stacked_dim=[...]`) will stack over\n            all dimensions.\n        create_index : bool or None, default: True\n            If True, create a multi-index for each of the stacked dimensions.\n            If False, don\'t create any index.\n            If None, create a multi-index only if exactly one single (1-d) coordinate\n            index is found for every dimension to stack.\n        index_cls: class, optional\n            Can be used to pass a custom multi-index type. Must be an Xarray index that\n            implements `.stack()`. By default, a pandas multi-index wrapper is used.\n        **dimensions_kwargs\n            The keyword arguments form of ``dimensions``.\n            One of dimensions or dimensions_kwargs must be provided.\n\n        Returns\n        -------\n        stacked : DataArray\n            DataArray with stacked data.\n\n        Examples\n        --------\n        >>> arr = xr.DataArray(\n        ...     np.arange(6).reshape(2, 3),\n        ...     coords=[("x", ["a", "b"]), ("y", [0, 1, 2])],\n        ... )\n        >>> arr\n        <xarray.DataArray (x: 2, y: 3)>\n        array([[0, 1, 2],\n               [3, 4, 5]])\n        Coordinates:\n          * x        (x) <U1 \'a\' \'b\'\n          * y        (y) int64 0 1 2\n        >>> stacked = arr.stack(z=("x", "y"))\n        >>> stacked.indexes["z"]\n        MultiIndex([(\'a\', 0),\n                    (\'a\', 1),\n                    (\'a\', 2),\n                    (\'b\', 0),\n                    (\'b\', 1),\n                    (\'b\', 2)],\n                   name=\'z\')\n\n        See Also\n        --------\n        DataArray.unstack\n        '
        ds = self._to_temp_dataset().stack(dimensions, create_index=create_index, index_cls=index_cls, **dimensions_kwargs)
        return self._from_temp_dataset(ds)

    @_deprecate_positional_args('v2023.10.0')
    def unstack(self, dim: Dims=None, *, fill_value: Any=dtypes.NA, sparse: bool=False) -> Self:
        if False:
            print('Hello World!')
        '\n        Unstack existing dimensions corresponding to MultiIndexes into\n        multiple new dimensions.\n\n        New dimensions will be added at the end.\n\n        Parameters\n        ----------\n        dim : str, Iterable of Hashable or None, optional\n            Dimension(s) over which to unstack. By default unstacks all\n            MultiIndexes.\n        fill_value : scalar or dict-like, default: nan\n            Value to be filled. If a dict-like, maps variable names to\n            fill values. Use the data array\'s name to refer to its\n            name. If not provided or if the dict-like does not contain\n            all variables, the dtype\'s NA value will be used.\n        sparse : bool, default: False\n            Use sparse-array if True\n\n        Returns\n        -------\n        unstacked : DataArray\n            Array with unstacked data.\n\n        Examples\n        --------\n        >>> arr = xr.DataArray(\n        ...     np.arange(6).reshape(2, 3),\n        ...     coords=[("x", ["a", "b"]), ("y", [0, 1, 2])],\n        ... )\n        >>> arr\n        <xarray.DataArray (x: 2, y: 3)>\n        array([[0, 1, 2],\n               [3, 4, 5]])\n        Coordinates:\n          * x        (x) <U1 \'a\' \'b\'\n          * y        (y) int64 0 1 2\n        >>> stacked = arr.stack(z=("x", "y"))\n        >>> stacked.indexes["z"]\n        MultiIndex([(\'a\', 0),\n                    (\'a\', 1),\n                    (\'a\', 2),\n                    (\'b\', 0),\n                    (\'b\', 1),\n                    (\'b\', 2)],\n                   name=\'z\')\n        >>> roundtripped = stacked.unstack()\n        >>> arr.identical(roundtripped)\n        True\n\n        See Also\n        --------\n        DataArray.stack\n        '
        ds = self._to_temp_dataset().unstack(dim, fill_value=fill_value, sparse=sparse)
        return self._from_temp_dataset(ds)

    def to_unstacked_dataset(self, dim: Hashable, level: int | Hashable=0) -> Dataset:
        if False:
            i = 10
            return i + 15
        'Unstack DataArray expanding to Dataset along a given level of a\n        stacked coordinate.\n\n        This is the inverse operation of Dataset.to_stacked_array.\n\n        Parameters\n        ----------\n        dim : Hashable\n            Name of existing dimension to unstack\n        level : int or Hashable, default: 0\n            The MultiIndex level to expand to a dataset along. Can either be\n            the integer index of the level or its name.\n\n        Returns\n        -------\n        unstacked: Dataset\n\n        Examples\n        --------\n        >>> arr = xr.DataArray(\n        ...     np.arange(6).reshape(2, 3),\n        ...     coords=[("x", ["a", "b"]), ("y", [0, 1, 2])],\n        ... )\n        >>> data = xr.Dataset({"a": arr, "b": arr.isel(y=0)})\n        >>> data\n        <xarray.Dataset>\n        Dimensions:  (x: 2, y: 3)\n        Coordinates:\n          * x        (x) <U1 \'a\' \'b\'\n          * y        (y) int64 0 1 2\n        Data variables:\n            a        (x, y) int64 0 1 2 3 4 5\n            b        (x) int64 0 3\n        >>> stacked = data.to_stacked_array("z", ["x"])\n        >>> stacked.indexes["z"]\n        MultiIndex([(\'a\',   0),\n                    (\'a\',   1),\n                    (\'a\',   2),\n                    (\'b\', nan)],\n                   name=\'z\')\n        >>> roundtripped = stacked.to_unstacked_dataset(dim="z")\n        >>> data.identical(roundtripped)\n        True\n\n        See Also\n        --------\n        Dataset.to_stacked_array\n        '
        idx = self._indexes[dim].to_pandas_index()
        if not isinstance(idx, pd.MultiIndex):
            raise ValueError(f"'{dim}' is not a stacked coordinate")
        level_number = idx._get_level_number(level)
        variables = idx.levels[level_number]
        variable_dim = idx.names[level_number]
        data_dict = {}
        for k in variables:
            data_dict[k] = self.sel({variable_dim: k}, drop=True).squeeze(drop=True)
        return Dataset(data_dict)

    def transpose(self, *dims: Hashable, transpose_coords: bool=True, missing_dims: ErrorOptionsWithWarn='raise') -> Self:
        if False:
            i = 10
            return i + 15
        'Return a new DataArray object with transposed dimensions.\n\n        Parameters\n        ----------\n        *dims : Hashable, optional\n            By default, reverse the dimensions. Otherwise, reorder the\n            dimensions to this order.\n        transpose_coords : bool, default: True\n            If True, also transpose the coordinates of this DataArray.\n        missing_dims : {"raise", "warn", "ignore"}, default: "raise"\n            What to do if dimensions that should be selected from are not present in the\n            DataArray:\n            - "raise": raise an exception\n            - "warn": raise a warning, and ignore the missing dimensions\n            - "ignore": ignore the missing dimensions\n\n        Returns\n        -------\n        transposed : DataArray\n            The returned DataArray\'s array is transposed.\n\n        Notes\n        -----\n        This operation returns a view of this array\'s data. It is\n        lazy for dask-backed DataArrays but not for numpy-backed DataArrays\n        -- the data will be fully loaded.\n\n        See Also\n        --------\n        numpy.transpose\n        Dataset.transpose\n        '
        if dims:
            dims = tuple(utils.infix_dims(dims, self.dims, missing_dims))
        variable = self.variable.transpose(*dims)
        if transpose_coords:
            coords: dict[Hashable, Variable] = {}
            for (name, coord) in self.coords.items():
                coord_dims = tuple((dim for dim in dims if dim in coord.dims))
                coords[name] = coord.variable.transpose(*coord_dims)
            return self._replace(variable, coords)
        else:
            return self._replace(variable)

    @property
    def T(self) -> Self:
        if False:
            i = 10
            return i + 15
        return self.transpose()

    def drop_vars(self, names: Hashable | Iterable[Hashable], *, errors: ErrorOptions='raise') -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Returns an array with dropped variables.\n\n        Parameters\n        ----------\n        names : Hashable or iterable of Hashable\n            Name(s) of variables to drop.\n        errors : {"raise", "ignore"}, default: "raise"\n            If \'raise\', raises a ValueError error if any of the variable\n            passed are not in the dataset. If \'ignore\', any given names that are in the\n            DataArray are dropped and no error is raised.\n\n        Returns\n        -------\n        dropped : Dataset\n            New Dataset copied from `self` with variables removed.\n\n        Examples\n        -------\n        >>> data = np.arange(12).reshape(4, 3)\n        >>> da = xr.DataArray(\n        ...     data=data,\n        ...     dims=["x", "y"],\n        ...     coords={"x": [10, 20, 30, 40], "y": [70, 80, 90]},\n        ... )\n        >>> da\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 0,  1,  2],\n               [ 3,  4,  5],\n               [ 6,  7,  8],\n               [ 9, 10, 11]])\n        Coordinates:\n          * x        (x) int64 10 20 30 40\n          * y        (y) int64 70 80 90\n\n        Removing a single variable:\n\n        >>> da.drop_vars("x")\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 0,  1,  2],\n               [ 3,  4,  5],\n               [ 6,  7,  8],\n               [ 9, 10, 11]])\n        Coordinates:\n          * y        (y) int64 70 80 90\n        Dimensions without coordinates: x\n\n        Removing a list of variables:\n\n        >>> da.drop_vars(["x", "y"])\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 0,  1,  2],\n               [ 3,  4,  5],\n               [ 6,  7,  8],\n               [ 9, 10, 11]])\n        Dimensions without coordinates: x, y\n        '
        ds = self._to_temp_dataset().drop_vars(names, errors=errors)
        return self._from_temp_dataset(ds)

    def drop_indexes(self, coord_names: Hashable | Iterable[Hashable], *, errors: ErrorOptions='raise') -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Drop the indexes assigned to the given coordinates.\n\n        Parameters\n        ----------\n        coord_names : hashable or iterable of hashable\n            Name(s) of the coordinate(s) for which to drop the index.\n        errors : {"raise", "ignore"}, default: "raise"\n            If \'raise\', raises a ValueError error if any of the coordinates\n            passed have no index or are not in the dataset.\n            If \'ignore\', no error is raised.\n\n        Returns\n        -------\n        dropped : DataArray\n            A new dataarray with dropped indexes.\n        '
        ds = self._to_temp_dataset().drop_indexes(coord_names, errors=errors)
        return self._from_temp_dataset(ds)

    def drop(self, labels: Mapping[Any, Any] | None=None, dim: Hashable | None=None, *, errors: ErrorOptions='raise', **labels_kwargs) -> Self:
        if False:
            while True:
                i = 10
        'Backward compatible method based on `drop_vars` and `drop_sel`\n\n        Using either `drop_vars` or `drop_sel` is encouraged\n\n        See Also\n        --------\n        DataArray.drop_vars\n        DataArray.drop_sel\n        '
        ds = self._to_temp_dataset().drop(labels, dim, errors=errors, **labels_kwargs)
        return self._from_temp_dataset(ds)

    def drop_sel(self, labels: Mapping[Any, Any] | None=None, *, errors: ErrorOptions='raise', **labels_kwargs) -> Self:
        if False:
            print('Hello World!')
        'Drop index labels from this DataArray.\n\n        Parameters\n        ----------\n        labels : mapping of Hashable to Any\n            Index labels to drop\n        errors : {"raise", "ignore"}, default: "raise"\n            If \'raise\', raises a ValueError error if\n            any of the index labels passed are not\n            in the dataset. If \'ignore\', any given labels that are in the\n            dataset are dropped and no error is raised.\n        **labels_kwargs : {dim: label, ...}, optional\n            The keyword arguments form of ``dim`` and ``labels``\n\n        Returns\n        -------\n        dropped : DataArray\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     np.arange(25).reshape(5, 5),\n        ...     coords={"x": np.arange(0, 9, 2), "y": np.arange(0, 13, 3)},\n        ...     dims=("x", "y"),\n        ... )\n        >>> da\n        <xarray.DataArray (x: 5, y: 5)>\n        array([[ 0,  1,  2,  3,  4],\n               [ 5,  6,  7,  8,  9],\n               [10, 11, 12, 13, 14],\n               [15, 16, 17, 18, 19],\n               [20, 21, 22, 23, 24]])\n        Coordinates:\n          * x        (x) int64 0 2 4 6 8\n          * y        (y) int64 0 3 6 9 12\n\n        >>> da.drop_sel(x=[0, 2], y=9)\n        <xarray.DataArray (x: 3, y: 4)>\n        array([[10, 11, 12, 14],\n               [15, 16, 17, 19],\n               [20, 21, 22, 24]])\n        Coordinates:\n          * x        (x) int64 4 6 8\n          * y        (y) int64 0 3 6 12\n\n        >>> da.drop_sel({"x": 6, "y": [0, 3]})\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 2,  3,  4],\n               [ 7,  8,  9],\n               [12, 13, 14],\n               [22, 23, 24]])\n        Coordinates:\n          * x        (x) int64 0 2 4 8\n          * y        (y) int64 6 9 12\n        '
        if labels_kwargs or isinstance(labels, dict):
            labels = either_dict_or_kwargs(labels, labels_kwargs, 'drop')
        ds = self._to_temp_dataset().drop_sel(labels, errors=errors)
        return self._from_temp_dataset(ds)

    def drop_isel(self, indexers: Mapping[Any, Any] | None=None, **indexers_kwargs) -> Self:
        if False:
            print('Hello World!')
        'Drop index positions from this DataArray.\n\n        Parameters\n        ----------\n        indexers : mapping of Hashable to Any or None, default: None\n            Index locations to drop\n        **indexers_kwargs : {dim: position, ...}, optional\n            The keyword arguments form of ``dim`` and ``positions``\n\n        Returns\n        -------\n        dropped : DataArray\n\n        Raises\n        ------\n        IndexError\n\n        Examples\n        --------\n        >>> da = xr.DataArray(np.arange(25).reshape(5, 5), dims=("X", "Y"))\n        >>> da\n        <xarray.DataArray (X: 5, Y: 5)>\n        array([[ 0,  1,  2,  3,  4],\n               [ 5,  6,  7,  8,  9],\n               [10, 11, 12, 13, 14],\n               [15, 16, 17, 18, 19],\n               [20, 21, 22, 23, 24]])\n        Dimensions without coordinates: X, Y\n\n        >>> da.drop_isel(X=[0, 4], Y=2)\n        <xarray.DataArray (X: 3, Y: 4)>\n        array([[ 5,  6,  8,  9],\n               [10, 11, 13, 14],\n               [15, 16, 18, 19]])\n        Dimensions without coordinates: X, Y\n\n        >>> da.drop_isel({"X": 3, "Y": 3})\n        <xarray.DataArray (X: 4, Y: 4)>\n        array([[ 0,  1,  2,  4],\n               [ 5,  6,  7,  9],\n               [10, 11, 12, 14],\n               [20, 21, 22, 24]])\n        Dimensions without coordinates: X, Y\n        '
        dataset = self._to_temp_dataset()
        dataset = dataset.drop_isel(indexers=indexers, **indexers_kwargs)
        return self._from_temp_dataset(dataset)

    @_deprecate_positional_args('v2023.10.0')
    def dropna(self, dim: Hashable, *, how: Literal['any', 'all']='any', thresh: int | None=None) -> Self:
        if False:
            i = 10
            return i + 15
        'Returns a new array with dropped labels for missing values along\n        the provided dimension.\n\n        Parameters\n        ----------\n        dim : Hashable\n            Dimension along which to drop missing values. Dropping along\n            multiple dimensions simultaneously is not yet supported.\n        how : {"any", "all"}, default: "any"\n            - any : if any NA values are present, drop that label\n            - all : if all values are NA, drop that label\n\n        thresh : int or None, default: None\n            If supplied, require this many non-NA values.\n\n        Returns\n        -------\n        dropped : DataArray\n\n        Examples\n        --------\n        >>> temperature = [\n        ...     [0, 4, 2, 9],\n        ...     [np.nan, np.nan, np.nan, np.nan],\n        ...     [np.nan, 4, 2, 0],\n        ...     [3, 1, 0, 0],\n        ... ]\n        >>> da = xr.DataArray(\n        ...     data=temperature,\n        ...     dims=["Y", "X"],\n        ...     coords=dict(\n        ...         lat=("Y", np.array([-20.0, -20.25, -20.50, -20.75])),\n        ...         lon=("X", np.array([10.0, 10.25, 10.5, 10.75])),\n        ...     ),\n        ... )\n        >>> da\n        <xarray.DataArray (Y: 4, X: 4)>\n        array([[ 0.,  4.,  2.,  9.],\n               [nan, nan, nan, nan],\n               [nan,  4.,  2.,  0.],\n               [ 3.,  1.,  0.,  0.]])\n        Coordinates:\n            lat      (Y) float64 -20.0 -20.25 -20.5 -20.75\n            lon      (X) float64 10.0 10.25 10.5 10.75\n        Dimensions without coordinates: Y, X\n\n        >>> da.dropna(dim="Y", how="any")\n        <xarray.DataArray (Y: 2, X: 4)>\n        array([[0., 4., 2., 9.],\n               [3., 1., 0., 0.]])\n        Coordinates:\n            lat      (Y) float64 -20.0 -20.75\n            lon      (X) float64 10.0 10.25 10.5 10.75\n        Dimensions without coordinates: Y, X\n\n        Drop values only if all values along the dimension are NaN:\n\n        >>> da.dropna(dim="Y", how="all")\n        <xarray.DataArray (Y: 3, X: 4)>\n        array([[ 0.,  4.,  2.,  9.],\n               [nan,  4.,  2.,  0.],\n               [ 3.,  1.,  0.,  0.]])\n        Coordinates:\n            lat      (Y) float64 -20.0 -20.5 -20.75\n            lon      (X) float64 10.0 10.25 10.5 10.75\n        Dimensions without coordinates: Y, X\n        '
        ds = self._to_temp_dataset().dropna(dim, how=how, thresh=thresh)
        return self._from_temp_dataset(ds)

    def fillna(self, value: Any) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Fill missing values in this object.\n\n        This operation follows the normal broadcasting and alignment rules that\n        xarray uses for binary arithmetic, except the result is aligned to this\n        object (``join=\'left\'``) instead of aligned to the intersection of\n        index coordinates (``join=\'inner\'``).\n\n        Parameters\n        ----------\n        value : scalar, ndarray or DataArray\n            Used to fill all matching missing values in this array. If the\n            argument is a DataArray, it is first aligned with (reindexed to)\n            this array.\n\n        Returns\n        -------\n        filled : DataArray\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     np.array([1, 4, np.nan, 0, 3, np.nan]),\n        ...     dims="Z",\n        ...     coords=dict(\n        ...         Z=("Z", np.arange(6)),\n        ...         height=("Z", np.array([0, 10, 20, 30, 40, 50])),\n        ...     ),\n        ... )\n        >>> da\n        <xarray.DataArray (Z: 6)>\n        array([ 1.,  4., nan,  0.,  3., nan])\n        Coordinates:\n          * Z        (Z) int64 0 1 2 3 4 5\n            height   (Z) int64 0 10 20 30 40 50\n\n        Fill all NaN values with 0:\n\n        >>> da.fillna(0)\n        <xarray.DataArray (Z: 6)>\n        array([1., 4., 0., 0., 3., 0.])\n        Coordinates:\n          * Z        (Z) int64 0 1 2 3 4 5\n            height   (Z) int64 0 10 20 30 40 50\n\n        Fill NaN values with corresponding values in array:\n\n        >>> da.fillna(np.array([2, 9, 4, 2, 8, 9]))\n        <xarray.DataArray (Z: 6)>\n        array([1., 4., 4., 0., 3., 9.])\n        Coordinates:\n          * Z        (Z) int64 0 1 2 3 4 5\n            height   (Z) int64 0 10 20 30 40 50\n        '
        if utils.is_dict_like(value):
            raise TypeError('cannot provide fill value as a dictionary with fillna on a DataArray')
        out = ops.fillna(self, value)
        return out

    def interpolate_na(self, dim: Hashable | None=None, method: InterpOptions='linear', limit: int | None=None, use_coordinate: bool | str=True, max_gap: None | int | float | str | pd.Timedelta | np.timedelta64 | datetime.timedelta=None, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        if False:
            return 10
        'Fill in NaNs by interpolating according to different methods.\n\n        Parameters\n        ----------\n        dim : Hashable or None, optional\n            Specifies the dimension along which to interpolate.\n        method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial",             "barycentric", "krogh", "pchip", "spline", "akima"}, default: "linear"\n            String indicating which method to use for interpolation:\n\n            - \'linear\': linear interpolation. Additional keyword\n              arguments are passed to :py:func:`numpy.interp`\n            - \'nearest\', \'zero\', \'slinear\', \'quadratic\', \'cubic\', \'polynomial\':\n              are passed to :py:func:`scipy.interpolate.interp1d`. If\n              ``method=\'polynomial\'``, the ``order`` keyword argument must also be\n              provided.\n            - \'barycentric\', \'krogh\', \'pchip\', \'spline\', \'akima\': use their\n              respective :py:class:`scipy.interpolate` classes.\n\n        use_coordinate : bool or str, default: True\n            Specifies which index to use as the x values in the interpolation\n            formulated as `y = f(x)`. If False, values are treated as if\n            equally-spaced along ``dim``. If True, the IndexVariable `dim` is\n            used. If ``use_coordinate`` is a string, it specifies the name of a\n            coordinate variable to use as the index.\n        limit : int or None, default: None\n            Maximum number of consecutive NaNs to fill. Must be greater than 0\n            or None for no limit. This filling is done regardless of the size of\n            the gap in the data. To only interpolate over gaps less than a given length,\n            see ``max_gap``.\n        max_gap : int, float, str, pandas.Timedelta, numpy.timedelta64, datetime.timedelta, default: None\n            Maximum size of gap, a continuous sequence of NaNs, that will be filled.\n            Use None for no limit. When interpolating along a datetime64 dimension\n            and ``use_coordinate=True``, ``max_gap`` can be one of the following:\n\n            - a string that is valid input for pandas.to_timedelta\n            - a :py:class:`numpy.timedelta64` object\n            - a :py:class:`pandas.Timedelta` object\n            - a :py:class:`datetime.timedelta` object\n\n            Otherwise, ``max_gap`` must be an int or a float. Use of ``max_gap`` with unlabeled\n            dimensions has not been implemented yet. Gap length is defined as the difference\n            between coordinate values at the first data point after a gap and the last value\n            before a gap. For gaps at the beginning (end), gap length is defined as the difference\n            between coordinate values at the first (last) valid data point and the first (last) NaN.\n            For example, consider::\n\n                <xarray.DataArray (x: 9)>\n                array([nan, nan, nan,  1., nan, nan,  4., nan, nan])\n                Coordinates:\n                  * x        (x) int64 0 1 2 3 4 5 6 7 8\n\n            The gap lengths are 3-0 = 3; 6-3 = 3; and 8-6 = 2 respectively\n        keep_attrs : bool or None, default: None\n            If True, the dataarray\'s attributes (`attrs`) will be copied from\n            the original object to the new one.  If False, the new\n            object will be returned without attributes.\n        **kwargs : dict, optional\n            parameters passed verbatim to the underlying interpolation function\n\n        Returns\n        -------\n        interpolated: DataArray\n            Filled in DataArray.\n\n        See Also\n        --------\n        numpy.interp\n        scipy.interpolate\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     [np.nan, 2, 3, np.nan, 0], dims="x", coords={"x": [0, 1, 2, 3, 4]}\n        ... )\n        >>> da\n        <xarray.DataArray (x: 5)>\n        array([nan,  2.,  3., nan,  0.])\n        Coordinates:\n          * x        (x) int64 0 1 2 3 4\n\n        >>> da.interpolate_na(dim="x", method="linear")\n        <xarray.DataArray (x: 5)>\n        array([nan, 2. , 3. , 1.5, 0. ])\n        Coordinates:\n          * x        (x) int64 0 1 2 3 4\n\n        >>> da.interpolate_na(dim="x", method="linear", fill_value="extrapolate")\n        <xarray.DataArray (x: 5)>\n        array([1. , 2. , 3. , 1.5, 0. ])\n        Coordinates:\n          * x        (x) int64 0 1 2 3 4\n        '
        from xarray.core.missing import interp_na
        return interp_na(self, dim=dim, method=method, limit=limit, use_coordinate=use_coordinate, max_gap=max_gap, keep_attrs=keep_attrs, **kwargs)

    def ffill(self, dim: Hashable, limit: int | None=None) -> Self:
        if False:
            while True:
                i = 10
        'Fill NaN values by propagating values forward\n\n        *Requires bottleneck.*\n\n        Parameters\n        ----------\n        dim : Hashable\n            Specifies the dimension along which to propagate values when\n            filling.\n        limit : int or None, default: None\n            The maximum number of consecutive NaN values to forward fill. In\n            other words, if there is a gap with more than this number of\n            consecutive NaNs, it will only be partially filled. Must be greater\n            than 0 or None for no limit. Must be None or greater than or equal\n            to axis length if filling along chunked axes (dimensions).\n\n        Returns\n        -------\n        filled : DataArray\n\n        Examples\n        --------\n        >>> temperature = np.array(\n        ...     [\n        ...         [np.nan, 1, 3],\n        ...         [0, np.nan, 5],\n        ...         [5, np.nan, np.nan],\n        ...         [3, np.nan, np.nan],\n        ...         [0, 2, 0],\n        ...     ]\n        ... )\n        >>> da = xr.DataArray(\n        ...     data=temperature,\n        ...     dims=["Y", "X"],\n        ...     coords=dict(\n        ...         lat=("Y", np.array([-20.0, -20.25, -20.50, -20.75, -21.0])),\n        ...         lon=("X", np.array([10.0, 10.25, 10.5])),\n        ...     ),\n        ... )\n        >>> da\n        <xarray.DataArray (Y: 5, X: 3)>\n        array([[nan,  1.,  3.],\n               [ 0., nan,  5.],\n               [ 5., nan, nan],\n               [ 3., nan, nan],\n               [ 0.,  2.,  0.]])\n        Coordinates:\n            lat      (Y) float64 -20.0 -20.25 -20.5 -20.75 -21.0\n            lon      (X) float64 10.0 10.25 10.5\n        Dimensions without coordinates: Y, X\n\n        Fill all NaN values:\n\n        >>> da.ffill(dim="Y", limit=None)\n        <xarray.DataArray (Y: 5, X: 3)>\n        array([[nan,  1.,  3.],\n               [ 0.,  1.,  5.],\n               [ 5.,  1.,  5.],\n               [ 3.,  1.,  5.],\n               [ 0.,  2.,  0.]])\n        Coordinates:\n            lat      (Y) float64 -20.0 -20.25 -20.5 -20.75 -21.0\n            lon      (X) float64 10.0 10.25 10.5\n        Dimensions without coordinates: Y, X\n\n        Fill only the first of consecutive NaN values:\n\n        >>> da.ffill(dim="Y", limit=1)\n        <xarray.DataArray (Y: 5, X: 3)>\n        array([[nan,  1.,  3.],\n               [ 0.,  1.,  5.],\n               [ 5., nan,  5.],\n               [ 3., nan, nan],\n               [ 0.,  2.,  0.]])\n        Coordinates:\n            lat      (Y) float64 -20.0 -20.25 -20.5 -20.75 -21.0\n            lon      (X) float64 10.0 10.25 10.5\n        Dimensions without coordinates: Y, X\n        '
        from xarray.core.missing import ffill
        return ffill(self, dim, limit=limit)

    def bfill(self, dim: Hashable, limit: int | None=None) -> Self:
        if False:
            return 10
        'Fill NaN values by propagating values backward\n\n        *Requires bottleneck.*\n\n        Parameters\n        ----------\n        dim : str\n            Specifies the dimension along which to propagate values when\n            filling.\n        limit : int or None, default: None\n            The maximum number of consecutive NaN values to backward fill. In\n            other words, if there is a gap with more than this number of\n            consecutive NaNs, it will only be partially filled. Must be greater\n            than 0 or None for no limit. Must be None or greater than or equal\n            to axis length if filling along chunked axes (dimensions).\n\n        Returns\n        -------\n        filled : DataArray\n\n        Examples\n        --------\n        >>> temperature = np.array(\n        ...     [\n        ...         [0, 1, 3],\n        ...         [0, np.nan, 5],\n        ...         [5, np.nan, np.nan],\n        ...         [3, np.nan, np.nan],\n        ...         [np.nan, 2, 0],\n        ...     ]\n        ... )\n        >>> da = xr.DataArray(\n        ...     data=temperature,\n        ...     dims=["Y", "X"],\n        ...     coords=dict(\n        ...         lat=("Y", np.array([-20.0, -20.25, -20.50, -20.75, -21.0])),\n        ...         lon=("X", np.array([10.0, 10.25, 10.5])),\n        ...     ),\n        ... )\n        >>> da\n        <xarray.DataArray (Y: 5, X: 3)>\n        array([[ 0.,  1.,  3.],\n               [ 0., nan,  5.],\n               [ 5., nan, nan],\n               [ 3., nan, nan],\n               [nan,  2.,  0.]])\n        Coordinates:\n            lat      (Y) float64 -20.0 -20.25 -20.5 -20.75 -21.0\n            lon      (X) float64 10.0 10.25 10.5\n        Dimensions without coordinates: Y, X\n\n        Fill all NaN values:\n\n        >>> da.bfill(dim="Y", limit=None)\n        <xarray.DataArray (Y: 5, X: 3)>\n        array([[ 0.,  1.,  3.],\n               [ 0.,  2.,  5.],\n               [ 5.,  2.,  0.],\n               [ 3.,  2.,  0.],\n               [nan,  2.,  0.]])\n        Coordinates:\n            lat      (Y) float64 -20.0 -20.25 -20.5 -20.75 -21.0\n            lon      (X) float64 10.0 10.25 10.5\n        Dimensions without coordinates: Y, X\n\n        Fill only the first of consecutive NaN values:\n\n        >>> da.bfill(dim="Y", limit=1)\n        <xarray.DataArray (Y: 5, X: 3)>\n        array([[ 0.,  1.,  3.],\n               [ 0., nan,  5.],\n               [ 5., nan, nan],\n               [ 3.,  2.,  0.],\n               [nan,  2.,  0.]])\n        Coordinates:\n            lat      (Y) float64 -20.0 -20.25 -20.5 -20.75 -21.0\n            lon      (X) float64 10.0 10.25 10.5\n        Dimensions without coordinates: Y, X\n        '
        from xarray.core.missing import bfill
        return bfill(self, dim, limit=limit)

    def combine_first(self, other: Self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        "Combine two DataArray objects, with union of coordinates.\n\n        This operation follows the normal broadcasting and alignment rules of\n        ``join='outer'``.  Default to non-null values of array calling the\n        method.  Use np.nan to fill in vacant cells after alignment.\n\n        Parameters\n        ----------\n        other : DataArray\n            Used to fill all matching missing values in this array.\n\n        Returns\n        -------\n        DataArray\n        "
        return ops.fillna(self, other, join='outer')

    def reduce(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, **kwargs: Any) -> Self:
        if False:
            while True:
                i = 10
        'Reduce this array by applying `func` along some dimension(s).\n\n        Parameters\n        ----------\n        func : callable\n            Function which can be called in the form\n            `f(x, axis=axis, **kwargs)` to return the result of reducing an\n            np.ndarray over an integer valued axis.\n        dim : "...", str, Iterable of Hashable or None, optional\n            Dimension(s) over which to apply `func`. By default `func` is\n            applied over all dimensions.\n        axis : int or sequence of int, optional\n            Axis(es) over which to repeatedly apply `func`. Only one of the\n            \'dim\' and \'axis\' arguments can be supplied. If neither are\n            supplied, then the reduction is calculated over the flattened array\n            (by calling `f(x)` without an axis argument).\n        keep_attrs : bool or None, optional\n            If True, the variable\'s attributes (`attrs`) will be copied from\n            the original object to the new one.  If False (default), the new\n            object will be returned without attributes.\n        keepdims : bool, default: False\n            If True, the dimensions which are reduced are left in the result\n            as dimensions of size one. Coordinates that use these dimensions\n            are removed.\n        **kwargs : dict\n            Additional keyword arguments passed on to `func`.\n\n        Returns\n        -------\n        reduced : DataArray\n            DataArray with this object\'s array replaced with an array with\n            summarized data and the indicated dimension(s) removed.\n        '
        var = self.variable.reduce(func, dim, axis, keep_attrs, keepdims, **kwargs)
        return self._replace_maybe_drop_dims(var)

    def to_pandas(self) -> Self | pd.Series | pd.DataFrame:
        if False:
            print('Hello World!')
        'Convert this array into a pandas object with the same shape.\n\n        The type of the returned object depends on the number of DataArray\n        dimensions:\n\n        * 0D -> `xarray.DataArray`\n        * 1D -> `pandas.Series`\n        * 2D -> `pandas.DataFrame`\n\n        Only works for arrays with 2 or fewer dimensions.\n\n        The DataArray constructor performs the inverse transformation.\n\n        Returns\n        -------\n        result : DataArray | Series | DataFrame\n            DataArray, pandas Series or pandas DataFrame.\n        '
        constructors = {0: lambda x: x, 1: pd.Series, 2: pd.DataFrame}
        try:
            constructor = constructors[self.ndim]
        except KeyError:
            raise ValueError(f'Cannot convert arrays with {self.ndim} dimensions into pandas objects. Requires 2 or fewer dimensions.')
        indexes = [self.get_index(dim) for dim in self.dims]
        return constructor(self.values, *indexes)

    def to_dataframe(self, name: Hashable | None=None, dim_order: Sequence[Hashable] | None=None) -> pd.DataFrame:
        if False:
            print('Hello World!')
        "Convert this array and its coordinates into a tidy pandas.DataFrame.\n\n        The DataFrame is indexed by the Cartesian product of index coordinates\n        (in the form of a :py:class:`pandas.MultiIndex`). Other coordinates are\n        included as columns in the DataFrame.\n\n        For 1D and 2D DataArrays, see also :py:func:`DataArray.to_pandas` which\n        doesn't rely on a MultiIndex to build the DataFrame.\n\n        Parameters\n        ----------\n        name: Hashable or None, optional\n            Name to give to this array (required if unnamed).\n        dim_order: Sequence of Hashable or None, optional\n            Hierarchical dimension order for the resulting dataframe.\n            Array content is transposed to this order and then written out as flat\n            vectors in contiguous order, so the last dimension in this list\n            will be contiguous in the resulting DataFrame. This has a major\n            influence on which operations are efficient on the resulting\n            dataframe.\n\n            If provided, must include all dimensions of this DataArray. By default,\n            dimensions are sorted according to the DataArray dimensions order.\n\n        Returns\n        -------\n        result: DataFrame\n            DataArray as a pandas DataFrame.\n\n        See also\n        --------\n        DataArray.to_pandas\n        DataArray.to_series\n        "
        if name is None:
            name = self.name
        if name is None:
            raise ValueError('cannot convert an unnamed DataArray to a DataFrame: use the ``name`` parameter')
        if self.ndim == 0:
            raise ValueError('cannot convert a scalar to a DataFrame')
        unique_name = '__unique_name_identifier_z98xfz98xugfg73ho__'
        ds = self._to_dataset_whole(name=unique_name)
        if dim_order is None:
            ordered_dims = dict(zip(self.dims, self.shape))
        else:
            ordered_dims = ds._normalize_dim_order(dim_order=dim_order)
        df = ds._to_dataframe(ordered_dims)
        df.columns = [name if c == unique_name else c for c in df.columns]
        return df

    def to_series(self) -> pd.Series:
        if False:
            i = 10
            return i + 15
        'Convert this array into a pandas.Series.\n\n        The Series is indexed by the Cartesian product of index coordinates\n        (in the form of a :py:class:`pandas.MultiIndex`).\n\n        Returns\n        -------\n        result : Series\n            DataArray as a pandas Series.\n\n        See also\n        --------\n        DataArray.to_pandas\n        DataArray.to_dataframe\n        '
        index = self.coords.to_index()
        return pd.Series(self.values.reshape(-1), index=index, name=self.name)

    def to_masked_array(self, copy: bool=True) -> np.ma.MaskedArray:
        if False:
            for i in range(10):
                print('nop')
        'Convert this array into a numpy.ma.MaskedArray\n\n        Parameters\n        ----------\n        copy : bool, default: True\n            If True make a copy of the array in the result. If False,\n            a MaskedArray view of DataArray.values is returned.\n\n        Returns\n        -------\n        result : MaskedArray\n            Masked where invalid values (nan or inf) occur.\n        '
        values = self.to_numpy()
        isnull = pd.isnull(values)
        return np.ma.MaskedArray(data=values, mask=isnull, copy=copy)

    @overload
    def to_netcdf(self, path: None=None, mode: Literal['w', 'a']='w', format: T_NetcdfTypes | None=None, group: str | None=None, engine: T_NetcdfEngine | None=None, encoding: Mapping[Hashable, Mapping[str, Any]] | None=None, unlimited_dims: Iterable[Hashable] | None=None, compute: bool=True, invalid_netcdf: bool=False) -> bytes:
        if False:
            while True:
                i = 10
        ...

    @overload
    def to_netcdf(self, path: str | PathLike, mode: Literal['w', 'a']='w', format: T_NetcdfTypes | None=None, group: str | None=None, engine: T_NetcdfEngine | None=None, encoding: Mapping[Hashable, Mapping[str, Any]] | None=None, unlimited_dims: Iterable[Hashable] | None=None, compute: Literal[True]=True, invalid_netcdf: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def to_netcdf(self, path: str | PathLike, mode: Literal['w', 'a']='w', format: T_NetcdfTypes | None=None, group: str | None=None, engine: T_NetcdfEngine | None=None, encoding: Mapping[Hashable, Mapping[str, Any]] | None=None, unlimited_dims: Iterable[Hashable] | None=None, *, compute: Literal[False], invalid_netcdf: bool=False) -> Delayed:
        if False:
            return 10
        ...

    def to_netcdf(self, path: str | PathLike | None=None, mode: Literal['w', 'a']='w', format: T_NetcdfTypes | None=None, group: str | None=None, engine: T_NetcdfEngine | None=None, encoding: Mapping[Hashable, Mapping[str, Any]] | None=None, unlimited_dims: Iterable[Hashable] | None=None, compute: bool=True, invalid_netcdf: bool=False) -> bytes | Delayed | None:
        if False:
            print('Hello World!')
        'Write DataArray contents to a netCDF file.\n\n        Parameters\n        ----------\n        path : str, path-like or None, optional\n            Path to which to save this dataset. File-like objects are only\n            supported by the scipy engine. If no path is provided, this\n            function returns the resulting netCDF file as bytes; in this case,\n            we need to use scipy, which does not support netCDF version 4 (the\n            default format becomes NETCDF3_64BIT).\n        mode : {"w", "a"}, default: "w"\n            Write (\'w\') or append (\'a\') mode. If mode=\'w\', any existing file at\n            this location will be overwritten. If mode=\'a\', existing variables\n            will be overwritten.\n        format : {"NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT",                   "NETCDF3_CLASSIC"}, optional\n            File format for the resulting netCDF file:\n\n            * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API\n              features.\n            * NETCDF4_CLASSIC: Data is stored in an HDF5 file, using only\n              netCDF 3 compatible API features.\n            * NETCDF3_64BIT: 64-bit offset version of the netCDF 3 file format,\n              which fully supports 2+ GB files, but is only compatible with\n              clients linked against netCDF version 3.6.0 or later.\n            * NETCDF3_CLASSIC: The classic netCDF 3 file format. It does not\n              handle 2+ GB files very well.\n\n            All formats are supported by the netCDF4-python library.\n            scipy.io.netcdf only supports the last two formats.\n\n            The default format is NETCDF4 if you are saving a file to disk and\n            have the netCDF4-python library available. Otherwise, xarray falls\n            back to using scipy to write netCDF files and defaults to the\n            NETCDF3_64BIT format (scipy does not support netCDF4).\n        group : str, optional\n            Path to the netCDF4 group in the given file to open (only works for\n            format=\'NETCDF4\'). The group(s) will be created if necessary.\n        engine : {"netcdf4", "scipy", "h5netcdf"}, optional\n            Engine to use when writing netCDF files. If not provided, the\n            default engine is chosen based on available dependencies, with a\n            preference for \'netcdf4\' if writing to a file on disk.\n        encoding : dict, optional\n            Nested dictionary with variable names as keys and dictionaries of\n            variable specific encodings as values, e.g.,\n            ``{"my_variable": {"dtype": "int16", "scale_factor": 0.1,\n            "zlib": True}, ...}``\n\n            The `h5netcdf` engine supports both the NetCDF4-style compression\n            encoding parameters ``{"zlib": True, "complevel": 9}`` and the h5py\n            ones ``{"compression": "gzip", "compression_opts": 9}``.\n            This allows using any compression plugin installed in the HDF5\n            library, e.g. LZF.\n\n        unlimited_dims : iterable of Hashable, optional\n            Dimension(s) that should be serialized as unlimited dimensions.\n            By default, no dimensions are treated as unlimited dimensions.\n            Note that unlimited_dims may also be set via\n            ``dataset.encoding["unlimited_dims"]``.\n        compute: bool, default: True\n            If true compute immediately, otherwise return a\n            ``dask.delayed.Delayed`` object that can be computed later.\n        invalid_netcdf: bool, default: False\n            Only valid along with ``engine="h5netcdf"``. If True, allow writing\n            hdf5 files which are invalid netcdf as described in\n            https://github.com/h5netcdf/h5netcdf.\n\n        Returns\n        -------\n        store: bytes or Delayed or None\n            * ``bytes`` if path is None\n            * ``dask.delayed.Delayed`` if compute is False\n            * None otherwise\n\n        Notes\n        -----\n        Only xarray.Dataset objects can be written to netCDF files, so\n        the xarray.DataArray is converted to a xarray.Dataset object\n        containing a single variable. If the DataArray has no name, or if the\n        name is the same as a coordinate name, then it is given the name\n        ``"__xarray_dataarray_variable__"``.\n\n        See Also\n        --------\n        Dataset.to_netcdf\n        '
        from xarray.backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE, to_netcdf
        if self.name is None:
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
        elif self.name in self.coords or self.name in self.dims:
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
            dataset.attrs[DATAARRAY_NAME] = self.name
        else:
            dataset = self.to_dataset()
        return to_netcdf(dataset, path, mode=mode, format=format, group=group, engine=engine, encoding=encoding, unlimited_dims=unlimited_dims, compute=compute, multifile=False, invalid_netcdf=invalid_netcdf)

    @overload
    def to_zarr(self, store: MutableMapping | str | PathLike[str] | None=None, chunk_store: MutableMapping | str | PathLike | None=None, mode: Literal['w', 'w-', 'a', 'r+', None]=None, synchronizer=None, group: str | None=None, *, encoding: Mapping | None=None, compute: Literal[True]=True, consolidated: bool | None=None, append_dim: Hashable | None=None, region: Mapping[str, slice] | None=None, safe_chunks: bool=True, storage_options: dict[str, str] | None=None, zarr_version: int | None=None) -> ZarrStore:
        if False:
            return 10
        ...

    @overload
    def to_zarr(self, store: MutableMapping | str | PathLike[str] | None=None, chunk_store: MutableMapping | str | PathLike | None=None, mode: Literal['w', 'w-', 'a', 'r+', None]=None, synchronizer=None, group: str | None=None, encoding: Mapping | None=None, *, compute: Literal[False], consolidated: bool | None=None, append_dim: Hashable | None=None, region: Mapping[str, slice] | None=None, safe_chunks: bool=True, storage_options: dict[str, str] | None=None, zarr_version: int | None=None) -> Delayed:
        if False:
            for i in range(10):
                print('nop')
        ...

    def to_zarr(self, store: MutableMapping | str | PathLike[str] | None=None, chunk_store: MutableMapping | str | PathLike | None=None, mode: Literal['w', 'w-', 'a', 'r+', None]=None, synchronizer=None, group: str | None=None, encoding: Mapping | None=None, *, compute: bool=True, consolidated: bool | None=None, append_dim: Hashable | None=None, region: Mapping[str, slice] | None=None, safe_chunks: bool=True, storage_options: dict[str, str] | None=None, zarr_version: int | None=None) -> ZarrStore | Delayed:
        if False:
            print('Hello World!')
        'Write DataArray contents to a Zarr store\n\n        Zarr chunks are determined in the following way:\n\n        - From the ``chunks`` attribute in each variable\'s ``encoding``\n          (can be set via `DataArray.chunk`).\n        - If the variable is a Dask array, from the dask chunks\n        - If neither Dask chunks nor encoding chunks are present, chunks will\n          be determined automatically by Zarr\n        - If both Dask chunks and encoding chunks are present, encoding chunks\n          will be used, provided that there is a many-to-one relationship between\n          encoding chunks and dask chunks (i.e. Dask chunks are bigger than and\n          evenly divide encoding chunks); otherwise raise a ``ValueError``.\n          This restriction ensures that no synchronization / locks are required\n          when writing. To disable this restriction, use ``safe_chunks=False``.\n\n        Parameters\n        ----------\n        store : MutableMapping, str or path-like, optional\n            Store or path to directory in local or remote file system.\n        chunk_store : MutableMapping, str or path-like, optional\n            Store or path to directory in local or remote file system only for Zarr\n            array chunks. Requires zarr-python v2.4.0 or later.\n        mode : {"w", "w-", "a", "r+", None}, optional\n            Persistence mode: "w" means create (overwrite if exists);\n            "w-" means create (fail if exists);\n            "a" means override existing variables (create if does not exist);\n            "r+" means modify existing array *values* only (raise an error if\n            any metadata or shapes would change).\n            The default mode is "a" if ``append_dim`` is set. Otherwise, it is\n            "r+" if ``region`` is set and ``w-`` otherwise.\n        synchronizer : object, optional\n            Zarr array synchronizer.\n        group : str, optional\n            Group path. (a.k.a. `path` in zarr terminology.)\n        encoding : dict, optional\n            Nested dictionary with variable names as keys and dictionaries of\n            variable specific encodings as values, e.g.,\n            ``{"my_variable": {"dtype": "int16", "scale_factor": 0.1,}, ...}``\n        compute : bool, default: True\n            If True write array data immediately, otherwise return a\n            ``dask.delayed.Delayed`` object that can be computed to write\n            array data later. Metadata is always updated eagerly.\n        consolidated : bool, optional\n            If True, apply zarr\'s `consolidate_metadata` function to the store\n            after writing metadata and read existing stores with consolidated\n            metadata; if False, do not. The default (`consolidated=None`) means\n            write consolidated metadata and attempt to read consolidated\n            metadata for existing stores (falling back to non-consolidated).\n\n            When the experimental ``zarr_version=3``, ``consolidated`` must be\n            either be ``None`` or ``False``.\n        append_dim : hashable, optional\n            If set, the dimension along which the data will be appended. All\n            other dimensions on overridden variables must remain the same size.\n        region : dict, optional\n            Optional mapping from dimension names to integer slices along\n            dataarray dimensions to indicate the region of existing zarr array(s)\n            in which to write this datarray\'s data. For example,\n            ``{\'x\': slice(0, 1000), \'y\': slice(10000, 11000)}`` would indicate\n            that values should be written to the region ``0:1000`` along ``x``\n            and ``10000:11000`` along ``y``.\n\n            Two restrictions apply to the use of ``region``:\n\n            - If ``region`` is set, _all_ variables in a dataarray must have at\n              least one dimension in common with the region. Other variables\n              should be written in a separate call to ``to_zarr()``.\n            - Dimensions cannot be included in both ``region`` and\n              ``append_dim`` at the same time. To create empty arrays to fill\n              in with ``region``, use a separate call to ``to_zarr()`` with\n              ``compute=False``. See "Appending to existing Zarr stores" in\n              the reference documentation for full details.\n        safe_chunks : bool, default: True\n            If True, only allow writes to when there is a many-to-one relationship\n            between Zarr chunks (specified in encoding) and Dask chunks.\n            Set False to override this restriction; however, data may become corrupted\n            if Zarr arrays are written in parallel. This option may be useful in combination\n            with ``compute=False`` to initialize a Zarr store from an existing\n            DataArray with arbitrary chunk structure.\n        storage_options : dict, optional\n            Any additional parameters for the storage backend (ignored for local\n            paths).\n        zarr_version : int or None, optional\n            The desired zarr spec version to target (currently 2 or 3). The\n            default of None will attempt to determine the zarr version from\n            ``store`` when possible, otherwise defaulting to 2.\n\n        Returns\n        -------\n            * ``dask.delayed.Delayed`` if compute is False\n            * ZarrStore otherwise\n\n        References\n        ----------\n        https://zarr.readthedocs.io/\n\n        Notes\n        -----\n        Zarr chunking behavior:\n            If chunks are found in the encoding argument or attribute\n            corresponding to any DataArray, those chunks are used.\n            If a DataArray is a dask array, it is written with those chunks.\n            If not other chunks are found, Zarr uses its own heuristics to\n            choose automatic chunk sizes.\n\n        encoding:\n            The encoding attribute (if exists) of the DataArray(s) will be\n            used. Override any existing encodings by providing the ``encoding`` kwarg.\n\n        See Also\n        --------\n        Dataset.to_zarr\n        :ref:`io.zarr`\n            The I/O user guide, with more details and examples.\n        '
        from xarray.backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE, to_zarr
        if self.name is None:
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
        elif self.name in self.coords or self.name in self.dims:
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
            dataset.attrs[DATAARRAY_NAME] = self.name
        else:
            dataset = self.to_dataset()
        return to_zarr(dataset, store=store, chunk_store=chunk_store, mode=mode, synchronizer=synchronizer, group=group, encoding=encoding, compute=compute, consolidated=consolidated, append_dim=append_dim, region=region, safe_chunks=safe_chunks, storage_options=storage_options, zarr_version=zarr_version)

    def to_dict(self, data: bool | Literal['list', 'array']='list', encoding: bool=False) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert this xarray.DataArray into a dictionary following xarray\n        naming conventions.\n\n        Converts all variables and attributes to native Python objects.\n        Useful for converting to json. To avoid datetime incompatibility\n        use decode_times=False kwarg in xarray.open_dataset.\n\n        Parameters\n        ----------\n        data : bool or {"list", "array"}, default: "list"\n            Whether to include the actual data in the dictionary. When set to\n            False, returns just the schema. If set to "array", returns data as\n            underlying array type. If set to "list" (or True for backwards\n            compatibility), returns data in lists of Python data types. Note\n            that for obtaining the "list" output efficiently, use\n            `da.compute().to_dict(data="list")`.\n\n        encoding : bool, default: False\n            Whether to include the Dataset\'s encoding in the dictionary.\n\n        Returns\n        -------\n        dict: dict\n\n        See Also\n        --------\n        DataArray.from_dict\n        Dataset.to_dict\n        '
        d = self.variable.to_dict(data=data)
        d.update({'coords': {}, 'name': self.name})
        for (k, coord) in self.coords.items():
            d['coords'][k] = coord.variable.to_dict(data=data)
        if encoding:
            d['encoding'] = dict(self.encoding)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        if False:
            i = 10
            return i + 15
        'Convert a dictionary into an xarray.DataArray\n\n        Parameters\n        ----------\n        d : dict\n            Mapping with a minimum structure of {"dims": [...], "data": [...]}\n\n        Returns\n        -------\n        obj : xarray.DataArray\n\n        See Also\n        --------\n        DataArray.to_dict\n        Dataset.from_dict\n\n        Examples\n        --------\n        >>> d = {"dims": "t", "data": [1, 2, 3]}\n        >>> da = xr.DataArray.from_dict(d)\n        >>> da\n        <xarray.DataArray (t: 3)>\n        array([1, 2, 3])\n        Dimensions without coordinates: t\n\n        >>> d = {\n        ...     "coords": {\n        ...         "t": {"dims": "t", "data": [0, 1, 2], "attrs": {"units": "s"}}\n        ...     },\n        ...     "attrs": {"title": "air temperature"},\n        ...     "dims": "t",\n        ...     "data": [10, 20, 30],\n        ...     "name": "a",\n        ... }\n        >>> da = xr.DataArray.from_dict(d)\n        >>> da\n        <xarray.DataArray \'a\' (t: 3)>\n        array([10, 20, 30])\n        Coordinates:\n          * t        (t) int64 0 1 2\n        Attributes:\n            title:    air temperature\n        '
        coords = None
        if 'coords' in d:
            try:
                coords = {k: (v['dims'], v['data'], v.get('attrs')) for (k, v) in d['coords'].items()}
            except KeyError as e:
                raise ValueError(f"cannot convert dict when coords are missing the key '{str(e.args[0])}'")
        try:
            data = d['data']
        except KeyError:
            raise ValueError("cannot convert dict without the key 'data''")
        else:
            obj = cls(data, coords, d.get('dims'), d.get('name'), d.get('attrs'))
        obj.encoding.update(d.get('encoding', {}))
        return obj

    @classmethod
    def from_series(cls, series: pd.Series, sparse: bool=False) -> DataArray:
        if False:
            for i in range(10):
                print('nop')
        "Convert a pandas.Series into an xarray.DataArray.\n\n        If the series's index is a MultiIndex, it will be expanded into a\n        tensor product of one-dimensional coordinates (filling in missing\n        values with NaN). Thus this operation should be the inverse of the\n        `to_series` method.\n\n        Parameters\n        ----------\n        series : Series\n            Pandas Series object to convert.\n        sparse : bool, default: False\n            If sparse=True, creates a sparse array instead of a dense NumPy array.\n            Requires the pydata/sparse package.\n\n        See Also\n        --------\n        DataArray.to_series\n        Dataset.from_dataframe\n        "
        temp_name = '__temporary_name'
        df = pd.DataFrame({temp_name: series})
        ds = Dataset.from_dataframe(df, sparse=sparse)
        result = ds[temp_name]
        result.name = series.name
        return result

    def to_iris(self) -> iris_Cube:
        if False:
            return 10
        'Convert this array into a iris.cube.Cube'
        from xarray.convert import to_iris
        return to_iris(self)

    @classmethod
    def from_iris(cls, cube: iris_Cube) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Convert a iris.cube.Cube into an xarray.DataArray'
        from xarray.convert import from_iris
        return from_iris(cube)

    def _all_compat(self, other: Self, compat_str: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Helper function for equals, broadcast_equals, and identical'

        def compat(x, y):
            if False:
                i = 10
                return i + 15
            return getattr(x.variable, compat_str)(y.variable)
        return utils.dict_equiv(self.coords, other.coords, compat=compat) and compat(self, other)

    def broadcast_equals(self, other: Self) -> bool:
        if False:
            i = 10
            return i + 15
        'Two DataArrays are broadcast equal if they are equal after\n        broadcasting them against each other such that they have the same\n        dimensions.\n\n        Parameters\n        ----------\n        other : DataArray\n            DataArray to compare to.\n\n        Returns\n        ----------\n        equal : bool\n            True if the two DataArrays are broadcast equal.\n\n        See Also\n        --------\n        DataArray.equals\n        DataArray.identical\n\n        Examples\n        --------\n        >>> a = xr.DataArray([1, 2], dims="X")\n        >>> b = xr.DataArray([[1, 1], [2, 2]], dims=["X", "Y"])\n        >>> a\n        <xarray.DataArray (X: 2)>\n        array([1, 2])\n        Dimensions without coordinates: X\n        >>> b\n        <xarray.DataArray (X: 2, Y: 2)>\n        array([[1, 1],\n               [2, 2]])\n        Dimensions without coordinates: X, Y\n\n        .equals returns True if two DataArrays have the same values, dimensions, and coordinates. .broadcast_equals returns True if the results of broadcasting two DataArrays against each other have the same values, dimensions, and coordinates.\n\n        >>> a.equals(b)\n        False\n        >>> a2, b2 = xr.broadcast(a, b)\n        >>> a2.equals(b2)\n        True\n        >>> a.broadcast_equals(b)\n        True\n        '
        try:
            return self._all_compat(other, 'broadcast_equals')
        except (TypeError, AttributeError):
            return False

    def equals(self, other: Self) -> bool:
        if False:
            while True:
                i = 10
        'True if two DataArrays have the same dimensions, coordinates and\n        values; otherwise False.\n\n        DataArrays can still be equal (like pandas objects) if they have NaN\n        values in the same locations.\n\n        This method is necessary because `v1 == v2` for ``DataArray``\n        does element-wise comparisons (like numpy.ndarrays).\n\n        Parameters\n        ----------\n        other : DataArray\n            DataArray to compare to.\n\n        Returns\n        ----------\n        equal : bool\n            True if the two DataArrays are equal.\n\n        See Also\n        --------\n        DataArray.broadcast_equals\n        DataArray.identical\n\n        Examples\n        --------\n        >>> a = xr.DataArray([1, 2, 3], dims="X")\n        >>> b = xr.DataArray([1, 2, 3], dims="X", attrs=dict(units="m"))\n        >>> c = xr.DataArray([1, 2, 3], dims="Y")\n        >>> d = xr.DataArray([3, 2, 1], dims="X")\n        >>> a\n        <xarray.DataArray (X: 3)>\n        array([1, 2, 3])\n        Dimensions without coordinates: X\n        >>> b\n        <xarray.DataArray (X: 3)>\n        array([1, 2, 3])\n        Dimensions without coordinates: X\n        Attributes:\n            units:    m\n        >>> c\n        <xarray.DataArray (Y: 3)>\n        array([1, 2, 3])\n        Dimensions without coordinates: Y\n        >>> d\n        <xarray.DataArray (X: 3)>\n        array([3, 2, 1])\n        Dimensions without coordinates: X\n\n        >>> a.equals(b)\n        True\n        >>> a.equals(c)\n        False\n        >>> a.equals(d)\n        False\n        '
        try:
            return self._all_compat(other, 'equals')
        except (TypeError, AttributeError):
            return False

    def identical(self, other: Self) -> bool:
        if False:
            print('Hello World!')
        'Like equals, but also checks the array name and attributes, and\n        attributes on all coordinates.\n\n        Parameters\n        ----------\n        other : DataArray\n            DataArray to compare to.\n\n        Returns\n        ----------\n        equal : bool\n            True if the two DataArrays are identical.\n\n        See Also\n        --------\n        DataArray.broadcast_equals\n        DataArray.equals\n\n        Examples\n        --------\n        >>> a = xr.DataArray([1, 2, 3], dims="X", attrs=dict(units="m"), name="Width")\n        >>> b = xr.DataArray([1, 2, 3], dims="X", attrs=dict(units="m"), name="Width")\n        >>> c = xr.DataArray([1, 2, 3], dims="X", attrs=dict(units="ft"), name="Width")\n        >>> a\n        <xarray.DataArray \'Width\' (X: 3)>\n        array([1, 2, 3])\n        Dimensions without coordinates: X\n        Attributes:\n            units:    m\n        >>> b\n        <xarray.DataArray \'Width\' (X: 3)>\n        array([1, 2, 3])\n        Dimensions without coordinates: X\n        Attributes:\n            units:    m\n        >>> c\n        <xarray.DataArray \'Width\' (X: 3)>\n        array([1, 2, 3])\n        Dimensions without coordinates: X\n        Attributes:\n            units:    ft\n\n        >>> a.equals(b)\n        True\n        >>> a.identical(b)\n        True\n\n        >>> a.equals(c)\n        True\n        >>> a.identical(c)\n        False\n        '
        try:
            return self.name == other.name and self._all_compat(other, 'identical')
        except (TypeError, AttributeError):
            return False

    def _result_name(self, other: Any=None) -> Hashable | None:
        if False:
            for i in range(10):
                print('nop')
        other_name = getattr(other, 'name', _default)
        if other_name is _default or other_name == self.name:
            return self.name
        else:
            return None

    def __array_wrap__(self, obj, context=None) -> Self:
        if False:
            print('Hello World!')
        new_var = self.variable.__array_wrap__(obj, context)
        return self._replace(new_var)

    def __matmul__(self, obj: T_Xarray) -> T_Xarray:
        if False:
            i = 10
            return i + 15
        return self.dot(obj)

    def __rmatmul__(self, other: T_Xarray) -> T_Xarray:
        if False:
            print('Hello World!')
        return computation.dot(other, self)

    def _unary_op(self, f: Callable, *args, **kwargs) -> Self:
        if False:
            return 10
        keep_attrs = kwargs.pop('keep_attrs', None)
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
            warnings.filterwarnings('ignore', 'Mean of empty slice', category=RuntimeWarning)
            with np.errstate(all='ignore'):
                da = self.__array_wrap__(f(self.variable.data, *args, **kwargs))
            if keep_attrs:
                da.attrs = self.attrs
            return da

    def _binary_op(self, other: DaCompatible, f: Callable, reflexive: bool=False) -> Self:
        if False:
            while True:
                i = 10
        from xarray.core.groupby import GroupBy
        if isinstance(other, (Dataset, GroupBy)):
            return NotImplemented
        if isinstance(other, DataArray):
            align_type = OPTIONS['arithmetic_join']
            (self, other) = align(self, other, join=align_type, copy=False)
        other_variable_or_arraylike: DaCompatible = getattr(other, 'variable', other)
        other_coords = getattr(other, 'coords', None)
        variable = f(self.variable, other_variable_or_arraylike) if not reflexive else f(other_variable_or_arraylike, self.variable)
        (coords, indexes) = self.coords._merge_raw(other_coords, reflexive)
        name = self._result_name(other)
        return self._replace(variable, coords, name, indexes=indexes)

    def _inplace_binary_op(self, other: DaCompatible, f: Callable) -> Self:
        if False:
            while True:
                i = 10
        from xarray.core.groupby import GroupBy
        if isinstance(other, GroupBy):
            raise TypeError('in-place operations between a DataArray and a grouped object are not permitted')
        other_coords = getattr(other, 'coords', None)
        other_variable = getattr(other, 'variable', other)
        try:
            with self.coords._merge_inplace(other_coords):
                f(self.variable, other_variable)
        except MergeError as exc:
            raise MergeError('Automatic alignment is not supported for in-place operations.\nConsider aligning the indices manually or using a not-in-place operation.\nSee https://github.com/pydata/xarray/issues/3910 for more explanations.') from exc
        return self

    def _copy_attrs_from(self, other: DataArray | Dataset | Variable) -> None:
        if False:
            while True:
                i = 10
        self.attrs = other.attrs
    plot = utils.UncachedAccessor(DataArrayPlotAccessor)

    def _title_for_slice(self, truncate: int=50) -> str:
        if False:
            i = 10
            return i + 15
        '\n        If the dataarray has 1 dimensional coordinates or comes from a slice\n        we can show that info in the title\n\n        Parameters\n        ----------\n        truncate : int, default: 50\n            maximum number of characters for title\n\n        Returns\n        -------\n        title : string\n            Can be used for plot titles\n\n        '
        one_dims = []
        for (dim, coord) in self.coords.items():
            if coord.size == 1:
                one_dims.append(f'{dim} = {format_item(coord.values)}{_get_units_from_attrs(coord)}')
        title = ', '.join(one_dims)
        if len(title) > truncate:
            title = title[:truncate - 3] + '...'
        return title

    @_deprecate_positional_args('v2023.10.0')
    def diff(self, dim: Hashable, n: int=1, *, label: Literal['upper', 'lower']='upper') -> Self:
        if False:
            while True:
                i = 10
        'Calculate the n-th order discrete difference along given axis.\n\n        Parameters\n        ----------\n        dim : Hashable\n            Dimension over which to calculate the finite difference.\n        n : int, default: 1\n            The number of times values are differenced.\n        label : {"upper", "lower"}, default: "upper"\n            The new coordinate in dimension ``dim`` will have the\n            values of either the minuend\'s or subtrahend\'s coordinate\n            for values \'upper\' and \'lower\', respectively.\n\n        Returns\n        -------\n        difference : DataArray\n            The n-th order finite difference of this object.\n\n        Notes\n        -----\n        `n` matches numpy\'s behavior and is different from pandas\' first argument named\n        `periods`.\n\n        Examples\n        --------\n        >>> arr = xr.DataArray([5, 5, 6, 6], [[1, 2, 3, 4]], ["x"])\n        >>> arr.diff("x")\n        <xarray.DataArray (x: 3)>\n        array([0, 1, 0])\n        Coordinates:\n          * x        (x) int64 2 3 4\n        >>> arr.diff("x", 2)\n        <xarray.DataArray (x: 2)>\n        array([ 1, -1])\n        Coordinates:\n          * x        (x) int64 3 4\n\n        See Also\n        --------\n        DataArray.differentiate\n        '
        ds = self._to_temp_dataset().diff(n=n, dim=dim, label=label)
        return self._from_temp_dataset(ds)

    def shift(self, shifts: Mapping[Any, int] | None=None, fill_value: Any=dtypes.NA, **shifts_kwargs: int) -> Self:
        if False:
            return 10
        'Shift this DataArray by an offset along one or more dimensions.\n\n        Only the data is moved; coordinates stay in place. This is consistent\n        with the behavior of ``shift`` in pandas.\n\n        Values shifted from beyond array bounds will appear at one end of\n        each dimension, which are filled according to `fill_value`. For periodic\n        offsets instead see `roll`.\n\n        Parameters\n        ----------\n        shifts : mapping of Hashable to int or None, optional\n            Integer offset to shift along each of the given dimensions.\n            Positive offsets shift to the right; negative offsets shift to the\n            left.\n        fill_value : scalar, optional\n            Value to use for newly missing values\n        **shifts_kwargs\n            The keyword arguments form of ``shifts``.\n            One of shifts or shifts_kwargs must be provided.\n\n        Returns\n        -------\n        shifted : DataArray\n            DataArray with the same coordinates and attributes but shifted\n            data.\n\n        See Also\n        --------\n        roll\n\n        Examples\n        --------\n        >>> arr = xr.DataArray([5, 6, 7], dims="x")\n        >>> arr.shift(x=1)\n        <xarray.DataArray (x: 3)>\n        array([nan,  5.,  6.])\n        Dimensions without coordinates: x\n        '
        variable = self.variable.shift(shifts=shifts, fill_value=fill_value, **shifts_kwargs)
        return self._replace(variable=variable)

    def roll(self, shifts: Mapping[Hashable, int] | None=None, roll_coords: bool=False, **shifts_kwargs: int) -> Self:
        if False:
            return 10
        'Roll this array by an offset along one or more dimensions.\n\n        Unlike shift, roll treats the given dimensions as periodic, so will not\n        create any missing values to be filled.\n\n        Unlike shift, roll may rotate all variables, including coordinates\n        if specified. The direction of rotation is consistent with\n        :py:func:`numpy.roll`.\n\n        Parameters\n        ----------\n        shifts : mapping of Hashable to int, optional\n            Integer offset to rotate each of the given dimensions.\n            Positive offsets roll to the right; negative offsets roll to the\n            left.\n        roll_coords : bool, default: False\n            Indicates whether to roll the coordinates by the offset too.\n        **shifts_kwargs : {dim: offset, ...}, optional\n            The keyword arguments form of ``shifts``.\n            One of shifts or shifts_kwargs must be provided.\n\n        Returns\n        -------\n        rolled : DataArray\n            DataArray with the same attributes but rolled data and coordinates.\n\n        See Also\n        --------\n        shift\n\n        Examples\n        --------\n        >>> arr = xr.DataArray([5, 6, 7], dims="x")\n        >>> arr.roll(x=1)\n        <xarray.DataArray (x: 3)>\n        array([7, 5, 6])\n        Dimensions without coordinates: x\n        '
        ds = self._to_temp_dataset().roll(shifts=shifts, roll_coords=roll_coords, **shifts_kwargs)
        return self._from_temp_dataset(ds)

    @property
    def real(self) -> Self:
        if False:
            print('Hello World!')
        '\n        The real part of the array.\n\n        See Also\n        --------\n        numpy.ndarray.real\n        '
        return self._replace(self.variable.real)

    @property
    def imag(self) -> Self:
        if False:
            while True:
                i = 10
        '\n        The imaginary part of the array.\n\n        See Also\n        --------\n        numpy.ndarray.imag\n        '
        return self._replace(self.variable.imag)

    def dot(self, other: T_Xarray, dims: Dims=None) -> T_Xarray:
        if False:
            print('Hello World!')
        'Perform dot product of two DataArrays along their shared dims.\n\n        Equivalent to taking taking tensordot over all shared dims.\n\n        Parameters\n        ----------\n        other : DataArray\n            The other array with which the dot product is performed.\n        dims : ..., str, Iterable of Hashable or None, optional\n            Which dimensions to sum over. Ellipsis (`...`) sums over all dimensions.\n            If not specified, then all the common dimensions are summed over.\n\n        Returns\n        -------\n        result : DataArray\n            Array resulting from the dot product over all shared dimensions.\n\n        See Also\n        --------\n        dot\n        numpy.tensordot\n\n        Examples\n        --------\n        >>> da_vals = np.arange(6 * 5 * 4).reshape((6, 5, 4))\n        >>> da = xr.DataArray(da_vals, dims=["x", "y", "z"])\n        >>> dm_vals = np.arange(4)\n        >>> dm = xr.DataArray(dm_vals, dims=["z"])\n\n        >>> dm.dims\n        (\'z\',)\n\n        >>> da.dims\n        (\'x\', \'y\', \'z\')\n\n        >>> dot_result = da.dot(dm)\n        >>> dot_result.dims\n        (\'x\', \'y\')\n\n        '
        if isinstance(other, Dataset):
            raise NotImplementedError('dot products are not yet supported with Dataset objects.')
        if not isinstance(other, DataArray):
            raise TypeError('dot only operates on DataArrays.')
        return computation.dot(self, other, dims=dims)

    def sortby(self, variables: Hashable | DataArray | Sequence[Hashable | DataArray] | Callable[[Self], Hashable | DataArray | Sequence[Hashable | DataArray]], ascending: bool=True) -> Self:
        if False:
            i = 10
            return i + 15
        'Sort object by labels or values (along an axis).\n\n        Sorts the dataarray, either along specified dimensions,\n        or according to values of 1-D dataarrays that share dimension\n        with calling object.\n\n        If the input variables are dataarrays, then the dataarrays are aligned\n        (via left-join) to the calling object prior to sorting by cell values.\n        NaNs are sorted to the end, following Numpy convention.\n\n        If multiple sorts along the same dimension is\n        given, numpy\'s lexsort is performed along that dimension:\n        https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html\n        and the FIRST key in the sequence is used as the primary sort key,\n        followed by the 2nd key, etc.\n\n        Parameters\n        ----------\n        variables : Hashable, DataArray, sequence of Hashable or DataArray, or Callable\n            1D DataArray objects or name(s) of 1D variable(s) in coords whose values are\n            used to sort this array. If a callable, the callable is passed this object,\n            and the result is used as the value for cond.\n        ascending : bool, default: True\n            Whether to sort by ascending or descending order.\n\n        Returns\n        -------\n        sorted : DataArray\n            A new dataarray where all the specified dims are sorted by dim\n            labels.\n\n        See Also\n        --------\n        Dataset.sortby\n        numpy.sort\n        pandas.sort_values\n        pandas.sort_index\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     np.arange(5, 0, -1),\n        ...     coords=[pd.date_range("1/1/2000", periods=5)],\n        ...     dims="time",\n        ... )\n        >>> da\n        <xarray.DataArray (time: 5)>\n        array([5, 4, 3, 2, 1])\n        Coordinates:\n          * time     (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2000-01-05\n\n        >>> da.sortby(da)\n        <xarray.DataArray (time: 5)>\n        array([1, 2, 3, 4, 5])\n        Coordinates:\n          * time     (time) datetime64[ns] 2000-01-05 2000-01-04 ... 2000-01-01\n\n        >>> da.sortby(lambda x: x)\n        <xarray.DataArray (time: 5)>\n        array([1, 2, 3, 4, 5])\n        Coordinates:\n          * time     (time) datetime64[ns] 2000-01-05 2000-01-04 ... 2000-01-01\n        '
        if callable(variables):
            variables = variables(self)
        ds = self._to_temp_dataset().sortby(variables, ascending=ascending)
        return self._from_temp_dataset(ds)

    @_deprecate_positional_args('v2023.10.0')
    def quantile(self, q: ArrayLike, dim: Dims=None, *, method: QuantileMethods='linear', keep_attrs: bool | None=None, skipna: bool | None=None, interpolation: QuantileMethods | None=None) -> Self:
        if False:
            i = 10
            return i + 15
        'Compute the qth quantile of the data along the specified dimension.\n\n        Returns the qth quantiles(s) of the array elements.\n\n        Parameters\n        ----------\n        q : float or array-like of float\n            Quantile to compute, which must be between 0 and 1 inclusive.\n        dim : str or Iterable of Hashable, optional\n            Dimension(s) over which to apply quantile.\n        method : str, default: "linear"\n            This optional parameter specifies the interpolation method to use when the\n            desired quantile lies between two data points. The options sorted by their R\n            type as summarized in the H&F paper [1]_ are:\n\n                1. "inverted_cdf"\n                2. "averaged_inverted_cdf"\n                3. "closest_observation"\n                4. "interpolated_inverted_cdf"\n                5. "hazen"\n                6. "weibull"\n                7. "linear"  (default)\n                8. "median_unbiased"\n                9. "normal_unbiased"\n\n            The first three methods are discontiuous. The following discontinuous\n            variations of the default "linear" (7.) option are also available:\n\n                * "lower"\n                * "higher"\n                * "midpoint"\n                * "nearest"\n\n            See :py:func:`numpy.quantile` or [1]_ for details. The "method" argument\n            was previously called "interpolation", renamed in accordance with numpy\n            version 1.22.0.\n\n        keep_attrs : bool or None, optional\n            If True, the dataset\'s attributes (`attrs`) will be copied from\n            the original object to the new one.  If False (default), the new\n            object will be returned without attributes.\n        skipna : bool or None, optional\n            If True, skip missing values (as marked by NaN). By default, only\n            skips missing values for float dtypes; other dtypes either do not\n            have a sentinel missing value (int) or skipna=True has not been\n            implemented (object, datetime64 or timedelta64).\n\n        Returns\n        -------\n        quantiles : DataArray\n            If `q` is a single quantile, then the result\n            is a scalar. If multiple percentiles are given, first axis of\n            the result corresponds to the quantile and a quantile dimension\n            is added to the return array. The other dimensions are the\n            dimensions that remain after the reduction of the array.\n\n        See Also\n        --------\n        numpy.nanquantile, numpy.quantile, pandas.Series.quantile, Dataset.quantile\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     data=[[0.7, 4.2, 9.4, 1.5], [6.5, 7.3, 2.6, 1.9]],\n        ...     coords={"x": [7, 9], "y": [1, 1.5, 2, 2.5]},\n        ...     dims=("x", "y"),\n        ... )\n        >>> da.quantile(0)  # or da.quantile(0, dim=...)\n        <xarray.DataArray ()>\n        array(0.7)\n        Coordinates:\n            quantile  float64 0.0\n        >>> da.quantile(0, dim="x")\n        <xarray.DataArray (y: 4)>\n        array([0.7, 4.2, 2.6, 1.5])\n        Coordinates:\n          * y         (y) float64 1.0 1.5 2.0 2.5\n            quantile  float64 0.0\n        >>> da.quantile([0, 0.5, 1])\n        <xarray.DataArray (quantile: 3)>\n        array([0.7, 3.4, 9.4])\n        Coordinates:\n          * quantile  (quantile) float64 0.0 0.5 1.0\n        >>> da.quantile([0, 0.5, 1], dim="x")\n        <xarray.DataArray (quantile: 3, y: 4)>\n        array([[0.7 , 4.2 , 2.6 , 1.5 ],\n               [3.6 , 5.75, 6.  , 1.7 ],\n               [6.5 , 7.3 , 9.4 , 1.9 ]])\n        Coordinates:\n          * y         (y) float64 1.0 1.5 2.0 2.5\n          * quantile  (quantile) float64 0.0 0.5 1.0\n\n        References\n        ----------\n        .. [1] R. J. Hyndman and Y. Fan,\n           "Sample quantiles in statistical packages,"\n           The American Statistician, 50(4), pp. 361-365, 1996\n        '
        ds = self._to_temp_dataset().quantile(q, dim=dim, keep_attrs=keep_attrs, method=method, skipna=skipna, interpolation=interpolation)
        return self._from_temp_dataset(ds)

    @_deprecate_positional_args('v2023.10.0')
    def rank(self, dim: Hashable, *, pct: bool=False, keep_attrs: bool | None=None) -> Self:
        if False:
            i = 10
            return i + 15
        'Ranks the data.\n\n        Equal values are assigned a rank that is the average of the ranks that\n        would have been otherwise assigned to all of the values within that\n        set.  Ranks begin at 1, not 0. If pct, computes percentage ranks.\n\n        NaNs in the input array are returned as NaNs.\n\n        The `bottleneck` library is required.\n\n        Parameters\n        ----------\n        dim : Hashable\n            Dimension over which to compute rank.\n        pct : bool, default: False\n            If True, compute percentage ranks, otherwise compute integer ranks.\n        keep_attrs : bool or None, optional\n            If True, the dataset\'s attributes (`attrs`) will be copied from\n            the original object to the new one.  If False (default), the new\n            object will be returned without attributes.\n\n        Returns\n        -------\n        ranked : DataArray\n            DataArray with the same coordinates and dtype \'float64\'.\n\n        Examples\n        --------\n        >>> arr = xr.DataArray([5, 6, 7], dims="x")\n        >>> arr.rank("x")\n        <xarray.DataArray (x: 3)>\n        array([1., 2., 3.])\n        Dimensions without coordinates: x\n        '
        ds = self._to_temp_dataset().rank(dim, pct=pct, keep_attrs=keep_attrs)
        return self._from_temp_dataset(ds)

    def differentiate(self, coord: Hashable, edge_order: Literal[1, 2]=1, datetime_unit: DatetimeUnitOptions=None) -> Self:
        if False:
            return 10
        ' Differentiate the array with the second order accurate central\n        differences.\n\n        .. note::\n            This feature is limited to simple cartesian geometry, i.e. coord\n            must be one dimensional.\n\n        Parameters\n        ----------\n        coord : Hashable\n            The coordinate to be used to compute the gradient.\n        edge_order : {1, 2}, default: 1\n            N-th order accurate differences at the boundaries.\n        datetime_unit : {"W", "D", "h", "m", "s", "ms",                          "us", "ns", "ps", "fs", "as", None}, optional\n            Unit to compute gradient. Only valid for datetime coordinate. "Y" and "M" are not available as\n            datetime_unit.\n\n        Returns\n        -------\n        differentiated: DataArray\n\n        See also\n        --------\n        numpy.gradient: corresponding numpy function\n\n        Examples\n        --------\n\n        >>> da = xr.DataArray(\n        ...     np.arange(12).reshape(4, 3),\n        ...     dims=["x", "y"],\n        ...     coords={"x": [0, 0.1, 1.1, 1.2]},\n        ... )\n        >>> da\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 0,  1,  2],\n               [ 3,  4,  5],\n               [ 6,  7,  8],\n               [ 9, 10, 11]])\n        Coordinates:\n          * x        (x) float64 0.0 0.1 1.1 1.2\n        Dimensions without coordinates: y\n        >>>\n        >>> da.differentiate("x")\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[30.        , 30.        , 30.        ],\n               [27.54545455, 27.54545455, 27.54545455],\n               [27.54545455, 27.54545455, 27.54545455],\n               [30.        , 30.        , 30.        ]])\n        Coordinates:\n          * x        (x) float64 0.0 0.1 1.1 1.2\n        Dimensions without coordinates: y\n        '
        ds = self._to_temp_dataset().differentiate(coord, edge_order, datetime_unit)
        return self._from_temp_dataset(ds)

    def integrate(self, coord: Hashable | Sequence[Hashable]=None, datetime_unit: DatetimeUnitOptions=None) -> Self:
        if False:
            while True:
                i = 10
        'Integrate along the given coordinate using the trapezoidal rule.\n\n        .. note::\n            This feature is limited to simple cartesian geometry, i.e. coord\n            must be one dimensional.\n\n        Parameters\n        ----------\n        coord : Hashable, or sequence of Hashable\n            Coordinate(s) used for the integration.\n        datetime_unit : {\'Y\', \'M\', \'W\', \'D\', \'h\', \'m\', \'s\', \'ms\', \'us\', \'ns\',                         \'ps\', \'fs\', \'as\', None}, optional\n            Specify the unit if a datetime coordinate is used.\n\n        Returns\n        -------\n        integrated : DataArray\n\n        See also\n        --------\n        Dataset.integrate\n        numpy.trapz : corresponding numpy function\n\n        Examples\n        --------\n\n        >>> da = xr.DataArray(\n        ...     np.arange(12).reshape(4, 3),\n        ...     dims=["x", "y"],\n        ...     coords={"x": [0, 0.1, 1.1, 1.2]},\n        ... )\n        >>> da\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 0,  1,  2],\n               [ 3,  4,  5],\n               [ 6,  7,  8],\n               [ 9, 10, 11]])\n        Coordinates:\n          * x        (x) float64 0.0 0.1 1.1 1.2\n        Dimensions without coordinates: y\n        >>>\n        >>> da.integrate("x")\n        <xarray.DataArray (y: 3)>\n        array([5.4, 6.6, 7.8])\n        Dimensions without coordinates: y\n        '
        ds = self._to_temp_dataset().integrate(coord, datetime_unit)
        return self._from_temp_dataset(ds)

    def cumulative_integrate(self, coord: Hashable | Sequence[Hashable]=None, datetime_unit: DatetimeUnitOptions=None) -> Self:
        if False:
            i = 10
            return i + 15
        'Integrate cumulatively along the given coordinate using the trapezoidal rule.\n\n        .. note::\n            This feature is limited to simple cartesian geometry, i.e. coord\n            must be one dimensional.\n\n            The first entry of the cumulative integral is always 0, in order to keep the\n            length of the dimension unchanged between input and output.\n\n        Parameters\n        ----------\n        coord : Hashable, or sequence of Hashable\n            Coordinate(s) used for the integration.\n        datetime_unit : {\'Y\', \'M\', \'W\', \'D\', \'h\', \'m\', \'s\', \'ms\', \'us\', \'ns\',                         \'ps\', \'fs\', \'as\', None}, optional\n            Specify the unit if a datetime coordinate is used.\n\n        Returns\n        -------\n        integrated : DataArray\n\n        See also\n        --------\n        Dataset.cumulative_integrate\n        scipy.integrate.cumulative_trapezoid : corresponding scipy function\n\n        Examples\n        --------\n\n        >>> da = xr.DataArray(\n        ...     np.arange(12).reshape(4, 3),\n        ...     dims=["x", "y"],\n        ...     coords={"x": [0, 0.1, 1.1, 1.2]},\n        ... )\n        >>> da\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[ 0,  1,  2],\n               [ 3,  4,  5],\n               [ 6,  7,  8],\n               [ 9, 10, 11]])\n        Coordinates:\n          * x        (x) float64 0.0 0.1 1.1 1.2\n        Dimensions without coordinates: y\n        >>>\n        >>> da.cumulative_integrate("x")\n        <xarray.DataArray (x: 4, y: 3)>\n        array([[0.  , 0.  , 0.  ],\n               [0.15, 0.25, 0.35],\n               [4.65, 5.75, 6.85],\n               [5.4 , 6.6 , 7.8 ]])\n        Coordinates:\n          * x        (x) float64 0.0 0.1 1.1 1.2\n        Dimensions without coordinates: y\n        '
        ds = self._to_temp_dataset().cumulative_integrate(coord, datetime_unit)
        return self._from_temp_dataset(ds)

    def unify_chunks(self) -> Self:
        if False:
            return 10
        'Unify chunk size along all chunked dimensions of this DataArray.\n\n        Returns\n        -------\n        DataArray with consistent chunk sizes for all dask-array variables\n\n        See Also\n        --------\n        dask.array.core.unify_chunks\n        '
        return unify_chunks(self)[0]

    def map_blocks(self, func: Callable[..., T_Xarray], args: Sequence[Any]=(), kwargs: Mapping[str, Any] | None=None, template: DataArray | Dataset | None=None) -> T_Xarray:
        if False:
            i = 10
            return i + 15
        '\n        Apply a function to each block of this DataArray.\n\n        .. warning::\n            This method is experimental and its signature may change.\n\n        Parameters\n        ----------\n        func : callable\n            User-provided function that accepts a DataArray as its first\n            parameter. The function will receive a subset or \'block\' of this DataArray (see below),\n            corresponding to one chunk along each chunked dimension. ``func`` will be\n            executed as ``func(subset_dataarray, *subset_args, **kwargs)``.\n\n            This function must return either a single DataArray or a single Dataset.\n\n            This function cannot add a new chunked dimension.\n        args : sequence\n            Passed to func after unpacking and subsetting any xarray objects by blocks.\n            xarray objects in args must be aligned with this object, otherwise an error is raised.\n        kwargs : mapping\n            Passed verbatim to func after unpacking. xarray objects, if any, will not be\n            subset to blocks. Passing dask collections in kwargs is not allowed.\n        template : DataArray or Dataset, optional\n            xarray object representing the final result after compute is called. If not provided,\n            the function will be first run on mocked-up data, that looks like this object but\n            has sizes 0, to determine properties of the returned object such as dtype,\n            variable names, attributes, new dimensions and new indexes (if any).\n            ``template`` must be provided if the function changes the size of existing dimensions.\n            When provided, ``attrs`` on variables in `template` are copied over to the result. Any\n            ``attrs`` set by ``func`` will be ignored.\n\n        Returns\n        -------\n        A single DataArray or Dataset with dask backend, reassembled from the outputs of the\n        function.\n\n        Notes\n        -----\n        This function is designed for when ``func`` needs to manipulate a whole xarray object\n        subset to each block. Each block is loaded into memory. In the more common case where\n        ``func`` can work on numpy arrays, it is recommended to use ``apply_ufunc``.\n\n        If none of the variables in this object is backed by dask arrays, calling this function is\n        equivalent to calling ``func(obj, *args, **kwargs)``.\n\n        See Also\n        --------\n        dask.array.map_blocks, xarray.apply_ufunc, xarray.Dataset.map_blocks\n        xarray.DataArray.map_blocks\n\n        :doc:`xarray-tutorial:advanced/map_blocks/map_blocks`\n            Advanced Tutorial on map_blocks with dask\n\n        Examples\n        --------\n        Calculate an anomaly from climatology using ``.groupby()``. Using\n        ``xr.map_blocks()`` allows for parallel operations with knowledge of ``xarray``,\n        its indices, and its methods like ``.groupby()``.\n\n        >>> def calculate_anomaly(da, groupby_type="time.month"):\n        ...     gb = da.groupby(groupby_type)\n        ...     clim = gb.mean(dim="time")\n        ...     return gb - clim\n        ...\n        >>> time = xr.cftime_range("1990-01", "1992-01", freq="M")\n        >>> month = xr.DataArray(time.month, coords={"time": time}, dims=["time"])\n        >>> np.random.seed(123)\n        >>> array = xr.DataArray(\n        ...     np.random.rand(len(time)),\n        ...     dims=["time"],\n        ...     coords={"time": time, "month": month},\n        ... ).chunk()\n        >>> array.map_blocks(calculate_anomaly, template=array).compute()\n        <xarray.DataArray (time: 24)>\n        array([ 0.12894847,  0.11323072, -0.0855964 , -0.09334032,  0.26848862,\n                0.12382735,  0.22460641,  0.07650108, -0.07673453, -0.22865714,\n               -0.19063865,  0.0590131 , -0.12894847, -0.11323072,  0.0855964 ,\n                0.09334032, -0.26848862, -0.12382735, -0.22460641, -0.07650108,\n                0.07673453,  0.22865714,  0.19063865, -0.0590131 ])\n        Coordinates:\n          * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00\n            month    (time) int64 1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12\n\n        Note that one must explicitly use ``args=[]`` and ``kwargs={}`` to pass arguments\n        to the function being applied in ``xr.map_blocks()``:\n\n        >>> array.map_blocks(\n        ...     calculate_anomaly, kwargs={"groupby_type": "time.year"}, template=array\n        ... )  # doctest: +ELLIPSIS\n        <xarray.DataArray (time: 24)>\n        dask.array<<this-array>-calculate_anomaly, shape=(24,), dtype=float64, chunksize=(24,), chunktype=numpy.ndarray>\n        Coordinates:\n          * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00\n            month    (time) int64 dask.array<chunksize=(24,), meta=np.ndarray>\n        '
        from xarray.core.parallel import map_blocks
        return map_blocks(func, self, args, kwargs, template)

    def polyfit(self, dim: Hashable, deg: int, skipna: bool | None=None, rcond: float | None=None, w: Hashable | Any | None=None, full: bool=False, cov: bool | Literal['unscaled']=False) -> Dataset:
        if False:
            while True:
                i = 10
        '\n        Least squares polynomial fit.\n\n        This replicates the behaviour of `numpy.polyfit` but differs by skipping\n        invalid values when `skipna = True`.\n\n        Parameters\n        ----------\n        dim : Hashable\n            Coordinate along which to fit the polynomials.\n        deg : int\n            Degree of the fitting polynomial.\n        skipna : bool or None, optional\n            If True, removes all invalid values before fitting each 1D slices of the array.\n            Default is True if data is stored in a dask.array or if there is any\n            invalid values, False otherwise.\n        rcond : float or None, optional\n            Relative condition number to the fit.\n        w : Hashable, array-like or None, optional\n            Weights to apply to the y-coordinate of the sample points.\n            Can be an array-like object or the name of a coordinate in the dataset.\n        full : bool, default: False\n            Whether to return the residuals, matrix rank and singular values in addition\n            to the coefficients.\n        cov : bool or "unscaled", default: False\n            Whether to return to the covariance matrix in addition to the coefficients.\n            The matrix is not scaled if `cov=\'unscaled\'`.\n\n        Returns\n        -------\n        polyfit_results : Dataset\n            A single dataset which contains:\n\n            polyfit_coefficients\n                The coefficients of the best fit.\n            polyfit_residuals\n                The residuals of the least-square computation (only included if `full=True`).\n                When the matrix rank is deficient, np.nan is returned.\n            [dim]_matrix_rank\n                The effective rank of the scaled Vandermonde coefficient matrix (only included if `full=True`)\n            [dim]_singular_value\n                The singular values of the scaled Vandermonde coefficient matrix (only included if `full=True`)\n            polyfit_covariance\n                The covariance matrix of the polynomial coefficient estimates (only included if `full=False` and `cov=True`)\n\n        See Also\n        --------\n        numpy.polyfit\n        numpy.polyval\n        xarray.polyval\n        DataArray.curvefit\n        '
        return self._to_temp_dataset().polyfit(dim, deg, skipna=skipna, rcond=rcond, w=w, full=full, cov=cov)

    def pad(self, pad_width: Mapping[Any, int | tuple[int, int]] | None=None, mode: PadModeOptions='constant', stat_length: int | tuple[int, int] | Mapping[Any, tuple[int, int]] | None=None, constant_values: float | tuple[float, float] | Mapping[Any, tuple[float, float]] | None=None, end_values: int | tuple[int, int] | Mapping[Any, tuple[int, int]] | None=None, reflect_type: PadReflectOptions=None, keep_attrs: bool | None=None, **pad_width_kwargs: Any) -> Self:
        if False:
            return 10
        'Pad this array along one or more dimensions.\n\n        .. warning::\n            This function is experimental and its behaviour is likely to change\n            especially regarding padding of dimension coordinates (or IndexVariables).\n\n        When using one of the modes ("edge", "reflect", "symmetric", "wrap"),\n        coordinates will be padded with the same mode, otherwise coordinates\n        are padded using the "constant" mode with fill_value dtypes.NA.\n\n        Parameters\n        ----------\n        pad_width : mapping of Hashable to tuple of int\n            Mapping with the form of {dim: (pad_before, pad_after)}\n            describing the number of values padded along each dimension.\n            {dim: pad} is a shortcut for pad_before = pad_after = pad\n        mode : {"constant", "edge", "linear_ramp", "maximum", "mean", "median",             "minimum", "reflect", "symmetric", "wrap"}, default: "constant"\n            How to pad the DataArray (taken from numpy docs):\n\n            - "constant": Pads with a constant value.\n            - "edge": Pads with the edge values of array.\n            - "linear_ramp": Pads with the linear ramp between end_value and the\n              array edge value.\n            - "maximum": Pads with the maximum value of all or part of the\n              vector along each axis.\n            - "mean": Pads with the mean value of all or part of the\n              vector along each axis.\n            - "median": Pads with the median value of all or part of the\n              vector along each axis.\n            - "minimum": Pads with the minimum value of all or part of the\n              vector along each axis.\n            - "reflect": Pads with the reflection of the vector mirrored on\n              the first and last values of the vector along each axis.\n            - "symmetric": Pads with the reflection of the vector mirrored\n              along the edge of the array.\n            - "wrap": Pads with the wrap of the vector along the axis.\n              The first values are used to pad the end and the\n              end values are used to pad the beginning.\n\n        stat_length : int, tuple or mapping of Hashable to tuple, default: None\n            Used in \'maximum\', \'mean\', \'median\', and \'minimum\'.  Number of\n            values at edge of each axis used to calculate the statistic value.\n            {dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)} unique\n            statistic lengths along each dimension.\n            ((before, after),) yields same before and after statistic lengths\n            for each dimension.\n            (stat_length,) or int is a shortcut for before = after = statistic\n            length for all axes.\n            Default is ``None``, to use the entire axis.\n        constant_values : scalar, tuple or mapping of Hashable to tuple, default: 0\n            Used in \'constant\'.  The values to set the padded values for each\n            axis.\n            ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique\n            pad constants along each dimension.\n            ``((before, after),)`` yields same before and after constants for each\n            dimension.\n            ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for\n            all dimensions.\n            Default is 0.\n        end_values : scalar, tuple or mapping of Hashable to tuple, default: 0\n            Used in \'linear_ramp\'.  The values used for the ending value of the\n            linear_ramp and that will form the edge of the padded array.\n            ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique\n            end values along each dimension.\n            ``((before, after),)`` yields same before and after end values for each\n            axis.\n            ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for\n            all axes.\n            Default is 0.\n        reflect_type : {"even", "odd", None}, optional\n            Used in "reflect", and "symmetric". The "even" style is the\n            default with an unaltered reflection around the edge value. For\n            the "odd" style, the extended part of the array is created by\n            subtracting the reflected values from two times the edge value.\n        keep_attrs : bool or None, optional\n            If True, the attributes (``attrs``) will be copied from the\n            original object to the new one. If False, the new object\n            will be returned without attributes.\n        **pad_width_kwargs\n            The keyword arguments form of ``pad_width``.\n            One of ``pad_width`` or ``pad_width_kwargs`` must be provided.\n\n        Returns\n        -------\n        padded : DataArray\n            DataArray with the padded coordinates and data.\n\n        See Also\n        --------\n        DataArray.shift, DataArray.roll, DataArray.bfill, DataArray.ffill, numpy.pad, dask.array.pad\n\n        Notes\n        -----\n        For ``mode="constant"`` and ``constant_values=None``, integer types will be\n        promoted to ``float`` and padded with ``np.nan``.\n\n        Padding coordinates will drop their corresponding index (if any) and will reset default\n        indexes for dimension coordinates.\n\n        Examples\n        --------\n        >>> arr = xr.DataArray([5, 6, 7], coords=[("x", [0, 1, 2])])\n        >>> arr.pad(x=(1, 2), constant_values=0)\n        <xarray.DataArray (x: 6)>\n        array([0, 5, 6, 7, 0, 0])\n        Coordinates:\n          * x        (x) float64 nan 0.0 1.0 2.0 nan nan\n\n        >>> da = xr.DataArray(\n        ...     [[0, 1, 2, 3], [10, 11, 12, 13]],\n        ...     dims=["x", "y"],\n        ...     coords={"x": [0, 1], "y": [10, 20, 30, 40], "z": ("x", [100, 200])},\n        ... )\n        >>> da.pad(x=1)\n        <xarray.DataArray (x: 4, y: 4)>\n        array([[nan, nan, nan, nan],\n               [ 0.,  1.,  2.,  3.],\n               [10., 11., 12., 13.],\n               [nan, nan, nan, nan]])\n        Coordinates:\n          * x        (x) float64 nan 0.0 1.0 nan\n          * y        (y) int64 10 20 30 40\n            z        (x) float64 nan 100.0 200.0 nan\n\n        Careful, ``constant_values`` are coerced to the data type of the array which may\n        lead to a loss of precision:\n\n        >>> da.pad(x=1, constant_values=1.23456789)\n        <xarray.DataArray (x: 4, y: 4)>\n        array([[ 1,  1,  1,  1],\n               [ 0,  1,  2,  3],\n               [10, 11, 12, 13],\n               [ 1,  1,  1,  1]])\n        Coordinates:\n          * x        (x) float64 nan 0.0 1.0 nan\n          * y        (y) int64 10 20 30 40\n            z        (x) float64 nan 100.0 200.0 nan\n        '
        ds = self._to_temp_dataset().pad(pad_width=pad_width, mode=mode, stat_length=stat_length, constant_values=constant_values, end_values=end_values, reflect_type=reflect_type, keep_attrs=keep_attrs, **pad_width_kwargs)
        return self._from_temp_dataset(ds)

    @_deprecate_positional_args('v2023.10.0')
    def idxmin(self, dim: Hashable | None=None, *, skipna: bool | None=None, fill_value: Any=dtypes.NA, keep_attrs: bool | None=None) -> Self:
        if False:
            i = 10
            return i + 15
        'Return the coordinate label of the minimum value along a dimension.\n\n        Returns a new `DataArray` named after the dimension with the values of\n        the coordinate labels along that dimension corresponding to minimum\n        values along that dimension.\n\n        In comparison to :py:meth:`~DataArray.argmin`, this returns the\n        coordinate label while :py:meth:`~DataArray.argmin` returns the index.\n\n        Parameters\n        ----------\n        dim : str, optional\n            Dimension over which to apply `idxmin`.  This is optional for 1D\n            arrays, but required for arrays with 2 or more dimensions.\n        skipna : bool or None, default: None\n            If True, skip missing values (as marked by NaN). By default, only\n            skips missing values for ``float``, ``complex``, and ``object``\n            dtypes; other dtypes either do not have a sentinel missing value\n            (``int``) or ``skipna=True`` has not been implemented\n            (``datetime64`` or ``timedelta64``).\n        fill_value : Any, default: NaN\n            Value to be filled in case all of the values along a dimension are\n            null.  By default this is NaN.  The fill value and result are\n            automatically converted to a compatible dtype if possible.\n            Ignored if ``skipna`` is False.\n        keep_attrs : bool or None, optional\n            If True, the attributes (``attrs``) will be copied from the\n            original object to the new one. If False, the new object\n            will be returned without attributes.\n\n        Returns\n        -------\n        reduced : DataArray\n            New `DataArray` object with `idxmin` applied to its data and the\n            indicated dimension removed.\n\n        See Also\n        --------\n        Dataset.idxmin, DataArray.idxmax, DataArray.min, DataArray.argmin\n\n        Examples\n        --------\n        >>> array = xr.DataArray(\n        ...     [0, 2, 1, 0, -2], dims="x", coords={"x": ["a", "b", "c", "d", "e"]}\n        ... )\n        >>> array.min()\n        <xarray.DataArray ()>\n        array(-2)\n        >>> array.argmin(...)\n        {\'x\': <xarray.DataArray ()>\n        array(4)}\n        >>> array.idxmin()\n        <xarray.DataArray \'x\' ()>\n        array(\'e\', dtype=\'<U1\')\n\n        >>> array = xr.DataArray(\n        ...     [\n        ...         [2.0, 1.0, 2.0, 0.0, -2.0],\n        ...         [-4.0, np.nan, 2.0, np.nan, -2.0],\n        ...         [np.nan, np.nan, 1.0, np.nan, np.nan],\n        ...     ],\n        ...     dims=["y", "x"],\n        ...     coords={"y": [-1, 0, 1], "x": np.arange(5.0) ** 2},\n        ... )\n        >>> array.min(dim="x")\n        <xarray.DataArray (y: 3)>\n        array([-2., -4.,  1.])\n        Coordinates:\n          * y        (y) int64 -1 0 1\n        >>> array.argmin(dim="x")\n        <xarray.DataArray (y: 3)>\n        array([4, 0, 2])\n        Coordinates:\n          * y        (y) int64 -1 0 1\n        >>> array.idxmin(dim="x")\n        <xarray.DataArray \'x\' (y: 3)>\n        array([16.,  0.,  4.])\n        Coordinates:\n          * y        (y) int64 -1 0 1\n        '
        return computation._calc_idxminmax(array=self, func=lambda x, *args, **kwargs: x.argmin(*args, **kwargs), dim=dim, skipna=skipna, fill_value=fill_value, keep_attrs=keep_attrs)

    @_deprecate_positional_args('v2023.10.0')
    def idxmax(self, dim: Hashable=None, *, skipna: bool | None=None, fill_value: Any=dtypes.NA, keep_attrs: bool | None=None) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Return the coordinate label of the maximum value along a dimension.\n\n        Returns a new `DataArray` named after the dimension with the values of\n        the coordinate labels along that dimension corresponding to maximum\n        values along that dimension.\n\n        In comparison to :py:meth:`~DataArray.argmax`, this returns the\n        coordinate label while :py:meth:`~DataArray.argmax` returns the index.\n\n        Parameters\n        ----------\n        dim : Hashable, optional\n            Dimension over which to apply `idxmax`.  This is optional for 1D\n            arrays, but required for arrays with 2 or more dimensions.\n        skipna : bool or None, default: None\n            If True, skip missing values (as marked by NaN). By default, only\n            skips missing values for ``float``, ``complex``, and ``object``\n            dtypes; other dtypes either do not have a sentinel missing value\n            (``int``) or ``skipna=True`` has not been implemented\n            (``datetime64`` or ``timedelta64``).\n        fill_value : Any, default: NaN\n            Value to be filled in case all of the values along a dimension are\n            null.  By default this is NaN.  The fill value and result are\n            automatically converted to a compatible dtype if possible.\n            Ignored if ``skipna`` is False.\n        keep_attrs : bool or None, optional\n            If True, the attributes (``attrs``) will be copied from the\n            original object to the new one. If False, the new object\n            will be returned without attributes.\n\n        Returns\n        -------\n        reduced : DataArray\n            New `DataArray` object with `idxmax` applied to its data and the\n            indicated dimension removed.\n\n        See Also\n        --------\n        Dataset.idxmax, DataArray.idxmin, DataArray.max, DataArray.argmax\n\n        Examples\n        --------\n        >>> array = xr.DataArray(\n        ...     [0, 2, 1, 0, -2], dims="x", coords={"x": ["a", "b", "c", "d", "e"]}\n        ... )\n        >>> array.max()\n        <xarray.DataArray ()>\n        array(2)\n        >>> array.argmax(...)\n        {\'x\': <xarray.DataArray ()>\n        array(1)}\n        >>> array.idxmax()\n        <xarray.DataArray \'x\' ()>\n        array(\'b\', dtype=\'<U1\')\n\n        >>> array = xr.DataArray(\n        ...     [\n        ...         [2.0, 1.0, 2.0, 0.0, -2.0],\n        ...         [-4.0, np.nan, 2.0, np.nan, -2.0],\n        ...         [np.nan, np.nan, 1.0, np.nan, np.nan],\n        ...     ],\n        ...     dims=["y", "x"],\n        ...     coords={"y": [-1, 0, 1], "x": np.arange(5.0) ** 2},\n        ... )\n        >>> array.max(dim="x")\n        <xarray.DataArray (y: 3)>\n        array([2., 2., 1.])\n        Coordinates:\n          * y        (y) int64 -1 0 1\n        >>> array.argmax(dim="x")\n        <xarray.DataArray (y: 3)>\n        array([0, 2, 2])\n        Coordinates:\n          * y        (y) int64 -1 0 1\n        >>> array.idxmax(dim="x")\n        <xarray.DataArray \'x\' (y: 3)>\n        array([0., 4., 4.])\n        Coordinates:\n          * y        (y) int64 -1 0 1\n        '
        return computation._calc_idxminmax(array=self, func=lambda x, *args, **kwargs: x.argmax(*args, **kwargs), dim=dim, skipna=skipna, fill_value=fill_value, keep_attrs=keep_attrs)

    @_deprecate_positional_args('v2023.10.0')
    def argmin(self, dim: Dims=None, *, axis: int | None=None, keep_attrs: bool | None=None, skipna: bool | None=None) -> Self | dict[Hashable, Self]:
        if False:
            return 10
        'Index or indices of the minimum of the DataArray over one or more dimensions.\n\n        If a sequence is passed to \'dim\', then result returned as dict of DataArrays,\n        which can be passed directly to isel(). If a single str is passed to \'dim\' then\n        returns a DataArray with dtype int.\n\n        If there are multiple minima, the indices of the first one found will be\n        returned.\n\n        Parameters\n        ----------\n        dim : "...", str, Iterable of Hashable or None, optional\n            The dimensions over which to find the minimum. By default, finds minimum over\n            all dimensions - for now returning an int for backward compatibility, but\n            this is deprecated, in future will return a dict with indices for all\n            dimensions; to return a dict with all dimensions now, pass \'...\'.\n        axis : int or None, optional\n            Axis over which to apply `argmin`. Only one of the \'dim\' and \'axis\' arguments\n            can be supplied.\n        keep_attrs : bool or None, optional\n            If True, the attributes (`attrs`) will be copied from the original\n            object to the new one. If False, the new object will be\n            returned without attributes.\n        skipna : bool or None, optional\n            If True, skip missing values (as marked by NaN). By default, only\n            skips missing values for float dtypes; other dtypes either do not\n            have a sentinel missing value (int) or skipna=True has not been\n            implemented (object, datetime64 or timedelta64).\n\n        Returns\n        -------\n        result : DataArray or dict of DataArray\n\n        See Also\n        --------\n        Variable.argmin, DataArray.idxmin\n\n        Examples\n        --------\n        >>> array = xr.DataArray([0, 2, -1, 3], dims="x")\n        >>> array.min()\n        <xarray.DataArray ()>\n        array(-1)\n        >>> array.argmin(...)\n        {\'x\': <xarray.DataArray ()>\n        array(2)}\n        >>> array.isel(array.argmin(...))\n        <xarray.DataArray ()>\n        array(-1)\n\n        >>> array = xr.DataArray(\n        ...     [[[3, 2, 1], [3, 1, 2], [2, 1, 3]], [[1, 3, 2], [2, -5, 1], [2, 3, 1]]],\n        ...     dims=("x", "y", "z"),\n        ... )\n        >>> array.min(dim="x")\n        <xarray.DataArray (y: 3, z: 3)>\n        array([[ 1,  2,  1],\n               [ 2, -5,  1],\n               [ 2,  1,  1]])\n        Dimensions without coordinates: y, z\n        >>> array.argmin(dim="x")\n        <xarray.DataArray (y: 3, z: 3)>\n        array([[1, 0, 0],\n               [1, 1, 1],\n               [0, 0, 1]])\n        Dimensions without coordinates: y, z\n        >>> array.argmin(dim=["x"])\n        {\'x\': <xarray.DataArray (y: 3, z: 3)>\n        array([[1, 0, 0],\n               [1, 1, 1],\n               [0, 0, 1]])\n        Dimensions without coordinates: y, z}\n        >>> array.min(dim=("x", "z"))\n        <xarray.DataArray (y: 3)>\n        array([ 1, -5,  1])\n        Dimensions without coordinates: y\n        >>> array.argmin(dim=["x", "z"])\n        {\'x\': <xarray.DataArray (y: 3)>\n        array([0, 1, 0])\n        Dimensions without coordinates: y, \'z\': <xarray.DataArray (y: 3)>\n        array([2, 1, 1])\n        Dimensions without coordinates: y}\n        >>> array.isel(array.argmin(dim=["x", "z"]))\n        <xarray.DataArray (y: 3)>\n        array([ 1, -5,  1])\n        Dimensions without coordinates: y\n        '
        result = self.variable.argmin(dim, axis, keep_attrs, skipna)
        if isinstance(result, dict):
            return {k: self._replace_maybe_drop_dims(v) for (k, v) in result.items()}
        else:
            return self._replace_maybe_drop_dims(result)

    @_deprecate_positional_args('v2023.10.0')
    def argmax(self, dim: Dims=None, *, axis: int | None=None, keep_attrs: bool | None=None, skipna: bool | None=None) -> Self | dict[Hashable, Self]:
        if False:
            while True:
                i = 10
        'Index or indices of the maximum of the DataArray over one or more dimensions.\n\n        If a sequence is passed to \'dim\', then result returned as dict of DataArrays,\n        which can be passed directly to isel(). If a single str is passed to \'dim\' then\n        returns a DataArray with dtype int.\n\n        If there are multiple maxima, the indices of the first one found will be\n        returned.\n\n        Parameters\n        ----------\n        dim : "...", str, Iterable of Hashable or None, optional\n            The dimensions over which to find the maximum. By default, finds maximum over\n            all dimensions - for now returning an int for backward compatibility, but\n            this is deprecated, in future will return a dict with indices for all\n            dimensions; to return a dict with all dimensions now, pass \'...\'.\n        axis : int or None, optional\n            Axis over which to apply `argmax`. Only one of the \'dim\' and \'axis\' arguments\n            can be supplied.\n        keep_attrs : bool or None, optional\n            If True, the attributes (`attrs`) will be copied from the original\n            object to the new one. If False, the new object will be\n            returned without attributes.\n        skipna : bool or None, optional\n            If True, skip missing values (as marked by NaN). By default, only\n            skips missing values for float dtypes; other dtypes either do not\n            have a sentinel missing value (int) or skipna=True has not been\n            implemented (object, datetime64 or timedelta64).\n\n        Returns\n        -------\n        result : DataArray or dict of DataArray\n\n        See Also\n        --------\n        Variable.argmax, DataArray.idxmax\n\n        Examples\n        --------\n        >>> array = xr.DataArray([0, 2, -1, 3], dims="x")\n        >>> array.max()\n        <xarray.DataArray ()>\n        array(3)\n        >>> array.argmax(...)\n        {\'x\': <xarray.DataArray ()>\n        array(3)}\n        >>> array.isel(array.argmax(...))\n        <xarray.DataArray ()>\n        array(3)\n\n        >>> array = xr.DataArray(\n        ...     [[[3, 2, 1], [3, 1, 2], [2, 1, 3]], [[1, 3, 2], [2, 5, 1], [2, 3, 1]]],\n        ...     dims=("x", "y", "z"),\n        ... )\n        >>> array.max(dim="x")\n        <xarray.DataArray (y: 3, z: 3)>\n        array([[3, 3, 2],\n               [3, 5, 2],\n               [2, 3, 3]])\n        Dimensions without coordinates: y, z\n        >>> array.argmax(dim="x")\n        <xarray.DataArray (y: 3, z: 3)>\n        array([[0, 1, 1],\n               [0, 1, 0],\n               [0, 1, 0]])\n        Dimensions without coordinates: y, z\n        >>> array.argmax(dim=["x"])\n        {\'x\': <xarray.DataArray (y: 3, z: 3)>\n        array([[0, 1, 1],\n               [0, 1, 0],\n               [0, 1, 0]])\n        Dimensions without coordinates: y, z}\n        >>> array.max(dim=("x", "z"))\n        <xarray.DataArray (y: 3)>\n        array([3, 5, 3])\n        Dimensions without coordinates: y\n        >>> array.argmax(dim=["x", "z"])\n        {\'x\': <xarray.DataArray (y: 3)>\n        array([0, 1, 0])\n        Dimensions without coordinates: y, \'z\': <xarray.DataArray (y: 3)>\n        array([0, 1, 2])\n        Dimensions without coordinates: y}\n        >>> array.isel(array.argmax(dim=["x", "z"]))\n        <xarray.DataArray (y: 3)>\n        array([3, 5, 3])\n        Dimensions without coordinates: y\n        '
        result = self.variable.argmax(dim, axis, keep_attrs, skipna)
        if isinstance(result, dict):
            return {k: self._replace_maybe_drop_dims(v) for (k, v) in result.items()}
        else:
            return self._replace_maybe_drop_dims(result)

    def query(self, queries: Mapping[Any, Any] | None=None, parser: QueryParserOptions='pandas', engine: QueryEngineOptions=None, missing_dims: ErrorOptionsWithWarn='raise', **queries_kwargs: Any) -> DataArray:
        if False:
            for i in range(10):
                print('nop')
        'Return a new data array indexed along the specified\n        dimension(s), where the indexers are given as strings containing\n        Python expressions to be evaluated against the values in the array.\n\n        Parameters\n        ----------\n        queries : dict-like or None, optional\n            A dict-like with keys matching dimensions and values given by strings\n            containing Python expressions to be evaluated against the data variables\n            in the dataset. The expressions will be evaluated using the pandas\n            eval() function, and can contain any valid Python expressions but cannot\n            contain any Python statements.\n        parser : {"pandas", "python"}, default: "pandas"\n            The parser to use to construct the syntax tree from the expression.\n            The default of \'pandas\' parses code slightly different than standard\n            Python. Alternatively, you can parse an expression using the \'python\'\n            parser to retain strict Python semantics.\n        engine : {"python", "numexpr", None}, default: None\n            The engine used to evaluate the expression. Supported engines are:\n\n            - None: tries to use numexpr, falls back to python\n            - "numexpr": evaluates expressions using numexpr\n            - "python": performs operations as if you had eval’d in top level python\n\n        missing_dims : {"raise", "warn", "ignore"}, default: "raise"\n            What to do if dimensions that should be selected from are not present in the\n            DataArray:\n\n            - "raise": raise an exception\n            - "warn": raise a warning, and ignore the missing dimensions\n            - "ignore": ignore the missing dimensions\n\n        **queries_kwargs : {dim: query, ...}, optional\n            The keyword arguments form of ``queries``.\n            One of queries or queries_kwargs must be provided.\n\n        Returns\n        -------\n        obj : DataArray\n            A new DataArray with the same contents as this dataset, indexed by\n            the results of the appropriate queries.\n\n        See Also\n        --------\n        DataArray.isel\n        Dataset.query\n        pandas.eval\n\n        Examples\n        --------\n        >>> da = xr.DataArray(np.arange(0, 5, 1), dims="x", name="a")\n        >>> da\n        <xarray.DataArray \'a\' (x: 5)>\n        array([0, 1, 2, 3, 4])\n        Dimensions without coordinates: x\n        >>> da.query(x="a > 2")\n        <xarray.DataArray \'a\' (x: 2)>\n        array([3, 4])\n        Dimensions without coordinates: x\n        '
        ds = self._to_dataset_whole(shallow_copy=True)
        ds = ds.query(queries=queries, parser=parser, engine=engine, missing_dims=missing_dims, **queries_kwargs)
        return ds[self.name]

    def curvefit(self, coords: str | DataArray | Iterable[str | DataArray], func: Callable[..., Any], reduce_dims: Dims=None, skipna: bool=True, p0: dict[str, float | DataArray] | None=None, bounds: dict[str, tuple[float | DataArray, float | DataArray]] | None=None, param_names: Sequence[str] | None=None, errors: ErrorOptions='raise', kwargs: dict[str, Any] | None=None) -> Dataset:
        if False:
            i = 10
            return i + 15
        '\n        Curve fitting optimization for arbitrary functions.\n\n        Wraps `scipy.optimize.curve_fit` with `apply_ufunc`.\n\n        Parameters\n        ----------\n        coords : Hashable, DataArray, or sequence of DataArray or Hashable\n            Independent coordinate(s) over which to perform the curve fitting. Must share\n            at least one dimension with the calling object. When fitting multi-dimensional\n            functions, supply `coords` as a sequence in the same order as arguments in\n            `func`. To fit along existing dimensions of the calling object, `coords` can\n            also be specified as a str or sequence of strs.\n        func : callable\n            User specified function in the form `f(x, *params)` which returns a numpy\n            array of length `len(x)`. `params` are the fittable parameters which are optimized\n            by scipy curve_fit. `x` can also be specified as a sequence containing multiple\n            coordinates, e.g. `f((x0, x1), *params)`.\n        reduce_dims : str, Iterable of Hashable or None, optional\n            Additional dimension(s) over which to aggregate while fitting. For example,\n            calling `ds.curvefit(coords=\'time\', reduce_dims=[\'lat\', \'lon\'], ...)` will\n            aggregate all lat and lon points and fit the specified function along the\n            time dimension.\n        skipna : bool, default: True\n            Whether to skip missing values when fitting. Default is True.\n        p0 : dict-like or None, optional\n            Optional dictionary of parameter names to initial guesses passed to the\n            `curve_fit` `p0` arg. If the values are DataArrays, they will be appropriately\n            broadcast to the coordinates of the array. If none or only some parameters are\n            passed, the rest will be assigned initial values following the default scipy\n            behavior.\n        bounds : dict-like, optional\n            Optional dictionary of parameter names to tuples of bounding values passed to the\n            `curve_fit` `bounds` arg. If any of the bounds are DataArrays, they will be\n            appropriately broadcast to the coordinates of the array. If none or only some\n            parameters are passed, the rest will be unbounded following the default scipy\n            behavior.\n        param_names : sequence of Hashable or None, optional\n            Sequence of names for the fittable parameters of `func`. If not supplied,\n            this will be automatically determined by arguments of `func`. `param_names`\n            should be manually supplied when fitting a function that takes a variable\n            number of parameters.\n        errors : {"raise", "ignore"}, default: "raise"\n            If \'raise\', any errors from the `scipy.optimize_curve_fit` optimization will\n            raise an exception. If \'ignore\', the coefficients and covariances for the\n            coordinates where the fitting failed will be NaN.\n        **kwargs : optional\n            Additional keyword arguments to passed to scipy curve_fit.\n\n        Returns\n        -------\n        curvefit_results : Dataset\n            A single dataset which contains:\n\n            [var]_curvefit_coefficients\n                The coefficients of the best fit.\n            [var]_curvefit_covariance\n                The covariance matrix of the coefficient estimates.\n\n        Examples\n        --------\n        Generate some exponentially decaying data, where the decay constant and amplitude are\n        different for different values of the coordinate ``x``:\n\n        >>> rng = np.random.default_rng(seed=0)\n        >>> def exp_decay(t, time_constant, amplitude):\n        ...     return np.exp(-t / time_constant) * amplitude\n        ...\n        >>> t = np.arange(11)\n        >>> da = xr.DataArray(\n        ...     np.stack(\n        ...         [\n        ...             exp_decay(t, 1, 0.1),\n        ...             exp_decay(t, 2, 0.2),\n        ...             exp_decay(t, 3, 0.3),\n        ...         ]\n        ...     )\n        ...     + rng.normal(size=(3, t.size)) * 0.01,\n        ...     coords={"x": [0, 1, 2], "time": t},\n        ... )\n        >>> da\n        <xarray.DataArray (x: 3, time: 11)>\n        array([[ 0.1012573 ,  0.0354669 ,  0.01993775,  0.00602771, -0.00352513,\n                 0.00428975,  0.01328788,  0.009562  , -0.00700381, -0.01264187,\n                -0.0062282 ],\n               [ 0.20041326,  0.09805582,  0.07138797,  0.03216692,  0.01974438,\n                 0.01097441,  0.00679441,  0.01015578,  0.01408826,  0.00093645,\n                 0.01501222],\n               [ 0.29334805,  0.21847449,  0.16305984,  0.11130396,  0.07164415,\n                 0.04744543,  0.03602333,  0.03129354,  0.01074885,  0.01284436,\n                 0.00910995]])\n        Coordinates:\n          * x        (x) int64 0 1 2\n          * time     (time) int64 0 1 2 3 4 5 6 7 8 9 10\n\n        Fit the exponential decay function to the data along the ``time`` dimension:\n\n        >>> fit_result = da.curvefit("time", exp_decay)\n        >>> fit_result["curvefit_coefficients"].sel(\n        ...     param="time_constant"\n        ... )  # doctest: +NUMBER\n        <xarray.DataArray \'curvefit_coefficients\' (x: 3)>\n        array([1.0569203, 1.7354963, 2.9421577])\n        Coordinates:\n          * x        (x) int64 0 1 2\n            param    <U13 \'time_constant\'\n        >>> fit_result["curvefit_coefficients"].sel(param="amplitude")\n        <xarray.DataArray \'curvefit_coefficients\' (x: 3)>\n        array([0.1005489 , 0.19631423, 0.30003579])\n        Coordinates:\n          * x        (x) int64 0 1 2\n            param    <U13 \'amplitude\'\n\n        An initial guess can also be given with the ``p0`` arg (although it does not make much\n        of a difference in this simple example). To have a different guess for different\n        coordinate points, the guess can be a DataArray. Here we use the same initial guess\n        for the amplitude but different guesses for the time constant:\n\n        >>> fit_result = da.curvefit(\n        ...     "time",\n        ...     exp_decay,\n        ...     p0={\n        ...         "amplitude": 0.2,\n        ...         "time_constant": xr.DataArray([1, 2, 3], coords=[da.x]),\n        ...     },\n        ... )\n        >>> fit_result["curvefit_coefficients"].sel(param="time_constant")\n        <xarray.DataArray \'curvefit_coefficients\' (x: 3)>\n        array([1.0569213 , 1.73550052, 2.94215733])\n        Coordinates:\n          * x        (x) int64 0 1 2\n            param    <U13 \'time_constant\'\n        >>> fit_result["curvefit_coefficients"].sel(param="amplitude")\n        <xarray.DataArray \'curvefit_coefficients\' (x: 3)>\n        array([0.10054889, 0.1963141 , 0.3000358 ])\n        Coordinates:\n          * x        (x) int64 0 1 2\n            param    <U13 \'amplitude\'\n\n        See Also\n        --------\n        DataArray.polyfit\n        scipy.optimize.curve_fit\n        '
        return self._to_temp_dataset().curvefit(coords, func, reduce_dims=reduce_dims, skipna=skipna, p0=p0, bounds=bounds, param_names=param_names, errors=errors, kwargs=kwargs)

    @_deprecate_positional_args('v2023.10.0')
    def drop_duplicates(self, dim: Hashable | Iterable[Hashable], *, keep: Literal['first', 'last', False]='first') -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Returns a new DataArray with duplicate dimension values removed.\n\n        Parameters\n        ----------\n        dim : dimension label or labels\n            Pass `...` to drop duplicates along all dimensions.\n        keep : {"first", "last", False}, default: "first"\n            Determines which duplicates (if any) to keep.\n\n            - ``"first"`` : Drop duplicates except for the first occurrence.\n            - ``"last"`` : Drop duplicates except for the last occurrence.\n            - False : Drop all duplicates.\n\n        Returns\n        -------\n        DataArray\n\n        See Also\n        --------\n        Dataset.drop_duplicates\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     np.arange(25).reshape(5, 5),\n        ...     dims=("x", "y"),\n        ...     coords={"x": np.array([0, 0, 1, 2, 3]), "y": np.array([0, 1, 2, 3, 3])},\n        ... )\n        >>> da\n        <xarray.DataArray (x: 5, y: 5)>\n        array([[ 0,  1,  2,  3,  4],\n               [ 5,  6,  7,  8,  9],\n               [10, 11, 12, 13, 14],\n               [15, 16, 17, 18, 19],\n               [20, 21, 22, 23, 24]])\n        Coordinates:\n          * x        (x) int64 0 0 1 2 3\n          * y        (y) int64 0 1 2 3 3\n\n        >>> da.drop_duplicates(dim="x")\n        <xarray.DataArray (x: 4, y: 5)>\n        array([[ 0,  1,  2,  3,  4],\n               [10, 11, 12, 13, 14],\n               [15, 16, 17, 18, 19],\n               [20, 21, 22, 23, 24]])\n        Coordinates:\n          * x        (x) int64 0 1 2 3\n          * y        (y) int64 0 1 2 3 3\n\n        >>> da.drop_duplicates(dim="x", keep="last")\n        <xarray.DataArray (x: 4, y: 5)>\n        array([[ 5,  6,  7,  8,  9],\n               [10, 11, 12, 13, 14],\n               [15, 16, 17, 18, 19],\n               [20, 21, 22, 23, 24]])\n        Coordinates:\n          * x        (x) int64 0 1 2 3\n          * y        (y) int64 0 1 2 3 3\n\n        Drop all duplicate dimension values:\n\n        >>> da.drop_duplicates(dim=...)\n        <xarray.DataArray (x: 4, y: 4)>\n        array([[ 0,  1,  2,  3],\n               [10, 11, 12, 13],\n               [15, 16, 17, 18],\n               [20, 21, 22, 23]])\n        Coordinates:\n          * x        (x) int64 0 1 2 3\n          * y        (y) int64 0 1 2 3\n        '
        deduplicated = self._to_temp_dataset().drop_duplicates(dim, keep=keep)
        return self._from_temp_dataset(deduplicated)

    def convert_calendar(self, calendar: str, dim: str='time', align_on: str | None=None, missing: Any | None=None, use_cftime: bool | None=None) -> Self:
        if False:
            print('Hello World!')
        'Convert the DataArray to another calendar.\n\n        Only converts the individual timestamps, does not modify any data except\n        in dropping invalid/surplus dates or inserting missing dates.\n\n        If the source and target calendars are either no_leap, all_leap or a\n        standard type, only the type of the time array is modified.\n        When converting to a leap year from a non-leap year, the 29th of February\n        is removed from the array. In the other direction the 29th of February\n        will be missing in the output, unless `missing` is specified,\n        in which case that value is inserted.\n\n        For conversions involving `360_day` calendars, see Notes.\n\n        This method is safe to use with sub-daily data as it doesn\'t touch the\n        time part of the timestamps.\n\n        Parameters\n        ---------\n        calendar : str\n            The target calendar name.\n        dim : str\n            Name of the time coordinate.\n        align_on : {None, \'date\', \'year\'}\n            Must be specified when either source or target is a `360_day` calendar,\n           ignored otherwise. See Notes.\n        missing : Optional[any]\n            By default, i.e. if the value is None, this method will simply attempt\n            to convert the dates in the source calendar to the same dates in the\n            target calendar, and drop any of those that are not possible to\n            represent.  If a value is provided, a new time coordinate will be\n            created in the target calendar with the same frequency as the original\n            time coordinate; for any dates that are not present in the source, the\n            data will be filled with this value.  Note that using this mode requires\n            that the source data have an inferable frequency; for more information\n            see :py:func:`xarray.infer_freq`.  For certain frequency, source, and\n            target calendar combinations, this could result in many missing values, see notes.\n        use_cftime : boolean, optional\n            Whether to use cftime objects in the output, only used if `calendar`\n            is one of {"proleptic_gregorian", "gregorian" or "standard"}.\n            If True, the new time axis uses cftime objects.\n            If None (default), it uses :py:class:`numpy.datetime64` values if the\n            date range permits it, and :py:class:`cftime.datetime` objects if not.\n            If False, it uses :py:class:`numpy.datetime64`  or fails.\n\n        Returns\n        -------\n        DataArray\n            Copy of the dataarray with the time coordinate converted to the\n            target calendar. If \'missing\' was None (default), invalid dates in\n            the new calendar are dropped, but missing dates are not inserted.\n            If `missing` was given, the new data is reindexed to have a time axis\n            with the same frequency as the source, but in the new calendar; any\n            missing datapoints are filled with `missing`.\n\n        Notes\n        -----\n        Passing a value to `missing` is only usable if the source\'s time coordinate as an\n        inferable frequencies (see :py:func:`~xarray.infer_freq`) and is only appropriate\n        if the target coordinate, generated from this frequency, has dates equivalent to the\n        source. It is usually **not** appropriate to use this mode with:\n\n        - Period-end frequencies : \'A\', \'Y\', \'Q\' or \'M\', in opposition to \'AS\' \'YS\', \'QS\' and \'MS\'\n        - Sub-monthly frequencies that do not divide a day evenly : \'W\', \'nD\' where `N != 1`\n            or \'mH\' where 24 % m != 0).\n\n        If one of the source or target calendars is `"360_day"`, `align_on` must\n        be specified and two options are offered.\n\n        - "year"\n            The dates are translated according to their relative position in the year,\n            ignoring their original month and day information, meaning that the\n            missing/surplus days are added/removed at regular intervals.\n\n            From a `360_day` to a standard calendar, the output will be missing the\n            following dates (day of year in parentheses):\n\n            To a leap year:\n                January 31st (31), March 31st (91), June 1st (153), July 31st (213),\n                September 31st (275) and November 30th (335).\n            To a non-leap year:\n                February 6th (36), April 19th (109), July 2nd (183),\n                September 12th (255), November 25th (329).\n\n            From a standard calendar to a `"360_day"`, the following dates in the\n            source array will be dropped:\n\n            From a leap year:\n                January 31st (31), April 1st (92), June 1st (153), August 1st (214),\n                September 31st (275), December 1st (336)\n            From a non-leap year:\n                February 6th (37), April 20th (110), July 2nd (183),\n                September 13th (256), November 25th (329)\n\n            This option is best used on daily and subdaily data.\n\n        - "date"\n            The month/day information is conserved and invalid dates are dropped\n            from the output. This means that when converting from a `"360_day"` to a\n            standard calendar, all 31st (Jan, March, May, July, August, October and\n            December) will be missing as there is no equivalent dates in the\n            `"360_day"` calendar and the 29th (on non-leap years) and 30th of February\n            will be dropped as there are no equivalent dates in a standard calendar.\n\n            This option is best used with data on a frequency coarser than daily.\n        '
        return convert_calendar(self, calendar, dim=dim, align_on=align_on, missing=missing, use_cftime=use_cftime)

    def interp_calendar(self, target: pd.DatetimeIndex | CFTimeIndex | DataArray, dim: str='time') -> Self:
        if False:
            return 10
        'Interpolates the DataArray to another calendar based on decimal year measure.\n\n        Each timestamp in `source` and `target` are first converted to their decimal\n        year equivalent then `source` is interpolated on the target coordinate.\n        The decimal year of a timestamp is its year plus its sub-year component\n        converted to the fraction of its year. For example "2000-03-01 12:00" is\n        2000.1653 in a standard calendar or 2000.16301 in a `"noleap"` calendar.\n\n        This method should only be used when the time (HH:MM:SS) information of\n        time coordinate is not important.\n\n        Parameters\n        ----------\n        target: DataArray or DatetimeIndex or CFTimeIndex\n            The target time coordinate of a valid dtype\n            (np.datetime64 or cftime objects)\n        dim : str\n            The time coordinate name.\n\n        Return\n        ------\n        DataArray\n            The source interpolated on the decimal years of target,\n        '
        return interp_calendar(self, target, dim=dim)

    def groupby(self, group: Hashable | DataArray | IndexVariable, squeeze: bool=True, restore_coord_dims: bool=False) -> DataArrayGroupBy:
        if False:
            for i in range(10):
                print('nop')
        'Returns a DataArrayGroupBy object for performing grouped operations.\n\n        Parameters\n        ----------\n        group : Hashable, DataArray or IndexVariable\n            Array whose unique values should be used to group this array. If a\n            Hashable, must be the name of a coordinate contained in this dataarray.\n        squeeze : bool, default: True\n            If "group" is a dimension of any arrays in this dataset, `squeeze`\n            controls whether the subarrays have a dimension of length 1 along\n            that dimension or if the dimension is squeezed out.\n        restore_coord_dims : bool, default: False\n            If True, also restore the dimension order of multi-dimensional\n            coordinates.\n\n        Returns\n        -------\n        grouped : DataArrayGroupBy\n            A `DataArrayGroupBy` object patterned after `pandas.GroupBy` that can be\n            iterated over in the form of `(unique_value, grouped_array)` pairs.\n\n        Examples\n        --------\n        Calculate daily anomalies for daily data:\n\n        >>> da = xr.DataArray(\n        ...     np.linspace(0, 1826, num=1827),\n        ...     coords=[pd.date_range("2000-01-01", "2004-12-31", freq="D")],\n        ...     dims="time",\n        ... )\n        >>> da\n        <xarray.DataArray (time: 1827)>\n        array([0.000e+00, 1.000e+00, 2.000e+00, ..., 1.824e+03, 1.825e+03,\n               1.826e+03])\n        Coordinates:\n          * time     (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2004-12-31\n        >>> da.groupby("time.dayofyear") - da.groupby("time.dayofyear").mean("time")\n        <xarray.DataArray (time: 1827)>\n        array([-730.8, -730.8, -730.8, ...,  730.2,  730.2,  730.5])\n        Coordinates:\n          * time       (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2004-12-31\n            dayofyear  (time) int64 1 2 3 4 5 6 7 8 ... 359 360 361 362 363 364 365 366\n\n        See Also\n        --------\n        :ref:`groupby`\n            Users guide explanation of how to group and bin data.\n\n        :doc:`xarray-tutorial:intermediate/01-high-level-computation-patterns`\n            Tutorial on :py:func:`~xarray.DataArray.Groupby` for windowed computation\n\n        :doc:`xarray-tutorial:fundamentals/03.2_groupby_with_xarray`\n            Tutorial on :py:func:`~xarray.DataArray.Groupby` demonstrating reductions, transformation and comparison with :py:func:`~xarray.DataArray.resample`\n\n        DataArray.groupby_bins\n        Dataset.groupby\n        core.groupby.DataArrayGroupBy\n        DataArray.coarsen\n        pandas.DataFrame.groupby\n        Dataset.resample\n        DataArray.resample\n        '
        from xarray.core.groupby import DataArrayGroupBy, ResolvedUniqueGrouper, UniqueGrouper, _validate_groupby_squeeze
        _validate_groupby_squeeze(squeeze)
        rgrouper = ResolvedUniqueGrouper(UniqueGrouper(), group, self)
        return DataArrayGroupBy(self, (rgrouper,), squeeze=squeeze, restore_coord_dims=restore_coord_dims)

    def groupby_bins(self, group: Hashable | DataArray | IndexVariable, bins: ArrayLike, right: bool=True, labels: ArrayLike | Literal[False] | None=None, precision: int=3, include_lowest: bool=False, squeeze: bool=True, restore_coord_dims: bool=False) -> DataArrayGroupBy:
        if False:
            i = 10
            return i + 15
        'Returns a DataArrayGroupBy object for performing grouped operations.\n\n        Rather than using all unique values of `group`, the values are discretized\n        first by applying `pandas.cut` [1]_ to `group`.\n\n        Parameters\n        ----------\n        group : Hashable, DataArray or IndexVariable\n            Array whose binned values should be used to group this array. If a\n            Hashable, must be the name of a coordinate contained in this dataarray.\n        bins : int or array-like\n            If bins is an int, it defines the number of equal-width bins in the\n            range of x. However, in this case, the range of x is extended by .1%\n            on each side to include the min or max values of x. If bins is a\n            sequence it defines the bin edges allowing for non-uniform bin\n            width. No extension of the range of x is done in this case.\n        right : bool, default: True\n            Indicates whether the bins include the rightmost edge or not. If\n            right == True (the default), then the bins [1,2,3,4] indicate\n            (1,2], (2,3], (3,4].\n        labels : array-like, False or None, default: None\n            Used as labels for the resulting bins. Must be of the same length as\n            the resulting bins. If False, string bin labels are assigned by\n            `pandas.cut`.\n        precision : int, default: 3\n            The precision at which to store and display the bins labels.\n        include_lowest : bool, default: False\n            Whether the first interval should be left-inclusive or not.\n        squeeze : bool, default: True\n            If "group" is a dimension of any arrays in this dataset, `squeeze`\n            controls whether the subarrays have a dimension of length 1 along\n            that dimension or if the dimension is squeezed out.\n        restore_coord_dims : bool, default: False\n            If True, also restore the dimension order of multi-dimensional\n            coordinates.\n\n        Returns\n        -------\n        grouped : DataArrayGroupBy\n            A `DataArrayGroupBy` object patterned after `pandas.GroupBy` that can be\n            iterated over in the form of `(unique_value, grouped_array)` pairs.\n            The name of the group has the added suffix `_bins` in order to\n            distinguish it from the original variable.\n\n        See Also\n        --------\n        :ref:`groupby`\n            Users guide explanation of how to group and bin data.\n        DataArray.groupby\n        Dataset.groupby_bins\n        core.groupby.DataArrayGroupBy\n        pandas.DataFrame.groupby\n\n        References\n        ----------\n        .. [1] http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html\n        '
        from xarray.core.groupby import BinGrouper, DataArrayGroupBy, ResolvedBinGrouper, _validate_groupby_squeeze
        _validate_groupby_squeeze(squeeze)
        grouper = BinGrouper(bins=bins, cut_kwargs={'right': right, 'labels': labels, 'precision': precision, 'include_lowest': include_lowest})
        rgrouper = ResolvedBinGrouper(grouper, group, self)
        return DataArrayGroupBy(self, (rgrouper,), squeeze=squeeze, restore_coord_dims=restore_coord_dims)

    def weighted(self, weights: DataArray) -> DataArrayWeighted:
        if False:
            i = 10
            return i + 15
        '\n        Weighted DataArray operations.\n\n        Parameters\n        ----------\n        weights : DataArray\n            An array of weights associated with the values in this Dataset.\n            Each value in the data contributes to the reduction operation\n            according to its associated weight.\n\n        Notes\n        -----\n        ``weights`` must be a DataArray and cannot contain missing values.\n        Missing values can be replaced by ``weights.fillna(0)``.\n\n        Returns\n        -------\n        core.weighted.DataArrayWeighted\n\n        See Also\n        --------\n        Dataset.weighted\n\n        :ref:`comput.weighted`\n            User guide on weighted array reduction using :py:func:`~xarray.DataArray.weighted`\n\n        :doc:`xarray-tutorial:fundamentals/03.4_weighted`\n            Tutorial on Weighted Reduction using :py:func:`~xarray.DataArray.weighted`\n\n        '
        from xarray.core.weighted import DataArrayWeighted
        return DataArrayWeighted(self, weights)

    def rolling(self, dim: Mapping[Any, int] | None=None, min_periods: int | None=None, center: bool | Mapping[Any, bool]=False, **window_kwargs: int) -> DataArrayRolling:
        if False:
            for i in range(10):
                print('nop')
        '\n        Rolling window object for DataArrays.\n\n        Parameters\n        ----------\n        dim : dict, optional\n            Mapping from the dimension name to create the rolling iterator\n            along (e.g. `time`) to its moving window size.\n        min_periods : int or None, default: None\n            Minimum number of observations in window required to have a value\n            (otherwise result is NA). The default, None, is equivalent to\n            setting min_periods equal to the size of the window.\n        center : bool or Mapping to int, default: False\n            Set the labels at the center of the window.\n        **window_kwargs : optional\n            The keyword arguments form of ``dim``.\n            One of dim or window_kwargs must be provided.\n\n        Returns\n        -------\n        core.rolling.DataArrayRolling\n\n        Examples\n        --------\n        Create rolling seasonal average of monthly data e.g. DJF, JFM, ..., SON:\n\n        >>> da = xr.DataArray(\n        ...     np.linspace(0, 11, num=12),\n        ...     coords=[\n        ...         pd.date_range(\n        ...             "1999-12-15",\n        ...             periods=12,\n        ...             freq=pd.DateOffset(months=1),\n        ...         )\n        ...     ],\n        ...     dims="time",\n        ... )\n        >>> da\n        <xarray.DataArray (time: 12)>\n        array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])\n        Coordinates:\n          * time     (time) datetime64[ns] 1999-12-15 2000-01-15 ... 2000-11-15\n        >>> da.rolling(time=3, center=True).mean()\n        <xarray.DataArray (time: 12)>\n        array([nan,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., nan])\n        Coordinates:\n          * time     (time) datetime64[ns] 1999-12-15 2000-01-15 ... 2000-11-15\n\n        Remove the NaNs using ``dropna()``:\n\n        >>> da.rolling(time=3, center=True).mean().dropna("time")\n        <xarray.DataArray (time: 10)>\n        array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])\n        Coordinates:\n          * time     (time) datetime64[ns] 2000-01-15 2000-02-15 ... 2000-10-15\n\n        See Also\n        --------\n        core.rolling.DataArrayRolling\n        Dataset.rolling\n        '
        from xarray.core.rolling import DataArrayRolling
        dim = either_dict_or_kwargs(dim, window_kwargs, 'rolling')
        return DataArrayRolling(self, dim, min_periods=min_periods, center=center)

    def coarsen(self, dim: Mapping[Any, int] | None=None, boundary: CoarsenBoundaryOptions='exact', side: SideOptions | Mapping[Any, SideOptions]='left', coord_func: str | Callable | Mapping[Any, str | Callable]='mean', **window_kwargs: int) -> DataArrayCoarsen:
        if False:
            while True:
                i = 10
        '\n        Coarsen object for DataArrays.\n\n        Parameters\n        ----------\n        dim : mapping of hashable to int, optional\n            Mapping from the dimension name to the window size.\n        boundary : {"exact", "trim", "pad"}, default: "exact"\n            If \'exact\', a ValueError will be raised if dimension size is not a\n            multiple of the window size. If \'trim\', the excess entries are\n            dropped. If \'pad\', NA will be padded.\n        side : {"left", "right"} or mapping of str to {"left", "right"}, default: "left"\n        coord_func : str or mapping of hashable to str, default: "mean"\n            function (name) that is applied to the coordinates,\n            or a mapping from coordinate name to function (name).\n\n        Returns\n        -------\n        core.rolling.DataArrayCoarsen\n\n        Examples\n        --------\n        Coarsen the long time series by averaging over every three days.\n\n        >>> da = xr.DataArray(\n        ...     np.linspace(0, 364, num=364),\n        ...     dims="time",\n        ...     coords={"time": pd.date_range("1999-12-15", periods=364)},\n        ... )\n        >>> da  # +doctest: ELLIPSIS\n        <xarray.DataArray (time: 364)>\n        array([  0.        ,   1.00275482,   2.00550964,   3.00826446,\n                 4.01101928,   5.0137741 ,   6.01652893,   7.01928375,\n                 8.02203857,   9.02479339,  10.02754821,  11.03030303,\n        ...\n               356.98071625, 357.98347107, 358.9862259 , 359.98898072,\n               360.99173554, 361.99449036, 362.99724518, 364.        ])\n        Coordinates:\n          * time     (time) datetime64[ns] 1999-12-15 1999-12-16 ... 2000-12-12\n        >>> da.coarsen(time=3, boundary="trim").mean()  # +doctest: ELLIPSIS\n        <xarray.DataArray (time: 121)>\n        array([  1.00275482,   4.01101928,   7.01928375,  10.02754821,\n                13.03581267,  16.04407713,  19.0523416 ,  22.06060606,\n                25.06887052,  28.07713499,  31.08539945,  34.09366391,\n        ...\n               349.96143251, 352.96969697, 355.97796143, 358.9862259 ,\n               361.99449036])\n        Coordinates:\n          * time     (time) datetime64[ns] 1999-12-16 1999-12-19 ... 2000-12-10\n        >>>\n\n        See Also\n        --------\n        core.rolling.DataArrayCoarsen\n        Dataset.coarsen\n\n        :ref:`reshape.coarsen`\n            User guide describing :py:func:`~xarray.DataArray.coarsen`\n\n        :ref:`compute.coarsen`\n            User guide on block arrgragation :py:func:`~xarray.DataArray.coarsen`\n\n        :doc:`xarray-tutorial:fundamentals/03.3_windowed`\n            Tutorial on windowed computation using :py:func:`~xarray.DataArray.coarsen`\n\n        '
        from xarray.core.rolling import DataArrayCoarsen
        dim = either_dict_or_kwargs(dim, window_kwargs, 'coarsen')
        return DataArrayCoarsen(self, dim, boundary=boundary, side=side, coord_func=coord_func)

    def resample(self, indexer: Mapping[Any, str] | None=None, skipna: bool | None=None, closed: SideOptions | None=None, label: SideOptions | None=None, base: int | None=None, offset: pd.Timedelta | datetime.timedelta | str | None=None, origin: str | DatetimeLike='start_day', loffset: datetime.timedelta | str | None=None, restore_coord_dims: bool | None=None, **indexer_kwargs: str) -> DataArrayResample:
        if False:
            for i in range(10):
                print('nop')
        'Returns a Resample object for performing resampling operations.\n\n        Handles both downsampling and upsampling. The resampled\n        dimension must be a datetime-like coordinate. If any intervals\n        contain no values from the original object, they will be given\n        the value ``NaN``.\n\n        Parameters\n        ----------\n        indexer : Mapping of Hashable to str, optional\n            Mapping from the dimension name to resample frequency [1]_. The\n            dimension must be datetime-like.\n        skipna : bool, optional\n            Whether to skip missing values when aggregating in downsampling.\n        closed : {"left", "right"}, optional\n            Side of each interval to treat as closed.\n        label : {"left", "right"}, optional\n            Side of each interval to use for labeling.\n        base : int, optional\n            For frequencies that evenly subdivide 1 day, the "origin" of the\n            aggregated intervals. For example, for "24H" frequency, base could\n            range from 0 through 23.\n        origin : {\'epoch\', \'start\', \'start_day\', \'end\', \'end_day\'}, pd.Timestamp, datetime.datetime, np.datetime64, or cftime.datetime, default \'start_day\'\n            The datetime on which to adjust the grouping. The timezone of origin\n            must match the timezone of the index.\n\n            If a datetime is not used, these values are also supported:\n            - \'epoch\': `origin` is 1970-01-01\n            - \'start\': `origin` is the first value of the timeseries\n            - \'start_day\': `origin` is the first day at midnight of the timeseries\n            - \'end\': `origin` is the last value of the timeseries\n            - \'end_day\': `origin` is the ceiling midnight of the last day\n        offset : pd.Timedelta, datetime.timedelta, or str, default is None\n            An offset timedelta added to the origin.\n        loffset : timedelta or str, optional\n            Offset used to adjust the resampled time labels. Some pandas date\n            offset strings are supported.\n        restore_coord_dims : bool, optional\n            If True, also restore the dimension order of multi-dimensional\n            coordinates.\n        **indexer_kwargs : str\n            The keyword arguments form of ``indexer``.\n            One of indexer or indexer_kwargs must be provided.\n\n        Returns\n        -------\n        resampled : core.resample.DataArrayResample\n            This object resampled.\n\n        Examples\n        --------\n        Downsample monthly time-series data to seasonal data:\n\n        >>> da = xr.DataArray(\n        ...     np.linspace(0, 11, num=12),\n        ...     coords=[\n        ...         pd.date_range(\n        ...             "1999-12-15",\n        ...             periods=12,\n        ...             freq=pd.DateOffset(months=1),\n        ...         )\n        ...     ],\n        ...     dims="time",\n        ... )\n        >>> da\n        <xarray.DataArray (time: 12)>\n        array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])\n        Coordinates:\n          * time     (time) datetime64[ns] 1999-12-15 2000-01-15 ... 2000-11-15\n        >>> da.resample(time="QS-DEC").mean()\n        <xarray.DataArray (time: 4)>\n        array([ 1.,  4.,  7., 10.])\n        Coordinates:\n          * time     (time) datetime64[ns] 1999-12-01 2000-03-01 2000-06-01 2000-09-01\n\n        Upsample monthly time-series data to daily data:\n\n        >>> da.resample(time="1D").interpolate("linear")  # +doctest: ELLIPSIS\n        <xarray.DataArray (time: 337)>\n        array([ 0.        ,  0.03225806,  0.06451613,  0.09677419,  0.12903226,\n                0.16129032,  0.19354839,  0.22580645,  0.25806452,  0.29032258,\n                0.32258065,  0.35483871,  0.38709677,  0.41935484,  0.4516129 ,\n        ...\n               10.80645161, 10.83870968, 10.87096774, 10.90322581, 10.93548387,\n               10.96774194, 11.        ])\n        Coordinates:\n          * time     (time) datetime64[ns] 1999-12-15 1999-12-16 ... 2000-11-15\n\n        Limit scope of upsampling method\n\n        >>> da.resample(time="1D").nearest(tolerance="1D")\n        <xarray.DataArray (time: 337)>\n        array([ 0.,  0., nan, ..., nan, 11., 11.])\n        Coordinates:\n          * time     (time) datetime64[ns] 1999-12-15 1999-12-16 ... 2000-11-15\n\n        See Also\n        --------\n        Dataset.resample\n        pandas.Series.resample\n        pandas.DataFrame.resample\n\n        References\n        ----------\n        .. [1] http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases\n        '
        from xarray.core.resample import DataArrayResample
        return self._resample(resample_cls=DataArrayResample, indexer=indexer, skipna=skipna, closed=closed, label=label, base=base, offset=offset, origin=origin, loffset=loffset, restore_coord_dims=restore_coord_dims, **indexer_kwargs)

    def to_dask_dataframe(self, dim_order: Sequence[Hashable] | None=None, set_index: bool=False) -> DaskDataFrame:
        if False:
            for i in range(10):
                print('nop')
        'Convert this array into a dask.dataframe.DataFrame.\n\n        Parameters\n        ----------\n        dim_order : Sequence of Hashable or None , optional\n            Hierarchical dimension order for the resulting dataframe.\n            Array content is transposed to this order and then written out as flat\n            vectors in contiguous order, so the last dimension in this list\n            will be contiguous in the resulting DataFrame. This has a major influence\n            on which operations are efficient on the resulting dask dataframe.\n        set_index : bool, default: False\n            If set_index=True, the dask DataFrame is indexed by this dataset\'s\n            coordinate. Since dask DataFrames do not support multi-indexes,\n            set_index only works if the dataset only contains one dimension.\n\n        Returns\n        -------\n        dask.dataframe.DataFrame\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     np.arange(4 * 2 * 2).reshape(4, 2, 2),\n        ...     dims=("time", "lat", "lon"),\n        ...     coords={\n        ...         "time": np.arange(4),\n        ...         "lat": [-30, -20],\n        ...         "lon": [120, 130],\n        ...     },\n        ...     name="eg_dataarray",\n        ...     attrs={"units": "Celsius", "description": "Random temperature data"},\n        ... )\n        >>> da.to_dask_dataframe(["lat", "lon", "time"]).compute()\n            lat  lon  time  eg_dataarray\n        0   -30  120     0             0\n        1   -30  120     1             4\n        2   -30  120     2             8\n        3   -30  120     3            12\n        4   -30  130     0             1\n        5   -30  130     1             5\n        6   -30  130     2             9\n        7   -30  130     3            13\n        8   -20  120     0             2\n        9   -20  120     1             6\n        10  -20  120     2            10\n        11  -20  120     3            14\n        12  -20  130     0             3\n        13  -20  130     1             7\n        14  -20  130     2            11\n        15  -20  130     3            15\n        '
        if self.name is None:
            raise ValueError('Cannot convert an unnamed DataArray to a dask dataframe : use the ``.rename`` method to assign a name.')
        name = self.name
        ds = self._to_dataset_whole(name, shallow_copy=False)
        return ds.to_dask_dataframe(dim_order, set_index)
    str = utils.UncachedAccessor(StringAccessor['DataArray'])