from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any as TAny, Sequence, Union, overload
import numpy as np
from ..core.has_props import abstract
from ..core.properties import JSON, Any, Bool, ColumnData, Dict, Enum, Instance, InstanceDefault, Int, Nullable, Object, Readonly, Required, Seq, String
from ..model import Model
from ..util.deprecation import deprecated
from ..util.serialization import convert_datetime_array
from ..util.warnings import BokehUserWarning, warn
from .callbacks import CustomJS
from .filters import AllIndices, Filter, IntersectionFilter
from .selections import Selection, SelectionPolicy, UnionRenderers
if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import TypeAlias
    from ..core.has_props import Setter
__all__ = ('AjaxDataSource', 'CDSView', 'ColumnarDataSource', 'ColumnDataSource', 'DataSource', 'GeoJSONDataSource', 'ServerSentDataSource', 'WebDataSource')
if TYPE_CHECKING:
    import numpy.typing as npt
    DataDict: TypeAlias = dict[str, Union[Sequence[TAny], npt.NDArray[TAny], pd.Series, pd.Index]]
    Index: TypeAlias = Union[int, slice, tuple[Union[int, slice], ...]]
    Patches: TypeAlias = dict[str, list[tuple[Index, Any]]]

@abstract
class DataSource(Model):
    """ A base class for data source types.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    selected = Readonly(Instance(Selection), default=InstanceDefault(Selection), help='\n    An instance of a ``Selection`` that indicates selected indices on this ``DataSource``.\n    This is a read-only property. You may only change the attributes of this object\n    to change the selection (e.g., ``selected.indices``).\n    ')

@abstract
class ColumnarDataSource(DataSource):
    """ A base class for data source types, which can be mapped onto
    a columnar format.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    selection_policy = Instance(SelectionPolicy, default=InstanceDefault(UnionRenderers), help='\n    An instance of a ``SelectionPolicy`` that determines how selections are set.\n    ')

class ColumnDataSource(ColumnarDataSource):
    """ Maps names of columns to sequences or arrays.

    The ``ColumnDataSource`` is a fundamental data structure of Bokeh. Most
    plots, data tables, etc. will be driven by a ``ColumnDataSource``.

    If the ``ColumnDataSource`` initializer is called with a single argument that
    can be any of the following:

    * A Python ``dict`` that maps string names to sequences of values, e.g.
      lists, arrays, etc.

      .. code-block:: python

          data = {'x': [1,2,3,4], 'y': np.array([10.0, 20.0, 30.0, 40.0])}

          source = ColumnDataSource(data)

    .. note::
        ``ColumnDataSource`` only creates a shallow copy of ``data``. Use e.g.
        ``ColumnDataSource(copy.deepcopy(data))`` if initializing from another
        ``ColumnDataSource.data`` object that you want to keep independent.

    * A Pandas ``DataFrame`` object

      .. code-block:: python

          source = ColumnDataSource(df)

      In this case the CDS will have columns corresponding to the columns of
      the ``DataFrame``. If the ``DataFrame`` columns have multiple levels,
      they will be flattened using an underscore (e.g. level_0_col_level_1_col).
      The index of the ``DataFrame`` will be flattened to an ``Index`` of tuples
      if it's a ``MultiIndex``, and then reset using ``reset_index``. The result
      will be a column with the same name if the index was named, or
      level_0_name_level_1_name if it was a named ``MultiIndex``. If the
      ``Index`` did not have a name or the ``MultiIndex`` name could not be
      flattened/determined, the ``reset_index`` function will name the index column
      ``index``, or ``level_0`` if the name ``index`` is not available.

    * A Pandas ``GroupBy`` object

      .. code-block:: python

          group = df.groupby(('colA', 'ColB'))

      In this case the CDS will have columns corresponding to the result of
      calling ``group.describe()``. The ``describe`` method generates columns
      for statistical measures such as ``mean`` and ``count`` for all the
      non-grouped original columns. The CDS columns are formed by joining
      original column names with the computed measure. For example, if a
      ``DataFrame`` has columns ``'year'`` and ``'mpg'``. Then passing
      ``df.groupby('year')`` to a CDS will result in columns such as
      ``'mpg_mean'``

      If the ``GroupBy.describe`` result has a named index column, then
      CDS will also have a column with this name. However, if the index name
      (or any subname of a ``MultiIndex``) is ``None``, then the CDS will have
      a column generically named ``index`` for the index.

      Note this capability to adapt ``GroupBy`` objects may only work with
      Pandas ``>=0.20.0``.

    .. note::
        There is an implicit assumption that all the columns in a given
        ``ColumnDataSource`` all have the same length at all times. For this
        reason, it is usually preferable to update the ``.data`` property
        of a data source "all at once".

    """
    data: DataDict = ColumnData(String, Seq(Any), help='\n    Mapping of column names to sequences of data. The columns can be, e.g,\n    Python lists or tuples, NumPy arrays, etc.\n\n    The .data attribute can also be set from Pandas DataFrames or GroupBy\n    objects. In these cases, the behaviour is identical to passing the objects\n    to the ``ColumnDataSource`` initializer.\n    ').accepts(Object('pandas.DataFrame'), lambda x: ColumnDataSource._data_from_df(x)).accepts(Object('pandas.core.groupby.GroupBy'), lambda x: ColumnDataSource._data_from_groupby(x)).asserts(lambda _, data: len({len(x) for x in data.values()}) <= 1, lambda obj, name, data: warn("ColumnDataSource's columns must be of the same length. " + 'Current lengths: %s' % ', '.join(sorted((str((k, len(v))) for (k, v) in data.items()))), BokehUserWarning))

    @overload
    def __init__(self, data: DataDict | pd.DataFrame | pd.core.groupby.GroupBy, **kwargs: TAny) -> None:
        if False:
            while True:
                i = 10
        ...

    @overload
    def __init__(self, **kwargs: TAny) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __init__(self, *args: TAny, **kwargs: TAny) -> None:
        if False:
            while True:
                i = 10
        ' If called with a single argument that is a dict or\n        ``pandas.DataFrame``, treat that implicitly as the "data" attribute.\n\n        '
        if len(args) == 1 and 'data' not in kwargs:
            kwargs['data'] = args[0]
        raw_data: DataDict = kwargs.pop('data', {})
        import pandas as pd
        if not isinstance(raw_data, dict):
            if isinstance(raw_data, pd.DataFrame):
                raw_data = self._data_from_df(raw_data)
            elif isinstance(raw_data, pd.core.groupby.GroupBy):
                raw_data = self._data_from_groupby(raw_data)
            else:
                raise ValueError(f'expected a dict or pandas.DataFrame, got {raw_data}')
        super().__init__(**kwargs)
        self.data.update(raw_data)

    @property
    def column_names(self) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        ' A list of the column names in this data source.\n\n        '
        return list(self.data)

    @staticmethod
    def _data_from_df(df: pd.DataFrame) -> DataDict:
        if False:
            for i in range(10):
                print('nop')
        ' Create a ``dict`` of columns from a Pandas ``DataFrame``,\n        suitable for creating a ColumnDataSource.\n\n        Args:\n            df (DataFrame) : data to convert\n\n        Returns:\n            dict[str, np.array]\n\n        '
        import pandas as pd
        _df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            try:
                _df.columns = ['_'.join(col) for col in _df.columns.values]
            except TypeError:
                raise TypeError('Could not flatten MultiIndex columns. use string column names or flatten manually')
        if isinstance(df.columns, pd.CategoricalIndex):
            _df.columns = df.columns.tolist()
        index_name = ColumnDataSource._df_index_name(df)
        if index_name == 'index':
            _df.index = pd.Index(_df.index.values)
        else:
            _df.index = pd.Index(_df.index.values, name=index_name)
        _df.reset_index(inplace=True)
        tmp_data = {c: v.values for (c, v) in _df.items()}
        new_data: DataDict = {}
        for (k, v) in tmp_data.items():
            new_data[k] = v
        return new_data

    @staticmethod
    def _data_from_groupby(group: pd.core.groupby.GroupBy) -> DataDict:
        if False:
            return 10
        ' Create a ``dict`` of columns from a Pandas ``GroupBy``,\n        suitable for creating a ``ColumnDataSource``.\n\n        The data generated is the result of running ``describe``\n        on the group.\n\n        Args:\n            group (GroupBy) : data to convert\n\n        Returns:\n            dict[str, np.array]\n\n        '
        return ColumnDataSource._data_from_df(group.describe())

    @staticmethod
    def _df_index_name(df: pd.DataFrame) -> str:
        if False:
            while True:
                i = 10
        ' Return the Bokeh-appropriate column name for a ``DataFrame`` index\n\n        If there is no named index, then `"index" is returned.\n\n        If there is a single named index, then ``df.index.name`` is returned.\n\n        If there is a multi-index, and the index names are all strings, then\n        the names are joined with \'_\' and the result is returned, e.g. for a\n        multi-index ``[\'ind1\', \'ind2\']`` the result will be "ind1_ind2".\n        Otherwise if any index name is not a string, the fallback name "index"\n        is returned.\n\n        Args:\n            df (DataFrame) : the ``DataFrame`` to find an index name for\n\n        Returns:\n            str\n\n        '
        if df.index.name:
            return df.index.name
        elif df.index.names:
            try:
                return '_'.join(df.index.names)
            except TypeError:
                return 'index'
        else:
            return 'index'

    @classmethod
    def from_df(cls, data: pd.DataFrame) -> DataDict:
        if False:
            return 10
        ' Create a ``dict`` of columns from a Pandas ``DataFrame``,\n        suitable for creating a ``ColumnDataSource``.\n\n        Args:\n            data (DataFrame) : data to convert\n\n        Returns:\n            dict[str, np.array]\n\n        '
        return cls._data_from_df(data)

    @classmethod
    def from_groupby(cls, data: pd.core.groupby.GroupBy) -> DataDict:
        if False:
            for i in range(10):
                print('nop')
        ' Create a ``dict`` of columns from a Pandas ``GroupBy``,\n        suitable for creating a ``ColumnDataSource``.\n\n        The data generated is the result of running ``describe``\n        on the group.\n\n        Args:\n            data (Groupby) : data to convert\n\n        Returns:\n            dict[str, np.array]\n\n        '
        return cls._data_from_df(data.describe())

    def to_df(self) -> pd.DataFrame:
        if False:
            return 10
        ' Convert this data source to pandas ``DataFrame``.\n\n        Returns:\n            DataFrame\n\n        '
        import pandas as pd
        return pd.DataFrame(self.data)

    def add(self, data: Sequence[Any], name: str | None=None) -> str:
        if False:
            print('Hello World!')
        ' Appends a new column of data to the data source.\n\n        Args:\n            data (seq) : new data to add\n            name (str, optional) : column name to use.\n                If not supplied, generate a name of the form "Series ####"\n\n        Returns:\n            str:  the column name used\n\n        '
        if name is None:
            n = len(self.data)
            while f'Series {n}' in self.data:
                n += 1
            name = f'Series {n}'
        self.data[name] = data
        return name

    def remove(self, name: str) -> None:
        if False:
            while True:
                i = 10
        ' Remove a column of data.\n\n        Args:\n            name (str) : name of the column to remove\n\n        Returns:\n            None\n\n        .. note::\n            If the column name does not exist, a warning is issued.\n\n        '
        try:
            del self.data[name]
        except (ValueError, KeyError):
            warn(f"Unable to find column '{name}' in data source")

    def stream(self, new_data: DataDict, rollover: int | None=None) -> None:
        if False:
            return 10
        " Efficiently update data source columns with new append-only data.\n\n        In cases where it is necessary to update data columns in, this method\n        can efficiently send only the new data, instead of requiring the\n        entire data set to be re-sent.\n\n        Args:\n            new_data (dict[str, seq]) : a mapping of column names to sequences of\n                new data to append to each column.\n\n                All columns of the data source must be present in ``new_data``,\n                with identical-length append data.\n\n            rollover (int, optional) : A maximum column size, above which data\n                from the start of the column begins to be discarded. If None,\n                then columns will continue to grow unbounded (default: None)\n\n        Returns:\n            None\n\n        Raises:\n            ValueError\n\n        Example:\n\n        .. code-block:: python\n\n            source = ColumnDataSource(data=dict(foo=[], bar=[]))\n\n            # has new, identical-length updates for all columns in source\n            new_data = {\n                'foo' : [10, 20],\n                'bar' : [100, 200],\n            }\n\n            source.stream(new_data)\n\n        "
        self._stream(new_data, rollover)

    def _stream(self, new_data: DataDict | pd.Series | pd.DataFrame, rollover: int | None=None, setter: Setter | None=None) -> None:
        if False:
            return 10
        ' Internal implementation to efficiently update data source columns\n        with new append-only data. The internal implementation adds the setter\n        attribute.  [https://github.com/bokeh/bokeh/issues/6577]\n\n        In cases where it is necessary to update data columns in, this method\n        can efficiently send only the new data, instead of requiring the\n        entire data set to be re-sent.\n\n        Args:\n            new_data (dict[str, seq] or DataFrame or Series) : a mapping of\n                column names to sequences of new data to append to each column,\n                a pandas DataFrame, or a pandas Series in case of a single row -\n                in this case the Series index is used as column names\n\n                All columns of the data source must be present in ``new_data``,\n                with identical-length append data.\n\n            rollover (int, optional) : A maximum column size, above which data\n                from the start of the column begins to be discarded. If None,\n                then columns will continue to grow unbounded (default: None)\n            setter (ClientSession or ServerSession or None, optional) :\n                This is used to prevent "boomerang" updates to Bokeh apps.\n                (default: None)\n                In the context of a Bokeh server application, incoming updates\n                to properties will be annotated with the session that is\n                doing the updating. This value is propagated through any\n                subsequent change notifications that the update triggers.\n                The session can compare the event setter to itself, and\n                suppress any updates that originate from itself.\n        Returns:\n            None\n\n        Raises:\n            ValueError\n\n        Example:\n\n        .. code-block:: python\n\n            source = ColumnDataSource(data=dict(foo=[], bar=[]))\n\n            # has new, identical-length updates for all columns in source\n            new_data = {\n                \'foo\' : [10, 20],\n                \'bar\' : [100, 200],\n            }\n\n            source.stream(new_data)\n\n        '
        import pandas as pd
        needs_length_check = True
        if isinstance(new_data, (pd.Series, pd.DataFrame)):
            if isinstance(new_data, pd.Series):
                new_data = new_data.to_frame().T
            needs_length_check = False
            _df = new_data
            newkeys = set(_df.columns)
            index_name = ColumnDataSource._df_index_name(_df)
            newkeys.add(index_name)
            new_data = dict(_df.items())
            new_data[index_name] = _df.index.values
        else:
            newkeys = set(new_data.keys())
        oldkeys = set(self.data.keys())
        if newkeys != oldkeys:
            missing = sorted(oldkeys - newkeys)
            extra = sorted(newkeys - oldkeys)
            if missing and extra:
                raise ValueError(f"Must stream updates to all existing columns (missing: {', '.join(missing)}, extra: {', '.join(extra)})")
            elif missing:
                raise ValueError(f"Must stream updates to all existing columns (missing: {', '.join(missing)})")
            else:
                raise ValueError(f"Must stream updates to all existing columns (extra: {', '.join(extra)})")
        if needs_length_check:
            lengths: set[int] = set()
            arr_types = (np.ndarray, pd.Series)
            for (_, x) in new_data.items():
                if isinstance(x, arr_types):
                    if len(x.shape) != 1:
                        raise ValueError(f'stream(...) only supports 1d sequences, got ndarray with size {x.shape!r}')
                    lengths.add(x.shape[0])
                else:
                    lengths.add(len(x))
            if len(lengths) > 1:
                raise ValueError('All streaming column updates must be the same length')
        for (key, values) in new_data.items():
            if pd and isinstance(values, (pd.Series, pd.Index)):
                values = values.values
            old_values = self.data[key]
            if isinstance(values, np.ndarray) and values.dtype.kind.lower() == 'm' and isinstance(old_values, np.ndarray) and (old_values.dtype.kind.lower() != 'm'):
                new_data[key] = convert_datetime_array(values)
            else:
                new_data[key] = values
        self.data._stream(self.document, self, new_data, rollover, setter)

    def patch(self, patches: Patches, setter: Setter | None=None) -> None:
        if False:
            while True:
                i = 10
        ' Efficiently update data source columns at specific locations\n\n        If it is only necessary to update a small subset of data in a\n        ``ColumnDataSource``, this method can be used to efficiently update only\n        the subset, instead of requiring the entire data set to be sent.\n\n        This method should be passed a dictionary that maps column names to\n        lists of tuples that describe a patch change to apply. To replace\n        individual items in columns entirely, the tuples should be of the\n        form:\n\n        .. code-block:: python\n\n            (index, new_value)  # replace a single column value\n\n            # or\n\n            (slice, new_values) # replace several column values\n\n        Values at an index or slice will be replaced with the corresponding\n        new values.\n\n        In the case of columns whose values are other arrays or lists, (e.g.\n        image or patches glyphs), it is also possible to patch "subregions".\n        In this case the first item of the tuple should be a whose first\n        element is the index of the array item in the CDS patch, and whose\n        subsequent elements are integer indices or slices into the array item:\n\n        .. code-block:: python\n\n            # replace the entire 10th column of the 2nd array:\n\n              +----------------- index of item in column data source\n              |\n              |       +--------- row subindex into array item\n              |       |\n              |       |       +- column subindex into array item\n              V       V       V\n            ([2, slice(None), 10], new_values)\n\n        Imagining a list of 2d NumPy arrays, the patch above is roughly\n        equivalent to:\n\n        .. code-block:: python\n\n            data = [arr1, arr2, ...]  # list of 2d arrays\n\n            data[2][:, 10] = new_data\n\n        There are some limitations to the kinds of slices and data that can\n        be accepted.\n\n        * Negative ``start``, ``stop``, or ``step`` values for slices will\n          result in a ``ValueError``.\n\n        * In a slice, ``start > stop`` will result in a ``ValueError``\n\n        * When patching 1d or 2d subitems, the subitems must be NumPy arrays.\n\n        * New values must be supplied as a **flattened one-dimensional array**\n          of the appropriate size.\n\n        Args:\n            patches (dict[str, list[tuple]]) : lists of patches for each column\n\n        Returns:\n            None\n\n        Raises:\n            ValueError\n\n        Example:\n\n        The following example shows how to patch entire column elements. In this case,\n\n        .. code-block:: python\n\n            source = ColumnDataSource(data=dict(foo=[10, 20, 30], bar=[100, 200, 300]))\n\n            patches = {\n                \'foo\' : [ (slice(2), [11, 12]) ],\n                \'bar\' : [ (0, 101), (2, 301) ],\n            }\n\n            source.patch(patches)\n\n        After this operation, the value of the ``source.data`` will be:\n\n        .. code-block:: python\n\n            dict(foo=[11, 12, 30], bar=[101, 200, 301])\n\n        For a more comprehensive example, see :bokeh-tree:`examples/server/app/patch_app.py`.\n\n        '
        extra = set(patches.keys()) - set(self.data.keys())
        if extra:
            raise ValueError('Can only patch existing columns (extra: %s)' % ', '.join(sorted(extra)))
        for (name, patch) in patches.items():
            col_len = len(self.data[name])
            for (ind, _) in patch:
                if isinstance(ind, int):
                    if ind > col_len or ind < 0:
                        raise ValueError('Out-of bounds index (%d) in patch for column: %s' % (ind, name))
                elif isinstance(ind, slice):
                    _check_slice(ind)
                    if ind.stop is not None and ind.stop > col_len:
                        raise ValueError('Out-of bounds slice index stop (%d) in patch for column: %s' % (ind.stop, name))
                elif isinstance(ind, (list, tuple)):
                    if len(ind) == 0:
                        raise ValueError('Empty (length zero) patch multi-index')
                    if len(ind) == 1:
                        raise ValueError('Patch multi-index must contain more than one subindex')
                    ind_0 = ind[0]
                    if not isinstance(ind_0, int):
                        raise ValueError('Initial patch sub-index may only be integer, got: %s' % ind_0)
                    if ind_0 > col_len or ind_0 < 0:
                        raise ValueError('Out-of bounds initial sub-index (%d) in patch for column: %s' % (ind, name))
                    if not isinstance(self.data[name][ind_0], np.ndarray):
                        raise ValueError('Can only sub-patch into columns with NumPy array items')
                    if len(self.data[name][ind_0].shape) != len(ind) - 1:
                        raise ValueError('Shape mismatch between patch slice and sliced data')
                    elif isinstance(ind_0, slice):
                        _check_slice(ind_0)
                        if ind_0.stop is not None and ind_0.stop > col_len:
                            raise ValueError('Out-of bounds initial slice sub-index stop (%d) in patch for column: %s' % (ind.stop, name))
                    for subind in ind[1:]:
                        if not isinstance(subind, (int, slice)):
                            raise ValueError('Invalid patch sub-index: %s' % subind)
                        if isinstance(subind, slice):
                            _check_slice(subind)
                else:
                    raise ValueError('Invalid patch index: %s' % ind)
        self.data._patch(self.document, self, patches, setter)

class CDSView(Model):
    """ A view into a ``ColumnDataSource`` that represents a row-wise subset.

    """

    def __init__(self, *args: TAny, **kwargs: TAny) -> None:
        if False:
            i = 10
            return i + 15
        if 'source' in kwargs:
            del kwargs['source']
            deprecated('CDSView.source is no longer needed, and is now ignored. In a future release, passing source will result an error.')
        super().__init__(*args, **kwargs)
    filter = Instance(Filter, default=InstanceDefault(AllIndices), help='\n    Defines the subset of indices to use from the data source this view applies to.\n\n    By default all indices are used (``AllIndices`` filter). This can be changed by\n    using specialized filters like ``IndexFilter``, ``BooleanFilter``, etc. Filters\n    can be composed using set operations to create non-trivial data masks. This can\n    be accomplished by directly using models like ``InversionFilter``, ``UnionFilter``,\n    etc., or by using set operators on filters, e.g.:\n\n    .. code-block:: python\n\n        # filters everything but indexes 10 and 11\n        cds_view.filter &= ~IndexFilter(indices=[10, 11])\n    ')

    @property
    def filters(self) -> list[Filter]:
        if False:
            return 10
        deprecated('CDSView.filters was deprecated in bokeh 3.0. Use CDSView.filter instead.')
        filter = self.filter
        if isinstance(filter, IntersectionFilter):
            return filter.operands
        elif isinstance(filter, AllIndices):
            return []
        else:
            return [filter]

    @filters.setter
    def filters(self, filters: list[Filter]) -> None:
        if False:
            print('Hello World!')
        deprecated('CDSView.filters was deprecated in bokeh 3.0. Use CDSView.filter instead.')
        if len(filters) == 0:
            self.filter = AllIndices()
        elif len(filters) == 1:
            self.filter = filters[0]
        else:
            self.filter = IntersectionFilter(operands=filters)

class GeoJSONDataSource(ColumnarDataSource):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    geojson = Required(JSON, help='\n    GeoJSON that contains features for plotting. Currently\n    ``GeoJSONDataSource`` can only process a ``FeatureCollection`` or\n    ``GeometryCollection``.\n    ')

@abstract
class WebDataSource(ColumnDataSource):
    """ Base class for web column data sources that can update from data
    URLs.

    .. note::
        This base class is typically not useful to instantiate on its own.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    adapter = Nullable(Instance(CustomJS), help='\n    A JavaScript callback to adapt raw JSON responses to Bokeh ``ColumnDataSource``\n    format.\n\n    If provided, this callback is executes immediately after the JSON data is\n    received, but before appending or replacing data in the data source. The\n    ``CustomJS`` callback will receive the ``AjaxDataSource`` as ``cb_obj`` and\n    will receive the raw JSON response as ``cb_data.response``. The callback\n    code should return a ``data`` object suitable for a Bokeh ``ColumnDataSource``\n    (i.e.  a mapping of string column names to arrays of data).\n    ')
    max_size = Nullable(Int, help='\n    Maximum size of the data columns. If a new fetch would result in columns\n    larger than ``max_size``, then earlier data is dropped to make room.\n    ')
    mode = Enum('replace', 'append', help='\n    Whether to append new data to existing data (up to ``max_size``), or to\n    replace existing data entirely.\n    ')
    data_url = Required(String, help='\n    A URL to to fetch data from.\n    ')

class ServerSentDataSource(WebDataSource):
    """ A data source that can populate columns by receiving server sent
    events endpoints.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

class AjaxDataSource(WebDataSource):
    """ A data source that can populate columns by making Ajax calls to REST
    endpoints.

    The ``AjaxDataSource`` can be especially useful if you want to make a
    standalone document (i.e. not backed by the Bokeh server) that can still
    dynamically update using an existing REST API.

    The response from the REST API should match the ``.data`` property of a
    standard ``ColumnDataSource``, i.e. a JSON dict that maps names to arrays
    of values:

    .. code-block:: python

        {
            'x' : [1, 2, 3, ...],
            'y' : [9, 3, 2, ...]
        }

    Alternatively, if the REST API returns a different format, a ``CustomJS``
    callback can be provided to convert the REST response into Bokeh format,
    via the ``adapter`` property of this data source.

    Initial data can be set by specifying the ``data`` property directly.
    This is necessary when used in conjunction with a ``FactorRange``, even
    if the columns in `data`` are empty.

    A full example can be seen at :bokeh-tree:`examples/basic/data/ajax_source.py`

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    polling_interval = Nullable(Int, help='\n    A polling interval (in milliseconds) for updating data source.\n    ')
    method = Enum('POST', 'GET', help='\n    Specify the HTTP method to use for the Ajax request (GET or POST)\n    ')
    if_modified = Bool(False, help='\n    Whether to include an ``If-Modified-Since`` header in Ajax requests\n    to the server. If this header is supported by the server, then only\n    new data since the last request will be returned.\n    ')
    content_type = String(default='application/json', help='\n    Set the "contentType" parameter for the Ajax request.\n    ')
    http_headers = Dict(String, String, help="\n    Specify HTTP headers to set for the Ajax request.\n\n    Example:\n\n    .. code-block:: python\n\n        ajax_source.headers = { 'x-my-custom-header': 'some value' }\n\n    ")

def _check_slice(s: slice) -> None:
    if False:
        print('Hello World!')
    if s.start is not None and s.stop is not None and (s.start > s.stop):
        raise ValueError('Patch slices must have start < end, got %s' % s)
    if s.start is not None and s.start < 0 or (s.stop is not None and s.stop < 0) or (s.step is not None and s.step < 0):
        raise ValueError('Patch slices must have non-negative (start, stop, step) values, got %s' % s)