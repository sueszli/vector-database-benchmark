import ast
import logging
import numpy as np
from bson.binary import Binary
from pandas import DataFrame, Series
from arctic._util import NP_OBJECT_DTYPE
from arctic.serialization.numpy_records import SeriesSerializer, DataFrameSerializer
from ._ndarray_store import NdarrayStore
from .._compression import compress, decompress
from .._config import FORCE_BYTES_TO_UNICODE
from ..date._util import to_pandas_closed_closed
from ..exceptions import ArcticException
log = logging.getLogger(__name__)
DTN64_DTYPE = 'datetime64[ns]'
INDEX_DTYPE = [('datetime', DTN64_DTYPE), ('index', 'i8')]

class PandasStore(NdarrayStore):

    def _segment_index(self, recarr, existing_index, start, new_segments):
        if False:
            while True:
                i = 10
        '\n        Generate index of datetime64 -> item offset.\n\n        Parameters:\n        -----------\n        new_data: new data being written (or appended)\n        existing_index: index field from the versions document of the previous version\n        start: first (0-based) offset of the new data\n        segments: list of offsets. Each offset is the row index of the\n                  the last row of a particular chunk relative to the start of the _original_ item.\n                  array(new_data) - segments = array(offsets in item)\n\n        Returns:\n        --------\n        Binary(compress(array([(index, datetime)]))\n            Where index is the 0-based index of the datetime in the DataFrame\n        '
        idx_col = self._datetime64_index(recarr)
        if idx_col is not None:
            new_segments = np.array(new_segments, dtype='i8')
            last_rows = recarr[new_segments - start]
            index = np.core.records.fromarrays([last_rows[idx_col]] + [new_segments], dtype=INDEX_DTYPE)
            if existing_index:
                existing_index_arr = np.frombuffer(decompress(existing_index), dtype=INDEX_DTYPE)
                if start > 0:
                    existing_index_arr = existing_index_arr[existing_index_arr['index'] < start]
                index = np.concatenate((existing_index_arr, index))
            return Binary(compress(index.tobytes()))
        elif existing_index:
            raise ArcticException('Could not find datetime64 index in item but existing data contains one')
        return None

    def _datetime64_index(self, recarr):
        if False:
            while True:
                i = 10
        ' Given a np.recarray find the first datetime64 column '
        names = recarr.dtype.names
        for name in names:
            if recarr[name].dtype == DTN64_DTYPE:
                return name
        return None

    def read_options(self):
        if False:
            return 10
        return ['date_range']

    def _index_range(self, version, symbol, date_range=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Given a version, read the segment_index and return the chunks associated\n        with the date_range. As the segment index is (id -> last datetime)\n        we need to take care in choosing the correct chunks. '
        if date_range and 'segment_index' in version:
            index = np.frombuffer(decompress(version['segment_index']), dtype=INDEX_DTYPE)
            dtcol = self._datetime64_index(index)
            if dtcol and len(index):
                dts = index[dtcol]
                (start, end) = _start_end(date_range, dts)
                if start > dts[-1]:
                    return (-1, -1)
                idxstart = min(np.searchsorted(dts, start), len(dts) - 1)
                idxend = min(np.searchsorted(dts, end, side='right'), len(dts) - 1)
                return (int(index['index'][idxstart]), int(index['index'][idxend] + 1))
        return super(PandasStore, self)._index_range(version, symbol, **kwargs)

    def _daterange(self, recarr, date_range):
        if False:
            i = 10
            return i + 15
        ' Given a recarr, slice out the given artic.date.DateRange if a\n        datetime64 index exists '
        idx = self._datetime64_index(recarr)
        if idx and len(recarr):
            dts = recarr[idx]
            mask = Series(np.zeros(len(dts)), index=dts)
            (start, end) = _start_end(date_range, dts)
            mask[start:end] = 1.0
            return recarr[mask.values.astype(bool)]
        return recarr

    def read(self, arctic_lib, version, symbol, read_preference=None, date_range=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        item = super(PandasStore, self).read(arctic_lib, version, symbol, read_preference, date_range=date_range, **kwargs)
        if date_range:
            item = self._daterange(item, date_range)
        return item

    def get_info(self, version):
        if False:
            while True:
                i = 10
        '\n        parses out the relevant information in version\n        and returns it to the user in a dictionary\n        '
        ret = super(PandasStore, self).get_info(version)
        ret['col_names'] = version['dtype_metadata']
        ret['handler'] = self.__class__.__name__
        ret['dtype'] = ast.literal_eval(version['dtype'])
        return ret

def _start_end(date_range, dts):
    if False:
        return 10
    '\n    Return tuple: [start, end] of np.datetime64 dates that are inclusive of the passed\n    in datetimes.\n    '
    assert len(dts)
    _assert_no_timezone(date_range)
    date_range = to_pandas_closed_closed(date_range, add_tz=False)
    start = np.datetime64(date_range.start) if date_range.start else dts[0]
    end = np.datetime64(date_range.end) if date_range.end else dts[-1]
    return (start, end)

def _assert_no_timezone(date_range):
    if False:
        for i in range(10):
            print('nop')
    for _dt in (date_range.start, date_range.end):
        if _dt and _dt.tzinfo is not None:
            raise ValueError('DateRange with timezone not supported')

class PandasSeriesStore(PandasStore):
    TYPE = 'pandasseries'
    SERIALIZER = SeriesSerializer()

    @staticmethod
    def can_write_type(data):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(data, Series)

    def can_write(self, version, symbol, data):
        if False:
            for i in range(10):
                print('nop')
        if self.can_write_type(data):
            if data.dtype is NP_OBJECT_DTYPE or data.index.dtype is NP_OBJECT_DTYPE:
                return self.SERIALIZER.can_convert_to_records_without_objects(data, symbol)
            return True
        return False

    def write(self, arctic_lib, version, symbol, item, previous_version):
        if False:
            i = 10
            return i + 15
        (item, md) = self.SERIALIZER.serialize(item)
        super(PandasSeriesStore, self).write(arctic_lib, version, symbol, item, previous_version, dtype=md)

    def append(self, arctic_lib, version, symbol, item, previous_version, **kwargs):
        if False:
            return 10
        (item, md) = self.SERIALIZER.serialize(item)
        super(PandasSeriesStore, self).append(arctic_lib, version, symbol, item, previous_version, dtype=md, **kwargs)

    def read_options(self):
        if False:
            i = 10
            return i + 15
        return super(PandasSeriesStore, self).read_options()

    def read(self, arctic_lib, version, symbol, **kwargs):
        if False:
            print('Hello World!')
        item = super(PandasSeriesStore, self).read(arctic_lib, version, symbol, **kwargs)
        force_bytes_to_unicode = kwargs.get('force_bytes_to_unicode', FORCE_BYTES_TO_UNICODE)
        return self.SERIALIZER.deserialize(item, force_bytes_to_unicode=force_bytes_to_unicode)

class PandasDataFrameStore(PandasStore):
    TYPE = 'pandasdf'
    SERIALIZER = DataFrameSerializer()

    @staticmethod
    def can_write_type(data):
        if False:
            while True:
                i = 10
        return isinstance(data, DataFrame)

    def can_write(self, version, symbol, data):
        if False:
            return 10
        if self.can_write_type(data):
            if NP_OBJECT_DTYPE in data.dtypes.values or data.index.dtype is NP_OBJECT_DTYPE:
                return self.SERIALIZER.can_convert_to_records_without_objects(data, symbol)
            return True
        return False

    def write(self, arctic_lib, version, symbol, item, previous_version):
        if False:
            while True:
                i = 10
        (item, md) = self.SERIALIZER.serialize(item)
        super(PandasDataFrameStore, self).write(arctic_lib, version, symbol, item, previous_version, dtype=md)

    def append(self, arctic_lib, version, symbol, item, previous_version, **kwargs):
        if False:
            return 10
        (item, md) = self.SERIALIZER.serialize(item)
        super(PandasDataFrameStore, self).append(arctic_lib, version, symbol, item, previous_version, dtype=md, **kwargs)

    def read(self, arctic_lib, version, symbol, **kwargs):
        if False:
            return 10
        item = super(PandasDataFrameStore, self).read(arctic_lib, version, symbol, **kwargs)
        force_bytes_to_unicode = kwargs.get('force_bytes_to_unicode', FORCE_BYTES_TO_UNICODE)
        return self.SERIALIZER.deserialize(item, force_bytes_to_unicode=force_bytes_to_unicode)

    def read_options(self):
        if False:
            return 10
        return super(PandasDataFrameStore, self).read_options()

class PandasPanelStore(PandasDataFrameStore):
    TYPE = 'pandaspan'

    @staticmethod
    def can_write_type(data):
        if False:
            for i in range(10):
                print('nop')
        from pandas import Panel
        return isinstance(data, Panel)

    def can_write(self, version, symbol, data):
        if False:
            return 10
        if self.can_write_type(data):
            frame = data.to_frame(filter_observations=False)
            if NP_OBJECT_DTYPE in frame.dtypes.values or (hasattr(data, 'index') and data.index.dtype is NP_OBJECT_DTYPE):
                return self.SERIALIZER.can_convert_to_records_without_objects(frame, symbol)
            return True
        return False

    def write(self, arctic_lib, version, symbol, item, previous_version):
        if False:
            i = 10
            return i + 15
        if np.product(item.shape) == 0:
            raise ValueError('Cannot insert a zero size panel into mongo.')
        if not np.all((len(i.names) == 1 for i in item.axes)):
            raise ValueError('Cannot insert panels with multiindexes')
        item = item.to_frame(filter_observations=False)
        if len(set(item.dtypes)) == 1:
            item = DataFrame(item.stack())
        elif item.columns.dtype != np.dtype('object'):
            raise ValueError('Cannot support non-object dtypes for columns')
        super(PandasPanelStore, self).write(arctic_lib, version, symbol, item, previous_version)

    def read(self, arctic_lib, version, symbol, **kwargs):
        if False:
            print('Hello World!')
        item = super(PandasPanelStore, self).read(arctic_lib, version, symbol, **kwargs)
        if len(item.index.names) == 3:
            return item.iloc[:, 0].unstack().to_panel()
        return item.to_panel()

    def read_options(self):
        if False:
            return 10
        return super(PandasPanelStore, self).read_options()

    def append(self, arctic_lib, version, symbol, item, previous_version, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError('Appending not supported for pandas.Panel')