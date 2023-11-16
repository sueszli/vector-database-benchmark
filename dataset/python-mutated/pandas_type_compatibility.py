"""Utilities for converting between Beam schemas and pandas DataFrames.

Imposes a mapping between native Python typings (specifically those compatible
with :mod:`apache_beam.typehints.schemas`), and common pandas dtypes::

  pandas dtype                    Python typing
  np.int{8,16,32,64}      <-----> np.int{8,16,32,64}*
  pd.Int{8,16,32,64}Dtype <-----> Optional[np.int{8,16,32,64}]*
  np.float{32,64}         <-----> Optional[np.float{32,64}]
                             \\--- np.float{32,64}
  Not supported           <------ Optional[bytes]
  np.bool                 <-----> np.bool
  np.dtype('S')           <-----> bytes
  pd.BooleanDType()       <-----> Optional[bool]
  pd.StringDType()        <-----> Optional[str]
                             \\--- str
  np.object               <-----> Any

  * int, float, bool are treated the same as np.int64, np.float64, np.bool

Note that when converting to pandas dtypes, any types not specified here are
shunted to ``np.object``.

Similarly when converting from pandas to Python types, types that aren't
otherwise specified here are shunted to ``Any``. Notably, this includes
``np.datetime64``.

Pandas does not support hierarchical data natively. Currently, all structured
types (``Sequence``, ``Mapping``, nested ``NamedTuple`` types), are
shunted to ``np.object`` like all other unknown types. In the future these
types may be given special consideration.

Note utilities in this package are for internal use only, we make no backward
compatibility guarantees, except for the type mapping itself.
"""
from typing import Any
from typing import List
from typing import Optional
import numpy as np
import pandas as pd
from apache_beam.typehints.batch import BatchConverter
from apache_beam.typehints.row_type import RowTypeConstraint
from apache_beam.typehints.typehints import is_nullable
from apache_beam.typehints.typehints import normalize
__all__ = []
INDEX_OPTION_NAME = 'beam:dataframe:index'
_BIDIRECTIONAL = [(bool, bool), (np.int8, np.int8), (np.int16, np.int16), (np.int32, np.int32), (np.int64, np.int64), (pd.Int8Dtype(), Optional[np.int8]), (pd.Int16Dtype(), Optional[np.int16]), (pd.Int32Dtype(), Optional[np.int32]), (pd.Int64Dtype(), Optional[np.int64]), (np.float32, Optional[np.float32]), (np.float64, Optional[np.float64]), (object, Any), (pd.StringDtype(), Optional[str]), (pd.BooleanDtype(), Optional[bool])]
PANDAS_TO_BEAM = {pd.Series([], dtype=dtype).dtype: fieldtype for (dtype, fieldtype) in _BIDIRECTIONAL}
BEAM_TO_PANDAS = {fieldtype: dtype for (dtype, fieldtype) in _BIDIRECTIONAL}
OPTIONAL_SHUNTS = [np.float32, np.float64, str]
for typehint in OPTIONAL_SHUNTS:
    BEAM_TO_PANDAS[typehint] = BEAM_TO_PANDAS[Optional[typehint]]
BEAM_TO_PANDAS[int] = BEAM_TO_PANDAS[np.int64]
BEAM_TO_PANDAS[Optional[int]] = BEAM_TO_PANDAS[Optional[np.int64]]
BEAM_TO_PANDAS[float] = BEAM_TO_PANDAS[np.float64]
BEAM_TO_PANDAS[Optional[float]] = BEAM_TO_PANDAS[Optional[np.float64]]
BEAM_TO_PANDAS[bytes] = 'bytes'
BEAM_TO_PANDAS.update({normalize(typehint): pandas_dtype for (typehint, pandas_dtype) in BEAM_TO_PANDAS.items()})

def dtype_from_typehint(typehint):
    if False:
        while True:
            i = 10
    return BEAM_TO_PANDAS.get(typehint, object)

def dtype_to_fieldtype(dtype):
    if False:
        i = 10
        return i + 15
    fieldtype = PANDAS_TO_BEAM.get(dtype)
    if fieldtype is not None:
        return fieldtype
    elif dtype.kind == 'S':
        return bytes
    else:
        return Any

@BatchConverter.register(name='pandas')
def create_pandas_batch_converter(element_type: type, batch_type: type) -> BatchConverter:
    if False:
        for i in range(10):
            print('nop')
    if batch_type == pd.DataFrame:
        return DataFrameBatchConverter.from_typehints(element_type=element_type, batch_type=batch_type)
    elif batch_type == pd.Series:
        return SeriesBatchConverter.from_typehints(element_type=element_type, batch_type=batch_type)
    raise TypeError('batch type must be pd.Series or pd.DataFrame')

class DataFrameBatchConverter(BatchConverter):

    def __init__(self, element_type: RowTypeConstraint):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(pd.DataFrame, element_type)
        self._columns = [name for (name, _) in element_type._fields]

    @staticmethod
    def from_typehints(element_type, batch_type) -> Optional['DataFrameBatchConverter']:
        if False:
            i = 10
            return i + 15
        assert batch_type == pd.DataFrame
        if not isinstance(element_type, RowTypeConstraint):
            element_type = RowTypeConstraint.from_user_type(element_type)
            if element_type is None:
                raise TypeError('Element type must be compatible with Beam Schemas (https://beam.apache.org/documentation/programming-guide/#schemas) for batch type pd.DataFrame')
        index_columns = [field_name for (field_name, field_options) in element_type._field_options.items() if any((key == INDEX_OPTION_NAME for (key, value) in field_options))]
        if index_columns:
            return DataFrameBatchConverterKeepIndex(element_type, index_columns)
        else:
            return DataFrameBatchConverterDropIndex(element_type)

    def _get_series(self, batch: pd.DataFrame):
        if False:
            return 10
        raise NotImplementedError

    def explode_batch(self, batch: pd.DataFrame):
        if False:
            while True:
                i = 10

        def make_null_checking_generator(series):
            if False:
                for i in range(10):
                    print('nop')
            nulls = pd.isnull(series)
            return (None if isnull else value for (isnull, value) in zip(nulls, series))
        all_series = self._get_series(batch)
        iterators = [make_null_checking_generator(series) for series in all_series]
        for values in zip(*iterators):
            yield self._element_type.user_type(**{column: value for (column, value) in zip(self._columns, values)})

    def combine_batches(self, batches: List[pd.DataFrame]):
        if False:
            i = 10
            return i + 15
        return pd.concat(batches)

    def estimate_byte_size(self, batch: pd.DataFrame):
        if False:
            while True:
                i = 10
        return batch.memory_usage().sum()

    def get_length(self, batch: pd.DataFrame):
        if False:
            print('Hello World!')
        return len(batch)

class DataFrameBatchConverterDropIndex(DataFrameBatchConverter):
    """A DataFrameBatchConverter that assumes the DataFrame index has no meaning.

  When producing a DataFrame from Rows, a meaningless index will be generated.
  When exploding a DataFrame into Rows, the index will be dropped.
  """

    def _get_series(self, batch: pd.DataFrame):
        if False:
            while True:
                i = 10
        return [batch[column] for column in batch.columns]

    def produce_batch(self, elements):
        if False:
            for i in range(10):
                print('nop')
        batch = pd.DataFrame.from_records(elements, columns=self._columns)
        for (column, typehint) in self._element_type._fields:
            batch[column] = batch[column].astype(dtype_from_typehint(typehint))
        return batch

class DataFrameBatchConverterKeepIndex(DataFrameBatchConverter):
    """A DataFrameBatchConverter that preserves the DataFrame index.

  This is tracked via options on the Beam schema. Each field in the schema that
  should map to the index is tagged in an option with name 'dataframe:index'.
  """

    def __init__(self, element_type: RowTypeConstraint, index_columns: List[Any]):
        if False:
            i = 10
            return i + 15
        super().__init__(element_type)
        self._index_columns = index_columns

    def _get_series(self, batch: pd.DataFrame):
        if False:
            while True:
                i = 10
        assert list(batch.index.names) == self._index_columns
        return [batch.index.get_level_values(i) for i in range(len(batch.index.names))] + [batch[column] for column in batch.columns]

    def produce_batch(self, elements):
        if False:
            return 10
        batch = pd.DataFrame.from_records(elements, columns=self._columns)
        for (column, typehint) in self._element_type._fields:
            batch[column] = batch[column].astype(dtype_from_typehint(typehint))
        return batch.set_index(self._index_columns)

class SeriesBatchConverter(BatchConverter):

    def __init__(self, element_type: type, dtype):
        if False:
            return 10
        super().__init__(pd.DataFrame, element_type)
        self._dtype = dtype
        if is_nullable(element_type):

            def unbatch(series):
                if False:
                    return 10
                for (isnull, value) in zip(pd.isnull(series), series):
                    yield (None if isnull else value)
        else:

            def unbatch(series):
                if False:
                    i = 10
                    return i + 15
                yield from series
        self.explode_batch = unbatch

    @staticmethod
    def from_typehints(element_type, batch_type) -> Optional['SeriesBatchConverter']:
        if False:
            for i in range(10):
                print('nop')
        assert batch_type == pd.Series
        dtype = dtype_from_typehint(element_type)
        return SeriesBatchConverter(element_type, dtype)

    def produce_batch(self, elements: List[Any]) -> pd.Series:
        if False:
            for i in range(10):
                print('nop')
        return pd.Series(elements, dtype=self._dtype)

    def explode_batch(self, batch: pd.Series):
        if False:
            return 10
        raise NotImplementedError('explode_batch should be generated in SeriesBatchConverter.__init__')

    def combine_batches(self, batches: List[pd.Series]):
        if False:
            i = 10
            return i + 15
        return pd.concat(batches)

    def estimate_byte_size(self, batch: pd.Series):
        if False:
            while True:
                i = 10
        return batch.memory_usage()

    def get_length(self, batch: pd.Series):
        if False:
            return 10
        return len(batch)