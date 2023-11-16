from typing import Sequence
import dask.base
import numpy as np
import pyarrow as pa
import vaex
import vaex.array_types

class DataType:
    """Wraps numpy and arrow data types in a uniform interface

    Examples:
    >>> import numpy as np
    >>> import pyarrow as pa
    >>> type1 = DataType(np.dtype('f8'))
    >>> type1
    float64
    >>> type2 = DataType(np.dtype('>f8'))
    >>> type2
    >f8
    >>> type1 in [float, int]
    True
    >>> type1 == type2
    False
    >>> type1 == pa.float64()
    True
    >>> type1 == pa.int64()
    False
    >>> DataType(np.dtype('f4'))
    float32
    >>> DataType(pa.float32())
    float32

    """

    def __init__(self, dtype):
        if False:
            i = 10
            return i + 15
        if isinstance(dtype, DataType):
            self.internal = dtype.internal
        elif isinstance(dtype, pa.DataType):
            self.internal = dtype
        else:
            self.internal = np.dtype(dtype)

    def to_native(self):
        if False:
            return 10
        'Removes non-native endianness'
        return DataType(vaex.utils.to_native_dtype(self.internal))

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.__class__.__name__, self.internal))

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if self.is_encoded:
            return self.value_type == other
        if other is str:
            return self.is_string
        if other is float:
            return self.is_float
        if other is int:
            return self.is_integer
        if other is list:
            return self.is_list
        if other is dict:
            return self.is_struct
        if other is object:
            return self.is_object
        if isinstance(other, str):
            tester = 'is_' + other
            if hasattr(self, tester):
                return getattr(self, tester)
        if not isinstance(other, DataType):
            other = DataType(other)
        if other.is_primitive:
            if self.is_arrow:
                other = DataType(other.arrow)
            if self.is_numpy:
                other = DataType(other.numpy)
        return vaex.array_types.same_type(self.internal, other.internal)

    def __repr__(self):
        if False:
            while True:
                i = 10
        'Standard representation for datatypes\n\n\n        >>> dtype = DataType(pa.float64())\n        >>> dtype.internal\n        DataType(double)\n        >>> dtype\n        float64\n        >>> DataType(pa.float32())\n        float32\n        >>> DataType(pa.dictionary(pa.int32(), pa.string()))\n        dictionary<values=string, indices=int32, ordered=0>\n        '
        internal = self.internal
        if self.is_datetime:
            internal = self.numpy
        repr = str(internal)
        translate = {'datetime64': 'datetime64[ns]', 'double': 'float64', 'float': 'float32'}
        return translate.get(repr, repr)

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        "Alias of dtype.numpy.name or str(dtype.arrow) if not primitive\n\n        >>> DataType(np.dtype('f8')).name\n        'float64'\n        >>> DataType(np.dtype('>f4')).name\n        'float32'\n        >>> DataType(pa.float64()).name\n        'float64'\n        >>> DataType(pa.large_string()).name\n        'large_string'\n        >>> DataType(pa.string()).name\n        'string'\n        >>> DataType(pa.bool_()).name\n        'bool'\n        >>> DataType(np.dtype('?')).name\n        'bool'\n        >>> DataType(pa.dictionary(pa.int32(), pa.string())).name\n        'dictionary<values=string, indices=int32, ordered=0>'\n        "
        return self.numpy.name if self.is_primitive or self.is_datetime else str(self.internal)

    @property
    def kind(self):
        if False:
            while True:
                i = 10
        return self.numpy.kind

    @property
    def numpy(self):
        if False:
            print('Hello World!')
        "Return the numpy equivalent type\n\n        >>> DataType(pa.float64()).numpy == np.dtype('f8')\n        True\n        "
        return vaex.array_types.to_numpy_type(self.internal)

    @property
    def arrow(self):
        if False:
            i = 10
            return i + 15
        "Return the Apache Arrow equivalent type\n\n        >>> DataType(np.dtype('f8')).arrow == pa.float64()\n        True\n        "
        return vaex.array_types.to_arrow_type(self.internal)

    @property
    def is_arrow(self):
        if False:
            for i in range(10):
                print('nop')
        "Return True if it wraps an Arrow type\n\n        >>> DataType(pa.string()).is_arrow\n        True\n        >>> DataType(pa.int32()).is_arrow\n        True\n        >>> DataType(np.dtype('f8')).is_arrow\n        False\n        "
        return isinstance(self.internal, pa.DataType)

    @property
    def is_numpy(self):
        if False:
            print('Hello World!')
        "Return True if it wraps an NumPy dtype\n\n        >>> DataType(np.dtype('f8')).is_numpy\n        True\n        >>> DataType(pa.string()).is_numpy\n        False\n        >>> DataType(pa.int32()).is_numpy\n        False\n        "
        return isinstance(self.internal, np.dtype)

    @property
    def is_numeric(self):
        if False:
            return 10
        "Tests if type is numerical (float, int)\n\n        >>> DataType(np.dtype('f8')).is_numeric\n        True\n        >>> DataType(pa.float32()).is_numeric\n        True\n        >>> DataType(pa.large_string()).is_numeric\n        False\n        "
        try:
            return self.kind in 'fiu'
        except NotImplementedError:
            return False

    @property
    def is_primitive(self):
        if False:
            return 10
        "Tests if type is numerical (float, int, bool)\n\n        >>> DataType(np.dtype('b')).is_primitive\n        True\n        >>> DataType(pa.bool_()).is_primitive\n        True\n        "
        if self.is_arrow:
            return pa.types.is_primitive(self.internal)
        else:
            return self.kind in 'fiub'

    @property
    def is_datetime(self):
        if False:
            while True:
                i = 10
        "Tests if dtype is datetime (numpy) or timestamp (arrow)\n\n        Date/Time:\n        >>> date_type = DataType(np.dtype('datetime64'))\n        >>> date_type\n        datetime64[ns]\n        >>> date_type == 'datetime'\n        True\n\n        Using Arrow:\n\n        >>> date_type = DataType(pa.timestamp('ns'))\n        >>> date_type\n        datetime64[ns]\n        >>> date_type == 'datetime'\n        True\n        >>> date_type = DataType(pa.large_string())\n        >>> date_type.is_datetime\n        False\n        "
        if self.is_arrow:
            return pa.types.is_timestamp(self.internal)
        else:
            return self.kind in 'M'

    @property
    def is_timedelta(self):
        if False:
            return 10
        "Test if timedelta\n\n        >>> dtype = DataType(np.dtype('timedelta64'))\n        >>> dtype\n        timedelta64\n        >>> dtype == 'timedelta'\n        True\n        >>> dtype.is_timedelta\n        True\n        >>> date_type = DataType(pa.large_string())\n        >>> date_type.is_timedelta\n        False\n        "
        if self.is_arrow:
            return isinstance(self.arrow, pa.DurationType)
        else:
            return self.kind in 'm'

    @property
    def is_temporal(self):
        if False:
            i = 10
            return i + 15
        'Alias of (is_datetime or is_timedelta)'
        return self.is_datetime or self.is_timedelta

    @property
    def is_float(self):
        if False:
            print('Hello World!')
        "Test if a float (float32 or float64)\n\n        >>> dtype = DataType(np.dtype('float32'))\n        >>> dtype\n        float32\n        >>> dtype == 'float'\n        True\n        >>> dtype == float\n        True\n        >>> dtype.is_float\n        True\n\n        Using Arrow:\n        >>> DataType(pa.float32()) == float\n        True\n        "
        return self.is_primitive and vaex.array_types.to_numpy_type(self.internal).kind in 'f'

    @property
    def is_unsigned(self):
        if False:
            print('Hello World!')
        "Test if an (unsigned) integer\n\n        >>> dtype = DataType(np.dtype('uint32'))\n        >>> dtype\n        uint32\n        >>> dtype == 'unsigned'\n        True\n        >>> dtype.is_unsigned\n        True\n\n        Using Arrow:\n        >>> DataType(pa.uint32()).is_unsigned\n        True\n        "
        return self.is_primitive and vaex.array_types.to_numpy_type(self.internal).kind in 'u'

    @property
    def is_signed(self):
        if False:
            for i in range(10):
                print('nop')
        "Test if a (signed) integer\n\n        >>> dtype = DataType(np.dtype('int32'))\n        >>> dtype\n        int32\n        >>> dtype == 'signed'\n        True\n\n        Using Arrow:\n        >>> DataType(pa.int32()).is_signed\n        True\n        "
        return self.is_primitive and vaex.array_types.to_numpy_type(self.internal).kind in 'i'

    @property
    def is_integer(self):
        if False:
            print('Hello World!')
        "Test if an (unsigned or signed) integer\n\n        >>> DataType(np.dtype('uint32')) == 'integer'\n        True\n        >>> DataType(np.dtype('int8')) == int\n        True\n        >>> DataType(np.dtype('int16')).is_integer\n        True\n\n        Using Arrow:\n        >>> DataType(pa.uint32()).is_integer\n        True\n        >>> DataType(pa.int16()) == int\n        True\n\n        "
        return self.is_primitive and vaex.array_types.to_numpy_type(self.internal).kind in 'iu'

    @property
    def is_string(self):
        if False:
            return 10
        "Test if an (arrow) string or large_string\n\n        >>> DataType(pa.string()) == str\n        True\n        >>> DataType(pa.large_string()) == str\n        True\n        >>> DataType(pa.large_string()).is_string\n        True\n        >>> DataType(pa.large_string()) == 'string'\n        True\n        "
        return vaex.array_types.is_string_type(self.internal)

    @property
    def is_list(self):
        if False:
            return 10
        "Test if an (arrow) list or large_string\n\n        >>> DataType(pa.list_(pa.string())) == list\n        True\n        >>> DataType(pa.large_list(pa.string())) == list\n        True\n        >>> DataType(pa.list_(pa.string())).is_list\n        True\n        >>> DataType(pa.list_(pa.string())) == 'list'\n        True\n        "
        return self.is_arrow and (pa.types.is_list(self.internal) or pa.types.is_large_list(self.internal))

    @property
    def is_struct(self) -> bool:
        if False:
            print('Hello World!')
        "Test if an (arrow) struct\n\n        >>> DataType(pa.struct([pa.field('a', pa.utf8())])) == dict\n        True\n        >>> DataType(pa.struct([pa.field('a', pa.utf8())])).is_struct\n        True\n        >>> DataType(pa.struct([pa.field('a', pa.utf8())])) == 'struct'\n        True\n        "
        return self.is_arrow and pa.types.is_struct(self.internal)

    @property
    def is_object(self):
        if False:
            i = 10
            return i + 15
        'Test if a NumPy dtype=object (avoid if possible)'
        return self.is_numpy and self.internal == object

    @property
    def is_encoded(self):
        if False:
            i = 10
            return i + 15
        'Test if an (arrow) dictionary type (encoded data)\n\n        >>> DataType(pa.dictionary(pa.int32(), pa.string())) == str\n        True\n        >>> DataType(pa.dictionary(pa.int32(), pa.string())).is_encoded\n        True\n        '
        return self.is_arrow and pa.types.is_dictionary(self.internal)

    @property
    def value_type(self):
        if False:
            i = 10
            return i + 15
        'Return the DataType of the list values or values of an encoded type\n\n        >>> DataType(pa.list_(pa.string())).value_type\n        string\n        >>> DataType(pa.list_(pa.float64())).value_type\n        float64\n        >>> DataType(pa.dictionary(pa.int32(), pa.string())).value_type\n        string\n        '
        if not (self.is_list or self.is_encoded):
            raise TypeError(f'{self} is not a list or encoded type')
        return DataType(self.internal.value_type)

    @property
    def index_type(self):
        if False:
            while True:
                i = 10
        'Return the DataType of the index of an encoded type, or simple the type\n\n        >>> DataType(pa.string()).index_type\n        string\n        >>> DataType(pa.dictionary(pa.int32(), pa.string())).index_type\n        int32\n        '
        type = self.internal
        if self.is_encoded:
            type = self.internal.index_type
        return DataType(type)

    def upcast(self):
        if False:
            i = 10
            return i + 15
        "Cast to the higest data type matching the type\n\n        >>> DataType(np.dtype('uint32')).upcast()\n        uint64\n        >>> DataType(np.dtype('int8')).upcast()\n        int64\n        >>> DataType(np.dtype('float32')).upcast()\n        float64\n\n        Using Arrow\n        >>> DataType(pa.float32()).upcast()\n        float64\n        "
        return DataType(vaex.array_types.upcast(self.internal))

    @property
    def byteorder(self):
        if False:
            return 10
        return self.numpy.byteorder

    def create_array(self, values: Sequence):
        if False:
            while True:
                i = 10
        "Create an array from a sequence with the same dtype\n\n        If values is a list containing None, it will map to a masked array (numpy) or null values (arrow)\n\n        >>> DataType(np.dtype('float32')).create_array([1., 2.5, None, np.nan])\n        masked_array(data=[1.0, 2.5, --, nan],\n                     mask=[False, False,  True, False],\n               fill_value=1e+20)\n        >>> DataType(pa.float32()).create_array([1., 2.5, None, np.nan])  # doctest:+ELLIPSIS\n        <pyarrow.lib.FloatArray object at ...>\n        [\n          1,\n          2.5,\n          null,\n          nan\n        ]\n        "
        if self.is_arrow:
            if vaex.array_types.is_arrow_array(values):
                return values
            else:
                return pa.array(values, type=self.arrow)
        else:
            if isinstance(values, np.ndarray):
                return values.astype(self.internal, copy=False)
            mask = [k is None for k in values]
            if any(mask):
                values = [values[0] if k is None else k for k in values]
                return np.ma.array(values, mask=mask)
            else:
                return np.array(values)
            return np.asarray(values, dtype=self.numpy)

@dask.base.normalize_token.register(DataType)
def normalize_DataType(t):
    if False:
        print('Hello World!')
    return (type(t).__name__, t.internal)