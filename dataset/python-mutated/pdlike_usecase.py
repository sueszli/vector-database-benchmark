"""
Implementation of a minimal Pandas-like API.
"""
import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import typeof_impl, type_callable, register_model, lower_builtin, box, unbox, NativeValue, overload, overload_attribute, overload_method, make_attribute_wrapper
from numba.core.imputils import impl_ret_borrowed

class Index(object):
    """
    A minimal pandas.Index-like object.
    """

    def __init__(self, data):
        if False:
            i = 10
            return i + 15
        assert isinstance(data, np.ndarray)
        assert data.ndim == 1
        self._data = data

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self._data)

    @property
    def dtype(self):
        if False:
            i = 10
            return i + 15
        return self._data.dtype

    @property
    def flags(self):
        if False:
            for i in range(10):
                print('nop')
        return self._data.flags

class IndexType(types.Buffer):
    """
    The type class for Index objects.
    """
    array_priority = 1000

    def __init__(self, dtype, layout, pyclass):
        if False:
            for i in range(10):
                print('nop')
        self.pyclass = pyclass
        super(IndexType, self).__init__(dtype, 1, layout)

    @property
    def key(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.pyclass, self.dtype, self.layout)

    @property
    def as_array(self):
        if False:
            for i in range(10):
                print('nop')
        return types.Array(self.dtype, 1, self.layout)

    def copy(self, dtype=None, ndim=1, layout=None):
        if False:
            while True:
                i = 10
        assert ndim == 1
        if dtype is None:
            dtype = self.dtype
        layout = layout or self.layout
        return type(self)(dtype, layout, self.pyclass)

class Series(object):
    """
    A minimal pandas.Series-like object.
    """

    def __init__(self, data, index):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(data, np.ndarray)
        assert isinstance(index, Index)
        assert data.ndim == 1
        self._values = data
        self._index = index

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self._values)

    @property
    def dtype(self):
        if False:
            for i in range(10):
                print('nop')
        return self._values.dtype

    @property
    def flags(self):
        if False:
            return 10
        return self._values.flags

class SeriesType(types.ArrayCompatible):
    """
    The type class for Series objects.
    """
    array_priority = 1000

    def __init__(self, dtype, index):
        if False:
            print('Hello World!')
        assert isinstance(index, IndexType)
        self.dtype = dtype
        self.index = index
        self.values = types.Array(self.dtype, 1, 'C')
        name = 'series(%s, %s)' % (dtype, index)
        super(SeriesType, self).__init__(name)

    @property
    def key(self):
        if False:
            return 10
        return (self.dtype, self.index)

    @property
    def as_array(self):
        if False:
            for i in range(10):
                print('nop')
        return self.values

    def copy(self, dtype=None, ndim=1, layout='C'):
        if False:
            for i in range(10):
                print('nop')
        assert ndim == 1
        assert layout == 'C'
        if dtype is None:
            dtype = self.dtype
        return type(self)(dtype, self.index)

@typeof_impl.register(Index)
def typeof_index(val, c):
    if False:
        for i in range(10):
            print('nop')
    arrty = typeof_impl(val._data, c)
    assert arrty.ndim == 1
    return IndexType(arrty.dtype, arrty.layout, type(val))

@typeof_impl.register(Series)
def typeof_series(val, c):
    if False:
        for i in range(10):
            print('nop')
    index = typeof_impl(val._index, c)
    arrty = typeof_impl(val._values, c)
    assert arrty.ndim == 1
    assert arrty.layout == 'C'
    return SeriesType(arrty.dtype, index)

@type_callable('__array_wrap__')
def type_array_wrap(context):
    if False:
        i = 10
        return i + 15

    def typer(input_type, result):
        if False:
            print('Hello World!')
        if isinstance(input_type, (IndexType, SeriesType)):
            return input_type.copy(dtype=result.dtype, ndim=result.ndim, layout=result.layout)
    return typer

@type_callable(Series)
def type_series_constructor(context):
    if False:
        return 10

    def typer(data, index):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(index, IndexType) and isinstance(data, types.Array):
            assert data.layout == 'C'
            assert data.ndim == 1
            return SeriesType(data.dtype, index)
    return typer

@register_model(IndexType)
class IndexModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        if False:
            return 10
        members = [('data', fe_type.as_array)]
        models.StructModel.__init__(self, dmm, fe_type, members)

@register_model(SeriesType)
class SeriesModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        if False:
            while True:
                i = 10
        members = [('index', fe_type.index), ('values', fe_type.as_array)]
        models.StructModel.__init__(self, dmm, fe_type, members)
make_attribute_wrapper(IndexType, 'data', '_data')
make_attribute_wrapper(SeriesType, 'index', '_index')
make_attribute_wrapper(SeriesType, 'values', '_values')

def make_index(context, builder, typ, **kwargs):
    if False:
        i = 10
        return i + 15
    return cgutils.create_struct_proxy(typ)(context, builder, **kwargs)

def make_series(context, builder, typ, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return cgutils.create_struct_proxy(typ)(context, builder, **kwargs)

@lower_builtin('__array__', IndexType)
def index_as_array(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    val = make_index(context, builder, sig.args[0], ref=args[0])
    return val._get_ptr_by_name('data')

@lower_builtin('__array__', SeriesType)
def series_as_array(context, builder, sig, args):
    if False:
        print('Hello World!')
    val = make_series(context, builder, sig.args[0], ref=args[0])
    return val._get_ptr_by_name('values')

@lower_builtin('__array_wrap__', IndexType, types.Array)
def index_wrap_array(context, builder, sig, args):
    if False:
        print('Hello World!')
    dest = make_index(context, builder, sig.return_type)
    dest.data = args[1]
    return impl_ret_borrowed(context, builder, sig.return_type, dest._getvalue())

@lower_builtin('__array_wrap__', SeriesType, types.Array)
def series_wrap_array(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    src = make_series(context, builder, sig.args[0], value=args[0])
    dest = make_series(context, builder, sig.return_type)
    dest.values = args[1]
    dest.index = src.index
    return impl_ret_borrowed(context, builder, sig.return_type, dest._getvalue())

@lower_builtin(Series, types.Array, IndexType)
def pdseries_constructor(context, builder, sig, args):
    if False:
        print('Hello World!')
    (data, index) = args
    series = make_series(context, builder, sig.return_type)
    series.index = index
    series.values = data
    return impl_ret_borrowed(context, builder, sig.return_type, series._getvalue())

@unbox(IndexType)
def unbox_index(typ, obj, c):
    if False:
        return 10
    '\n    Convert a Index object to a native structure.\n    '
    data = c.pyapi.object_getattr_string(obj, '_data')
    index = make_index(c.context, c.builder, typ)
    index.data = c.unbox(typ.as_array, data).value
    return NativeValue(index._getvalue())

@unbox(SeriesType)
def unbox_series(typ, obj, c):
    if False:
        while True:
            i = 10
    '\n    Convert a Series object to a native structure.\n    '
    index = c.pyapi.object_getattr_string(obj, '_index')
    values = c.pyapi.object_getattr_string(obj, '_values')
    series = make_series(c.context, c.builder, typ)
    series.index = c.unbox(typ.index, index).value
    series.values = c.unbox(typ.values, values).value
    return NativeValue(series._getvalue())

@box(IndexType)
def box_index(typ, val, c):
    if False:
        print('Hello World!')
    '\n    Convert a native index structure to a Index object.\n    '
    index = make_index(c.context, c.builder, typ, value=val)
    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.pyclass))
    arrayobj = c.box(typ.as_array, index.data)
    indexobj = c.pyapi.call_function_objargs(classobj, (arrayobj,))
    return indexobj

@box(SeriesType)
def box_series(typ, val, c):
    if False:
        i = 10
        return i + 15
    '\n    Convert a native series structure to a Series object.\n    '
    series = make_series(c.context, c.builder, typ, value=val)
    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(Series))
    indexobj = c.box(typ.index, series.index)
    arrayobj = c.box(typ.as_array, series.values)
    seriesobj = c.pyapi.call_function_objargs(classobj, (arrayobj, indexobj))
    return seriesobj

@overload_attribute(IndexType, 'is_monotonic_increasing')
def index_is_monotonic_increasing(index):
    if False:
        while True:
            i = 10
    '\n    Index.is_monotonic_increasing\n    '

    def getter(index):
        if False:
            while True:
                i = 10
        data = index._data
        if len(data) == 0:
            return True
        u = data[0]
        for v in data:
            if v < u:
                return False
            v = u
        return True
    return getter

@overload(len)
def series_len(series):
    if False:
        print('Hello World!')
    '\n    len(Series)\n    '
    if isinstance(series, SeriesType):

        def len_impl(series):
            if False:
                return 10
            return len(series._values)
        return len_impl

@overload_method(SeriesType, 'clip')
def series_clip(series, lower, upper):
    if False:
        while True:
            i = 10
    '\n    Series.clip(...)\n    '

    def clip_impl(series, lower, upper):
        if False:
            while True:
                i = 10
        data = series._values.copy()
        for i in range(len(data)):
            v = data[i]
            if v < lower:
                data[i] = lower
            elif v > upper:
                data[i] = upper
        return Series(data, series._index)
    return clip_impl