import numpy as np
from xarray import DataArray, Dataset, Variable

def test_variable_typed_ops() -> None:
    if False:
        while True:
            i = 10
    'Tests for type checking of typed_ops on Variable'
    var = Variable(dims=['t'], data=[1, 2, 3])

    def _test(var: Variable) -> None:
        if False:
            i = 10
            return i + 15
        assert isinstance(var, Variable)
    _int: int = 1
    _list = [1, 2, 3]
    _ndarray = np.array([1, 2, 3])
    _test(var + _int)
    _test(var + _list)
    _test(var + _ndarray)
    _test(var + var)
    _test(_int + var)
    _test(_list + var)
    _test(_ndarray + var)
    _test(var == _int)
    _test(var == _list)
    _test(var == _ndarray)
    _test(_int == var)
    _test(_list == var)
    _test(_ndarray == var)
    _test(var < _int)
    _test(var < _list)
    _test(var < _ndarray)
    _test(_int > var)
    _test(_list > var)
    _test(_ndarray > var)
    var += _int
    var += _list
    var += _ndarray
    _test(-var)

def test_dataarray_typed_ops() -> None:
    if False:
        return 10
    'Tests for type checking of typed_ops on DataArray'
    da = DataArray([1, 2, 3], dims=['t'])

    def _test(da: DataArray) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(da, DataArray)
    _int: int = 1
    _list = [1, 2, 3]
    _ndarray = np.array([1, 2, 3])
    _var = Variable(dims=['t'], data=[1, 2, 3])
    _test(da + _int)
    _test(da + _list)
    _test(da + _ndarray)
    _test(da + _var)
    _test(da + da)
    _test(_int + da)
    _test(_list + da)
    _test(_ndarray + da)
    _test(_var + da)
    _test(da == _int)
    _test(da == _list)
    _test(da == _ndarray)
    _test(da == _var)
    _test(_int == da)
    _test(_list == da)
    _test(_ndarray == da)
    _test(_var == da)
    _test(da < _int)
    _test(da < _list)
    _test(da < _ndarray)
    _test(da < _var)
    _test(_int > da)
    _test(_list > da)
    _test(_ndarray > da)
    _test(_var > da)
    da += _int
    da += _list
    da += _ndarray
    da += _var
    _test(-da)

def test_dataset_typed_ops() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Tests for type checking of typed_ops on Dataset'
    ds = Dataset({'a': ('t', [1, 2, 3])})

    def _test(ds: Dataset) -> None:
        if False:
            print('Hello World!')
        assert isinstance(ds, Dataset)
    _int: int = 1
    _list = [1, 2, 3]
    _ndarray = np.array([1, 2, 3])
    _var = Variable(dims=['t'], data=[1, 2, 3])
    _da = DataArray([1, 2, 3], dims=['t'])
    _test(ds + _int)
    _test(ds + _list)
    _test(ds + _ndarray)
    _test(ds + _var)
    _test(ds + _da)
    _test(ds + ds)
    _test(_int + ds)
    _test(_list + ds)
    _test(_ndarray + ds)
    _test(_var + ds)
    _test(_da + ds)
    _test(ds == _int)
    _test(ds == _list)
    _test(ds == _ndarray)
    _test(ds == _var)
    _test(ds == _da)
    _test(_int == ds)
    _test(_list == ds)
    _test(_ndarray == ds)
    _test(_var == ds)
    _test(_da == ds)
    _test(ds < _int)
    _test(ds < _list)
    _test(ds < _ndarray)
    _test(ds < _var)
    _test(ds < _da)
    _test(_int > ds)
    _test(_list > ds)
    _test(_ndarray > ds)
    _test(_var > ds)
    _test(_da > ds)
    ds += _int
    ds += _list
    ds += _ndarray
    ds += _var
    ds += _da
    _test(-ds)

def test_dataarray_groupy_typed_ops() -> None:
    if False:
        return 10
    'Tests for type checking of typed_ops on DataArrayGroupBy'
    da = DataArray([1, 2, 3], coords={'x': ('t', [1, 2, 2])}, dims=['t'])
    grp = da.groupby('x')

    def _testda(da: DataArray) -> None:
        if False:
            i = 10
            return i + 15
        assert isinstance(da, DataArray)

    def _testds(ds: Dataset) -> None:
        if False:
            return 10
        assert isinstance(ds, Dataset)
    _da = DataArray([5, 6], coords={'x': [1, 2]}, dims='x')
    _ds = _da.to_dataset(name='a')
    _testda(grp + _da)
    _testds(grp + _ds)
    _testda(_da + grp)
    _testds(_ds + grp)
    _testda(grp == _da)
    _testda(_da == grp)
    _testds(grp == _ds)
    _testds(_ds == grp)
    _testda(grp < _da)
    _testda(_da > grp)
    _testds(grp < _ds)
    _testds(_ds > grp)

def test_dataset_groupy_typed_ops() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Tests for type checking of typed_ops on DatasetGroupBy'
    ds = Dataset({'a': ('t', [1, 2, 3])}, coords={'x': ('t', [1, 2, 2])})
    grp = ds.groupby('x')

    def _test(ds: Dataset) -> None:
        if False:
            return 10
        assert isinstance(ds, Dataset)
    _da = DataArray([5, 6], coords={'x': [1, 2]}, dims='x')
    _ds = _da.to_dataset(name='a')
    _test(grp + _da)
    _test(grp + _ds)
    _test(_da + grp)
    _test(_ds + grp)
    _test(grp == _da)
    _test(_da == grp)
    _test(grp == _ds)
    _test(_ds == grp)
    _test(grp < _da)
    _test(_da > grp)
    _test(grp < _ds)
    _test(_ds > grp)