from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates

@pytest.fixture
def dataarray() -> xr.DataArray:
    if False:
        print('Hello World!')
    return xr.DataArray(np.random.RandomState(0).randn(4, 6))

@pytest.fixture
def dask_dataarray(dataarray: xr.DataArray) -> xr.DataArray:
    if False:
        return 10
    pytest.importorskip('dask')
    return dataarray.chunk()

@pytest.fixture
def multiindex() -> xr.Dataset:
    if False:
        while True:
            i = 10
    midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=('level_1', 'level_2'))
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
    return xr.Dataset({}, midx_coords)

@pytest.fixture
def dataset() -> xr.Dataset:
    if False:
        i = 10
        return i + 15
    times = pd.date_range('2000-01-01', '2001-12-31', name='time')
    annual_cycle = np.sin(2 * np.pi * (times.dayofyear.values / 365.25 - 0.28))
    base = 10 + 15 * annual_cycle.reshape(-1, 1)
    tmin_values = base + 3 * np.random.randn(annual_cycle.size, 3)
    tmax_values = base + 10 + 3 * np.random.randn(annual_cycle.size, 3)
    return xr.Dataset({'tmin': (('time', 'location'), tmin_values), 'tmax': (('time', 'location'), tmax_values)}, {'time': times, 'location': ['<IA>', 'IN', 'IL']}, attrs={'description': 'Test data.'})

def test_short_data_repr_html(dataarray: xr.DataArray) -> None:
    if False:
        while True:
            i = 10
    data_repr = fh.short_data_repr_html(dataarray)
    assert data_repr.startswith('<pre>array')

def test_short_data_repr_html_non_str_keys(dataset: xr.Dataset) -> None:
    if False:
        return 10
    ds = dataset.assign({2: lambda x: x['tmin']})
    fh.dataset_repr(ds)

def test_short_data_repr_html_dask(dask_dataarray: xr.DataArray) -> None:
    if False:
        for i in range(10):
            print('nop')
    assert hasattr(dask_dataarray.data, '_repr_html_')
    data_repr = fh.short_data_repr_html(dask_dataarray)
    assert data_repr == dask_dataarray.data._repr_html_()

def test_format_dims_no_dims() -> None:
    if False:
        print('Hello World!')
    dims: dict = {}
    dims_with_index: list = []
    formatted = fh.format_dims(dims, dims_with_index)
    assert formatted == ''

def test_format_dims_unsafe_dim_name() -> None:
    if False:
        return 10
    dims = {'<x>': 3, 'y': 2}
    dims_with_index: list = []
    formatted = fh.format_dims(dims, dims_with_index)
    assert '&lt;x&gt;' in formatted

def test_format_dims_non_index() -> None:
    if False:
        return 10
    (dims, dims_with_index) = ({'x': 3, 'y': 2}, ['time'])
    formatted = fh.format_dims(dims, dims_with_index)
    assert "class='xr-has-index'" not in formatted

def test_format_dims_index() -> None:
    if False:
        return 10
    (dims, dims_with_index) = ({'x': 3, 'y': 2}, ['x'])
    formatted = fh.format_dims(dims, dims_with_index)
    assert "class='xr-has-index'" in formatted

def test_summarize_attrs_with_unsafe_attr_name_and_value() -> None:
    if False:
        while True:
            i = 10
    attrs = {'<x>': 3, 'y': '<pd.DataFrame>'}
    formatted = fh.summarize_attrs(attrs)
    assert '<dt><span>&lt;x&gt; :</span></dt>' in formatted
    assert '<dt><span>y :</span></dt>' in formatted
    assert '<dd>3</dd>' in formatted
    assert '<dd>&lt;pd.DataFrame&gt;</dd>' in formatted

def test_repr_of_dataarray(dataarray: xr.DataArray) -> None:
    if False:
        i = 10
        return i + 15
    formatted = fh.array_repr(dataarray)
    assert 'dim_0' in formatted
    assert formatted.count("class='xr-array-in' type='checkbox' checked>") == 1
    assert formatted.count("class='xr-section-summary-in' type='checkbox' disabled >") == 3
    with xr.set_options(display_expand_data=False):
        formatted = fh.array_repr(dataarray)
        assert 'dim_0' in formatted
        assert formatted.count("class='xr-array-in' type='checkbox' checked>") == 0
        assert formatted.count("class='xr-section-summary-in' type='checkbox' disabled >") == 3

def test_repr_of_multiindex(multiindex: xr.Dataset) -> None:
    if False:
        return 10
    formatted = fh.dataset_repr(multiindex)
    assert '(x)' in formatted

def test_repr_of_dataset(dataset: xr.Dataset) -> None:
    if False:
        for i in range(10):
            print('nop')
    formatted = fh.dataset_repr(dataset)
    assert formatted.count("class='xr-section-summary-in' type='checkbox'  checked>") == 3
    assert formatted.count("class='xr-section-summary-in' type='checkbox'  >") == 1
    assert '&lt;U4' in formatted or '&gt;U4' in formatted
    assert '&lt;IA&gt;' in formatted
    with xr.set_options(display_expand_coords=False, display_expand_data_vars=False, display_expand_attrs=False, display_expand_indexes=True):
        formatted = fh.dataset_repr(dataset)
        assert formatted.count("class='xr-section-summary-in' type='checkbox'  checked>") == 1
        assert '&lt;U4' in formatted or '&gt;U4' in formatted
        assert '&lt;IA&gt;' in formatted

def test_repr_text_fallback(dataset: xr.Dataset) -> None:
    if False:
        print('Hello World!')
    formatted = fh.dataset_repr(dataset)
    assert "<pre class='xr-text-repr-fallback'>" in formatted

def test_variable_repr_html() -> None:
    if False:
        for i in range(10):
            print('nop')
    v = xr.Variable(['time', 'x'], [[1, 2, 3], [4, 5, 6]], {'foo': 'bar'})
    assert hasattr(v, '_repr_html_')
    with xr.set_options(display_style='html'):
        html = v._repr_html_().strip()
    assert html.startswith('<div') and html.endswith('</div>')
    assert 'xarray.Variable' in html

def test_repr_of_nonstr_dataset(dataset: xr.Dataset) -> None:
    if False:
        print('Hello World!')
    ds = dataset.copy()
    ds.attrs[1] = 'Test value'
    ds[2] = ds['tmin']
    formatted = fh.dataset_repr(ds)
    assert '<dt><span>1 :</span></dt><dd>Test value</dd>' in formatted
    assert "<div class='xr-var-name'><span>2</span>" in formatted

def test_repr_of_nonstr_dataarray(dataarray: xr.DataArray) -> None:
    if False:
        i = 10
        return i + 15
    da = dataarray.rename(dim_0=15)
    da.attrs[1] = 'value'
    formatted = fh.array_repr(da)
    assert '<dt><span>1 :</span></dt><dd>value</dd>' in formatted
    assert '<li><span>15</span>: 4</li>' in formatted

def test_nonstr_variable_repr_html() -> None:
    if False:
        return 10
    v = xr.Variable(['time', 10], [[1, 2, 3], [4, 5, 6]], {22: 'bar'})
    assert hasattr(v, '_repr_html_')
    with xr.set_options(display_style='html'):
        html = v._repr_html_().strip()
    assert '<dt><span>22 :</span></dt><dd>bar</dd>' in html
    assert '<li><span>10</span>: 3</li></ul>' in html