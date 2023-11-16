from __future__ import annotations
from numbers import Number
import numpy as np
import pytest
import xarray as xr
from xarray.backends.api import _get_default_engine
from xarray.tests import assert_identical, assert_no_warnings, requires_dask, requires_netCDF4, requires_scipy

@requires_netCDF4
@requires_scipy
def test__get_default_engine() -> None:
    if False:
        for i in range(10):
            print('nop')
    engine_remote = _get_default_engine('http://example.org/test.nc', allow_remote=True)
    assert engine_remote == 'netcdf4'
    engine_gz = _get_default_engine('/example.gz')
    assert engine_gz == 'scipy'
    engine_default = _get_default_engine('/example')
    assert engine_default == 'netcdf4'

def test_custom_engine() -> None:
    if False:
        i = 10
        return i + 15
    expected = xr.Dataset(dict(a=2 * np.arange(5)), coords=dict(x=('x', np.arange(5), dict(units='s'))))

    class CustomBackend(xr.backends.BackendEntrypoint):

        def open_dataset(self, filename_or_obj, drop_variables=None, **kwargs) -> xr.Dataset:
            if False:
                i = 10
                return i + 15
            return expected.copy(deep=True)
    actual = xr.open_dataset('fake_filename', engine=CustomBackend)
    assert_identical(expected, actual)

def test_multiindex() -> None:
    if False:
        print('Hello World!')
    dataset = xr.Dataset(coords={'coord1': ['A', 'B'], 'coord2': [1, 2]})
    dataset = dataset.stack(z=['coord1', 'coord2'])

    class MultiindexBackend(xr.backends.BackendEntrypoint):

        def open_dataset(self, filename_or_obj, drop_variables=None, **kwargs) -> xr.Dataset:
            if False:
                print('Hello World!')
            return dataset.copy(deep=True)
    loaded = xr.open_dataset('fake_filename', engine=MultiindexBackend)
    assert_identical(dataset, loaded)

class PassThroughBackendEntrypoint(xr.backends.BackendEntrypoint):
    """Access an object passed to the `open_dataset` method."""

    def open_dataset(self, dataset, *, drop_variables=None):
        if False:
            print('Hello World!')
        'Return the first argument.'
        return dataset

def explicit_chunks(chunks, shape):
    if False:
        return 10
    'Return explicit chunks, expanding any integer member to a tuple of integers.'
    return tuple((size // chunk * (chunk,) + ((size % chunk,) if size % chunk or size == 0 else ()) if isinstance(chunk, Number) else chunk for (chunk, size) in zip(chunks, shape)))

@requires_dask
class TestPreferredChunks:
    """Test behaviors related to the backend's preferred chunks."""
    var_name = 'data'

    def create_dataset(self, shape, pref_chunks):
        if False:
            print('Hello World!')
        'Return a dataset with a variable with the given shape and preferred chunks.'
        dims = tuple((f'dim_{idx}' for idx in range(len(shape))))
        return xr.Dataset({self.var_name: xr.Variable(dims, np.empty(shape, dtype=np.dtype('V1')), encoding={'preferred_chunks': dict(zip(dims, pref_chunks))})})

    def check_dataset(self, initial, final, expected_chunks):
        if False:
            for i in range(10):
                print('nop')
        assert_identical(initial, final)
        assert final[self.var_name].chunks == expected_chunks

    @pytest.mark.parametrize('shape,pref_chunks', [((5,), (2,)), ((5,), ((2, 2, 1),)), ((5, 6), (4, 2)), ((5, 6), (4, (2, 2, 2)))])
    @pytest.mark.parametrize('request_with_empty_map', [False, True])
    def test_honor_chunks(self, shape, pref_chunks, request_with_empty_map):
        if False:
            while True:
                i = 10
        "Honor the backend's preferred chunks when opening a dataset."
        initial = self.create_dataset(shape, pref_chunks)
        chunks = {} if request_with_empty_map else dict.fromkeys(initial[self.var_name].dims, None)
        final = xr.open_dataset(initial, engine=PassThroughBackendEntrypoint, chunks=chunks)
        self.check_dataset(initial, final, explicit_chunks(pref_chunks, shape))

    @pytest.mark.parametrize('shape,pref_chunks,req_chunks', [((5,), (2,), (3,)), ((5,), (2,), ((2, 1, 1, 1),)), ((5,), ((2, 2, 1),), (3,)), ((5,), ((2, 2, 1),), ((2, 1, 1, 1),)), ((1, 5), (1, 2), (1, 3))])
    def test_split_chunks(self, shape, pref_chunks, req_chunks):
        if False:
            while True:
                i = 10
        "Warn when the requested chunks separate the backend's preferred chunks."
        initial = self.create_dataset(shape, pref_chunks)
        with pytest.warns(UserWarning):
            final = xr.open_dataset(initial, engine=PassThroughBackendEntrypoint, chunks=dict(zip(initial[self.var_name].dims, req_chunks)))
        self.check_dataset(initial, final, explicit_chunks(req_chunks, shape))

    @pytest.mark.parametrize('shape,pref_chunks,req_chunks', [((5,), (2,), (2,)), ((5,), (2,), ((2, 2, 1),)), ((5,), (2,), (4,)), ((5,), (2,), (6,)), ((5,), (1,), ((1, 1, 2, 1),)), ((5,), ((1, 1, 2, 1),), (2,)), ((5,), ((1, 1, 2, 1),), ((2, 3),)), ((5, 5), (2, (1, 1, 2, 1)), (4, (2, 3)))])
    def test_join_chunks(self, shape, pref_chunks, req_chunks):
        if False:
            i = 10
            return i + 15
        "Don't warn when the requested chunks join or keep the preferred chunks."
        initial = self.create_dataset(shape, pref_chunks)
        with assert_no_warnings():
            final = xr.open_dataset(initial, engine=PassThroughBackendEntrypoint, chunks=dict(zip(initial[self.var_name].dims, req_chunks)))
        self.check_dataset(initial, final, explicit_chunks(req_chunks, shape))