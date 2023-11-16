import pytest
import itertools
from hypothesis import strategies as st, given, settings, HealthCheck
import ivy
import numpy as np
from ivy.utils.backend.handler import _backend_dict

def test_is_local(backend_fw):
    if False:
        i = 10
        return i + 15
    local_ivy = ivy.with_backend(backend_fw)
    assert local_ivy.is_local()

@settings(suppress_health_check=[HealthCheck(9)])
@given(name=st.sampled_from(['add', 'Array', 'Container', 'globals_vars']))
def test_memory_id(name, traced_backends):
    if False:
        while True:
            i = 10
    for b in traced_backends:
        assert id(getattr(ivy, name)) != id(getattr(b, name)), f'Shared object {name} between global Ivy and backend {b.backend}'
    for comb in itertools.combinations(traced_backends, 2):
        assert id(getattr(comb[0], name)) != id(getattr(comb[1], name)), f'Shared object {name} between {comb[0].backend} and backend {comb[1].backend}'

def test_prevent_access(backend_fw):
    if False:
        return 10
    local_ivy = ivy.with_backend(backend_fw)
    with pytest.raises(RuntimeError):
        local_ivy.with_backend(backend_fw)
    with pytest.raises(RuntimeError):
        local_ivy.set_backend(backend_fw)

def test_with_backend_array(backend_fw):
    if False:
        print('Hello World!')
    local_ivy = ivy.with_backend(backend_fw)
    local_x = local_ivy.array([1, 2, 3, 4])
    ivy.set_backend(backend_fw)
    x = ivy.array([1, 2, 3, 4])
    assert np.allclose(x._data, local_x._data)

def test_with_backend_cached(backend_fw):
    if False:
        return 10
    non_cached_local_ivy = ivy.with_backend(backend_fw)
    cached_local_ivy = ivy.with_backend(backend_fw)
    assert non_cached_local_ivy == cached_local_ivy

@pytest.fixture
def traced_backends():
    if False:
        return 10
    traced_backends = []
    for b in _backend_dict:
        _b = ivy.with_backend(b)
        traced_backends.append(_b)
    return traced_backends