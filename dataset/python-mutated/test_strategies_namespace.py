from types import SimpleNamespace
from weakref import WeakValueDictionary
import pytest
from hypothesis.extra import array_api
from hypothesis.extra.array_api import NOMINAL_VERSIONS, make_strategies_namespace, mock_xp
from hypothesis.strategies import SearchStrategy
pytestmark = pytest.mark.filterwarnings('ignore::hypothesis.errors.HypothesisWarning')

class HashableArrayModuleFactory:
    """
    mock_xp cannot be hashed and thus cannot be used in our cache. So just for
    the purposes of testing the cache, we wrap it with an unsafe hash method.
    """

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        return getattr(mock_xp, name)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(tuple(sorted(mock_xp.__dict__)))

@pytest.mark.parametrize('api_version', ['2021.12', None])
def test_caching(api_version, monkeypatch):
    if False:
        return 10
    'Caches namespaces respective to arguments.'
    xp = HashableArrayModuleFactory()
    assert isinstance(array_api._args_to_xps, WeakValueDictionary)
    monkeypatch.setattr(array_api, '_args_to_xps', WeakValueDictionary())
    assert len(array_api._args_to_xps) == 0
    xps1 = array_api.make_strategies_namespace(xp, api_version=api_version)
    assert len(array_api._args_to_xps) == 1
    xps2 = array_api.make_strategies_namespace(xp, api_version=api_version)
    assert len(array_api._args_to_xps) == 1
    assert isinstance(xps2, SimpleNamespace)
    assert xps2 is xps1
    del xps1
    del xps2
    assert len(array_api._args_to_xps) == 0

@pytest.mark.parametrize('api_version1, api_version2', [(None, '2021.12'), ('2021.12', None)])
def test_inferred_namespace_shares_cache(api_version1, api_version2, monkeypatch):
    if False:
        print('Hello World!')
    'Results from inferred versions share the same cache key as results\n    from specified versions.'
    xp = HashableArrayModuleFactory()
    xp.__array_api_version__ = '2021.12'
    assert isinstance(array_api._args_to_xps, WeakValueDictionary)
    monkeypatch.setattr(array_api, '_args_to_xps', WeakValueDictionary())
    assert len(array_api._args_to_xps) == 0
    xps1 = array_api.make_strategies_namespace(xp, api_version=api_version1)
    assert xps1.api_version == '2021.12'
    assert len(array_api._args_to_xps) == 1
    xps2 = array_api.make_strategies_namespace(xp, api_version=api_version2)
    assert xps2.api_version == '2021.12'
    assert len(array_api._args_to_xps) == 1
    assert xps2 is xps1

def test_complex_dtypes_raises_on_2021_12():
    if False:
        while True:
            i = 10
    'Accessing complex_dtypes() for 2021.12 strategy namespace raises helpful\n    error, but accessing on future versions returns expected strategy.'
    first_xps = make_strategies_namespace(mock_xp, api_version='2021.12')
    with pytest.raises(AttributeError, match='attempted to access'):
        first_xps.complex_dtypes()
    for api_version in NOMINAL_VERSIONS[1:]:
        xps = make_strategies_namespace(mock_xp, api_version=api_version)
        assert isinstance(xps.complex_dtypes(), SearchStrategy)