import logging
import pytest
import salt.cache
import salt.loader
from tests.pytests.functional.cache.helpers import run_common_cache_tests
from tests.support.pytest.etcd import *
docker = pytest.importorskip('docker')
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.slow_test, pytest.mark.skip_if_binaries_missing('dockerd')]

@pytest.fixture(scope='module', params=(EtcdVersion.v2, EtcdVersion.v3_v2_mode), ids=etcd_version_ids)
def etcd_version(request):
    if False:
        while True:
            i = 10
    if request.param and (not HAS_ETCD_V2):
        pytest.skip('No etcd library installed')
    if not request.param and (not HAS_ETCD_V3):
        pytest.skip('No etcd3 library installed')
    return request.param

@pytest.fixture
def cache(minion_opts, etcd_port):
    if False:
        return 10
    opts = minion_opts.copy()
    opts['cache'] = 'etcd'
    opts['etcd.host'] = '127.0.0.1'
    opts['etcd.port'] = etcd_port
    opts['etcd.protocol'] = 'http'
    opts['etcd.timestamp_suffix'] = '.frobnosticate'
    cache = salt.cache.factory(opts)
    return cache

def test_caching(subtests, cache):
    if False:
        return 10
    run_common_cache_tests(subtests, cache)