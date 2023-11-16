import logging
import pytest
import salt.sdb.etcd_db as etcd_db
from salt.utils.etcd_util import get_conn
from tests.support.pytest.etcd import *
pytest.importorskip('docker')
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.slow_test, pytest.mark.skip_if_binaries_missing('docker', 'dockerd', check_all=False)]

@pytest.fixture(scope='module')
def etcd_client(minion_opts, profile_name):
    if False:
        for i in range(10):
            print('nop')
    return get_conn(minion_opts, profile=profile_name)

@pytest.fixture(scope='module')
def prefix():
    if False:
        while True:
            i = 10
    return '/salt/sdb/test'

@pytest.fixture(autouse=True)
def cleanup_prefixed_entries(etcd_client, prefix):
    if False:
        return 10
    '\n    Cleanup after each test to ensure a consistent starting state.\n    '
    try:
        assert etcd_client.get(prefix, recurse=True) is None
        yield
    finally:
        etcd_client.delete(prefix, recurse=True)

def test_basic_operations(etcd_profile, prefix, profile_name):
    if False:
        while True:
            i = 10
    '\n    Ensure we can do the basic CRUD operations available in sdb.etcd_db\n    '
    assert etcd_db.set_('{}/1'.format(prefix), 'one', profile=etcd_profile[profile_name]) == 'one'
    etcd_db.delete('{}/1'.format(prefix), profile=etcd_profile[profile_name])
    assert etcd_db.get('{}/1'.format(prefix), profile=etcd_profile[profile_name]) is None