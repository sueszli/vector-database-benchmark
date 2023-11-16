from logging import getLogger
from os.path import join
from pathlib import Path
from time import sleep
from unittest.mock import patch
import pytest
from conda import CondaError
from conda.base.context import conda_tests_ctxt_mgmt_def_pol, context
from conda.common.io import env_var, env_vars
from conda.core.index import get_index
from conda.core.subdir_data import SubdirData, cache_fn_url, fetch_repodata_remote_request
from conda.exceptions import CondaUpgradeError
from conda.exports import url_path
from conda.gateways.repodata import CondaRepoInterface, RepodataCache, RepodataFetch
from conda.models.channel import Channel
from conda.models.records import PackageRecord
from conda.testing.helpers import CHANNEL_DIR
from conda.testing.integration import make_temp_env
log = getLogger(__name__)
OVERRIDE_PLATFORM = 'linux-64' if context.subdir not in ('win-64', 'linux-64', 'osx-64') else context.subdir

def platform_in_record(platform, record):
    if False:
        print('Hello World!')
    return record.name.endswith('@') or '/%s/' % platform in record.url or '/noarch/' in record.url

@pytest.mark.integration
def test_get_index_no_platform_with_offline_cache(platform=OVERRIDE_PLATFORM):
    if False:
        while True:
            i = 10
    with env_vars({'CONDA_REPODATA_TIMEOUT_SECS': '0', 'CONDA_PLATFORM': platform}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        channel_urls = ('https://repo.anaconda.com/pkgs/pro',)
        this_platform = context.subdir
        index = get_index(channel_urls=channel_urls, prepend=False)
        for (dist, record) in index.items():
            assert platform_in_record(this_platform, record), (this_platform, record.url)
    for unknown in (None, False, True):
        with env_var('CONDA_OFFLINE', 'yes', stack_callback=conda_tests_ctxt_mgmt_def_pol):
            index2 = get_index(channel_urls=channel_urls, prepend=False, unknown=unknown)
            assert all((index2.get(k) == rec for (k, rec) in index.items()))
            assert unknown is not False or len(index) == len(index2)
    for unknown in (False, True):
        with env_vars({'CONDA_REPODATA_TIMEOUT_SECS': '0', 'CONDA_PLATFORM': 'linux-64'}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
            index3 = get_index(channel_urls=channel_urls, prepend=False, unknown=unknown)
            assert all((index3.get(k) == rec for (k, rec) in index.items()))
            assert unknown or len(index) == len(index3)
    with env_vars({'CONDA_OFFLINE': 'yes', 'CONDA_PLATFORM': platform}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        local_channel = Channel(join(CHANNEL_DIR, platform))
        sd = SubdirData(channel=local_channel)
        assert len(sd.query_all('zlib', channels=[local_channel])) > 0
        assert len(sd.query_all('zlib')) == 0
    assert len(sd.query_all('zlib')) > 1
    with env_vars({'CONDA_USE_INDEX_CACHE': 'true'}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        sd.clear_cached_local_channel_data()
        sd._load()

def test_cache_fn_url_repo_continuum_io():
    if False:
        i = 10
        return i + 15
    hash1 = cache_fn_url('http://repo.continuum.io/pkgs/free/osx-64/')
    hash2 = cache_fn_url('http://repo.continuum.io/pkgs/free/osx-64')
    assert 'aa99d924.json' == hash1 == hash2
    hash3 = cache_fn_url('https://repo.continuum.io/pkgs/free/osx-64/')
    hash4 = cache_fn_url('https://repo.continuum.io/pkgs/free/osx-64')
    assert 'd85a531e.json' == hash3 == hash4 != hash1
    hash5 = cache_fn_url('https://repo.continuum.io/pkgs/free/linux-64/')
    assert hash4 != hash5
    hash6 = cache_fn_url('https://repo.continuum.io/pkgs/r/osx-64')
    assert hash4 != hash6

def test_cache_fn_url_repo_anaconda_com():
    if False:
        i = 10
        return i + 15
    hash1 = cache_fn_url('http://repo.anaconda.com/pkgs/free/osx-64/')
    hash2 = cache_fn_url('http://repo.anaconda.com/pkgs/free/osx-64')
    assert '1e817819.json' == hash1 == hash2
    hash3 = cache_fn_url('https://repo.anaconda.com/pkgs/free/osx-64/')
    hash4 = cache_fn_url('https://repo.anaconda.com/pkgs/free/osx-64')
    assert '3ce78580.json' == hash3 == hash4 != hash1
    hash5 = cache_fn_url('https://repo.anaconda.com/pkgs/free/linux-64/')
    assert hash4 != hash5
    hash6 = cache_fn_url('https://repo.anaconda.com/pkgs/r/osx-64')
    assert hash4 != hash6

def test_fetch_repodata_remote_request_invalid_arch():
    if False:
        while True:
            i = 10
    url = 'file:///fake/fake/fake/linux-64'
    etag = None
    mod_stamp = 'Mon, 28 Jan 2019 01:01:01 GMT'
    result = fetch_repodata_remote_request(url, etag, mod_stamp)
    assert result is None

def test_subdir_data_prefers_conda_to_tar_bz2(platform=OVERRIDE_PLATFORM):
    if False:
        for i in range(10):
            print('nop')
    with env_vars({'CONDA_USE_ONLY_TAR_BZ2': False, 'CONDA_PLATFORM': platform}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        channel = Channel(join(CHANNEL_DIR, platform))
        sd = SubdirData(channel)
        precs = tuple(sd.query('zlib'))
        assert precs[0].fn.endswith('.conda')

def test_use_only_tar_bz2(platform=OVERRIDE_PLATFORM):
    if False:
        print('Hello World!')
    channel = Channel(join(CHANNEL_DIR, platform))
    SubdirData.clear_cached_local_channel_data()
    with env_var('CONDA_USE_ONLY_TAR_BZ2', True, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        sd = SubdirData(channel)
        precs = tuple(sd.query('zlib'))
        assert precs[0].fn.endswith('.tar.bz2')
    SubdirData.clear_cached_local_channel_data()
    with env_var('CONDA_USE_ONLY_TAR_BZ2', False, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        sd = SubdirData(channel)
        precs = tuple(sd.query('zlib'))
        assert precs[0].fn.endswith('.conda')

def test_subdir_data_coverage(platform=OVERRIDE_PLATFORM):
    if False:
        return 10

    class ChannelCacheClear:

        def __enter__(self):
            if False:
                return 10
            return

        def __exit__(self, *exc):
            if False:
                print('Hello World!')
            Channel._cache_.clear()
    with ChannelCacheClear(), make_temp_env(), env_vars({'CONDA_PLATFORM': platform, 'CONDA_SSL_VERIFY': 'false'}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        channel = Channel(url_path(join(CHANNEL_DIR, platform)))
        sd = SubdirData(channel)
        sd.load()
        assert all((isinstance(p, PackageRecord) for p in sd._package_records[1:]))
        assert all((r.name == 'zlib' for r in sd._iter_records_by_name('zlib')))
        sd.reload()
        assert all((r.name == 'zlib' for r in sd._iter_records_by_name('zlib')))

def test_repodata_version_error(platform=OVERRIDE_PLATFORM):
    if False:
        for i in range(10):
            print('nop')
    channel = Channel(url_path(join(CHANNEL_DIR, platform)))
    SubdirData.clear_cached_local_channel_data(exclude_file=False)

    class SubdirDataRepodataTooNew(SubdirData):

        def _load(self):
            if False:
                return 10
            return {'repodata_version': 1024}
    with pytest.raises(CondaUpgradeError):
        SubdirDataRepodataTooNew(channel).load()
    SubdirData.clear_cached_local_channel_data(exclude_file=False)

def test_metadata_cache_works(platform=OVERRIDE_PLATFORM):
    if False:
        for i in range(10):
            print('nop')
    channel = Channel(join(CHANNEL_DIR, platform))
    SubdirData.clear_cached_local_channel_data()
    sleep(3)
    with env_vars({'CONDA_PLATFORM': platform}, stack_callback=conda_tests_ctxt_mgmt_def_pol), patch.object(CondaRepoInterface, 'repodata', return_value='{}') as fetcher:
        sd_a = SubdirData(channel)
        tuple(sd_a.query('zlib'))
        assert fetcher.call_count == 1
        sd_b = SubdirData(channel)
        assert sd_b is sd_a
        tuple(sd_b.query('zlib'))
        assert fetcher.call_count == 1

def test_metadata_cache_clearing(platform=OVERRIDE_PLATFORM):
    if False:
        i = 10
        return i + 15
    channel = Channel(join(CHANNEL_DIR, platform))
    SubdirData.clear_cached_local_channel_data()
    with env_vars({'CONDA_PLATFORM': platform}, stack_callback=conda_tests_ctxt_mgmt_def_pol), patch.object(CondaRepoInterface, 'repodata', return_value='{}') as fetcher:
        sd_a = SubdirData(channel)
        precs_a = tuple(sd_a.query('zlib'))
        assert fetcher.call_count == 1
        SubdirData.clear_cached_local_channel_data()
        sd_b = SubdirData(channel)
        assert sd_b is not sd_a
        precs_b = tuple(sd_b.query('zlib'))
        assert fetcher.call_count == 2
        assert precs_b == precs_a

def test_search_by_packagerecord(platform=OVERRIDE_PLATFORM):
    if False:
        while True:
            i = 10
    local_channel = Channel(join(CHANNEL_DIR, platform))
    sd = SubdirData(channel=local_channel)
    assert len(tuple(sd.query('*[version=1.2.11]'))) >= 1
    assert any(sd.query(next(sd.query('zlib'))))

def test_state_is_not_json(tmp_path, platform=OVERRIDE_PLATFORM):
    if False:
        for i in range(10):
            print('nop')
    '\n    SubdirData has a ValueError exception handler, that is hard to invoke\n    currently.\n    '
    local_channel = Channel(join(CHANNEL_DIR, platform))
    bad_cache = tmp_path / 'not_json.json'
    bad_cache.write_text('{}')

    class BadRepodataCache(RepodataCache):
        cache_path_state = bad_cache

    class BadRepodataFetch(RepodataFetch):

        @property
        def repo_cache(self) -> RepodataCache:
            if False:
                return 10
            return BadRepodataCache(self.cache_path_base, self.repodata_fn)

    class BadCacheSubdirData(SubdirData):

        @property
        def repo_fetch(self):
            if False:
                i = 10
                return i + 15
            return BadRepodataFetch(Path(self.cache_path_base), self.channel, self.repodata_fn, repo_interface_cls=CondaRepoInterface)
    SubdirData.clear_cached_local_channel_data(exclude_file=False)
    sd: SubdirData = BadCacheSubdirData(channel=local_channel)
    with pytest.raises(CondaError):
        state = sd.repo_cache.load_state()
        bad_cache.write_text('NOT JSON')
        sd._read_local_repodata(state)

def test_subdir_data_dict_state(platform=OVERRIDE_PLATFORM):
    if False:
        while True:
            i = 10
    'SubdirData can accept a dict instead of a RepodataState, for compatibility.'
    local_channel = Channel(join(CHANNEL_DIR, platform))
    sd = SubdirData(channel=local_channel)
    sd._read_pickled({})