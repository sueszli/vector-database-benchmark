"""
Test that SubdirData is able to use (or skip) incremental jlap downloads.
"""
import datetime
import json
import time
from pathlib import Path
from socket import socket
from unittest.mock import Mock
import jsonpatch
import pytest
import requests
import zstandard
import conda.gateways.repodata
from conda.base.context import conda_tests_ctxt_mgmt_def_pol, context, reset_context
from conda.common.io import env_vars
from conda.core.subdir_data import SubdirData
from conda.exceptions import CondaHTTPError, CondaSSLError
from conda.gateways.connection.session import CondaSession, get_session
from conda.gateways.repodata import CACHE_CONTROL_KEY, CACHE_STATE_SUFFIX, ETAG_KEY, LAST_MODIFIED_KEY, URL_KEY, CondaRepoInterface, RepodataCache, RepodataOnDisk, RepodataState, Response304ContentUnchanged, get_repo_interface
from conda.gateways.repodata.jlap import core, fetch, interface
from conda.models.channel import Channel

def test_server_available(package_server: socket):
    if False:
        return 10
    port = package_server.getsockname()[1]
    response = requests.get(f'http://127.0.0.1:{port}/notfound')
    assert response.status_code == 404

def test_jlap_fetch(package_server: socket, tmp_path: Path, mocker):
    if False:
        print('Hello World!')
    "Check that JlapRepoInterface doesn't raise exceptions."
    (host, port) = package_server.getsockname()
    base = f'http://{host}:{port}/test'
    cache = RepodataCache(base=tmp_path / 'cache', repodata_fn='repodata.json')
    url = f'{base}/osx-64'
    repo = interface.JlapRepoInterface(url, repodata_fn='repodata.json', cache=cache, cache_path_json=Path(tmp_path, 'repodata.json'), cache_path_state=Path(tmp_path, f'repodata{CACHE_STATE_SUFFIX}'))
    patched = mocker.patch('conda.gateways.repodata.jlap.fetch.download_and_hash', wraps=fetch.download_and_hash)
    state = {}
    with pytest.raises(RepodataOnDisk):
        repo.repodata(state)
    assert patched.call_count == 2
    with pytest.raises(RepodataOnDisk):
        repo.repodata(state)
    with pytest.raises(RepodataOnDisk):
        repo.repodata(state)
    assert patched.call_count == 4

def test_jlap_fetch_file(package_repository_base: Path, tmp_path: Path, mocker):
    if False:
        for i in range(10):
            print('nop')
    'Check that JlapRepoInterface can fetch from a file:/// URL'
    base = package_repository_base.as_uri()
    cache = RepodataCache(base=tmp_path / 'cache', repodata_fn='repodata.json')
    url = f'{base}/osx-64'
    repo = interface.JlapRepoInterface(url, repodata_fn='repodata.json', cache=cache, cache_path_json=Path(tmp_path, 'repodata.json'), cache_path_state=Path(tmp_path, f'repodata{CACHE_STATE_SUFFIX}'))
    test_jlap = make_test_jlap((package_repository_base / 'osx-64' / 'repodata.json').read_bytes(), 8)
    test_jlap.terminate()
    test_jlap.write(package_repository_base / 'osx-64' / 'repodata.jlap')
    patched = mocker.patch('conda.gateways.repodata.jlap.fetch.download_and_hash', wraps=fetch.download_and_hash)
    state = {}
    with pytest.raises(RepodataOnDisk):
        repo.repodata(state)
    with pytest.raises(RepodataOnDisk):
        repo.repodata(state)
    with pytest.raises(RepodataOnDisk):
        repo.repodata(state)
    assert patched.call_count == 2

@pytest.mark.parametrize('verify_ssl', [True, False])
def test_jlap_fetch_ssl(package_server_ssl: socket, tmp_path: Path, monkeypatch, verify_ssl: bool):
    if False:
        return 10
    "Check that JlapRepoInterface doesn't raise exceptions."
    (host, port) = package_server_ssl.getsockname()
    base = f'https://{host}:{port}/test'
    cache = RepodataCache(base=tmp_path / 'cache', repodata_fn='repodata.json')
    url = f'{base}/osx-64'
    repo = interface.JlapRepoInterface(url, repodata_fn='repodata.json', cache=cache, cache_path_json=Path(tmp_path, f'repodata_{verify_ssl}.json'), cache_path_state=Path(tmp_path, f'repodata_{verify_ssl}{CACHE_STATE_SUFFIX}'))
    expected_exception = CondaSSLError if verify_ssl else RepodataOnDisk
    try:
        CondaSession._thread_local.sessions = {}
    except AttributeError:
        pass
    state = {}
    with pytest.raises(expected_exception), pytest.warns() as record:
        monkeypatch.setenv('CONDA_SSL_VERIFY', str(verify_ssl).lower())
        reset_context()
        repo.repodata(state)
    get_session.cache_clear()
    assert len(record) == 0, f'Unexpected warning {record[0]._category_name}'
    try:
        CondaSession._thread_local.sessions = {}
    except AttributeError:
        pass

def test_download_and_hash(package_server: socket, tmp_path: Path, package_repository_base: Path):
    if False:
        while True:
            i = 10
    (host, port) = package_server.getsockname()
    base = f'http://{host}:{port}/test'
    url = base + '/notfound.json.zst'
    session = CondaSession()
    state = RepodataState()
    destination = tmp_path / 'download_not_found'
    try:
        fetch.download_and_hash(fetch.hash(), url, destination, session, state)
    except requests.HTTPError as e:
        assert e.response.status_code == 404
        assert not destination.exists()
    else:
        assert False, 'must raise'
    destination = tmp_path / 'repodata.json'
    url2 = base + '/osx-64/repodata.json'
    hasher2 = fetch.hash()
    response = fetch.download_and_hash(hasher2, url2, destination, session, state, dest_path=destination)
    print(response)
    print(state)
    t = destination.read_text()
    assert len(t)
    response2 = fetch.download_and_hash(fetch.hash(), url2, destination, session, RepodataState(dict={'_etag': response.headers['etag']}))
    assert response2.status_code == 304
    assert destination.read_text() == t
    (package_repository_base / 'osx-64' / 'repodata.json.zst').write_bytes(zstandard.ZstdCompressor().compress((package_repository_base / 'osx-64' / 'repodata.json').read_bytes()))
    url3 = base + '/osx-64/repodata.json.zst'
    dest_zst = tmp_path / 'repodata.json.from-zst'
    assert not dest_zst.exists()
    hasher3 = fetch.hash()
    response3 = fetch.download_and_hash(hasher3, url3, dest_zst, session, RepodataState(), is_zst=True)
    assert response3.status_code == 200
    assert int(response3.headers['content-length']) < dest_zst.stat().st_size
    assert destination.read_text() == dest_zst.read_text()
    assert hasher2.digest() == hasher3.digest()

@pytest.mark.parametrize('use_jlap', [True, False])
def test_repodata_state(package_server: socket, use_jlap: bool):
    if False:
        while True:
            i = 10
    'Test that cache metadata file works correctly.'
    (host, port) = package_server.getsockname()
    base = f'http://{host}:{port}/test'
    channel_url = f'{base}/osx-64'
    if use_jlap:
        repo_cls = interface.JlapRepoInterface
    else:
        repo_cls = CondaRepoInterface
    with env_vars({'CONDA_PLATFORM': 'osx-64', 'CONDA_EXPERIMENTAL': 'jlap' if use_jlap else ''}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        SubdirData.clear_cached_local_channel_data(exclude_file=False)
        test_channel = Channel(channel_url)
        sd = SubdirData(channel=test_channel)
        assert isinstance(sd._repo, repo_cls)
        print(sd.repodata_fn)
        assert sd._loaded is False
        assert len(list(sd.iter_records()))
        assert sd._loaded is True
        state = json.loads(Path(sd.cache_path_state).read_text())
        for field in (LAST_MODIFIED_KEY, ETAG_KEY, CACHE_CONTROL_KEY, URL_KEY, 'size', 'mtime_ns'):
            assert field in state
            assert f'_{field}' not in state

@pytest.mark.parametrize('use_jlap', [True, False])
def test_repodata_info_jsondecodeerror(package_server: socket, use_jlap: bool, monkeypatch):
    if False:
        while True:
            i = 10
    'Test that cache metadata file works correctly.'
    (host, port) = package_server.getsockname()
    base = f'http://{host}:{port}/test'
    channel_url = f'{base}/osx-64'
    if use_jlap:
        repo_cls = interface.JlapRepoInterface
    else:
        repo_cls = CondaRepoInterface
    with env_vars({'CONDA_PLATFORM': 'osx-64', 'CONDA_EXPERIMENTAL': 'jlap' if use_jlap else ''}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        SubdirData.clear_cached_local_channel_data(exclude_file=False)
        test_channel = Channel(channel_url)
        sd = SubdirData(channel=test_channel)
        assert isinstance(sd._repo, repo_cls)
        print(sd.repodata_fn)
        assert sd._loaded is False
        assert len(list(sd.iter_records()))
        assert sd._loaded is True
        sd.cache_path_state.write_text(sd.cache_path_state.read_text() * 2)
        SubdirData.clear_cached_local_channel_data(exclude_file=False)
        sd2 = SubdirData(channel=test_channel)
        records = []

        def warning(*args, **kwargs):
            if False:
                print('Hello World!')
            records.append(args)
        monkeypatch.setattr(conda.gateways.repodata.log, 'warning', warning)
        sd2.load()
        assert any((record[0].startswith('JSONDecodeError') for record in records))

@pytest.mark.parametrize('use_jlap', ['jlap', 'jlapopotamus', 'jlap,another', ''])
def test_jlap_flag(use_jlap):
    if False:
        return 10
    'Test that CONDA_EXPERIMENTAL is a comma-delimited list.'
    with env_vars({'CONDA_EXPERIMENTAL': use_jlap}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        expected = 'jlap' in use_jlap.split(',')
        assert ('jlap' in context.experimental) is expected
        expected_cls = interface.JlapRepoInterface if expected else CondaRepoInterface
        assert get_repo_interface() is expected_cls

def test_jlap_sought(package_server: socket, tmp_path: Path, package_repository_base: Path):
    if False:
        for i in range(10):
            print('nop')
    'Test that we try to fetch the .jlap file.'
    (package_repository_base / 'osx-64' / 'repodata.jlap').unlink(missing_ok=True)
    (host, port) = package_server.getsockname()
    base = f'http://{host}:{port}/test'
    channel_url = f'{base}/osx-64'
    with env_vars({'CONDA_PLATFORM': 'osx-64', 'CONDA_EXPERIMENTAL': 'jlap', 'CONDA_PKGS_DIRS': str(tmp_path)}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        SubdirData.clear_cached_local_channel_data(exclude_file=False)
        test_channel = Channel(channel_url)
        sd = SubdirData(channel=test_channel)
        assert not sd.cache_path_state.exists()
        assert not sd.cache_path_json.exists()
        sd.load()
        cache = sd.repo_cache
        state = json.loads(Path(cache.cache_path_state).read_text())
        print('first fetch', state)
        SubdirData.clear_cached_local_channel_data(exclude_file=False)
        state['refresh_ns'] = state['refresh_ns'] - int(1000000000.0 * 60)
        cache.cache_path_state.write_text(json.dumps(state))
        sd = SubdirData(channel=test_channel)
        sd.load()
        print(list(sd.iter_records()))
        state_object = cache.load_state()
        print(state_object)
        assert state_object.should_check_format('jlap') is False
        test_jlap = make_test_jlap(cache.cache_path_json.read_bytes(), 8)
        test_jlap.terminate()
        test_jlap.write(package_repository_base / 'osx-64' / 'repodata.jlap')
        SubdirData.clear_cached_local_channel_data(exclude_file=False)
        test_channel = Channel(channel_url)
        sd = SubdirData(channel=test_channel)
        state = cache.load_state()
        state.clear_has_format('jlap')
        state['refresh_ns'] = state['refresh_ns'] - int(1000000000.0 * 60)
        cache.cache_path_state.write_text(json.dumps(dict(state)))
        sd.load()
        patched = json.loads(sd.cache_path_json.read_text())
        assert len(patched['info']) == 9
        with pytest.raises(RepodataOnDisk):
            sd._repo.repodata(cache.load_state())
        test_jlap = make_test_jlap(cache.cache_path_json.read_bytes(), 4)
        footer = test_jlap.pop()
        test_jlap.pop()
        test_jlap.add(footer[1])
        test_jlap.terminate()
        test_jlap.write(package_repository_base / 'osx-64' / 'repodata.jlap')
        SubdirData.clear_cached_local_channel_data(exclude_file=False)
        test_channel = Channel(channel_url)
        sd = SubdirData(channel=test_channel)
        state = cache.load_state()
        assert state.has_format('jlap')[0] is True
        state['refresh_ns'] = state['refresh_ns'] - int(1000000000.0 * 60)
        cache.cache_path_state.write_text(json.dumps(dict(state)))
        sd.load()
        patched = json.loads(sd.cache_path_json.read_text())
        assert len(patched['info']) == 9
        (package_repository_base / 'osx-64' / 'repodata.jlap').write_text('')
        state = cache.load_state()
        state.etag = ''
        assert fetch.JLAP_UNAVAILABLE not in state
        state['refresh_ns'] = state['refresh_ns'] - int(1000000000.0 * 60)
        cache.cache_path_state.write_text(json.dumps(dict(state)))
        sd.load()
        patched = json.loads(sd.cache_path_json.read_text())
        assert len(patched['info']) == 1

def test_jlap_coverage():
    if False:
        while True:
            i = 10
    '\n    Force raise RepodataOnDisk() at end of JlapRepoInterface.repodata() function.\n    '

    class JlapCoverMe(interface.JlapRepoInterface):

        def repodata_parsed(self, state):
            if False:
                i = 10
                return i + 15
            return
    with pytest.raises(RepodataOnDisk):
        JlapCoverMe('', '', cache=None).repodata({})

def test_jlap_errors(package_server: socket, tmp_path: Path, package_repository_base: Path, mocker):
    if False:
        i = 10
        return i + 15
    'Test that we handle 304 Not Modified responses, other errors.'
    (host, port) = package_server.getsockname()
    base = f'http://{host}:{port}/test'
    channel_url = f'{base}/osx-64'
    with env_vars({'CONDA_PLATFORM': 'osx-64', 'CONDA_EXPERIMENTAL': 'jlap', 'CONDA_PKGS_DIRS': str(tmp_path)}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        SubdirData.clear_cached_local_channel_data(exclude_file=False)
        test_channel = Channel(channel_url)
        sd = SubdirData(channel=test_channel)
        sd.load()
        cache = sd.repo_cache
        state = cache.load_state()
        SubdirData.clear_cached_local_channel_data(exclude_file=False)
        cache.refresh(state['refresh_ns'] - int(1000000000.0 * 60))
        test_jlap = make_test_jlap(cache.cache_path_json.read_bytes(), 8)
        test_jlap.terminate()
        test_jlap_path = package_repository_base / 'osx-64' / 'repodata.jlap'
        test_jlap.write(test_jlap_path)
        sd.load()
        state = cache.load_state()
        (has, when) = state.has_format('jlap')
        assert has is True and isinstance(when, datetime.datetime)
        patched = json.loads(sd.cache_path_json.read_text())
        assert len(patched['info']) == 9
        with test_jlap_path.open('a') as test_jlap_file:
            test_jlap_file.write('x')
        state = cache.load_state()
        state['refresh_ns'] -= int(60 * 1000000000.0)
        with pytest.raises(RepodataOnDisk):
            sd._repo.repodata(state)
        test_jlap_path.write_text(core.DEFAULT_IV.hex())
        state.pop('has_jlap', None)
        state.pop('jlap', None)
        with pytest.raises(RepodataOnDisk):
            sd._repo.repodata(state)
        state.pop('has_jlap', None)
        state.pop('jlap', None)
        cache.cache_path_state.write_text(json.dumps(dict(state)))
        with mocker.patch.object(CondaSession, 'get', return_value=Mock(status_code=304, headers={})), pytest.raises(Response304ContentUnchanged):
            sd._repo.repodata(cache.load_state())

@pytest.mark.parametrize('use_jlap', [True, False])
def test_jlap_cache_clock(package_server: socket, tmp_path: Path, package_repository_base: Path, mocker, use_jlap: bool):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that we add another "local_repodata_ttl" (an alternative to\n    "cache-control: max-age=x") seconds to the clock once the cache expires,\n    whether the response was "200" or "304 Not Modified".\n    '
    (host, port) = package_server.getsockname()
    base = f'http://{host}:{port}/test'
    channel_url = f'{base}/osx-64'
    now = time.time_ns()
    mocker.patch('time.time_ns', return_value=now)
    assert time.time_ns() == now
    local_repodata_ttl = 30
    with env_vars({'CONDA_PLATFORM': 'osx-64', 'CONDA_EXPERIMENTAL': 'jlap' if use_jlap else '', 'CONDA_PKGS_DIRS': str(tmp_path), 'CONDA_LOCAL_REPODATA_TTL': local_repodata_ttl}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        SubdirData.clear_cached_local_channel_data(exclude_file=False)
        test_channel = Channel(channel_url)
        sd = SubdirData(channel=test_channel)
        cache = sd.repo_cache
        sd.load()
        assert cache.load_state()['refresh_ns'] == time.time_ns()
        SubdirData.clear_cached_local_channel_data(exclude_file=False)
        test_jlap = make_test_jlap(cache.cache_path_json.read_bytes(), 8)
        test_jlap.terminate()
        test_jlap_path = package_repository_base / 'osx-64' / 'repodata.jlap'
        test_jlap.write(test_jlap_path)
        later0 = now + (local_repodata_ttl + 1) * int(1000000000.0)
        mocker.patch('time.time_ns', return_value=later0)
        assert cache.stale()
        sd.load()
        later1 = now + (2 * local_repodata_ttl + 2) * int(1000000000.0)
        mocker.patch('time.time_ns', return_value=later1)
        with mocker.patch.object(CondaSession, 'get', return_value=Mock(status_code=304, headers={})):
            assert cache.stale()
            sd.load()
        assert cache.load_state()['refresh_ns'] == later1
        assert not cache.stale()
        later2 = now + (3 * local_repodata_ttl + 3) * int(1000000000.0)
        mocker.patch('time.time_ns', return_value=later2)
        assert cache.stale()
        sd.load()
        assert cache.load_state()['refresh_ns'] == later2
        mocker.patch('time.time_ns', return_value=now + (3 * local_repodata_ttl + 4) * int(1000000000.0))
        sd.load()
        assert cache.load_state()['refresh_ns'] == later2

def test_jlap_zst_not_404(mocker, package_server, tmp_path):
    if False:
        print('Hello World!')
    '\n    Test that exception is raised if `repodata.json.zst` produces something\n    other than a 404. For code coverage.\n    '
    (host, port) = package_server.getsockname()
    base = f'http://{host}:{port}/test'
    url = f'{base}/osx-64'
    cache = RepodataCache(base=tmp_path / 'cache', repodata_fn='repodata.json')
    repo = interface.JlapRepoInterface(url, repodata_fn='repodata.json', cache=cache, cache_path_json=Path(tmp_path, 'repodata.json'), cache_path_state=Path(tmp_path, f'repodata{CACHE_STATE_SUFFIX}'))

    def error(*args, **kwargs):
        if False:
            while True:
                i = 10

        class Response:
            status_code = 405
        raise fetch.HTTPError(response=Response())
    mocker.patch('conda.gateways.repodata.jlap.fetch.download_and_hash', side_effect=error)
    with pytest.raises(CondaHTTPError, match='HTTP 405'):
        repo.repodata({})

def test_jlap_core(tmp_path: Path):
    if False:
        return 10
    'Code paths not excercised by other tests.'
    with pytest.raises(ValueError):
        core.JLAP.from_lines([core.DEFAULT_IV.hex().encode('utf-8')] * 3, iv=core.DEFAULT_IV, verify=True)
    with pytest.raises(IndexError):
        core.JLAP.from_lines([core.DEFAULT_IV.hex().encode('utf-8')] * 1, iv=core.DEFAULT_IV, verify=True)
    jlap = core.JLAP.from_lines([core.DEFAULT_IV.hex().encode('utf-8')] * 2, iv=core.DEFAULT_IV, verify=True)
    with pytest.raises(ValueError):
        jlap.add('two\nlines')
    test_jlap = tmp_path / 'minimal.jlap'
    jlap.write(test_jlap)
    jlap2 = jlap.from_path(test_jlap)
    assert jlap2 == jlap
    assert jlap2.last == jlap2[-1]
    assert jlap2.penultimate == jlap2[-2]
    assert jlap2.body == jlap2[1:-2]

def make_test_jlap(original: bytes, changes=1):
    if False:
        return 10
    ':original: as bytes, to avoid any newline confusion.'

    def jlap_lines():
        if False:
            return 10
        yield core.DEFAULT_IV.hex().encode('utf-8')
        before = json.loads(original)
        after = json.loads(original)
        h = fetch.hash()
        h.update(original)
        starting_digest = h.digest().hex()
        for i in range(changes):
            after['info'][f'test{i}'] = i
            patch = jsonpatch.make_patch(before, after)
            row = {'from': starting_digest}
            h = fetch.hash()
            h.update(json.dumps(after).encode('utf-8'))
            starting_digest = h.digest().hex()
            row['to'] = starting_digest
            before = json.loads(json.dumps(after))
            row['patch'] = patch.patch
            yield json.dumps(row).encode('utf-8')
        yield json.dumps({'from': core.DEFAULT_IV.hex(), 'to': core.DEFAULT_IV.hex(), 'patch': []}).encode('utf-8')
        footer = {'url': 'repodata.json', 'latest': starting_digest}
        yield json.dumps(footer).encode('utf-8')
    j = core.JLAP.from_lines(jlap_lines(), iv=core.DEFAULT_IV, verify=False)
    return j

def test_hashwriter():
    if False:
        return 10
    'Test that HashWriter closes its backing file in a context manager.'
    closed = False

    class backing:

        def close(self):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal closed
            closed = True
    writer = fetch.HashWriter(backing(), None)
    with writer:
        pass
    assert closed

def test_request_url_jlap_state(tmp_path, package_server, package_repository_base):
    if False:
        while True:
            i = 10
    '\n    Code coverage for case intended to catch "repodata.json written while we\n    were downloading its patches".\n\n    When this happens, we do not write a new repodata.json and instruct the\n    caller to defer to the on-disk cache.\n    '
    (host, port) = package_server.getsockname()
    base = f'http://{host}:{port}/test'
    url = f'{base}/osx-64/repodata.json'
    cache = RepodataCache(base=tmp_path / 'cache', repodata_fn='repodata.json')
    cache.state.set_has_format('jlap', True)
    cache.save(json.dumps({'info': {}}))
    test_jlap = make_test_jlap(cache.cache_path_json.read_bytes(), 8)
    test_jlap.terminate()
    test_jlap_path = package_repository_base / 'osx-64' / 'repodata.jlap'
    test_jlap.write(test_jlap_path)
    temp_path = tmp_path / 'new_repodata.json'
    outdated_state = cache.load_state()
    hasher = fetch.hash()
    hasher.update(cache.cache_path_json.read_bytes())
    outdated_state[fetch.NOMINAL_HASH] = hasher.hexdigest()
    outdated_state[fetch.ON_DISK_HASH] = hasher.hexdigest()
    on_disk_state = json.loads(cache.cache_path_state.read_text())
    on_disk_state[fetch.NOMINAL_HASH] = '0' * 64
    on_disk_state[fetch.ON_DISK_HASH] = '0' * 64
    cache.cache_path_state.write_text(json.dumps(on_disk_state))
    result = fetch.request_url_jlap_state(url, outdated_state, session=CondaSession(), cache=cache, temp_path=temp_path)
    assert result is None