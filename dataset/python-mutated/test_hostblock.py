import pathlib
import zipfile
import logging
import pytest
from qutebrowser.qt.core import QUrl
from qutebrowser.components import hostblock
from qutebrowser.utils import urlmatch
from helpers import testutils
pytestmark = pytest.mark.usefixtures('qapp')
WHITELISTED_HOSTS = ('qutebrowser.org', 'mediumhost.io', 'http://*.edu')
BLOCKLIST_HOSTS = ('localhost', 'mediumhost.io', 'malware.badhost.org', '4-verybadhost.com', 'ads.worsthostever.net')
CLEAN_HOSTS = ('goodhost.gov', 'verygoodhost.com')
URLS_TO_CHECK = ('http://localhost', 'http://mediumhost.io', 'ftp://malware.badhost.org', 'http://4-verybadhost.com', 'http://ads.worsthostever.net', 'http://goodhost.gov', 'ftp://verygoodhost.com', 'http://qutebrowser.org', 'http://veryverygoodhost.edu')

@pytest.fixture
def host_blocker_factory(config_tmpdir, data_tmpdir, download_stub, config_stub):
    if False:
        return 10

    def factory():
        if False:
            i = 10
            return i + 15
        return hostblock.HostBlocker(config_dir=config_tmpdir, data_dir=data_tmpdir)
    return factory

def create_zipfile(directory, files, zipname='test'):
    if False:
        i = 10
        return i + 15
    'Return a path to a newly created zip file.\n\n    Args:\n        directory: path object where to create the zip file.\n        files: list of pathlib.Paths (relative to directory) to each file to add.\n        zipname: name to give to the zip file.\n    '
    zipfile_path = (directory / zipname).with_suffix('.zip')
    with zipfile.ZipFile(zipfile_path, 'w') as new_zipfile:
        for file_path in files:
            new_zipfile.write(directory / file_path, arcname=file_path.name)
    return pathlib.Path(zipname + '.zip')

def create_blocklist(directory, blocked_hosts=BLOCKLIST_HOSTS, name='hosts', line_format='one_per_line'):
    if False:
        print('Hello World!')
    "Return a path to a blocklist file.\n\n    Args:\n        directory: path object where to create the blocklist file\n        blocked_hosts: an iterable of string hosts to add to the blocklist\n        name: name to give to the blocklist file\n        line_format: 'etc_hosts'  -->  /etc/hosts format\n                    'one_per_line'  -->  one host per line format\n                    'not_correct'  -->  Not a correct hosts file format.\n    "
    blocklist_file = directory / name
    with blocklist_file.open('w', encoding='UTF-8') as blocklist:
        blocklist.write('# Blocked Hosts List #\n\n')
        if line_format == 'etc_hosts':
            for host in blocked_hosts:
                blocklist.write('127.0.0.1  ' + host + '\n')
        elif line_format == 'one_per_line':
            for host in blocked_hosts:
                blocklist.write(host + '\n')
        elif line_format == 'not_correct':
            for host in blocked_hosts:
                blocklist.write(host + ' This is not a correct hosts file\n')
        else:
            raise ValueError('Incorrect line_format argument')
    return pathlib.Path(name)

def assert_urls(host_blocker, blocked=BLOCKLIST_HOSTS, whitelisted=WHITELISTED_HOSTS, urls_to_check=URLS_TO_CHECK):
    if False:
        for i in range(10):
            print('nop')
    "Test if Urls to check are blocked or not by HostBlocker.\n\n    Ensure URLs in 'blocked' and not in 'whitelisted' are blocked.\n    All other URLs must not be blocked.\n\n    localhost is an example of a special case that shouldn't be blocked.\n    "
    whitelisted = list(whitelisted) + ['localhost']
    for str_url in urls_to_check:
        url = QUrl(str_url)
        host = url.host()
        if host in blocked and host not in whitelisted:
            assert host_blocker._is_blocked(url)
        else:
            assert not host_blocker._is_blocked(url)

def blocklist_to_url(path):
    if False:
        while True:
            i = 10
    'Get an example.com-URL with the given filename as path.'
    assert not path.is_absolute(), path
    url = QUrl('http://example.com/')
    url.setPath('/' + str(path))
    assert url.isValid(), url.errorString()
    return url

def generic_blocklists(directory):
    if False:
        i = 10
        return i + 15
    'Return a generic list of files to be used in hosts-block-lists option.\n\n    This list contains :\n    - a remote zip file with 1 hosts file and 2 useless files\n    - a remote zip file with only useless files\n        (Should raise a FileNotFoundError)\n    - a remote zip file with only one valid hosts file\n    - a local text file with valid hosts\n    - a remote text file without valid hosts format.\n    '
    file1 = create_blocklist(directory, blocked_hosts=CLEAN_HOSTS, name='README', line_format='not_correct')
    file2 = create_blocklist(directory, blocked_hosts=BLOCKLIST_HOSTS[:3], name='hosts', line_format='etc_hosts')
    file3 = create_blocklist(directory, blocked_hosts=CLEAN_HOSTS, name='false_positive', line_format='one_per_line')
    files_to_zip = [file1, file2, file3]
    blocklist1 = blocklist_to_url(create_zipfile(directory, files_to_zip, 'block1'))
    file1 = create_blocklist(directory, blocked_hosts=CLEAN_HOSTS, name='md5sum', line_format='etc_hosts')
    file2 = create_blocklist(directory, blocked_hosts=CLEAN_HOSTS, name='README', line_format='not_correct')
    file3 = create_blocklist(directory, blocked_hosts=CLEAN_HOSTS, name='false_positive', line_format='one_per_line')
    files_to_zip = [file1, file2, file3]
    blocklist2 = blocklist_to_url(create_zipfile(directory, files_to_zip, 'block2'))
    file1 = create_blocklist(directory, blocked_hosts=[BLOCKLIST_HOSTS[3]], name='malwarelist', line_format='etc_hosts')
    blocklist3 = blocklist_to_url(create_zipfile(directory, [file1], 'block3'))
    blocklist4 = QUrl.fromLocalFile(str(directory / create_blocklist(directory, blocked_hosts=[BLOCKLIST_HOSTS[4]], name='mycustomblocklist', line_format='one_per_line')))
    assert blocklist4.isValid(), blocklist4.errorString()
    blocklist5 = blocklist_to_url(create_blocklist(directory, blocked_hosts=CLEAN_HOSTS, name='notcorrectlist', line_format='not_correct'))
    return [blocklist1.toString(), blocklist2.toString(), blocklist3.toString(), blocklist4.toString(), blocklist5.toString()]

@pytest.mark.parametrize('blocking_enabled, method', [(True, 'auto'), (True, 'adblock'), (False, 'auto'), (False, 'adblock'), (False, 'both'), (False, 'hosts')])
def test_disabled_blocking_update(config_stub, tmp_path, caplog, host_blocker_factory, blocking_enabled, method):
    if False:
        i = 10
        return i + 15
    'Ensure no URL is blocked when host blocking should be disabled.'
    if blocking_enabled and method == 'auto':
        pytest.importorskip('adblock')
    config_stub.val.content.blocking.hosts.lists = generic_blocklists(tmp_path)
    config_stub.val.content.blocking.enabled = blocking_enabled
    config_stub.val.content.blocking.method = method
    host_blocker = host_blocker_factory()
    downloads = host_blocker.adblock_update()
    while downloads._in_progress:
        current_download = downloads._in_progress[0]
        with caplog.at_level(logging.ERROR):
            current_download.successful = True
            current_download.finished.emit()
    host_blocker.read_hosts()
    for str_url in URLS_TO_CHECK:
        assert not host_blocker._is_blocked(QUrl(str_url))

def test_disabled_blocking_per_url(config_stub, host_blocker_factory):
    if False:
        for i in range(10):
            print('nop')
    example_com = 'https://www.example.com/'
    config_stub.val.content.blocking.method = 'hosts'
    config_stub.val.content.blocking.hosts.lists = []
    pattern = urlmatch.UrlPattern(example_com)
    config_stub.set_obj('content.blocking.enabled', False, pattern=pattern)
    url = QUrl('https://blocked.example.com')
    host_blocker = host_blocker_factory()
    host_blocker._blocked_hosts.add(url.host())
    assert host_blocker._is_blocked(url)
    assert not host_blocker._is_blocked(url, first_party_url=QUrl(example_com))

def test_no_blocklist_update(config_stub, download_stub, host_blocker_factory):
    if False:
        print('Hello World!')
    'Ensure no URL is blocked when no block list exists.'
    config_stub.val.content.blocking.hosts.lists = None
    config_stub.val.content.blocking.method = 'hosts'
    config_stub.val.content.blocking.enabled = True
    host_blocker = host_blocker_factory()
    host_blocker.adblock_update()
    host_blocker.read_hosts()
    for dl in download_stub.downloads:
        dl.successful = True
    for str_url in URLS_TO_CHECK:
        assert not host_blocker._is_blocked(QUrl(str_url))

def test_successful_update(config_stub, tmp_path, caplog, host_blocker_factory):
    if False:
        return 10
    'Ensure hosts from host_blocking.lists are blocked after an update.'
    config_stub.val.content.blocking.hosts.lists = generic_blocklists(tmp_path)
    config_stub.val.content.blocking.method = 'hosts'
    config_stub.val.content.blocking.enabled = True
    config_stub.val.content.blocking.whitelist = None
    host_blocker = host_blocker_factory()
    downloads = host_blocker.adblock_update()
    while downloads._in_progress:
        current_download = downloads._in_progress[0]
        with caplog.at_level(logging.ERROR):
            current_download.successful = True
            current_download.finished.emit()
    host_blocker.read_hosts()
    assert_urls(host_blocker, whitelisted=[])

def test_parsing_multiple_hosts_on_line(config_stub, host_blocker_factory):
    if False:
        print('Hello World!')
    'Ensure multiple hosts on a line get parsed correctly.'
    config_stub.val.content.blocking.method = 'hosts'
    config_stub.val.content.blocking.enabled = True
    host_blocker = host_blocker_factory()
    bytes_host_line = ' '.join(BLOCKLIST_HOSTS).encode('utf-8')
    parsed_hosts = host_blocker._read_hosts_line(bytes_host_line)
    host_blocker._blocked_hosts |= parsed_hosts
    assert_urls(host_blocker, whitelisted=[])

@pytest.mark.parametrize('ip, host', [('127.0.0.1', 'localhost'), ('27.0.0.1', 'localhost.localdomain'), ('27.0.0.1', 'local'), ('55.255.255.255', 'broadcasthost'), (':1', 'localhost'), (':1', 'ip6-localhost'), (':1', 'ip6-loopback'), ('e80::1%lo0', 'localhost'), ('f00::0', 'ip6-localnet'), ('f00::0', 'ip6-mcastprefix'), ('f02::1', 'ip6-allnodes'), ('f02::2', 'ip6-allrouters'), ('ff02::3', 'ip6-allhosts'), ('.0.0.0', '0.0.0.0'), ('127.0.1.1', 'myhostname'), ('127.0.0.53', 'myhostname')])
def test_whitelisted_lines(host_blocker_factory, ip, host):
    if False:
        return 10
    "Make sure we don't block hosts we don't want to."
    host_blocker = host_blocker_factory()
    line = '{} {}'.format(ip, host).encode('ascii')
    parsed_hosts = host_blocker._read_hosts_line(line)
    assert host not in parsed_hosts

def test_failed_dl_update(config_stub, tmp_path, caplog, host_blocker_factory):
    if False:
        print('Hello World!')
    'One blocklist fails to download.\n\n    Ensure hosts from this list are not blocked.\n    '
    dl_fail_blocklist = blocklist_to_url(create_blocklist(tmp_path, blocked_hosts=CLEAN_HOSTS, name='download_will_fail', line_format='one_per_line'))
    hosts_to_block = generic_blocklists(tmp_path) + [dl_fail_blocklist.toString()]
    config_stub.val.content.blocking.hosts.lists = hosts_to_block
    config_stub.val.content.blocking.enabled = True
    config_stub.val.content.blocking.method = 'hosts'
    config_stub.val.content.blocking.whitelist = None
    host_blocker = host_blocker_factory()
    downloads = host_blocker.adblock_update()
    while downloads._in_progress:
        current_download = downloads._in_progress[0]
        if current_download.name == dl_fail_blocklist.path():
            current_download.successful = False
        else:
            current_download.successful = True
        with caplog.at_level(logging.ERROR):
            current_download.finished.emit()
    host_blocker.read_hosts()
    assert_urls(host_blocker, whitelisted=[])

@pytest.mark.parametrize('location', ['content', 'comment'])
def test_invalid_utf8(config_stub, tmp_path, caplog, host_blocker_factory, location):
    if False:
        for i in range(10):
            print('nop')
    'Make sure invalid UTF-8 is handled correctly.\n\n    See https://github.com/qutebrowser/qutebrowser/issues/2301\n    '
    blocklist = tmp_path / 'blocklist'
    if location == 'comment':
        blocklist.write_bytes(b'# nbsp: \xa0\n')
    else:
        assert location == 'content'
        blocklist.write_bytes(b'https://www.example.org/\xa0')
    with blocklist.open('a') as f:
        for url in BLOCKLIST_HOSTS:
            f.write(url + '\n')
    url = blocklist_to_url(pathlib.Path('blocklist'))
    config_stub.val.content.blocking.hosts.lists = [url.toString()]
    config_stub.val.content.blocking.enabled = True
    config_stub.val.content.blocking.method = 'hosts'
    config_stub.val.content.blocking.whitelist = None
    host_blocker = host_blocker_factory()
    downloads = host_blocker.adblock_update()
    current_download = downloads._in_progress[0]
    if location == 'content':
        with caplog.at_level(logging.ERROR):
            current_download.successful = True
            current_download.finished.emit()
        expected = "Failed to decode: b'https://www.example.org/\\xa0localhost"
        assert caplog.messages[-2].startswith(expected)
    else:
        current_download.successful = True
        current_download.finished.emit()
    host_blocker.read_hosts()
    assert_urls(host_blocker, whitelisted=[])

def test_invalid_utf8_compiled(config_stub, config_tmpdir, data_tmpdir, monkeypatch, caplog, host_blocker_factory):
    if False:
        while True:
            i = 10
    'Make sure invalid UTF-8 in the compiled file is handled.'
    config_stub.val.content.blocking.hosts.lists = []
    monkeypatch.setattr(hostblock.HostBlocker, 'update_files', lambda _self: None)
    (config_tmpdir / 'blocked-hosts').write_binary(b'https://www.example.org/\xa0')
    (data_tmpdir / 'blocked-hosts').ensure()
    host_blocker = host_blocker_factory()
    with caplog.at_level(logging.ERROR):
        host_blocker.read_hosts()
    assert caplog.messages[-1] == 'Failed to read host blocklist!'

def test_blocking_with_whitelist(config_stub, data_tmpdir, host_blocker_factory):
    if False:
        i = 10
        return i + 15
    'Ensure hosts in content.blocking.whitelist are never blocked.'
    filtered_blocked_hosts = BLOCKLIST_HOSTS[1:]
    blocklist = create_blocklist(data_tmpdir, blocked_hosts=filtered_blocked_hosts, name='blocked-hosts', line_format='one_per_line')
    config_stub.val.content.blocking.hosts.lists = [str(blocklist)]
    config_stub.val.content.blocking.enabled = True
    config_stub.val.content.blocking.method = 'hosts'
    config_stub.val.content.blocking.whitelist = list(WHITELISTED_HOSTS)
    host_blocker = host_blocker_factory()
    host_blocker.read_hosts()
    assert_urls(host_blocker)

def test_config_change_initial(config_stub, tmp_path, host_blocker_factory):
    if False:
        return 10
    'Test emptying host_blocking.lists with existing blocked_hosts.\n\n    - A blocklist is present in host_blocking.lists and blocked_hosts is\n      populated\n    - User quits qutebrowser, empties host_blocking.lists from his config\n    - User restarts qutebrowser, does adblock-update\n    '
    create_blocklist(tmp_path, blocked_hosts=BLOCKLIST_HOSTS, name='blocked-hosts', line_format='one_per_line')
    config_stub.val.content.blocking.hosts.lists = None
    config_stub.val.content.blocking.enabled = True
    config_stub.val.content.blocking.method = 'hosts'
    config_stub.val.content.blocking.whitelist = None
    host_blocker = host_blocker_factory()
    host_blocker.read_hosts()
    for str_url in URLS_TO_CHECK:
        assert not host_blocker._is_blocked(QUrl(str_url))

def test_config_change(config_stub, tmp_path, host_blocker_factory):
    if False:
        print('Hello World!')
    'Ensure blocked-hosts resets if host-block-list is changed to None.'
    filtered_blocked_hosts = BLOCKLIST_HOSTS[1:]
    blocklist = blocklist_to_url(create_blocklist(tmp_path, blocked_hosts=filtered_blocked_hosts, name='blocked-hosts', line_format='one_per_line'))
    config_stub.val.content.blocking.hosts.lists = [blocklist.toString()]
    config_stub.val.content.blocking.enabled = True
    config_stub.val.content.blocking.method = 'hosts'
    config_stub.val.content.blocking.whitelist = None
    host_blocker = host_blocker_factory()
    host_blocker.read_hosts()
    config_stub.val.content.blocking.hosts.lists = None
    host_blocker.read_hosts()
    for str_url in URLS_TO_CHECK:
        assert not host_blocker._is_blocked(QUrl(str_url))

def test_add_directory(config_stub, tmp_path, host_blocker_factory):
    if False:
        print('Hello World!')
    'Ensure adblocker can import all files in a directory.'
    blocklist_hosts2 = []
    for i in BLOCKLIST_HOSTS[1:]:
        blocklist_hosts2.append('1' + i)
    create_blocklist(tmp_path, blocked_hosts=BLOCKLIST_HOSTS, name='blocked-hosts', line_format='one_per_line')
    create_blocklist(tmp_path, blocked_hosts=blocklist_hosts2, name='blocked-hosts2', line_format='one_per_line')
    config_stub.val.content.blocking.hosts.lists = [str(tmp_path)]
    config_stub.val.content.blocking.enabled = True
    config_stub.val.content.blocking.method = 'hosts'
    host_blocker = host_blocker_factory()
    host_blocker.adblock_update()
    assert len(host_blocker._blocked_hosts) == len(blocklist_hosts2) * 2

def test_adblock_benchmark(data_tmpdir, benchmark, host_blocker_factory):
    if False:
        print('Hello World!')
    blocked_hosts = data_tmpdir / 'blocked-hosts'
    blocked_hosts.write_text('\n'.join(testutils.blocked_hosts()), encoding='utf-8')
    url = QUrl('https://www.example.org/')
    blocker = host_blocker_factory()
    blocker.read_hosts()
    assert blocker._blocked_hosts
    benchmark(lambda : blocker._is_blocked(url))

@pytest.mark.parametrize('block_subdomains', [True, False])
def test_subdomain_blocking(config_stub, host_blocker_factory, block_subdomains):
    if False:
        for i in range(10):
            print('nop')
    config_stub.val.content.blocking.method = 'hosts'
    config_stub.val.content.blocking.hosts.lists = None
    config_stub.val.content.blocking.hosts.block_subdomains = block_subdomains
    host_blocker = host_blocker_factory()
    host_blocker._blocked_hosts.add('example.com')
    is_blocked = host_blocker._is_blocked(QUrl('https://subdomain.example.com'))
    assert is_blocked == block_subdomains