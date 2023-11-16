"""
Test the downloaders.
"""
from coursera import downloaders
from coursera import coursera_dl
from coursera.filtering import find_resources_to_get
import pytest

@pytest.fixture
def sample_bag():
    if False:
        while True:
            i = 10
    sample_bag = {'mp4': [['h://url1/lc1.mp4', 'video']], 'pdf': [['h://url2/lc2.pdf', 'slides']], 'txt': [['h://url3/lc3.txt', 'subtitle']]}
    return sample_bag

def test_collect_all_resources(sample_bag):
    if False:
        while True:
            i = 10
    res = find_resources_to_get(sample_bag, 'all', None)
    assert [('mp4', 'h://url1/lc1.mp4', 'video'), ('pdf', 'h://url2/lc2.pdf', 'slides'), ('txt', 'h://url3/lc3.txt', 'subtitle')] == sorted(res)

def test_collect_only_pdfs(sample_bag):
    if False:
        for i in range(10):
            print('nop')
    res = find_resources_to_get(sample_bag, 'pdf', None)
    assert [('pdf', 'h://url2/lc2.pdf', 'slides')] == sorted(res)

def test_collect_with_filtering(sample_bag):
    if False:
        return 10
    res = find_resources_to_get(sample_bag, 'all', 'de')
    res = sorted(res)
    assert [('mp4', 'h://url1/lc1.mp4', 'video'), ('pdf', 'h://url2/lc2.pdf', 'slides')] == res

def _ext_get_session():
    if False:
        return 10
    import time
    import requests
    expires = int(time.time() + 60 * 60 * 24 * 365 * 50)
    s = requests.Session()
    s.cookies.set('csrf_token', 'csrfclass001', domain='www.coursera.org', expires=expires)
    s.cookies.set('session', 'sessionclass1', domain='www.coursera.org', expires=expires)
    s.cookies.set('k', 'v', domain='www.example.org', expires=expires)
    return s

def test_bin_not_specified():
    if False:
        while True:
            i = 10
    pytest.raises(RuntimeError, downloaders.ExternalDownloader, None)

def test_bin_not_found_raises_exception():
    if False:
        print('Hello World!')
    d = downloaders.ExternalDownloader(None, bin='no_way_this_exists')
    d._prepare_cookies = lambda cmd, cv: None
    d._create_command = lambda x, y: ['no_way_this_exists']
    pytest.raises(OSError, d._start_download, 'url', 'filename', False)

def test_bin_is_set():
    if False:
        i = 10
        return i + 15
    d = downloaders.ExternalDownloader(None, bin='test')
    assert d.bin == 'test'

def test_prepare_cookies():
    if False:
        i = 10
        return i + 15
    s = _ext_get_session()
    d = downloaders.ExternalDownloader(s, bin='test')

    def mock_add_cookies(cmd, cv):
        if False:
            while True:
                i = 10
        cmd.append(cv)
    d._add_cookies = mock_add_cookies
    command = []
    d._prepare_cookies(command, 'http://www.coursera.org')
    assert 'csrf_token=csrfclass001' in command[0]
    assert 'session=sessionclass1' in command[0]

def test_prepare_cookies_does_nothing():
    if False:
        print('Hello World!')
    s = _ext_get_session()
    s.cookies.clear(domain='www.coursera.org')
    d = downloaders.ExternalDownloader(s, bin='test')
    command = []

    def mock_add_cookies(cmd, cookie_values):
        if False:
            while True:
                i = 10
        pass
    d._add_cookies = mock_add_cookies
    d._prepare_cookies(command, 'http://www.coursera.org')
    assert command == []

def test_start_command_raises_exception():
    if False:
        i = 10
        return i + 15
    d = downloaders.ExternalDownloader(None, bin='test')
    d._add_cookies = lambda cmd, cookie_values: None
    pytest.raises(NotImplementedError, d._create_command, 'url', 'filename')

def test_wget():
    if False:
        while True:
            i = 10
    s = _ext_get_session()
    d = downloaders.WgetDownloader(s)
    command = d._create_command('download_url', 'save_to')
    assert command[0] == 'wget'
    assert 'download_url' in command
    assert 'save_to' in command
    d._prepare_cookies(command, 'http://www.coursera.org')
    assert any(('Cookie: ' in e for e in command))
    assert any(('csrf_token=csrfclass001' in e for e in command))
    assert any(('session=sessionclass1' in e for e in command))

def test_curl():
    if False:
        i = 10
        return i + 15
    s = _ext_get_session()
    d = downloaders.CurlDownloader(s)
    command = d._create_command('download_url', 'save_to')
    assert command[0] == 'curl'
    assert 'download_url' in command
    assert 'save_to' in command
    d._prepare_cookies(command, 'http://www.coursera.org')
    assert any(('csrf_token=csrfclass001' in e for e in command))
    assert any(('session=sessionclass1' in e for e in command))

def test_aria2():
    if False:
        i = 10
        return i + 15
    s = _ext_get_session()
    d = downloaders.Aria2Downloader(s)
    command = d._create_command('download_url', 'save_to')
    assert command[0] == 'aria2c'
    assert 'download_url' in command
    assert 'save_to' in command
    d._prepare_cookies(command, 'http://www.coursera.org')
    assert any(('Cookie: ' in e for e in command))
    assert any(('csrf_token=csrfclass001' in e for e in command))
    assert any(('session=sessionclass1' in e for e in command))

def test_axel():
    if False:
        return 10
    s = _ext_get_session()
    d = downloaders.AxelDownloader(s)
    command = d._create_command('download_url', 'save_to')
    assert command[0] == 'axel'
    assert 'download_url' in command
    assert 'save_to' in command
    d._prepare_cookies(command, 'http://www.coursera.org')
    assert any(('Cookie: ' in e for e in command))
    assert any(('csrf_token=csrfclass001' in e for e in command))
    assert any(('session=sessionclass1' in e for e in command))

def test_all_attempts_have_failed():
    if False:
        i = 10
        return i + 15
    import time

    class IObject(object):
        pass

    class MockSession(object):

        def get(self, url, stream=True, headers={}):
            if False:
                while True:
                    i = 10
            object_ = IObject()
            object_.status_code = 400
            object_.reason = None
            return object_
    _sleep = time.sleep
    time.sleep = lambda interval: 0
    session = MockSession()
    d = downloaders.NativeDownloader(session)
    assert d._start_download('download_url', 'save_to', False) is False
    time.sleep = _sleep

def _get_progress(total):
    if False:
        return 10
    p = downloaders.DownloadProgress(total)
    p.report_progress = lambda : None
    return p

def test_calc_percent_if_total_is_zero():
    if False:
        for i in range(10):
            print('nop')
    p = _get_progress(0)
    assert p.calc_percent() == '--%'
    p.read(10)
    assert p.calc_percent() == '--%'

def test_calc_percent_if_not_yet_read():
    if False:
        print('Hello World!')
    p = _get_progress(100)
    assert p.calc_percent() == '[                                                  ] 0%'

def test_calc_percent_if_read():
    if False:
        i = 10
        return i + 15
    p = _get_progress(100)
    p.read(2)
    assert p.calc_percent() == '[#                                                 ] 2%'
    p.read(18)
    assert p.calc_percent() == '[##########                                        ] 20%'
    p = _get_progress(2300)
    p.read(177)
    assert p.calc_percent() == '[###                                               ] 7%'

def test_calc_speed_if_total_is_zero():
    if False:
        while True:
            i = 10
    p = _get_progress(0)
    assert p.calc_speed() == '---b/s'

def test_calc_speed_if_not_yet_read():
    if False:
        return 10
    p = _get_progress(100)
    assert p.calc_speed() == '---b/s'

def test_calc_speed_ifread():
    if False:
        return 10
    p = _get_progress(10000)
    p.read(2000)
    p._now = p._start + 1000
    assert p.calc_speed() == '2.00B/s'