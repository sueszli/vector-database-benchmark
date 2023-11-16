import collections
import os
import sys
import pytest
import llnl.util.tty as tty
from llnl.util.filesystem import is_exe, working_dir
import spack.config
import spack.error
import spack.fetch_strategy as fs
import spack.repo
import spack.util.crypto as crypto
import spack.util.executable
import spack.util.web as web_util
from spack.spec import Spec
from spack.stage import Stage
from spack.util.executable import which

@pytest.fixture(params=list(crypto.hashes.keys()))
def checksum_type(request):
    if False:
        for i in range(10):
            print('nop')
    return request.param

@pytest.fixture
def pkg_factory():
    if False:
        while True:
            i = 10
    Pkg = collections.namedtuple('Pkg', ['url_for_version', 'all_urls_for_version', 'find_valid_url_for_version', 'urls', 'url', 'versions', 'fetch_options'])

    def factory(url, urls, fetch_options={}):
        if False:
            print('Hello World!')

        def fn(v):
            if False:
                print('Hello World!')
            main_url = url or urls[0]
            return spack.url.substitute_version(main_url, v)

        def fn_urls(v):
            if False:
                i = 10
                return i + 15
            urls_loc = urls or [url]
            return [spack.url.substitute_version(u, v) for u in urls_loc]
        return Pkg(find_valid_url_for_version=fn, url_for_version=fn, all_urls_for_version=fn_urls, url=url, urls=(urls,), versions=collections.defaultdict(dict), fetch_options=fetch_options)
    return factory

@pytest.mark.parametrize('_fetch_method', ['curl', 'urllib'])
def test_urlfetchstrategy_sans_url(_fetch_method):
    if False:
        i = 10
        return i + 15
    'Ensure constructor with no URL fails.'
    with spack.config.override('config:url_fetch_method', _fetch_method):
        with pytest.raises(ValueError):
            with fs.URLFetchStrategy(None):
                pass

@pytest.mark.parametrize('_fetch_method', ['curl', 'urllib'])
def test_urlfetchstrategy_bad_url(tmpdir, _fetch_method):
    if False:
        i = 10
        return i + 15
    'Ensure fetch with bad URL fails as expected.'
    testpath = str(tmpdir)
    with spack.config.override('config:url_fetch_method', _fetch_method):
        with pytest.raises(fs.FailedDownloadError):
            fetcher = fs.URLFetchStrategy(url='file:///does-not-exist')
            assert fetcher is not None
            with Stage(fetcher, path=testpath) as stage:
                assert stage is not None
                assert fetcher.archive_file is None
                fetcher.fetch()

def test_fetch_options(tmpdir, mock_archive):
    if False:
        for i in range(10):
            print('nop')
    testpath = str(tmpdir)
    with spack.config.override('config:url_fetch_method', 'curl'):
        fetcher = fs.URLFetchStrategy(url=mock_archive.url, fetch_options={'cookie': 'True', 'timeout': 10})
        assert fetcher is not None
        with Stage(fetcher, path=testpath) as stage:
            assert stage is not None
            assert fetcher.archive_file is None
            fetcher.fetch()

@pytest.mark.parametrize('_fetch_method', ['curl', 'urllib'])
def test_archive_file_errors(tmpdir, mock_archive, _fetch_method):
    if False:
        return 10
    'Ensure FetchStrategy commands may only be used as intended'
    testpath = str(tmpdir)
    with spack.config.override('config:url_fetch_method', _fetch_method):
        fetcher = fs.URLFetchStrategy(url=mock_archive.url)
        assert fetcher is not None
        with pytest.raises(fs.FailedDownloadError):
            with Stage(fetcher, path=testpath) as stage:
                assert stage is not None
                assert fetcher.archive_file is None
                with pytest.raises(fs.NoArchiveFileError):
                    fetcher.archive(testpath)
                with pytest.raises(fs.NoArchiveFileError):
                    fetcher.expand()
                with pytest.raises(fs.NoArchiveFileError):
                    fetcher.reset()
                stage.fetch()
                with pytest.raises(fs.NoDigestError):
                    fetcher.check()
                assert fetcher.archive_file is not None
                fetcher._fetch_from_url('file:///does-not-exist')
files = [('.tar.gz', 'z'), ('.tgz', 'z')]
if sys.platform != 'win32':
    files += [('.tar.bz2', 'j'), ('.tbz2', 'j'), ('.tar.xz', 'J'), ('.txz', 'J')]

@pytest.mark.parametrize('secure', [True, False])
@pytest.mark.parametrize('_fetch_method', ['curl', 'urllib'])
@pytest.mark.parametrize('mock_archive', files, indirect=True)
def test_fetch(mock_archive, secure, _fetch_method, checksum_type, default_mock_concretization, mutable_mock_repo):
    if False:
        print('Hello World!')
    'Fetch an archive and make sure we can checksum it.'
    algo = crypto.hash_fun_for_algo(checksum_type)()
    with open(mock_archive.archive_file, 'rb') as f:
        algo.update(f.read())
    checksum = algo.hexdigest()
    s = default_mock_concretization('url-test')
    s.package.url = mock_archive.url
    s.package.versions[spack.version.Version('test')] = {checksum_type: checksum, 'url': s.package.url}
    with s.package.stage:
        with spack.config.override('config:verify_ssl', secure):
            with spack.config.override('config:url_fetch_method', _fetch_method):
                s.package.do_stage()
        with working_dir(s.package.stage.source_path):
            assert os.path.exists('configure')
            assert is_exe('configure')
            with open('configure') as f:
                contents = f.read()
            assert contents.startswith('#!/bin/sh')
            assert 'echo Building...' in contents

@pytest.mark.not_on_windows('Not supported on Windows (yet)')
@pytest.mark.parametrize('spec,url,digest', [('url-list-test @=0.0.0', 'foo-0.0.0.tar.gz', '00000000000000000000000000000000'), ('url-list-test @=1.0.0', 'foo-1.0.0.tar.gz', '00000000000000000000000000000100'), ('url-list-test @=3.0', 'foo-3.0.tar.gz', '00000000000000000000000000000030'), ('url-list-test @=4.5', 'foo-4.5.tar.gz', '00000000000000000000000000000450'), ('url-list-test @=2.0.0b2', 'foo-2.0.0b2.tar.gz', '000000000000000000000000000200b2'), ('url-list-test @=3.0a1', 'foo-3.0a1.tar.gz', '000000000000000000000000000030a1'), ('url-list-test @=4.5-rc5', 'foo-4.5-rc5.tar.gz', '000000000000000000000000000045c5')])
@pytest.mark.parametrize('_fetch_method', ['curl', 'urllib'])
def test_from_list_url(mock_packages, config, spec, url, digest, _fetch_method):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test URLs in the url-list-test package, which means they should\n    have checksums in the package.\n    '
    with spack.config.override('config:url_fetch_method', _fetch_method):
        s = Spec(spec).concretized()
        fetch_strategy = fs.from_list_url(s.package)
        assert isinstance(fetch_strategy, fs.URLFetchStrategy)
        assert os.path.basename(fetch_strategy.url) == url
        assert fetch_strategy.digest == digest
        assert fetch_strategy.extra_options == {}
        s.package.fetch_options = {'timeout': 60}
        fetch_strategy = fs.from_list_url(s.package)
        assert fetch_strategy.extra_options == {'timeout': 60}

@pytest.mark.not_on_windows('Not supported on Windows (yet)')
@pytest.mark.parametrize('_fetch_method', ['curl', 'urllib'])
@pytest.mark.parametrize('requested_version,tarball,digest', [('=4.5.0', 'foo-4.5.0.tar.gz', None), ('2.0.0', 'foo-2.0.0b2.tar.gz', '000000000000000000000000000200b2')])
@pytest.mark.only_clingo("Original concretizer doesn't resolve concrete versions to known ones")
def test_new_version_from_list_url(mock_packages, config, _fetch_method, requested_version, tarball, digest):
    if False:
        for i in range(10):
            print('nop')
    'Test non-specific URLs from the url-list-test package.'
    with spack.config.override('config:url_fetch_method', _fetch_method):
        s = Spec('url-list-test @%s' % requested_version).concretized()
        fetch_strategy = fs.from_list_url(s.package)
        assert isinstance(fetch_strategy, fs.URLFetchStrategy)
        assert os.path.basename(fetch_strategy.url) == tarball
        assert fetch_strategy.digest == digest
        assert fetch_strategy.extra_options == {}
        s.package.fetch_options = {'timeout': 60}
        fetch_strategy = fs.from_list_url(s.package)
        assert fetch_strategy.extra_options == {'timeout': 60}

def test_nosource_from_list_url(mock_packages, config):
    if False:
        i = 10
        return i + 15
    'This test confirms BundlePackages do not have list url.'
    s = Spec('nosource').concretized()
    fetch_strategy = fs.from_list_url(s.package)
    assert fetch_strategy is None

def test_hash_detection(checksum_type):
    if False:
        i = 10
        return i + 15
    algo = crypto.hash_fun_for_algo(checksum_type)()
    h = 'f' * (algo.digest_size * 2)
    checker = crypto.Checker(h)
    assert checker.hash_name == checksum_type

def test_unknown_hash(checksum_type):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):
        crypto.Checker('a')

@pytest.mark.skipif(which('curl') is None, reason='Urllib does not have built-in status bar')
def test_url_with_status_bar(tmpdir, mock_archive, monkeypatch, capfd):
    if False:
        while True:
            i = 10
    'Ensure fetch with status bar option succeeds.'

    def is_true():
        if False:
            for i in range(10):
                print('nop')
        return True
    testpath = str(tmpdir)
    monkeypatch.setattr(sys.stdout, 'isatty', is_true)
    monkeypatch.setattr(tty, 'msg_enabled', is_true)
    with spack.config.override('config:url_fetch_method', 'curl'):
        fetcher = fs.URLFetchStrategy(mock_archive.url)
        with Stage(fetcher, path=testpath) as stage:
            assert fetcher.archive_file is None
            stage.fetch()
        status = capfd.readouterr()[1]
        assert '##### 100' in status

@pytest.mark.parametrize('_fetch_method', ['curl', 'urllib'])
def test_url_extra_fetch(tmpdir, mock_archive, _fetch_method):
    if False:
        return 10
    'Ensure a fetch after downloading is effectively a no-op.'
    with spack.config.override('config:url_fetch_method', _fetch_method):
        testpath = str(tmpdir)
        fetcher = fs.URLFetchStrategy(mock_archive.url)
        with Stage(fetcher, path=testpath) as stage:
            assert fetcher.archive_file is None
            stage.fetch()
            assert fetcher.archive_file is not None
            fetcher.fetch()

@pytest.mark.parametrize('url,urls,version,expected', [(None, ['https://ftpmirror.gnu.org/autoconf/autoconf-2.69.tar.gz', 'https://ftp.gnu.org/gnu/autoconf/autoconf-2.69.tar.gz'], '2.62', ['https://ftpmirror.gnu.org/autoconf/autoconf-2.62.tar.gz', 'https://ftp.gnu.org/gnu/autoconf/autoconf-2.62.tar.gz'])])
@pytest.mark.parametrize('_fetch_method', ['curl', 'urllib'])
def test_candidate_urls(pkg_factory, url, urls, version, expected, _fetch_method):
    if False:
        print('Hello World!')
    'Tests that candidate urls include mirrors and that they go through\n    pattern matching and substitution for versions.\n    '
    with spack.config.override('config:url_fetch_method', _fetch_method):
        pkg = pkg_factory(url, urls)
        f = fs._from_merged_attrs(fs.URLFetchStrategy, pkg, version)
        assert f.candidate_urls == expected
        assert f.extra_options == {}
        pkg = pkg_factory(url, urls, fetch_options={'timeout': 60})
        f = fs._from_merged_attrs(fs.URLFetchStrategy, pkg, version)
        assert f.extra_options == {'timeout': 60}

@pytest.mark.regression('19673')
def test_missing_curl(tmpdir, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'Ensure a fetch involving missing curl package reports the error.'
    err_fmt = 'No such command {0}'

    def _which(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        err_msg = err_fmt.format(args[0])
        raise spack.util.executable.CommandNotFoundError(err_msg)
    monkeypatch.setattr(fs, 'which', _which)
    testpath = str(tmpdir)
    url = 'http://github.com/spack/spack'
    with spack.config.override('config:url_fetch_method', 'curl'):
        fetcher = fs.URLFetchStrategy(url=url)
        assert fetcher is not None
        with pytest.raises(TypeError, match='object is not callable'):
            with Stage(fetcher, path=testpath) as stage:
                out = stage.fetch()
            assert err_fmt.format('curl') in out

def test_url_fetch_text_without_url(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(spack.error.FetchError, match='URL is required'):
        web_util.fetch_url_text(None)

def test_url_fetch_text_curl_failures(tmpdir, monkeypatch):
    if False:
        while True:
            i = 10
    "Check fetch_url_text if URL's curl is missing."
    err_fmt = 'No such command {0}'

    def _which(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        err_msg = err_fmt.format(args[0])
        raise spack.util.executable.CommandNotFoundError(err_msg)
    monkeypatch.setattr(spack.util.web, 'which', _which)
    with spack.config.override('config:url_fetch_method', 'curl'):
        with pytest.raises(spack.error.FetchError, match='Missing required curl'):
            web_util.fetch_url_text('https://github.com/')

def test_url_check_curl_errors():
    if False:
        i = 10
        return i + 15
    'Check that standard curl error returncodes raise expected errors.'
    with pytest.raises(spack.error.FetchError, match='not found'):
        web_util.check_curl_code(22)
    with pytest.raises(spack.error.FetchError, match='invalid certificate'):
        web_util.check_curl_code(60)

def test_url_missing_curl(tmpdir, monkeypatch):
    if False:
        print('Hello World!')
    "Check url_exists failures if URL's curl is missing."
    err_fmt = 'No such command {0}'

    def _which(*args, **kwargs):
        if False:
            print('Hello World!')
        err_msg = err_fmt.format(args[0])
        raise spack.util.executable.CommandNotFoundError(err_msg)
    monkeypatch.setattr(spack.util.web, 'which', _which)
    with spack.config.override('config:url_fetch_method', 'curl'):
        with pytest.raises(spack.error.FetchError, match='Missing required curl'):
            web_util.url_exists('https://github.com/')

def test_url_fetch_text_urllib_bad_returncode(tmpdir, monkeypatch):
    if False:
        print('Hello World!')

    class response:

        def getcode(self):
            if False:
                while True:
                    i = 10
            return 404

    def _read_from_url(*args, **kwargs):
        if False:
            return 10
        return (None, None, response())
    monkeypatch.setattr(spack.util.web, 'read_from_url', _read_from_url)
    with spack.config.override('config:url_fetch_method', 'urllib'):
        with pytest.raises(spack.error.FetchError, match='failed with error code'):
            web_util.fetch_url_text('https://github.com/')

def test_url_fetch_text_urllib_web_error(tmpdir, monkeypatch):
    if False:
        print('Hello World!')

    def _raise_web_error(*args, **kwargs):
        if False:
            print('Hello World!')
        raise web_util.SpackWebError('bad url')
    monkeypatch.setattr(spack.util.web, 'read_from_url', _raise_web_error)
    with spack.config.override('config:url_fetch_method', 'urllib'):
        with pytest.raises(spack.error.FetchError, match='fetch failed to verify'):
            web_util.fetch_url_text('https://github.com/')