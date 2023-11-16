import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError

def urlopen_stub(url, data=None):
    if False:
        for i in range(10):
            print('nop')
    'Stub to replace urlopen for testing.'
    if url == valid_httpurl():
        tmpfile = NamedTemporaryFile(prefix='urltmp_')
        return tmpfile
    else:
        raise URLError('Name or service not known')
old_urlopen = None

def setup_module():
    if False:
        print('Hello World!')
    global old_urlopen
    old_urlopen = urllib_request.urlopen
    urllib_request.urlopen = urlopen_stub

def teardown_module():
    if False:
        i = 10
        return i + 15
    urllib_request.urlopen = old_urlopen
http_path = 'http://www.google.com/'
http_file = 'index.html'
http_fakepath = 'http://fake.abc.web/site/'
http_fakefile = 'fake.txt'
malicious_files = ['/etc/shadow', '../../shadow', '..\\system.dat', 'c:\\windows\\system.dat']
magic_line = b'three is the magic number'

def valid_textfile(filedir):
    if False:
        for i in range(10):
            print('nop')
    (fd, path) = mkstemp(suffix='.txt', prefix='dstmp_', dir=filedir, text=True)
    os.close(fd)
    return path

def invalid_textfile(filedir):
    if False:
        while True:
            i = 10
    (fd, path) = mkstemp(suffix='.txt', prefix='dstmp_', dir=filedir)
    os.close(fd)
    os.remove(path)
    return path

def valid_httpurl():
    if False:
        for i in range(10):
            print('nop')
    return http_path + http_file

def invalid_httpurl():
    if False:
        for i in range(10):
            print('nop')
    return http_fakepath + http_fakefile

def valid_baseurl():
    if False:
        return 10
    return http_path

def invalid_baseurl():
    if False:
        i = 10
        return i + 15
    return http_fakepath

def valid_httpfile():
    if False:
        for i in range(10):
            print('nop')
    return http_file

def invalid_httpfile():
    if False:
        print('Hello World!')
    return http_fakefile

class TestDataSourceOpen:

    def setup_method(self):
        if False:
            return 10
        self.tmpdir = mkdtemp()
        self.ds = datasource.DataSource(self.tmpdir)

    def teardown_method(self):
        if False:
            while True:
                i = 10
        rmtree(self.tmpdir)
        del self.ds

    def test_ValidHTTP(self):
        if False:
            for i in range(10):
                print('nop')
        fh = self.ds.open(valid_httpurl())
        assert_(fh)
        fh.close()

    def test_InvalidHTTP(self):
        if False:
            for i in range(10):
                print('nop')
        url = invalid_httpurl()
        assert_raises(OSError, self.ds.open, url)
        try:
            self.ds.open(url)
        except OSError as e:
            assert_(e.errno is None)

    def test_InvalidHTTPCacheURLError(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(URLError, self.ds._cache, invalid_httpurl())

    def test_ValidFile(self):
        if False:
            while True:
                i = 10
        local_file = valid_textfile(self.tmpdir)
        fh = self.ds.open(local_file)
        assert_(fh)
        fh.close()

    def test_InvalidFile(self):
        if False:
            return 10
        invalid_file = invalid_textfile(self.tmpdir)
        assert_raises(OSError, self.ds.open, invalid_file)

    def test_ValidGzipFile(self):
        if False:
            while True:
                i = 10
        try:
            import gzip
        except ImportError:
            pytest.skip()
        filepath = os.path.join(self.tmpdir, 'foobar.txt.gz')
        fp = gzip.open(filepath, 'w')
        fp.write(magic_line)
        fp.close()
        fp = self.ds.open(filepath)
        result = fp.readline()
        fp.close()
        assert_equal(magic_line, result)

    def test_ValidBz2File(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            import bz2
        except ImportError:
            pytest.skip()
        filepath = os.path.join(self.tmpdir, 'foobar.txt.bz2')
        fp = bz2.BZ2File(filepath, 'w')
        fp.write(magic_line)
        fp.close()
        fp = self.ds.open(filepath)
        result = fp.readline()
        fp.close()
        assert_equal(magic_line, result)

class TestDataSourceExists:

    def setup_method(self):
        if False:
            print('Hello World!')
        self.tmpdir = mkdtemp()
        self.ds = datasource.DataSource(self.tmpdir)

    def teardown_method(self):
        if False:
            return 10
        rmtree(self.tmpdir)
        del self.ds

    def test_ValidHTTP(self):
        if False:
            for i in range(10):
                print('nop')
        assert_(self.ds.exists(valid_httpurl()))

    def test_InvalidHTTP(self):
        if False:
            while True:
                i = 10
        assert_equal(self.ds.exists(invalid_httpurl()), False)

    def test_ValidFile(self):
        if False:
            print('Hello World!')
        tmpfile = valid_textfile(self.tmpdir)
        assert_(self.ds.exists(tmpfile))
        localdir = mkdtemp()
        tmpfile = valid_textfile(localdir)
        assert_(self.ds.exists(tmpfile))
        rmtree(localdir)

    def test_InvalidFile(self):
        if False:
            while True:
                i = 10
        tmpfile = invalid_textfile(self.tmpdir)
        assert_equal(self.ds.exists(tmpfile), False)

class TestDataSourceAbspath:

    def setup_method(self):
        if False:
            print('Hello World!')
        self.tmpdir = os.path.abspath(mkdtemp())
        self.ds = datasource.DataSource(self.tmpdir)

    def teardown_method(self):
        if False:
            for i in range(10):
                print('nop')
        rmtree(self.tmpdir)
        del self.ds

    def test_ValidHTTP(self):
        if False:
            while True:
                i = 10
        (scheme, netloc, upath, pms, qry, frg) = urlparse(valid_httpurl())
        local_path = os.path.join(self.tmpdir, netloc, upath.strip(os.sep).strip('/'))
        assert_equal(local_path, self.ds.abspath(valid_httpurl()))

    def test_ValidFile(self):
        if False:
            i = 10
            return i + 15
        tmpfile = valid_textfile(self.tmpdir)
        tmpfilename = os.path.split(tmpfile)[-1]
        assert_equal(tmpfile, self.ds.abspath(tmpfilename))
        assert_equal(tmpfile, self.ds.abspath(tmpfile))

    def test_InvalidHTTP(self):
        if False:
            print('Hello World!')
        (scheme, netloc, upath, pms, qry, frg) = urlparse(invalid_httpurl())
        invalidhttp = os.path.join(self.tmpdir, netloc, upath.strip(os.sep).strip('/'))
        assert_(invalidhttp != self.ds.abspath(valid_httpurl()))

    def test_InvalidFile(self):
        if False:
            i = 10
            return i + 15
        invalidfile = valid_textfile(self.tmpdir)
        tmpfile = valid_textfile(self.tmpdir)
        tmpfilename = os.path.split(tmpfile)[-1]
        assert_(invalidfile != self.ds.abspath(tmpfilename))
        assert_(invalidfile != self.ds.abspath(tmpfile))

    def test_sandboxing(self):
        if False:
            for i in range(10):
                print('nop')
        tmpfile = valid_textfile(self.tmpdir)
        tmpfilename = os.path.split(tmpfile)[-1]
        tmp_path = lambda x: os.path.abspath(self.ds.abspath(x))
        assert_(tmp_path(valid_httpurl()).startswith(self.tmpdir))
        assert_(tmp_path(invalid_httpurl()).startswith(self.tmpdir))
        assert_(tmp_path(tmpfile).startswith(self.tmpdir))
        assert_(tmp_path(tmpfilename).startswith(self.tmpdir))
        for fn in malicious_files:
            assert_(tmp_path(http_path + fn).startswith(self.tmpdir))
            assert_(tmp_path(fn).startswith(self.tmpdir))

    def test_windows_os_sep(self):
        if False:
            for i in range(10):
                print('nop')
        orig_os_sep = os.sep
        try:
            os.sep = '\\'
            self.test_ValidHTTP()
            self.test_ValidFile()
            self.test_InvalidHTTP()
            self.test_InvalidFile()
            self.test_sandboxing()
        finally:
            os.sep = orig_os_sep

class TestRepositoryAbspath:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.tmpdir = os.path.abspath(mkdtemp())
        self.repos = datasource.Repository(valid_baseurl(), self.tmpdir)

    def teardown_method(self):
        if False:
            print('Hello World!')
        rmtree(self.tmpdir)
        del self.repos

    def test_ValidHTTP(self):
        if False:
            for i in range(10):
                print('nop')
        (scheme, netloc, upath, pms, qry, frg) = urlparse(valid_httpurl())
        local_path = os.path.join(self.repos._destpath, netloc, upath.strip(os.sep).strip('/'))
        filepath = self.repos.abspath(valid_httpfile())
        assert_equal(local_path, filepath)

    def test_sandboxing(self):
        if False:
            print('Hello World!')
        tmp_path = lambda x: os.path.abspath(self.repos.abspath(x))
        assert_(tmp_path(valid_httpfile()).startswith(self.tmpdir))
        for fn in malicious_files:
            assert_(tmp_path(http_path + fn).startswith(self.tmpdir))
            assert_(tmp_path(fn).startswith(self.tmpdir))

    def test_windows_os_sep(self):
        if False:
            i = 10
            return i + 15
        orig_os_sep = os.sep
        try:
            os.sep = '\\'
            self.test_ValidHTTP()
            self.test_sandboxing()
        finally:
            os.sep = orig_os_sep

class TestRepositoryExists:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.tmpdir = mkdtemp()
        self.repos = datasource.Repository(valid_baseurl(), self.tmpdir)

    def teardown_method(self):
        if False:
            while True:
                i = 10
        rmtree(self.tmpdir)
        del self.repos

    def test_ValidFile(self):
        if False:
            print('Hello World!')
        tmpfile = valid_textfile(self.tmpdir)
        assert_(self.repos.exists(tmpfile))

    def test_InvalidFile(self):
        if False:
            i = 10
            return i + 15
        tmpfile = invalid_textfile(self.tmpdir)
        assert_equal(self.repos.exists(tmpfile), False)

    def test_RemoveHTTPFile(self):
        if False:
            return 10
        assert_(self.repos.exists(valid_httpurl()))

    def test_CachedHTTPFile(self):
        if False:
            while True:
                i = 10
        localfile = valid_httpurl()
        (scheme, netloc, upath, pms, qry, frg) = urlparse(localfile)
        local_path = os.path.join(self.repos._destpath, netloc)
        os.mkdir(local_path, 448)
        tmpfile = valid_textfile(local_path)
        assert_(self.repos.exists(tmpfile))

class TestOpenFunc:

    def setup_method(self):
        if False:
            return 10
        self.tmpdir = mkdtemp()

    def teardown_method(self):
        if False:
            i = 10
            return i + 15
        rmtree(self.tmpdir)

    def test_DataSourceOpen(self):
        if False:
            return 10
        local_file = valid_textfile(self.tmpdir)
        fp = datasource.open(local_file, destpath=self.tmpdir)
        assert_(fp)
        fp.close()
        fp = datasource.open(local_file)
        assert_(fp)
        fp.close()

def test_del_attr_handling():
    if False:
        i = 10
        return i + 15
    ds = datasource.DataSource()
    del ds._istmpdest
    ds.__del__()