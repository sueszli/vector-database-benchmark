"""Test Spack's URL handling utility functions."""
import os
import os.path
import urllib.parse
import spack.util.path
import spack.util.url as url_util

def test_url_local_file_path(tmpdir):
    if False:
        print('Hello World!')
    path = str(tmpdir.join('hello.txt'))
    with open(path, 'wb') as f:
        f.write(b'hello world')
    roundtrip = url_util.local_file_path(url_util.path_to_file_url(path))
    assert os.path.samefile(roundtrip, path)
    parsed = urllib.parse.urlparse(url_util.path_to_file_url(path))
    assert os.path.samefile(url_util.local_file_path(parsed), path)

def test_url_local_file_path_no_file_scheme():
    if False:
        return 10
    assert url_util.local_file_path('https://example.com/hello.txt') is None
    assert url_util.local_file_path('C:\\Program Files\\hello.txt') is None

def test_relative_path_to_file_url(tmpdir):
    if False:
        while True:
            i = 10
    path = str(tmpdir.join('hello.txt'))
    with open(path, 'wb') as f:
        f.write(b'hello world')
    with tmpdir.as_cwd():
        roundtrip = url_util.local_file_path(url_util.path_to_file_url('hello.txt'))
        assert os.path.samefile(roundtrip, path)

def test_url_join_local_paths():
    if False:
        print('Hello World!')
    assert url_util.join('s3://bucket/index.html', '../other-bucket/document.txt') == 's3://bucket/other-bucket/document.txt'
    assert url_util.join('s3://bucket/index.html', '../other-bucket/document.txt', resolve_href=True) == 's3://other-bucket/document.txt'
    assert url_util.join('s3://bucket/index.html', '..', 'other-bucket', 'document.txt', resolve_href=True) == 's3://other-bucket/document.txt'
    assert url_util.join('https://mirror.spack.io/build_cache', 'my-package', resolve_href=True) == 'https://mirror.spack.io/my-package'
    assert url_util.join('https://mirror.spack.io/build_cache', 'my-package', resolve_href=False) == 'https://mirror.spack.io/build_cache/my-package'
    assert url_util.join('https://mirror.spack.io/build_cache', 'my-package') == 'https://mirror.spack.io/build_cache/my-package'
    assert url_util.join('https://mirror.spack.io', 'build_cache', 'my-package') == 'https://mirror.spack.io/build_cache/my-package'
    args = ['s3://bucket/a/b', 'new-bucket', 'c']
    assert url_util.join(*args) == 's3://bucket/a/b/new-bucket/c'
    args.insert(1, '..')
    assert url_util.join(*args) == 's3://bucket/a/new-bucket/c'
    args.insert(1, '..')
    assert url_util.join(*args) == 's3://bucket/new-bucket/c'
    args.insert(1, '..')
    assert url_util.join(*args) == 's3://new-bucket/c'

def test_url_join_absolute_paths():
    if False:
        return 10
    p = '/path/to/resource'
    assert url_util.join('http://example.com/a/b/c', p) == 'http://example.com/path/to/resource'
    assert url_util.join('s3://example.com/a/b/c', p) == 's3://path/to/resource'
    p = 'http://example.com/path/to'
    join_result = url_util.join(p, 'resource')
    assert join_result == 'http://example.com/path/to/resource'
    assert url_util.join('literally', 'does', 'not', 'matter', p, 'resource') == join_result
    assert url_util.join('file:///a/b/c', './d') == 'file:///a/b/c/d'
    args = ['s3://does', 'not', 'matter', 'http://example.com', 'also', 'does', 'not', 'matter', '/path']
    expected = 'http://example.com/path'
    assert url_util.join(*args, resolve_href=True) == expected
    assert url_util.join(*args, resolve_href=False) == expected
    args[-1] = '/path/to/page'
    args.extend(('..', '..', 'resource'))
    assert url_util.join(*args, resolve_href=True) == 'http://example.com/resource'
    assert url_util.join(*args, resolve_href=False) == 'http://example.com/path/resource'

def test_default_download_name():
    if False:
        print('Hello World!')
    url = 'https://example.com:1234/path/to/file.txt;params?abc=def#file=blob.tar'
    filename = url_util.default_download_filename(url)
    assert filename == spack.util.path.sanitize_filename(filename)

def test_default_download_name_dot_dot():
    if False:
        print('Hello World!')
    'Avoid that downloaded files get names computed as ., .. or any hidden file.'
    assert url_util.default_download_filename('https://example.com/.') == '_'
    assert url_util.default_download_filename('https://example.com/..') == '_.'
    assert url_util.default_download_filename('https://example.com/.abcdef') == '_abcdef'