import os

def test_unprotect(tmp_dir, dvc):
    if False:
        while True:
            i = 10
    tmp_dir.gen('foo', 'foo')
    dvc.cache.local.cache_types = ['hardlink']
    dvc.add('foo')
    cache = os.path.join('.dvc', 'cache', 'files', 'md5', 'ac', 'bd18db4cc2f85cedef654fccc4a4d8')
    assert not os.access('foo', os.W_OK)
    assert not os.access(cache, os.W_OK)
    dvc.unprotect('foo')
    assert os.access('foo', os.W_OK)
    if os.name == 'nt':
        assert os.access(cache, os.W_OK)
        dvc.status()
    assert not os.access(cache, os.W_OK)