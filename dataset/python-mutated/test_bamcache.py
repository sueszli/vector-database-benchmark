from panda3d import core

def test_bamcache_flush_index():
    if False:
        for i in range(10):
            print('nop')
    cache = core.BamCache()
    cache.flush_index()