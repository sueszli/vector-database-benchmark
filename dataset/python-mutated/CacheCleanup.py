""" Cleanup of caches for Nuitka.

This is triggered by "--clean-cache=" usage, and can cleanup all kinds of
caches and is supposed to run before or instead of Nuitka compilation.
"""
import os
from nuitka.BytecodeCaching import getBytecodeCacheDir
from nuitka.Tracing import cache_logger
from nuitka.utils.AppDirs import getCacheDir
from nuitka.utils.FileOperations import removeDirectory

def _cleanCacheDirectory(cache_name, cache_dir):
    if False:
        while True:
            i = 10
    from nuitka.Options import shallCleanCache
    if shallCleanCache(cache_name) and os.path.exists(cache_dir):
        cache_logger.info("Cleaning cache '%s' directory '%s'." % (cache_name, cache_dir))
        removeDirectory(cache_dir, ignore_errors=False)
        cache_logger.info('Done.')

def cleanCaches():
    if False:
        i = 10
        return i + 15
    _cleanCacheDirectory('ccache', os.path.join(getCacheDir(), 'ccache'))
    _cleanCacheDirectory('clcache', os.path.join(getCacheDir(), 'clcache'))
    _cleanCacheDirectory('bytecode', getBytecodeCacheDir())
    _cleanCacheDirectory('dll-dependencies', os.path.join(getCacheDir(), 'library_dependencies'))