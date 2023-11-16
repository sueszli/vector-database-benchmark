"""
Common helper and cache for pwndbg tempdir
"""
from __future__ import annotations
import os
import tempfile
import pwndbg.lib.cache

@pwndbg.lib.cache.cache_until('forever')
def tempdir():
    if False:
        i = 10
        return i + 15
    '\n    Returns a safe and unpredictable temporary directory with pwndbg prefix.\n    '
    return tempfile.mkdtemp(prefix='pwndbg-')

@pwndbg.lib.cache.cache_until('forever')
def cachedir(namespace=None):
    if False:
        while True:
            i = 10
    '\n    Returns and potentially creates a persistent safe cachedir location\n    based on XDG_CACHE_HOME or ~/.cache\n\n    Optionally creates a sub namespace inside the pwndbg cache folder.\n    '
    cachehome = os.getenv('XDG_CACHE_HOME')
    if not cachehome:
        cachehome = os.path.join(os.getenv('HOME'), '.cache')
    cachedir = os.path.join(cachehome, 'pwndbg')
    if namespace:
        cachedir = os.path.join(cachedir, namespace)
    os.makedirs(cachedir, exist_ok=True)
    return cachedir