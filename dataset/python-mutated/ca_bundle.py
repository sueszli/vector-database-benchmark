"""Auto-detect of CA bundle for SSL connections"""
from __future__ import absolute_import
import os
import sys
from bzrlib.trace import mutter
_ca_path = None

def get_ca_path(use_cache=True):
    if False:
        while True:
            i = 10
    'Return location of CA bundle'
    global _ca_path
    if _ca_path is not None and use_cache:
        return _ca_path
    path = os.environ.get('CURL_CA_BUNDLE')
    if not path and sys.platform == 'win32':
        dirs = [os.path.realpath(os.path.dirname(sys.argv[0]))]
        paths = os.environ.get('PATH')
        if paths:
            paths = [i for i in paths.split(os.pathsep) if i not in ('', '.')]
            dirs.extend(paths)
        for d in dirs:
            fname = os.path.join(d, 'curl-ca-bundle.crt')
            if os.path.isfile(fname):
                path = fname
                break
    if path:
        mutter('using CA bundle: %r', path)
    else:
        path = ''
    if use_cache:
        _ca_path = path
    return path