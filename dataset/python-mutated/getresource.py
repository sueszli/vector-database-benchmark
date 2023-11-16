"""
Functionality to ensure up-to-date versions of resources. This module
must be able to run standalone, since it will be used from setup.py to
pack resources in dist packages.
"""
from __future__ import absolute_import, print_function, division
import os
try:
    from .logging import logger
    (warning, info) = (logger.warning, logger.info)
except Exception:
    warning = info = print
try:
    from urllib.request import urlopen
except ImportError:
    try:
        from urllib2 import urlopen
    except ImportError:
        raise RuntimeError('Could not import urlopen.')
phosphor_url = 'https://raw.githubusercontent.com/flexxui/phosphor-all/{}/dist/'
RESOURCES = {'phosphor-all.js': (phosphor_url + 'phosphor-all.js', '94d59b003849f'), 'phosphor-all.css': (phosphor_url + 'phosphor-all.css', '94d59b003849f')}

def get_resoure_path(filename):
    if False:
        return 10
    ' Get the full path to a resource, corresponding to the given\n    filename. Will use cached version if available. Otherwise will\n    download and cache.\n    '
    dest = os.path.abspath(os.path.join(__file__, '..', '..', 'resources'))
    if not os.path.isdir(dest):
        raise ValueError('Resource dest dir %r is not a directory.' % dest)
    path = os.path.join(dest, filename)
    url = ''
    if filename in RESOURCES:
        (url, tag) = RESOURCES[filename]
        if tag:
            url = url.replace('{}', tag)
            (basename, ext) = path.rsplit('.', 1)
            path = basename + '.' + tag + '.' + ext
        if not os.path.isfile(path):
            data = _fetch_file(url)
            with open(path, 'wb') as f:
                f.write(data)
    elif not os.path.isfile(path):
        raise ValueError('Unknown/unavailable resource %r' % filename)
    return path

def get_resource(filename):
    if False:
        print('Hello World!')
    ' Get the bytes of the resource corresponding to the given filename.\n    '
    return open(get_resoure_path(filename), 'rb').read()

def _fetch_file(url):
    if False:
        while True:
            i = 10
    ' Fetches a file from the internet. Retry a few times before\n    giving up on failure.\n    '
    info('Downloading %s' % url)
    for tries in range(4):
        try:
            return urlopen(url, timeout=5.0).read()
        except Exception as e:
            warning('Error while fetching file: %s' % str(e))
    raise IOError('Unable to download %r. Perhaps there is a no internet connection? If there is, please report this problem.' % url)
if __name__ == '__main__':
    for key in RESOURCES:
        get_resource(key)