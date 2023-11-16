__license__ = 'GPL 3'
__copyright__ = '2009, Kovid Goyal <kovid@kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
from calibre import guess_type

def _mt(path):
    if False:
        while True:
            i = 10
    mt = guess_type(path)[0]
    if not mt:
        mt = 'application/octet-stream'
    return mt

def mime_type_ext(ext):
    if False:
        return 10
    if not ext.startswith('.'):
        ext = '.' + ext
    return _mt('a' + ext)

def mime_type_path(path):
    if False:
        for i in range(10):
            print('nop')
    return _mt(path)