if a:
    b = c
else:
    b = d
b = c if a else d
if a:
    b = c
elif c:
    b = a
else:
    b = d
if True:
    pass
elif a:
    b = 1
else:
    b = 2
if True:
    pass
elif a:
    b = 1
else:
    b = 2
import sys
if sys.version_info >= (3, 9):
    randbytes = random.randbytes
else:
    randbytes = _get_random_bytes
if sys.platform == 'darwin':
    randbytes = random.randbytes
else:
    randbytes = _get_random_bytes
if sys.platform.startswith('linux'):
    randbytes = random.randbytes
else:
    randbytes = _get_random_bytes
if x > 0:
    abc = x
else:
    abc = -x
if parser.errno == BAD_FIRST_LINE:
    req = wrappers.Request(sock, server=self._server)
else:
    req = wrappers.Request(sock, parser.get_method(), parser.get_scheme() or _scheme, parser.get_path(), parser.get_version(), parser.get_query_string(), server=self._server)
if a:
    b = 'cccccccccccccccccccccccccccccccccß'
else:
    b = 'ddddddddddddddddddddddddddddddddd💣'
if True:
    if a:
        b = ccccccccccccccccccccccccccccccccccc
    else:
        b = ddddddddddddddddddddddddddddddddddd
if True:
    if a:
        b = ccccccccccccccccccccccccccccccccccc
    else:
        b = ddddddddddddddddddddddddddddddddddd
if True:
    exitcode = 0
else:
    exitcode = 1
if True:
    x = 3
else:
    x = 5
if True:
    x = 3
else:
    x = 5

def f():
    if False:
        return 10
    if True:
        x = (yield 3)
    else:
        x = (yield 5)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    x = 3
else:
    x = 5