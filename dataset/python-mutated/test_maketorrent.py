from tribler.core.utilities.maketorrent import pathlist2filename
from tribler.core.utilities.path_util import Path

def test_pathlist2filename_utf8():
    if False:
        print('Hello World!')
    path_list = ['test', 'path']
    path = pathlist2filename(path_list)
    assert path == Path('test') / 'path'

def test_pathlist2filename_not_utf8():
    if False:
        for i in range(10):
            print('nop')
    part = b'\xb0\xe7'
    path_list = ['test', part]
    path = pathlist2filename(path_list)
    assert path == Path('test') / '°ç'