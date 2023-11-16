from ..platform import swidth
from .platform import skipif_not_posix
pytestmark = skipif_not_posix

def test_posix_swidth_ascii():
    if False:
        i = 10
        return i + 15
    assert swidth('borg') == 4

def test_posix_swidth_cjk():
    if False:
        print('Hello World!')
    assert swidth('バックアップ') == 6 * 2

def test_posix_swidth_mixed():
    if False:
        while True:
            i = 10
    assert swidth('borgバックアップ') == 4 + 6 * 2