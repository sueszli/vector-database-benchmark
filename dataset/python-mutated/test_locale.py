import locale
import pytest
from PIL import Image
path = 'Tests/images/hopper.jpg'

def test_sanity():
    if False:
        print('Hello World!')
    with Image.open(path):
        pass
    try:
        locale.setlocale(locale.LC_ALL, 'polish')
    except locale.Error:
        pytest.skip('Polish locale not available')
    try:
        with Image.open(path):
            pass
    finally:
        locale.setlocale(locale.LC_ALL, (None, None))