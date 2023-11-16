"""Tests for conda.py"""
import pytest
from qtpy.QtGui import QIcon
from spyder.utils.icon_manager import ima
from spyder.utils.qthelpers import qapplication

def test_icon_mapping():
    if False:
        return 10
    'Test that all the entries on the icon dict for QtAwesome are valid.'
    qapp = qapplication()
    icons_dict = ima._qtaargs
    for key in icons_dict:
        try:
            assert isinstance(ima.icon(key), QIcon)
        except Exception as e:
            print('Invalid icon name:', key)
            raise e
if __name__ == '__main__':
    pytest.main()