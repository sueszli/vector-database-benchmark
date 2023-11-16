"""Test `tqdm.tk`."""
from .tests_tqdm import importorskip

def test_tk_import():
    if False:
        for i in range(10):
            print('nop')
    'Test `tqdm.tk` import'
    importorskip('tqdm.tk')