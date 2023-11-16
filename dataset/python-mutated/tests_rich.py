"""Test `tqdm.rich`."""
from .tests_tqdm import importorskip

def test_rich_import():
    if False:
        print('Hello World!')
    'Test `tqdm.rich` import'
    importorskip('tqdm.rich')