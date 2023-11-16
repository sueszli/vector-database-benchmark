import pytest
pytest.importorskip('Pmw')
from direct.tkwidgets import AppShell

def test_TestAppShell(tk_toplevel):
    if False:
        i = 10
        return i + 15
    test = AppShell.TestAppShell(balloon_state='none')