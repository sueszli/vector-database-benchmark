import sys
from importlib import reload
import pytest
from test.widgets.test_cpu import MockPsutil

@pytest.fixture
def widget(monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setitem(sys.modules, 'psutil', MockPsutil('psutil'))
    from libqtile.widget import cpu
    reload(cpu)
    yield cpu.CPU

def ss_cpu(screenshot_manager):
    if False:
        i = 10
        return i + 15
    screenshot_manager.take_screenshot()