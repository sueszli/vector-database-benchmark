import random
import sys
from importlib import reload
from types import ModuleType
import pytest
values = []
for _ in range(100):
    odds = random.randint(0, 10)
    val = 0 if odds < 6 else random.randint(100, 20000)
    values.append(val)

class MockPsutil(ModuleType):
    pass

@pytest.fixture
def widget(monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setitem(sys.modules, 'psutil', MockPsutil('psutil'))
    from libqtile.widget import graph
    reload(graph)
    yield graph.HDDBusyGraph

@pytest.mark.parametrize('screenshot_manager', [{}, {'type': 'box'}, {'type': 'line'}, {'type': 'line', 'line_width': 1}, {'start_pos': 'top'}], indirect=True)
def ss_hddbusygraph(screenshot_manager):
    if False:
        i = 10
        return i + 15
    widget = screenshot_manager.c.widget['hddbusygraph']
    widget.eval(f'self.values={values}')
    widget.eval(f'self.maxvalue={max(values)}')
    widget.eval('self.draw()')
    screenshot_manager.take_screenshot()