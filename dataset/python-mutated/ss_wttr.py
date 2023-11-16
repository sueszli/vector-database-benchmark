import pytest
from libqtile.widget import wttr
RESPONSE = 'London: +17°C'

@pytest.fixture
def widget(monkeypatch):
    if False:
        for i in range(10):
            print('nop')

    def result(self):
        if False:
            return 10
        return RESPONSE
    monkeypatch.setattr('libqtile.widget.wttr.Wttr.fetch', result)
    yield wttr.Wttr

@pytest.mark.parametrize('screenshot_manager', [{'location': {'London': 'Home'}}], indirect=True)
def ss_wttr(screenshot_manager):
    if False:
        while True:
            i = 10
    screenshot_manager.take_screenshot()