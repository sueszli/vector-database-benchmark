import pytest
from libqtile.widget import idlerpg
from test.widgets.test_idlerpg import online_response

@pytest.fixture
def widget(monkeypatch):
    if False:
        while True:
            i = 10

    def no_op(*args, **kwargs):
        if False:
            print('Hello World!')
        return ''
    idler = idlerpg.IdleRPG
    idler.RESPONSE = online_response
    monkeypatch.setattr(idler, 'fetch', no_op)
    yield idler

@pytest.mark.parametrize('screenshot_manager', [{'url': 'http://idlerpg.qtile.org?player=elParaguayo'}], indirect=True)
def ss_idlerpg(screenshot_manager):
    if False:
        i = 10
        return i + 15
    screenshot_manager.c.widget['idlerpg'].eval('self.update(self.parse(self.RESPONSE))')
    screenshot_manager.take_screenshot()