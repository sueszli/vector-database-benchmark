import pytest
from libqtile.widget import moc
from test.widgets.test_moc import MockMocpProcess

@pytest.fixture
def widget(fake_qtile, monkeypatch, fake_window):
    if False:
        for i in range(10):
            print('nop')
    MockMocpProcess.reset()
    monkeypatch.setattr(moc.Moc, 'call_process', MockMocpProcess.run)
    monkeypatch.setattr('libqtile.widget.moc.subprocess.Popen', MockMocpProcess.run)
    yield moc.Moc

def ss_moc(screenshot_manager):
    if False:
        i = 10
        return i + 15
    screenshot_manager.take_screenshot()