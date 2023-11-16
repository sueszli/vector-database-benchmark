import subprocess
import pytest
import libqtile.widget.cmus
from test.widgets.test_cmus import MockCmusRemoteProcess

@pytest.fixture
def widget(monkeypatch):
    if False:
        print('Hello World!')
    MockCmusRemoteProcess.reset()
    monkeypatch.setattr('libqtile.widget.cmus.subprocess', MockCmusRemoteProcess)
    monkeypatch.setattr('libqtile.widget.cmus.subprocess.CalledProcessError', subprocess.CalledProcessError)
    monkeypatch.setattr('libqtile.widget.cmus.base.ThreadPoolText.call_process', MockCmusRemoteProcess.call_process)
    yield libqtile.widget.cmus.Cmus

@pytest.mark.parametrize('screenshot_manager', [{}], indirect=True)
def ss_cmus(screenshot_manager):
    if False:
        print('Hello World!')
    screenshot_manager.take_screenshot()