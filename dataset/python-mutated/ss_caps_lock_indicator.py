import subprocess
import pytest
from libqtile.widget import CapsNumLockIndicator
from test.widgets.test_caps_num_lock_indicator import MockCapsNumLockIndicator

@pytest.fixture
def widget(monkeypatch):
    if False:
        return 10
    MockCapsNumLockIndicator.reset()
    monkeypatch.setattr('libqtile.widget.caps_num_lock_indicator.subprocess', MockCapsNumLockIndicator)
    monkeypatch.setattr('libqtile.widget.caps_num_lock_indicator.subprocess.CalledProcessError', subprocess.CalledProcessError)
    monkeypatch.setattr('libqtile.widget.caps_num_lock_indicator.base.ThreadPoolText.call_process', MockCapsNumLockIndicator.call_process)
    return CapsNumLockIndicator

@pytest.mark.parametrize('screenshot_manager', [{}], indirect=True)
def ss_caps_num_lock_indicator(screenshot_manager):
    if False:
        return 10
    screenshot_manager.take_screenshot()