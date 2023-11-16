import pytest
import libqtile.widget
import libqtile.widget.battery
from libqtile.widget.battery import BatteryState, BatteryStatus
from test.widgets.test_battery import dummy_load_battery

@pytest.fixture
def widget(monkeypatch):
    if False:
        print('Hello World!')
    loaded_bat = BatteryStatus(state=BatteryState.DISCHARGING, percent=0.5, power=15.0, time=1729)
    monkeypatch.setattr('libqtile.widget.battery.load_battery', dummy_load_battery(loaded_bat))
    yield libqtile.widget.battery.BatteryIcon

@pytest.mark.parametrize('screenshot_manager', [{}], indirect=True)
def ss_batteryicon(screenshot_manager):
    if False:
        print('Hello World!')
    screenshot_manager.take_screenshot()