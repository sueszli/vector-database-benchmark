import pytest
import libqtile.widget.open_weather
from test.widgets.test_openweather import mock_fetch

@pytest.fixture
def widget(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setattr('libqtile.widget.generic_poll_text.GenPollUrl.fetch', mock_fetch)
    yield libqtile.widget.open_weather.OpenWeather

@pytest.mark.parametrize('screenshot_manager', [{'location': 'London'}, {'location': 'London', 'format': '{location_city}: {sunrise} {sunset}'}, {'location': 'London', 'format': '{location_city}: {wind_speed} {wind_deg} {wind_direction}'}, {'location': 'London', 'format': '{location_city}: {icon}'}], indirect=True)
def ss_openweather(screenshot_manager):
    if False:
        print('Hello World!')
    screenshot_manager.take_screenshot()