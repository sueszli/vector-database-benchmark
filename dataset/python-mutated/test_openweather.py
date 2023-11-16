import time
import pytest
import libqtile.config
import libqtile.widget.open_weather
from libqtile.bar import Bar

def mock_fetch(*args, **kwargs):
    if False:
        return 10
    return {'coord': {'lon': -0.13, 'lat': 51.51}, 'weather': [{'id': 300, 'main': 'Drizzle', 'description': 'light intensity drizzle', 'icon': '09d'}], 'base': 'stations', 'main': {'temp': 280.15 - 273.15, 'pressure': 1012, 'humidity': 81, 'temp_min': 279.15 - 273.15, 'temp_max': 281.15 - 273.15}, 'visibility': 10000, 'wind': {'speed': 4.1, 'deg': 80}, 'clouds': {'all': 90}, 'dt': 1485789600, 'sys': {'type': 1, 'id': 5091, 'message': 0.0103, 'country': 'GB', 'sunrise': 1485762037, 'sunset': 1485794875}, 'id': 2643743, 'name': 'London', 'cod': 200}

@pytest.fixture
def patch_openweather(request, monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setattr('libqtile.widget.generic_poll_text.GenPollUrl.fetch', mock_fetch)
    monkeypatch.setattr('libqtile.widget.open_weather.time.localtime', time.gmtime)
    yield libqtile.widget.open_weather

@pytest.mark.parametrize('params,expected', [({'location': 'London'}, 'London: 7.0 °C 81% light intensity drizzle'), ({'location': 'London', 'format': '{location_city}: {sunrise} {sunset}'}, 'London: 07:40 16:47'), ({'location': 'London', 'format': '{location_city}: {wind_speed} {wind_deg} {wind_direction}'}, 'London: 4.1 80 E'), ({'location': 'London', 'format': '{location_city}: {icon}'}, 'London: 🌧️')])
def test_openweather_parse(patch_openweather, minimal_conf_noscreen, manager_nospawn, params, expected):
    if False:
        while True:
            i = 10
    'Check widget parses output correctly for display.'
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=Bar([patch_openweather.OpenWeather(**params)], 10))]
    manager_nospawn.start(config)
    manager_nospawn.c.widget['openweather'].eval('self.update(self.poll())')
    info = manager_nospawn.c.widget['openweather'].info()['text']
    assert info == expected

@pytest.mark.parametrize('params,vals', [({'location': 'London'}, ['q=London']), ({'cityid': 2643743}, ['id=2643743']), ({'zip': 90210}, ['zip=90210']), ({'coordinates': {'longitude': '77.22', 'latitude': '28.67'}}, ['lat=28.67', 'lon=77.22'])])
def test_url(patch_openweather, params, vals):
    if False:
        print('Hello World!')
    'Test that url is created correctly.'
    widget = patch_openweather.OpenWeather(**params)
    url = widget.url
    for val in vals:
        assert val in url