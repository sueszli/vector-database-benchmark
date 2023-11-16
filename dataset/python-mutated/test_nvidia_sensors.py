import pytest
from libqtile.widget.nvidia_sensors import NvidiaSensors, _all_sensors_names_correct
from test.widgets.conftest import FakeBar

def test_nvidia_sensors_input_regex():
    if False:
        return 10
    correct_sensors = NvidiaSensors(format='temp:{temp}°C,fan{fan_speed}asd,performance{perf}fds')._parse_format_string()
    incorrect_sensors = {'tem', 'fan_speed', 'perf'}
    assert correct_sensors == {'temp', 'fan_speed', 'perf'}
    assert _all_sensors_names_correct(correct_sensors)
    assert not _all_sensors_names_correct(incorrect_sensors)

class MockNvidiaSMI:
    temperature = '20'

    @classmethod
    def get_temperature(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        return cls.temperature

@pytest.fixture
def fake_nvidia(fake_qtile, monkeypatch, fake_window):
    if False:
        while True:
            i = 10
    n = NvidiaSensors()
    monkeypatch.setattr(n, 'call_process', MockNvidiaSMI.get_temperature)
    monkeypatch.setattr('libqtile.widget.moc.subprocess.Popen', MockNvidiaSMI.get_temperature)
    fakebar = FakeBar([n], window=fake_window)
    n._configure(fake_qtile, fakebar)
    return n

def test_nvidia_sensors_foreground_colour(fake_nvidia):
    if False:
        while True:
            i = 10
    fake_nvidia.poll()
    assert fake_nvidia.layout.colour == fake_nvidia.foreground_normal
    MockNvidiaSMI.temperature = '90'
    fake_nvidia.poll()
    assert fake_nvidia.layout.colour == fake_nvidia.foreground_alert
    MockNvidiaSMI.temperature = '20'
    fake_nvidia.poll()
    assert fake_nvidia.layout.colour == fake_nvidia.foreground_normal