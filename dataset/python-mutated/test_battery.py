import pytest
from libqtile import images
from libqtile.widget import battery
from libqtile.widget.battery import Battery, BatteryIcon, BatteryState, BatteryStatus
from test.widgets.conftest import TEST_DIR, FakeBar

class DummyBattery:

    def __init__(self, status):
        if False:
            return 10
        self._status = status

    def update_status(self):
        if False:
            for i in range(10):
                print('nop')
        return self._status

class DummyErrorBattery:

    def __init__(self, **config):
        if False:
            print('Hello World!')
        pass

    def update_status(self):
        if False:
            print('Hello World!')
        raise RuntimeError('err')

def dummy_load_battery(bat):
    if False:
        while True:
            i = 10

    def load_battery(**config):
        if False:
            i = 10
            return i + 15
        return DummyBattery(bat)
    return load_battery

def test_text_battery_charging(monkeypatch):
    if False:
        i = 10
        return i + 15
    loaded_bat = BatteryStatus(state=BatteryState.CHARGING, percent=0.5, power=15.0, time=1729)
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', dummy_load_battery(loaded_bat))
        batt = Battery()
    text = batt.poll()
    assert text == '^ 50% 0:28 15.00 W'

def test_text_battery_discharging(monkeypatch):
    if False:
        print('Hello World!')
    loaded_bat = BatteryStatus(state=BatteryState.DISCHARGING, percent=0.5, power=15.0, time=1729)
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', dummy_load_battery(loaded_bat))
        batt = Battery()
    text = batt.poll()
    assert text == 'V 50% 0:28 15.00 W'

def test_text_battery_full(monkeypatch):
    if False:
        print('Hello World!')
    loaded_bat = BatteryStatus(state=BatteryState.FULL, percent=0.5, power=15.0, time=1729)
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', dummy_load_battery(loaded_bat))
        batt = Battery()
    text = batt.poll()
    assert text == 'Full'
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', dummy_load_battery(loaded_bat))
        batt = Battery(show_short_text=False)
    text = batt.poll()
    assert text == '= 50% 0:28 15.00 W'

def test_text_battery_empty(monkeypatch):
    if False:
        while True:
            i = 10
    loaded_bat = BatteryStatus(state=BatteryState.EMPTY, percent=0.5, power=15.0, time=1729)
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', dummy_load_battery(loaded_bat))
        batt = Battery()
    text = batt.poll()
    assert text == 'Empty'
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', dummy_load_battery(loaded_bat))
        batt = Battery(show_short_text=False)
    text = batt.poll()
    assert text == 'x 50% 0:28 15.00 W'
    loaded_bat = BatteryStatus(state=BatteryState.UNKNOWN, percent=0.0, power=15.0, time=1729)
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', dummy_load_battery(loaded_bat))
        batt = Battery()
    text = batt.poll()
    assert text == 'Empty'

def test_text_battery_not_charging(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    loaded_bat = BatteryStatus(state=BatteryState.NOT_CHARGING, percent=0.5, power=15.0, time=1729)
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', dummy_load_battery(loaded_bat))
        batt = Battery()
    text = batt.poll()
    assert text == '* 50% 0:28 15.00 W'

def test_text_battery_unknown(monkeypatch):
    if False:
        print('Hello World!')
    loaded_bat = BatteryStatus(state=BatteryState.UNKNOWN, percent=0.5, power=15.0, time=1729)
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', dummy_load_battery(loaded_bat))
        batt = Battery()
    text = batt.poll()
    assert text == '? 50% 0:28 15.00 W'

def test_text_battery_hidden(monkeypatch):
    if False:
        i = 10
        return i + 15
    loaded_bat = BatteryStatus(state=BatteryState.DISCHARGING, percent=0.5, power=15.0, time=1729)
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', dummy_load_battery(loaded_bat))
        batt = Battery(hide_threshold=0.6)
    text = batt.poll()
    assert text != ''
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', dummy_load_battery(loaded_bat))
        batt = Battery(hide_threshold=0.4)
    text = batt.poll()
    assert text == ''

def test_text_battery_error(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', DummyErrorBattery)
        batt = Battery()
    text = batt.poll()
    assert text == 'Error: err'

def test_images_fail():
    if False:
        for i in range(10):
            print('nop')
    "Test BatteryIcon() with a bad theme_path\n\n    This theme path doesn't contain all of the required images.\n    "
    batt = BatteryIcon(theme_path=TEST_DIR)
    with pytest.raises(images.LoadingError):
        batt.setup_images()

def test_images_good(tmpdir, fake_bar, svg_img_as_pypath):
    if False:
        for i in range(10):
            print('nop')
    'Test BatteryIcon() with a good theme_path\n\n    This theme path does contain all of the required images.\n    '
    for name in BatteryIcon.icon_names:
        target = tmpdir.join(name + '.svg')
        svg_img_as_pypath.copy(target)
    batt = BatteryIcon(theme_path=str(tmpdir))
    batt.fontsize = 12
    batt.bar = fake_bar
    batt.setup_images()
    assert len(batt.images) == len(BatteryIcon.icon_names)
    for (name, img) in batt.images.items():
        assert isinstance(img, images.Img)

def test_images_default(fake_bar):
    if False:
        print('Hello World!')
    'Test BatteryIcon() with the default theme_path\n\n    Ensure that the default images are successfully loaded.\n    '
    batt = BatteryIcon()
    batt.fontsize = 12
    batt.bar = fake_bar
    batt.setup_images()
    assert len(batt.images) == len(BatteryIcon.icon_names)
    for (name, img) in batt.images.items():
        assert isinstance(img, images.Img)

def test_battery_background(fake_qtile, fake_window, monkeypatch):
    if False:
        i = 10
        return i + 15
    ok = BatteryStatus(state=BatteryState.DISCHARGING, percent=0.5, power=15.0, time=1729)
    low = BatteryStatus(state=BatteryState.DISCHARGING, percent=0.1, power=15.0, time=1729)
    low_background = 'ff0000'
    background = '000000'
    with monkeypatch.context() as manager:
        manager.setattr(battery, 'load_battery', dummy_load_battery(ok))
        batt = Battery(low_percentage=0.2, low_background=low_background, background=background)
    fakebar = FakeBar([batt], window=fake_window)
    batt._configure(fake_qtile, fakebar)
    assert batt.background == background
    batt._battery._status = low
    batt.poll()
    assert batt.background == low_background
    batt._battery._status = ok
    batt.poll()
    assert batt.background == background