import libqtile.config
from libqtile.widget.check_updates import CheckUpdates, Popen
from test.widgets.conftest import FakeBar
wrong_distro = 'Barch'
good_distro = 'Arch'
cmd_0_line = 'export toto'
cmd_1_line = 'echo toto'
cmd_error = 'false'
nus = 'No Update Available'

class MockPopen:

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.call_count = 0

    def poll(self):
        if False:
            i = 10
            return i + 15
        if self.call_count == 0:
            self.call_count += 1
            return None
        return 0

class MockSpawn:
    call_count = 0

    @classmethod
    def call_process(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if cls.call_count == 0:
            cls.call_count += 1
            return 'Updates'
        return ''

def test_unknown_distro():
    if False:
        return 10
    'test an unknown distribution'
    cu = CheckUpdates(distro=wrong_distro)
    text = cu.poll()
    assert text == 'N/A'

def test_update_available(fake_qtile, fake_window):
    if False:
        return 10
    'test output with update (check number of updates and color)'
    cu2 = CheckUpdates(distro=good_distro, custom_command=cmd_1_line, colour_have_updates='#123456')
    fakebar = FakeBar([cu2], window=fake_window)
    cu2._configure(fake_qtile, fakebar)
    text = cu2.poll()
    assert text == 'Updates: 1'
    assert cu2.layout.colour == cu2.colour_have_updates

def test_no_update_available_without_no_update_string(fake_qtile, fake_window):
    if False:
        for i in range(10):
            print('nop')
    'test output with no update (without dedicated string nor color)'
    cu3 = CheckUpdates(distro=good_distro, custom_command=cmd_0_line)
    fakebar = FakeBar([cu3], window=fake_window)
    cu3._configure(fake_qtile, fakebar)
    text = cu3.poll()
    assert text == ''

def test_no_update_available_with_no_update_string_and_color_no_updates(fake_qtile, fake_window):
    if False:
        print('Hello World!')
    'test output with no update (with dedicated string and color)'
    cu4 = CheckUpdates(distro=good_distro, custom_command=cmd_0_line, no_update_string=nus, colour_no_updates='#654321')
    fakebar = FakeBar([cu4], window=fake_window)
    cu4._configure(fake_qtile, fakebar)
    text = cu4.poll()
    assert text == nus
    assert cu4.layout.colour == cu4.colour_no_updates

def test_update_available_with_restart_indicator(monkeypatch, fake_qtile, fake_window):
    if False:
        return 10
    'test output with no indicator where restart needed'
    cu5 = CheckUpdates(distro=good_distro, custom_command=cmd_1_line, restart_indicator='*')
    monkeypatch.setattr('os.path.exists', lambda x: True)
    fakebar = FakeBar([cu5], window=fake_window)
    cu5._configure(fake_qtile, fakebar)
    text = cu5.poll()
    assert text == 'Updates: 1*'

def test_update_available_with_execute(manager_nospawn, minimal_conf_noscreen, monkeypatch):
    if False:
        return 10
    'test polling after executing command'

    class MockPopen:

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            self.call_count = 0

        def poll(self):
            if False:
                return 10
            if self.call_count == 0:
                self.call_count += 1
                return None
            return 0

    class MockSpawn:
        call_count = 0

        @classmethod
        def call_process(cls, *args, **kwargs):
            if False:
                while True:
                    i = 10
            if cls.call_count == 0:
                cls.call_count += 1
                return 'Updates'
            return ''
    cu6 = CheckUpdates(distro=good_distro, custom_command='dummy', execute='dummy', no_update_string=nus)
    monkeypatch.setattr(cu6, 'call_process', MockSpawn.call_process)
    monkeypatch.setattr('libqtile.widget.check_updates.Popen', MockPopen)
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([cu6], 10))]
    manager_nospawn.start(config)
    topbar = manager_nospawn.c.bar['top']
    assert topbar.info()['widgets'][0]['text'] == 'Updates: 1'
    topbar.fake_button_press(0, 'top', 0, 0, button=1)
    (_, result) = manager_nospawn.c.widget['checkupdates'].eval('self.poll()')
    assert result == nus

def test_update_process_error(fake_qtile, fake_window):
    if False:
        for i in range(10):
            print('nop')
    'test output where update check gives error'
    cu7 = CheckUpdates(distro=good_distro, custom_command=cmd_error, no_update_string='ERROR')
    fakebar = FakeBar([cu7], window=fake_window)
    cu7._configure(fake_qtile, fakebar)
    text = cu7.poll()
    assert text == 'ERROR'

def test_line_truncations(fake_qtile, monkeypatch, fake_window):
    if False:
        return 10
    'test update count is reduced'

    def mock_process(*args, **kwargs):
        if False:
            print('Hello World!')
        return '1\n2\n3\n4\n5\n'
    cu8 = CheckUpdates(distro='Fedora')
    monkeypatch.setattr(cu8, 'call_process', mock_process)
    fakebar = FakeBar([cu8], window=fake_window)
    cu8._configure(fake_qtile, fakebar)
    text = cu8.poll()
    assert text == 'Updates: 4'