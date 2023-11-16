import subprocess
import pytest
import libqtile.config
from libqtile.widget import moc
from test.widgets.conftest import FakeBar

class MockMocpProcess:
    info = {}
    is_error = False
    index = 0

    @classmethod
    def reset(cls):
        if False:
            i = 10
            return i + 15
        cls.info = [{'State': 'PLAY', 'File': '/playing/file/rickroll.mp3', 'SongTitle': 'Never Gonna Give You Up', 'Artist': 'Rick Astley', 'Album': 'Whenever You Need Somebody'}, {'State': 'PLAY', 'File': '/playing/file/sweetcaroline.mp3', 'SongTitle': 'Sweet Caroline', 'Artist': 'Neil Diamond', 'Album': 'Greatest Hits'}, {'State': 'STOP', 'File': '/playing/file/itsnotunusual.mp3', 'SongTitle': "It's Not Unusual", 'Artist': 'Tom Jones', 'Album': 'Along Came Jones'}]
        cls.index = 0

    @classmethod
    def run(cls, cmd):
        if False:
            return 10
        if cls.is_error:
            raise subprocess.CalledProcessError(-1, cmd=cmd, output="Couldn't connect to moc.")
        arg = cmd[1]
        if arg == '-i':
            output = '\n'.join(('{k}: {v}'.format(k=k, v=v) for (k, v) in cls.info[cls.index].items()))
            return output
        elif arg == '-p':
            cls.info[cls.index]['State'] = 'PLAY'
        elif arg == '-G':
            if cls.info[cls.index]['State'] == 'PLAY':
                cls.info[cls.index]['State'] = 'PAUSE'
            elif cls.info[cls.index]['State'] == 'PAUSE':
                cls.info[cls.index]['State'] = 'PLAY'
        elif arg == '-f':
            cls.index = (cls.index + 1) % len(cls.info)
        elif arg == '-r':
            cls.index = (cls.index - 1) % len(cls.info)

def no_op(*args, **kwargs):
    if False:
        return 10
    pass

@pytest.fixture
def patched_moc(fake_qtile, monkeypatch, fake_window):
    if False:
        return 10
    widget = moc.Moc()
    MockMocpProcess.reset()
    monkeypatch.setattr(widget, 'call_process', MockMocpProcess.run)
    monkeypatch.setattr('libqtile.widget.moc.subprocess.Popen', MockMocpProcess.run)
    fakebar = FakeBar([widget], window=fake_window)
    widget._configure(fake_qtile, fakebar)
    return widget

def test_moc_poll_string_formatting(patched_moc):
    if False:
        print('Hello World!')
    assert patched_moc.poll() == '♫ Rick Astley - Never Gonna Give You Up'
    MockMocpProcess.info[0]['Artist'] = ''
    assert patched_moc.poll() == '♫ Never Gonna Give You Up'
    MockMocpProcess.info[0]['SongTitle'] = ''
    assert patched_moc.poll() == '♫ rickroll'

def test_moc_state_and_colours(patched_moc):
    if False:
        while True:
            i = 10
    patched_moc.poll()
    assert patched_moc.layout.colour == patched_moc.play_color
    patched_moc.play()
    patched_moc.poll()
    assert patched_moc.layout.colour == patched_moc.noplay_color
    patched_moc.play()
    patched_moc.poll()
    assert patched_moc.layout.colour == patched_moc.play_color

def test_moc_button_presses(manager_nospawn, minimal_conf_noscreen, monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setattr('subprocess.Popen', MockMocpProcess.run)
    mocwidget = moc.Moc(update_interval=30)
    MockMocpProcess.reset()
    monkeypatch.setattr(mocwidget, 'call_process', MockMocpProcess.run)
    monkeypatch.setattr('libqtile.widget.moc.subprocess.Popen', MockMocpProcess.run)
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([mocwidget], 10))]
    manager_nospawn.start(config)
    topbar = manager_nospawn.c.bar['top']
    info = manager_nospawn.c.widget['moc'].info
    assert info()['text'] == '♫ Rick Astley - Never Gonna Give You Up'
    topbar.fake_button_press(0, 'top', 0, 0, button=4)
    manager_nospawn.c.widget['moc'].eval('self.update(self.poll())')
    assert info()['text'] == '♫ Neil Diamond - Sweet Caroline'
    topbar.fake_button_press(0, 'top', 0, 0, button=4)
    manager_nospawn.c.widget['moc'].eval('self.update(self.poll())')
    assert info()['text'] == '♫'
    topbar.fake_button_press(0, 'top', 0, 0, button=1)
    manager_nospawn.c.widget['moc'].eval('self.update(self.poll())')
    assert info()['text'] == "♫ Tom Jones - It's Not Unusual"
    topbar.fake_button_press(0, 'top', 0, 0, button=5)
    manager_nospawn.c.widget['moc'].eval('self.update(self.poll())')
    assert info()['text'] == '♫ Neil Diamond - Sweet Caroline'

def test_moc_error_handling(patched_moc):
    if False:
        i = 10
        return i + 15
    MockMocpProcess.is_error = True
    assert patched_moc.poll() == ''