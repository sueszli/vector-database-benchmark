import subprocess
import pytest
import libqtile.config
from libqtile.widget import cmus
from test.widgets.conftest import FakeBar

class MockCmusRemoteProcess:
    CalledProcessError = None
    EXTRA = ['set aaa_mode all', 'set continue true', 'set play_library true', 'set play_sorted false', 'set replaygain disabled', 'set replaygain_limit true', 'set replaygain_preamp 0.000000', 'set repeat false', 'set repeat_current false', 'set shuffle false', 'set softvol false', 'set vol_left 100', 'set vol_right 100']
    info = {}
    is_error = False
    index = 0

    @classmethod
    def reset(cls):
        if False:
            return 10
        cls.info = [['status playing', 'file /playing/file/rickroll.mp3', 'duration 222', 'position 14', 'tag artist Rick Astley', 'tag album Whenever You Need Somebody', 'tag title Never Gonna Give You Up'], ['status playing', 'file http://playing/file/sweetcaroline.mp3', 'duration 222', 'position 14', 'tag artist Neil Diamond', 'tag album Greatest Hits', 'tag title Sweet Caroline'], ['status stopped', 'file http://streaming.source/tomjones.m3u', 'duration -1', 'position -9', "tag title It's Not Unusual", 'stream tomjones'], ['status playing', 'file /playing/file/always.mp3', 'duration 222', 'position 14', 'tag artist Above & Beyond', 'tag album Anjunabeats 14', 'tag title Always - Tinlicker Extended Mix'], ['status playing', 'file /playing/file/always.mp3', 'duration 222', 'position 14']]
        cls.index = 0
        cls.is_error = False

    @classmethod
    def call_process(cls, cmd):
        if False:
            i = 10
            return i + 15
        if cls.is_error:
            raise subprocess.CalledProcessError(-1, cmd=cmd, output="Couldn't connect to cmus.")
        if cmd[1:] == ['-C', 'status']:
            track = cls.info[cls.index]
            track.extend(cls.EXTRA)
            output = '\n'.join(track)
            return output
        elif cmd[1] == '-p':
            cls.info[cls.index][0] = 'status playing'
        elif cmd[1] == '-u':
            if cls.info[cls.index][0] == 'status playing':
                cls.info[cls.index][0] = 'status paused'
            elif cls.info[cls.index][0] == 'status paused':
                cls.info[cls.index][0] = 'status playing'
        elif cmd[1] == '-n':
            cls.index = (cls.index + 1) % len(cls.info)
        elif cmd[1] == '-r':
            cls.index = (cls.index - 1) % len(cls.info)

    @classmethod
    def Popen(cls, cmd):
        if False:
            i = 10
            return i + 15
        cls.call_process(cmd)

def no_op(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    pass

@pytest.fixture
def patched_cmus(monkeypatch):
    if False:
        print('Hello World!')
    MockCmusRemoteProcess.reset()
    monkeypatch.setattr('libqtile.widget.cmus.subprocess', MockCmusRemoteProcess)
    monkeypatch.setattr('libqtile.widget.cmus.subprocess.CalledProcessError', subprocess.CalledProcessError)
    monkeypatch.setattr('libqtile.widget.cmus.base.ThreadPoolText.call_process', MockCmusRemoteProcess.call_process)
    return cmus

def test_cmus(fake_qtile, patched_cmus, fake_window):
    if False:
        for i in range(10):
            print('nop')
    widget = patched_cmus.Cmus()
    fakebar = FakeBar([widget], window=fake_window)
    widget._configure(fake_qtile, fakebar)
    text = widget.poll()
    assert text == '♫ Rick Astley - Never Gonna Give You Up'
    assert widget.layout.colour == widget.play_color
    widget.play()
    text = widget.poll()
    assert text == '♫ Rick Astley - Never Gonna Give You Up'
    assert widget.layout.colour == widget.noplay_color

def test_cmus_play_stopped(fake_qtile, patched_cmus, fake_window):
    if False:
        print('Hello World!')
    widget = patched_cmus.Cmus()
    MockCmusRemoteProcess.index = 2
    fakebar = FakeBar([widget], window=fake_window)
    widget._configure(fake_qtile, fakebar)
    text = widget.poll()
    assert text == '♫ tomjones'
    assert widget.layout.colour == widget.noplay_color
    widget.play()
    text = widget.poll()
    assert text == '♫ tomjones'
    assert widget.layout.colour == widget.play_color

def test_cmus_buttons(minimal_conf_noscreen, manager_nospawn, patched_cmus):
    if False:
        print('Hello World!')
    widget = patched_cmus.Cmus(update_interval=30)
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([widget], 10))]
    manager_nospawn.start(config)
    topbar = manager_nospawn.c.bar['top']
    cmuswidget = manager_nospawn.c.widget['cmus']
    assert cmuswidget.info()['text'] == '♫ Rick Astley - Never Gonna Give You Up'
    topbar.fake_button_press(0, 'top', 0, 0, button=4)
    cmuswidget.eval('self.update(self.poll())')
    assert cmuswidget.info()['text'] == '♫ Sweet Caroline'
    topbar.fake_button_press(0, 'top', 0, 0, button=4)
    cmuswidget.eval('self.update(self.poll())')
    assert cmuswidget.info()['text'] == '♫ tomjones'
    topbar.fake_button_press(0, 'top', 0, 0, button=5)
    cmuswidget.eval('self.update(self.poll())')
    assert cmuswidget.info()['text'] == '♫ Sweet Caroline'

def test_cmus_error_handling(fake_qtile, patched_cmus, fake_window):
    if False:
        print('Hello World!')
    widget = patched_cmus.Cmus()
    MockCmusRemoteProcess.is_error = True
    fakebar = FakeBar([widget], window=fake_window)
    widget._configure(fake_qtile, fakebar)
    text = widget.poll()
    assert text == ''

def test_escape_text(fake_qtile, patched_cmus, fake_window):
    if False:
        for i in range(10):
            print('nop')
    widget = patched_cmus.Cmus()
    MockCmusRemoteProcess.index = 3
    fakebar = FakeBar([widget], window=fake_window)
    widget._configure(fake_qtile, fakebar)
    text = widget.poll()
    assert text == '♫ Above &amp; Beyond - Always - Tinlicker Extended Mix'

def test_missing_metadata(fake_qtile, patched_cmus, fake_window):
    if False:
        print('Hello World!')
    widget = patched_cmus.Cmus()
    MockCmusRemoteProcess.index = 4
    fakebar = FakeBar([widget], window=fake_window)
    widget._configure(fake_qtile, fakebar)
    text = widget.poll()
    assert text == '♫ always.mp3'