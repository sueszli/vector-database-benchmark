import sys
from types import ModuleType
import pytest
import libqtile.config
from libqtile import widget

class MockMPD(ModuleType):

    class ConnectionError(Exception):
        pass

    class CommandError(Exception):
        pass

    class MPDClient:
        tracks = [{'title': 'Never gonna give you up', 'artist': 'Rick Astley', 'song': '0'}, {'title': 'Sweet Caroline', 'artist': 'Neil Diamond'}, {'title': 'Marea', 'artist': 'Fred Again..'}, {}, {'title': 'Sweden', 'performer': 'C418'}]

        def __init__(self):
            if False:
                print('Hello World!')
            self._index = 0
            self._connected = False
            self._state_override = True
            self._status = {'state': 'pause'}

        @property
        def _current_song(self):
            if False:
                while True:
                    i = 10
            return self.tracks[self._index]

        def ping(self):
            if False:
                return 10
            if not self._connected:
                raise ConnectionError()
            return self._state_override

        def connect(self, host, port):
            if False:
                while True:
                    i = 10
            return True

        def command_list_ok_begin(self):
            if False:
                print('Hello World!')
            pass

        def status(self):
            if False:
                i = 10
                return i + 15
            return self._status

        def currentsong(self):
            if False:
                print('Hello World!')
            return self._index + 1

        def command_list_end(self):
            if False:
                return 10
            return (self.status(), self._current_song)

        def close(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def disconnect(self):
            if False:
                return 10
            pass

        def pause(self):
            if False:
                while True:
                    i = 10
            self._status['state'] = 'pause'

        def play(self):
            if False:
                print('Hello World!')
            print('PLAYING')
            self._status['state'] = 'play'

        def stop(self):
            if False:
                print('Hello World!')
            self._status['state'] = 'stop'

        def next(self):
            if False:
                while True:
                    i = 10
            self._index = (self._index + 1) % len(self.tracks)

        def previous(self):
            if False:
                return 10
            self._index = (self._index - 1) % len(self.tracks)

        def add_states(self):
            if False:
                while True:
                    i = 10
            self._status.update({'repeat': '1', 'random': '1', 'single': '1', 'consume': '1', 'updating_db': '1'})

        def force_idle(self):
            if False:
                return 10
            self._status['state'] = 'stop'
            self._index = 3

@pytest.fixture
def mpd2_manager(manager_nospawn, monkeypatch, minimal_conf_noscreen, request):
    if False:
        while True:
            i = 10
    monkeypatch.setitem(sys.modules, 'mpd', MockMPD('mpd'))
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([widget.Mpd2(**getattr(request, 'param', dict()))], 50))]
    manager_nospawn.start(config)
    yield manager_nospawn

def test_mpd2_widget_display_and_actions(mpd2_manager):
    if False:
        for i in range(10):
            print('nop')
    widget = mpd2_manager.c.widget['mpd2']
    assert widget.info()['text'] == '⏸ Rick Astley/Never gonna give you up [-----]'
    mpd2_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, 1)
    widget.eval('self.update(self.poll())')
    assert widget.info()['text'] == '▶ Rick Astley/Never gonna give you up [-----]'
    mpd2_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, 3)
    widget.eval('self.update(self.poll())')
    assert widget.info()['text'] == '■ Rick Astley/Never gonna give you up [-----]'
    mpd2_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, 1)
    widget.eval('self.update(self.poll())')
    assert widget.info()['text'] == '▶ Rick Astley/Never gonna give you up [-----]'
    mpd2_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, 1)
    widget.eval('self.update(self.poll())')
    assert widget.info()['text'] == '⏸ Rick Astley/Never gonna give you up [-----]'
    mpd2_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, 5)
    widget.eval('self.update(self.poll())')
    assert widget.info()['text'] == '⏸ Neil Diamond/Sweet Caroline [-----]'
    mpd2_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, 5)
    widget.eval('self.update(self.poll())')
    assert widget.info()['text'] == '⏸ Fred Again../Marea [-----]'
    mpd2_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, 4)
    widget.eval('self.update(self.poll())')
    assert widget.info()['text'] == '⏸ Neil Diamond/Sweet Caroline [-----]'
    mpd2_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, 4)
    widget.eval('self.update(self.poll())')
    assert widget.info()['text'] == '⏸ Rick Astley/Never gonna give you up [-----]'

def test_mpd2_widget_extra_info(mpd2_manager):
    if False:
        print('Hello World!')
    'Quick test to check extra info is displayed ok.'
    widget = mpd2_manager.c.widget['mpd2']
    widget.eval('self.client.add_states()')
    widget.eval('self.update(self.poll())')
    assert widget.info()['text'] == '⏸ Rick Astley/Never gonna give you up [rz1cU]'

def test_mpd2_widget_idle_message(mpd2_manager):
    if False:
        while True:
            i = 10
    'Quick test to check idle message.'
    widget = mpd2_manager.c.widget['mpd2']
    widget.eval('self.client.force_idle()')
    widget.eval('self.update(self.poll())')
    assert widget.info()['text'] == '■ MPD IDLE[-----]'

@pytest.mark.parametrize('mpd2_manager', [{'status_format': '{currentsong}: {artist}/{title}'}], indirect=True)
def test_mpd2_widget_current_song(mpd2_manager):
    if False:
        while True:
            i = 10
    'Quick test to check currentsong info'
    widget = mpd2_manager.c.widget['mpd2']
    assert widget.info()['text'] == '1: Rick Astley/Never gonna give you up'

@pytest.mark.parametrize('mpd2_manager', [{'undefined_value': 'Unknown', 'status_format': '{title} ({year})'}], indirect=True)
def test_mpd2_widget_custom_undefined_value(mpd2_manager):
    if False:
        while True:
            i = 10
    'Quick test to check undefined_value option'
    widget = mpd2_manager.c.widget['mpd2']
    assert widget.info()['text'] == 'Never gonna give you up (Unknown)'

def test_mpd2_widget_dynamic_artist_value(mpd2_manager):
    if False:
        i = 10
        return i + 15
    'Quick test to check dynamic artist value'
    widget = mpd2_manager.c.widget['mpd2']
    widget.eval('self.client._index = 4')
    widget.eval('self.update(self.poll())')
    assert widget.info()['text'] == '⏸ C418/Sweden [-----]'