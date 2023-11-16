from __future__ import annotations
import asyncio
import re
import string
from typing import TYPE_CHECKING
from dbus_next import Message, Variant
from dbus_next.constants import MessageType
from libqtile.command.base import expose_command
from libqtile.log_utils import logger
from libqtile.utils import _send_dbus_message, add_signal_receiver, create_task
from libqtile.widget import base
if TYPE_CHECKING:
    from typing import Any
MPRIS_PATH = '/org/mpris/MediaPlayer2'
MPRIS_OBJECT = 'org.mpris.MediaPlayer2'
MPRIS_PLAYER = 'org.mpris.MediaPlayer2.Player'
PROPERTIES_INTERFACE = 'org.freedesktop.DBus.Properties'
MPRIS_REGEX = re.compile('(\\{(.*?):(.*?)(:.*?)?\\})')

class Mpris2Formatter(string.Formatter):
    """
    Custom string formatter for MPRIS2 metadata.

    Keys have a colon (e.g. "xesam:title") which causes issues with python's string
    formatting as the colon splits the identifier from the format specification.

    This formatter handles this issue by changing the first colon to an underscore and
    then formatting the incoming kwargs to match.

    Additionally, a default value is returned when an identifier is not provided by the
    kwarg data.
    """

    def __init__(self, default=''):
        if False:
            for i in range(10):
                print('nop')
        string.Formatter.__init__(self)
        self._default = default

    def get_value(self, key, args, kwargs):
        if False:
            print('Hello World!')
        '\n        Replaces colon in kwarg keys with an underscore before getting value.\n\n        Missing identifiers are replaced with the default value.\n        '
        kwargs = {k.replace(':', '_'): v for (k, v) in kwargs.items()}
        try:
            return string.Formatter.get_value(self, key, args, kwargs)
        except (IndexError, KeyError):
            return self._default

    def parse(self, format_string):
        if False:
            print('Hello World!')
        '\n        Replaces first colon in format string with an underscore.\n\n        This will cause issues if any identifier is provided that does not\n        contain a colon. This should not happen according to the MPRIS2\n        specification!\n        '
        format_string = MPRIS_REGEX.sub('{\\2_\\3\\4}', format_string)
        return string.Formatter.parse(self, format_string)

class Mpris2(base._TextBox):
    """An MPRIS 2 widget

    A widget which displays the current track/artist of your favorite MPRIS
    player. This widget scrolls the text if neccessary and information that
    is displayed is configurable.

    The widget relies on players broadcasting signals when the metadata or playback
    status changes. If you are getting inconsistent results then you can enable background
    polling of the player by setting the `poll_interval` parameter. This is disabled by
    default.

    Basic mouse controls are also available: button 1 = play/pause,
    scroll up = next track, scroll down = previous track.

    Widget requirements: dbus-next_.

    .. _dbus-next: https://pypi.org/project/dbus-next/
    """
    defaults = [('name', 'audacious', 'Name of the MPRIS widget.'), ('objname', None, 'DBUS MPRIS 2 compatible player identifier- Find it out with dbus-monitor - Also see: http://specifications.freedesktop.org/mpris-spec/latest/#Bus-Name-Policy. ``None`` will listen for notifications from all MPRIS2 compatible players.'), ('format', '{xesam:title} - {xesam:album} - {xesam:artist}', 'Format string for displaying metadata. See http://www.freedesktop.org/wiki/Specifications/mpris-spec/metadata/#index5h3 for available values'), ('separator', ', ', 'Separator for metadata fields that are a list.'), ('display_metadata', ['xesam:title', 'xesam:album', 'xesam:artist'], '(Deprecated) Which metadata identifiers to display. '), ('scroll', True, 'Whether text should scroll.'), ('playing_text', '{track}', 'Text to show when playing'), ('paused_text', 'Paused: {track}', 'Text to show when paused'), ('stopped_text', '', 'Text to show when stopped'), ('stop_pause_text', None, '(Deprecated) Optional text to display when in the stopped/paused state'), ('no_metadata_text', 'No metadata for current track', 'Text to show when track has no metadata'), ('poll_interval', 0, 'Periodic background polling interval of player (0 to disable polling).')]

    def __init__(self, **config):
        if False:
            for i in range(10):
                print('nop')
        base._TextBox.__init__(self, '', **config)
        self.add_defaults(Mpris2.defaults)
        self.is_playing = False
        self.count = 0
        self.displaytext = ''
        self.track_info = ''
        self.status = '{track}'
        self.add_callbacks({'Button1': self.play_pause, 'Button4': self.next, 'Button5': self.previous})
        paused = ''
        stopped = ''
        if 'stop_pause_text' in config:
            logger.warning("The use of 'stop_pause_text' is deprecated. Please use 'paused_text' and 'stopped_text' instead.")
            if 'paused_text' not in config:
                paused = self.stop_pause_text
            if 'stopped_text' not in config:
                stopped = self.stop_pause_text
        if 'display_metadata' in config:
            logger.warning('The use of `display_metadata is deprecated. Please use `format` instead.')
            self.format = ' - '.join((f'{{{s}}}' for s in config['display_metadata']))
        self._formatter = Mpris2Formatter()
        self.prefixes = {'Playing': self.playing_text, 'Paused': paused or self.paused_text, 'Stopped': stopped or self.stopped_text}
        self._current_player: str | None = None
        self.player_names: dict[str, str] = {}
        self._background_poll: asyncio.TimerHandle | None = None

    @property
    def player(self) -> str:
        if False:
            print('Hello World!')
        if self._current_player is None:
            return 'None'
        else:
            return self.player_names.get(self._current_player, 'Unknown')

    async def _config_async(self):
        await add_signal_receiver(self._name_owner_changed, session_bus=True, signal_name='NameOwnerChanged', dbus_interface='org.freedesktop.DBus')
        subscribe = await add_signal_receiver(self.message, session_bus=True, signal_name='PropertiesChanged', bus_name=self.objname, path='/org/mpris/MediaPlayer2', dbus_interface='org.freedesktop.DBus.Properties')
        if not subscribe:
            logger.warning('Unable to add signal receiver for Mpris2 players')
        if self.objname is not None:
            await self._check_player()

    def _name_owner_changed(self, message):
        if False:
            i = 10
            return i + 15
        (name, _, new_owner) = message.body
        if new_owner == '' and name == self._current_player:
            self._current_player = None
            self.update('')
            self._set_background_poll(False)

    def message(self, message):
        if False:
            i = 10
            return i + 15
        if message.message_type != MessageType.SIGNAL:
            return
        create_task(self.process_message(message))

    async def process_message(self, message):
        current_player = message.sender
        if current_player not in self.player_names:
            self.player_names[current_player] = await self.get_player_name(current_player)
        self._current_player = current_player
        self.parse_message(*message.body)

    async def _check_player(self):
        """Check for player at startup and retrieve metadata."""
        if not (self.objname or self._current_player):
            return
        (bus, message) = await _send_dbus_message(True, MessageType.METHOD_CALL, self.objname if self.objname else self._current_player, PROPERTIES_INTERFACE, MPRIS_PATH, 'GetAll', 's', [MPRIS_PLAYER])
        if bus:
            bus.disconnect()
        if message.message_type != MessageType.METHOD_RETURN:
            self._current_player = None
            self.update('')
            return
        if message.body:
            self._current_player = message.sender
            self.parse_message(self.objname, message.body[0], [])

    def _set_background_poll(self, poll=True):
        if False:
            print('Hello World!')
        if self._background_poll is not None:
            self._background_poll.cancel()
        if poll:
            self._background_poll = self.timeout_add(self.poll_interval, self._check_player)

    async def get_player_name(self, player):
        (bus, message) = await _send_dbus_message(True, MessageType.METHOD_CALL, player, PROPERTIES_INTERFACE, MPRIS_PATH, 'Get', 'ss', [MPRIS_OBJECT, 'Identity'])
        if bus:
            bus.disconnect()
        if message.message_type != MessageType.METHOD_RETURN:
            logger.warning('Could not retrieve identity of player on %s.', player)
            return ''
        return message.body[0].value

    def parse_message(self, _interface_name: str, changed_properties: dict[str, Any], _invalidated_properties: list[str]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        http://specifications.freedesktop.org/mpris-spec/latest/Track_List_Interface.html#Mapping:Metadata_Map\n        '
        if not self.configured:
            return
        if 'Metadata' not in changed_properties and 'PlaybackStatus' not in changed_properties:
            return
        self.displaytext = ''
        metadata = changed_properties.get('Metadata')
        if metadata:
            self.track_info = self.get_track_info(metadata.value)
        playbackstatus = getattr(changed_properties.get('PlaybackStatus'), 'value', None)
        if playbackstatus:
            self.is_playing = playbackstatus == 'Playing'
            self.status = self.prefixes.get(playbackstatus, '{track}')
        if not self.track_info:
            self.track_info = self.no_metadata_text
        self.displaytext = self.status.format(track=self.track_info)
        if self.text != self.displaytext:
            self.update(self.displaytext)
        if self.poll_interval:
            self._set_background_poll()

    def get_track_info(self, metadata: dict[str, Variant]) -> str:
        if False:
            i = 10
            return i + 15
        self.metadata = {}
        for key in metadata:
            new_key = key
            val = getattr(metadata.get(key), 'value', None)
            if isinstance(val, str):
                self.metadata[new_key] = val
            elif isinstance(val, list):
                self.metadata[new_key] = self.separator.join((y for y in val if isinstance(y, str)))
        return self._formatter.format(self.format, **self.metadata).replace('\n', '')

    def _player_cmd(self, cmd: str) -> None:
        if False:
            print('Hello World!')
        if self._current_player is None:
            return
        task = create_task(self._send_player_cmd(cmd))
        assert task
        task.add_done_callback(self._task_callback)

    async def _send_player_cmd(self, cmd: str) -> Message | None:
        (bus, message) = await _send_dbus_message(True, MessageType.METHOD_CALL, self._current_player, MPRIS_PLAYER, MPRIS_PATH, cmd, '', [])
        if bus:
            bus.disconnect()
        return message

    def _task_callback(self, task: asyncio.Task) -> None:
        if False:
            print('Hello World!')
        message = task.result()
        if message is None:
            return
        if message.message_type != MessageType.METHOD_RETURN:
            logger.warning('Unable to send command to player.')

    @expose_command()
    def play_pause(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Toggle the playback status.'
        self._player_cmd('PlayPause')

    @expose_command()
    def next(self) -> None:
        if False:
            while True:
                i = 10
        'Play the next track.'
        self._player_cmd('Next')

    @expose_command()
    def previous(self) -> None:
        if False:
            print('Hello World!')
        'Play the previous track.'
        self._player_cmd('Previous')

    @expose_command()
    def stop(self) -> None:
        if False:
            print('Hello World!')
        'Stop playback.'
        self._player_cmd('Stop')

    @expose_command()
    def info(self):
        if False:
            print('Hello World!')
        "What's the current state of the widget?"
        d = base._TextBox.info(self)
        d.update(dict(isplaying=self.is_playing, player=self.player))
        return d