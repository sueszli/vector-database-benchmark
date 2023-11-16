from __future__ import unicode_literals, division, absolute_import, print_function
import sys
import re
from powerline.lib.shell import asrun, run_cmd
from powerline.lib.unicode import out_u
from powerline.segments import Segment, with_docstring
STATE_SYMBOLS = {'fallback': '', 'play': '>', 'pause': '~', 'stop': 'X'}

def _convert_state(state):
    if False:
        return 10
    'Guess player state'
    state = state.lower()
    if 'play' in state:
        return 'play'
    if 'pause' in state:
        return 'pause'
    if 'stop' in state:
        return 'stop'
    return 'fallback'

def _convert_seconds(seconds):
    if False:
        for i in range(10):
            print('nop')
    'Convert seconds to minutes:seconds format'
    if isinstance(seconds, str):
        seconds = seconds.replace(',', '.')
    return '{0:.0f}:{1:02.0f}'.format(*divmod(float(seconds), 60))

class PlayerSegment(Segment):

    def __call__(self, format='{state_symbol} {artist} - {title} ({total})', state_symbols=STATE_SYMBOLS, **kwargs):
        if False:
            i = 10
            return i + 15
        stats = {'state': 'fallback', 'album': None, 'artist': None, 'title': None, 'elapsed': None, 'total': None}
        func_stats = self.get_player_status(**kwargs)
        if not func_stats:
            return None
        stats.update(func_stats)
        stats['state_symbol'] = state_symbols.get(stats['state'])
        return [{'contents': format.format(**stats), 'highlight_groups': ['player_' + (stats['state'] or 'fallback'), 'player']}]

    def get_player_status(self, pl):
        if False:
            return 10
        pass

    def argspecobjs(self):
        if False:
            print('Hello World!')
        for ret in super(PlayerSegment, self).argspecobjs():
            yield ret
        yield ('get_player_status', self.get_player_status)

    def omitted_args(self, name, method):
        if False:
            for i in range(10):
                print('nop')
        return ()
_common_args = '\nThis player segment should be added like this:\n\n.. code-block:: json\n\n\t{{\n\t\t"function": "powerline.segments.common.players.{0}",\n\t\t"name": "player"\n\t}}\n\n(with additional ``"args": {{…}}`` if needed).\n\nHighlight groups used: ``player_fallback`` or ``player``, ``player_play`` or ``player``, ``player_pause`` or ``player``, ``player_stop`` or ``player``.\n\n:param str format:\n\tFormat used for displaying data from player. Should be a str.format-like \n\tstring with the following keyword parameters:\n\n\t+------------+-------------------------------------------------------------+\n\t|Parameter   |Description                                                  |\n\t+============+=============================================================+\n\t|state_symbol|Symbol displayed for play/pause/stop states. There is also   |\n\t|            |“fallback” state used in case function failed to get player  |\n\t|            |state. For this state symbol is by default empty. All        |\n\t|            |symbols are defined in ``state_symbols`` argument.           |\n\t+------------+-------------------------------------------------------------+\n\t|album       |Album that is currently played.                              |\n\t+------------+-------------------------------------------------------------+\n\t|artist      |Artist whose song is currently played                        |\n\t+------------+-------------------------------------------------------------+\n\t|title       |Currently played composition.                                |\n\t+------------+-------------------------------------------------------------+\n\t|elapsed     |Composition duration in format M:SS (minutes:seconds).       |\n\t+------------+-------------------------------------------------------------+\n\t|total       |Composition length in format M:SS.                           |\n\t+------------+-------------------------------------------------------------+\n:param dict state_symbols:\n\tSymbols used for displaying state. Must contain all of the following keys:\n\n\t========  ========================================================\n\tKey       Description\n\t========  ========================================================\n\tplay      Displayed when player is playing.\n\tpause     Displayed when player is paused.\n\tstop      Displayed when player is not playing anything.\n\tfallback  Displayed if state is not one of the above or not known.\n\t========  ========================================================\n'
_player = with_docstring(PlayerSegment(), _common_args.format('_player'))

class CmusPlayerSegment(PlayerSegment):

    def get_player_status(self, pl):
        if False:
            while True:
                i = 10
        'Return cmus player information.\n\n\t\tcmus-remote -Q returns data with multi-level information i.e.\n\t\t\tstatus playing\n\t\t\tfile <file_name>\n\t\t\ttag artist <artist_name>\n\t\t\ttag title <track_title>\n\t\t\ttag ..\n\t\t\ttag n\n\t\t\tset continue <true|false>\n\t\t\tset repeat <true|false>\n\t\t\tset ..\n\t\t\tset n\n\n\t\tFor the information we are looking for we don’t really care if we’re on\n\t\tthe tag level or the set level. The dictionary comprehension in this\n\t\tmethod takes anything in ignore_levels and brings the key inside that\n\t\tto the first level of the dictionary.\n\t\t'
        now_playing_str = run_cmd(pl, ['cmus-remote', '-Q'])
        if not now_playing_str:
            return
        ignore_levels = ('tag', 'set')
        now_playing = dict(((token[0] if token[0] not in ignore_levels else token[1], ' '.join(token[1:]) if token[0] not in ignore_levels else ' '.join(token[2:])) for token in [line.split(' ') for line in now_playing_str.split('\n')[:-1]]))
        state = _convert_state(now_playing.get('status'))
        return {'state': state, 'album': now_playing.get('album'), 'artist': now_playing.get('artist'), 'title': now_playing.get('title'), 'elapsed': _convert_seconds(now_playing.get('position', 0)), 'total': _convert_seconds(now_playing.get('duration', 0))}
cmus = with_docstring(CmusPlayerSegment(), 'Return CMUS player information\n\nRequires cmus-remote command be accessible from $PATH.\n\n{0}\n'.format(_common_args.format('cmus')))

class MpdPlayerSegment(PlayerSegment):

    def get_player_status(self, pl, host='localhost', password=None, port=6600):
        if False:
            for i in range(10):
                print('nop')
        try:
            import mpd
        except ImportError:
            if password:
                host = password + '@' + host
            now_playing = run_cmd(pl, ['mpc', '-h', host, '-p', str(port)], strip=False)
            album = run_cmd(pl, ['mpc', 'current', '-f', '%album%', '-h', host, '-p', str(port)])
            if not now_playing or now_playing.count('\n') != 3:
                return
            now_playing = re.match('(.*) - (.*)\\n\\[([a-z]+)\\] +[#0-9\\/]+ +([0-9\\:]+)\\/([0-9\\:]+)', now_playing)
            return {'state': _convert_state(now_playing[3]), 'album': album, 'artist': now_playing[1], 'title': now_playing[2], 'elapsed': now_playing[4], 'total': now_playing[5]}
        else:
            try:
                client = mpd.MPDClient(use_unicode=True)
            except TypeError:
                client = mpd.MPDClient()
            client.connect(host, port)
            if password:
                client.password(password)
            now_playing = client.currentsong()
            if not now_playing:
                return
            status = client.status()
            client.close()
            client.disconnect()
            return {'state': status.get('state'), 'album': now_playing.get('album'), 'artist': now_playing.get('artist'), 'title': now_playing.get('title'), 'elapsed': _convert_seconds(status.get('elapsed', 0)), 'total': _convert_seconds(now_playing.get('time', 0))}
mpd = with_docstring(MpdPlayerSegment(), 'Return Music Player Daemon information\n\nRequires ``mpd`` Python module (e.g. |python-mpd2|_ or |python-mpd|_ Python\npackage) or alternatively the ``mpc`` command to be accessible from $PATH.\n\n.. |python-mpd| replace:: ``python-mpd``\n.. _python-mpd: https://pypi.python.org/pypi/python-mpd\n\n.. |python-mpd2| replace:: ``python-mpd2``\n.. _python-mpd2: https://pypi.python.org/pypi/python-mpd2\n\n{0}\n:param str host:\n\tHost on which mpd runs.\n:param str password:\n\tPassword used for connecting to daemon.\n:param int port:\n\tPort which should be connected to.\n'.format(_common_args.format('mpd')))
try:
    import dbus
except ImportError:

    def _get_dbus_player_status(pl, player_name, **kwargs):
        if False:
            i = 10
            return i + 15
        pl.error('Could not add {0} segment: requires dbus module', player_name)
        return
else:

    def _get_dbus_player_status(pl, bus_name=None, iface_prop='org.freedesktop.DBus.Properties', iface_player='org.mpris.MediaPlayer2.Player', player_path='/org/mpris/MediaPlayer2', player_name='player'):
        if False:
            i = 10
            return i + 15
        bus = dbus.SessionBus()
        if bus_name is None:
            for service in bus.list_names():
                if re.match('org.mpris.MediaPlayer2.', service):
                    bus_name = service
                    break
        try:
            player = bus.get_object(bus_name, player_path)
            iface = dbus.Interface(player, iface_prop)
            info = iface.Get(iface_player, 'Metadata')
            status = iface.Get(iface_player, 'PlaybackStatus')
        except dbus.exceptions.DBusException:
            return
        if not info:
            return
        try:
            elapsed = iface.Get(iface_player, 'Position')
        except dbus.exceptions.DBusException:
            pl.warning('Missing player elapsed time')
            elapsed = None
        else:
            elapsed = _convert_seconds(elapsed / 1000000.0)
        album = info.get('xesam:album')
        title = info.get('xesam:title')
        artist = info.get('xesam:artist')
        state = _convert_state(status)
        if album:
            album = out_u(album)
        if title:
            title = out_u(title)
        if artist:
            artist = out_u(artist[0])
        length = info.get('mpris:length')
        parsed_length = length and _convert_seconds(length / 1000000.0)
        return {'state': state, 'album': album, 'artist': artist, 'title': title, 'elapsed': elapsed, 'total': parsed_length}

class DbusPlayerSegment(PlayerSegment):
    get_player_status = staticmethod(_get_dbus_player_status)
dbus_player = with_docstring(DbusPlayerSegment(), 'Return generic dbus player state\n\nRequires ``dbus`` python module. Only for players that support specific protocol \n (e.g. like :py:func:`spotify` and :py:func:`clementine`).\n\n{0}\n:param str player_name:\n\tPlayer name. Used in error messages only.\n:param str bus_name:\n\tDbus bus name.\n:param str player_path:\n\tPath to the player on the given bus.\n:param str iface_prop:\n\tInterface properties name for use with dbus.Interface.\n:param str iface_player:\n\tPlayer name.\n'.format(_common_args.format('dbus_player')))

class SpotifyDbusPlayerSegment(PlayerSegment):

    def get_player_status(self, pl):
        if False:
            while True:
                i = 10
        player_status = _get_dbus_player_status(pl=pl, player_name='Spotify', bus_name='org.mpris.MediaPlayer2.spotify', player_path='/org/mpris/MediaPlayer2', iface_prop='org.freedesktop.DBus.Properties', iface_player='org.mpris.MediaPlayer2.Player')
        if player_status is not None:
            return player_status
        return _get_dbus_player_status(pl=pl, player_name='Spotify', bus_name='com.spotify.qt', player_path='/', iface_prop='org.freedesktop.DBus.Properties', iface_player='org.freedesktop.MediaPlayer2')
spotify_dbus = with_docstring(SpotifyDbusPlayerSegment(), 'Return spotify player information\n\nRequires ``dbus`` python module.\n\n{0}\n'.format(_common_args.format('spotify_dbus')))

class SpotifyAppleScriptPlayerSegment(PlayerSegment):

    def get_player_status(self, pl):
        if False:
            print('Hello World!')
        status_delimiter = '-~`/='
        ascript = '\n\t\t\ttell application "System Events"\n\t\t\t\tset process_list to (name of every process)\n\t\t\tend tell\n\n\t\t\tif process_list contains "Spotify" then\n\t\t\t\ttell application "Spotify"\n\t\t\t\t\tif player state is playing or player state is paused then\n\t\t\t\t\t\tset track_name to name of current track\n\t\t\t\t\t\tset artist_name to artist of current track\n\t\t\t\t\t\tset album_name to album of current track\n\t\t\t\t\t\tset track_length to duration of current track\n\t\t\t\t\t\tset now_playing to "" & player state & "{0}" & album_name & "{0}" & artist_name & "{0}" & track_name & "{0}" & track_length & "{0}" & player position\n\t\t\t\t\t\treturn now_playing\n\t\t\t\t\telse\n\t\t\t\t\t\treturn player state\n\t\t\t\t\tend if\n\n\t\t\t\tend tell\n\t\t\telse\n\t\t\t\treturn "stopped"\n\t\t\tend if\n\t\t'.format(status_delimiter)
        spotify = asrun(pl, ascript)
        if not asrun:
            return None
        spotify_status = spotify.split(status_delimiter)
        state = _convert_state(spotify_status[0])
        if state == 'stop':
            return None
        return {'state': state, 'album': spotify_status[1], 'artist': spotify_status[2], 'title': spotify_status[3], 'total': _convert_seconds(int(spotify_status[4]) / 1000), 'elapsed': _convert_seconds(spotify_status[5])}
spotify_apple_script = with_docstring(SpotifyAppleScriptPlayerSegment(), 'Return spotify player information\n\nRequires ``osascript`` available in $PATH.\n\n{0}\n'.format(_common_args.format('spotify_apple_script')))
if not sys.platform.startswith('darwin'):
    spotify = spotify_dbus
    _old_name = 'spotify_dbus'
else:
    spotify = spotify_apple_script
    _old_name = 'spotify_apple_script'
spotify = with_docstring(spotify, spotify.__doc__.replace(_old_name, 'spotify'))

class ClementinePlayerSegment(PlayerSegment):

    def get_player_status(self, pl):
        if False:
            while True:
                i = 10
        return _get_dbus_player_status(pl=pl, player_name='Clementine', bus_name='org.mpris.MediaPlayer2.clementine', player_path='/org/mpris/MediaPlayer2', iface_prop='org.freedesktop.DBus.Properties', iface_player='org.mpris.MediaPlayer2.Player')
clementine = with_docstring(ClementinePlayerSegment(), 'Return clementine player information\n\nRequires ``dbus`` python module.\n\n{0}\n'.format(_common_args.format('clementine')))

class RhythmboxPlayerSegment(PlayerSegment):

    def get_player_status(self, pl):
        if False:
            return 10
        now_playing = run_cmd(pl, ['rhythmbox-client', '--no-start', '--no-present', '--print-playing-format', '%at\n%aa\n%tt\n%te\n%td'], strip=False)
        if not now_playing:
            return
        now_playing = now_playing.split('\n')
        return {'album': now_playing[0], 'artist': now_playing[1], 'title': now_playing[2], 'elapsed': now_playing[3], 'total': now_playing[4]}
rhythmbox = with_docstring(RhythmboxPlayerSegment(), 'Return rhythmbox player information\n\nRequires ``rhythmbox-client`` available in $PATH.\n\n{0}\n'.format(_common_args.format('rhythmbox')))

class RDIOPlayerSegment(PlayerSegment):

    def get_player_status(self, pl):
        if False:
            while True:
                i = 10
        status_delimiter = '-~`/='
        ascript = '\n\t\t\ttell application "System Events"\n\t\t\t\tset rdio_active to the count(every process whose name is "Rdio")\n\t\t\t\tif rdio_active is 0 then\n\t\t\t\t\treturn\n\t\t\t\tend if\n\t\t\tend tell\n\t\t\ttell application "Rdio"\n\t\t\t\tset rdio_name to the name of the current track\n\t\t\t\tset rdio_artist to the artist of the current track\n\t\t\t\tset rdio_album to the album of the current track\n\t\t\t\tset rdio_duration to the duration of the current track\n\t\t\t\tset rdio_state to the player state\n\t\t\t\tset rdio_elapsed to the player position\n\t\t\t\treturn rdio_name & "{0}" & rdio_artist & "{0}" & rdio_album & "{0}" & rdio_elapsed & "{0}" & rdio_duration & "{0}" & rdio_state\n\t\t\tend tell\n\t\t'.format(status_delimiter)
        now_playing = asrun(pl, ascript)
        if not now_playing:
            return
        now_playing = now_playing.split(status_delimiter)
        if len(now_playing) != 6:
            return
        state = _convert_state(now_playing[5])
        total = _convert_seconds(now_playing[4])
        elapsed = _convert_seconds(float(now_playing[3]) * float(now_playing[4]) / 100)
        return {'title': now_playing[0], 'artist': now_playing[1], 'album': now_playing[2], 'elapsed': elapsed, 'total': total, 'state': state}
rdio = with_docstring(RDIOPlayerSegment(), 'Return rdio player information\n\nRequires ``osascript`` available in $PATH.\n\n{0}\n'.format(_common_args.format('rdio')))

class ITunesPlayerSegment(PlayerSegment):

    def get_player_status(self, pl):
        if False:
            print('Hello World!')
        status_delimiter = '-~`/='
        ascript = '\n\t\t\ttell application "System Events"\n\t\t\t\tset process_list to (name of every process)\n\t\t\tend tell\n\n\t\t\tif process_list contains "iTunes" then\n\t\t\t\ttell application "iTunes"\n\t\t\t\t\tif player state is playing then\n\t\t\t\t\t\tset t_title to name of current track\n\t\t\t\t\t\tset t_artist to artist of current track\n\t\t\t\t\t\tset t_album to album of current track\n\t\t\t\t\t\tset t_duration to duration of current track\n\t\t\t\t\t\tset t_elapsed to player position\n\t\t\t\t\t\tset t_state to player state\n\t\t\t\t\t\treturn t_title & "{0}" & t_artist & "{0}" & t_album & "{0}" & t_elapsed & "{0}" & t_duration & "{0}" & t_state\n\t\t\t\t\tend if\n\t\t\t\tend tell\n\t\t\tend if\n\t\t'.format(status_delimiter)
        now_playing = asrun(pl, ascript)
        if not now_playing:
            return
        now_playing = now_playing.split(status_delimiter)
        if len(now_playing) != 6:
            return
        (title, artist, album) = (now_playing[0], now_playing[1], now_playing[2])
        state = _convert_state(now_playing[5])
        total = _convert_seconds(now_playing[4])
        elapsed = _convert_seconds(now_playing[3])
        return {'title': title, 'artist': artist, 'album': album, 'total': total, 'elapsed': elapsed, 'state': state}
itunes = with_docstring(ITunesPlayerSegment(), 'Return iTunes now playing information\n\nRequires ``osascript``.\n\n{0}\n'.format(_common_args.format('itunes')))

class MocPlayerSegment(PlayerSegment):

    def get_player_status(self, pl):
        if False:
            for i in range(10):
                print('nop')
        'Return Music On Console (mocp) player information.\n\n\t\t``mocp -i`` returns current information i.e.\n\n\t\t.. code-block::\n\n\t\t   File: filename.format\n\t\t   Title: full title\n\t\t   Artist: artist name\n\t\t   SongTitle: song title\n\t\t   Album: album name\n\t\t   TotalTime: 00:00\n\t\t   TimeLeft: 00:00\n\t\t   TotalSec: 000\n\t\t   CurrentTime: 00:00\n\t\t   CurrentSec: 000\n\t\t   Bitrate: 000kbps\n\t\t   AvgBitrate: 000kbps\n\t\t   Rate: 00kHz\n\n\t\tFor the information we are looking for we don’t really care if we have \n\t\textra-timing information or bit rate level. The dictionary comprehension \n\t\tin this method takes anything in ignore_info and brings the key inside \n\t\tthat to the right info of the dictionary.\n\t\t'
        now_playing_str = run_cmd(pl, ['mocp', '-i'])
        if not now_playing_str:
            return
        now_playing = dict((line.split(': ', 1) for line in now_playing_str.split('\n')[:-1]))
        state = _convert_state(now_playing.get('State', 'stop'))
        return {'state': state, 'album': now_playing.get('Album', ''), 'artist': now_playing.get('Artist', ''), 'title': now_playing.get('SongTitle', ''), 'elapsed': _convert_seconds(now_playing.get('CurrentSec', 0)), 'total': _convert_seconds(now_playing.get('TotalSec', 0))}
mocp = with_docstring(MocPlayerSegment(), 'Return MOC (Music On Console) player information\n\nRequires version >= 2.3.0 and ``mocp`` executable in ``$PATH``.\n\n{0}\n'.format(_common_args.format('mocp')))