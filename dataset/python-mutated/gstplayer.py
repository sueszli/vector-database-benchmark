"""A wrapper for the GStreamer Python bindings that exposes a simple
music player.
"""
import _thread
import copy
import os
import sys
import time
import urllib
import gi
from beets import ui
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
Gst.init(None)

class QueryError(Exception):
    pass

class GstPlayer:
    """A music player abstracting GStreamer's Playbin element.

    Create a player object, then call run() to start a thread with a
    runloop. Then call play_file to play music. Use player.playing
    to check whether music is currently playing.

    A basic play queue is also implemented (just a Python list,
    player.queue, whose last element is next to play). To use it,
    just call enqueue() and then play(). When a track finishes and
    another is available on the queue, it is played automatically.
    """

    def __init__(self, finished_callback=None):
        if False:
            return 10
        'Initialize a player.\n\n        If a finished_callback is provided, it is called every time a\n        track started with play_file finishes.\n\n        Once the player has been created, call run() to begin the main\n        runloop in a separate thread.\n        '
        self.player = Gst.ElementFactory.make('playbin', 'player')
        if self.player is None:
            raise ui.UserError('Could not create playbin')
        fakesink = Gst.ElementFactory.make('fakesink', 'fakesink')
        if fakesink is None:
            raise ui.UserError('Could not create fakesink')
        self.player.set_property('video-sink', fakesink)
        bus = self.player.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self._handle_message)
        self.playing = False
        self.finished_callback = finished_callback
        self.cached_time = None
        self._volume = 1.0

    def _get_state(self):
        if False:
            return 10
        'Returns the current state flag of the playbin.'
        return self.player.get_state(Gst.CLOCK_TIME_NONE)[1]

    def _handle_message(self, bus, message):
        if False:
            print('Hello World!')
        'Callback for status updates from GStreamer.'
        if message.type == Gst.MessageType.EOS:
            self.player.set_state(Gst.State.NULL)
            self.playing = False
            self.cached_time = None
            if self.finished_callback:
                self.finished_callback()
        elif message.type == Gst.MessageType.ERROR:
            self.player.set_state(Gst.State.NULL)
            (err, debug) = message.parse_error()
            print(f'Error: {err}')
            self.playing = False

    def _set_volume(self, volume):
        if False:
            for i in range(10):
                print('nop')
        'Set the volume level to a value in the range [0, 1.5].'
        self._volume = volume
        self.player.set_property('volume', volume)

    def _get_volume(self):
        if False:
            while True:
                i = 10
        'Get the volume as a float in the range [0, 1.5].'
        return self._volume
    volume = property(_get_volume, _set_volume)

    def play_file(self, path):
        if False:
            return 10
        'Immediately begin playing the audio file at the given\n        path.\n        '
        self.player.set_state(Gst.State.NULL)
        if isinstance(path, str):
            path = path.encode('utf-8')
        uri = 'file://' + urllib.parse.quote(path)
        self.player.set_property('uri', uri)
        self.player.set_state(Gst.State.PLAYING)
        self.playing = True

    def play(self):
        if False:
            i = 10
            return i + 15
        'If paused, resume playback.'
        if self._get_state() == Gst.State.PAUSED:
            self.player.set_state(Gst.State.PLAYING)
            self.playing = True

    def pause(self):
        if False:
            print('Hello World!')
        'Pause playback.'
        self.player.set_state(Gst.State.PAUSED)

    def stop(self):
        if False:
            while True:
                i = 10
        'Halt playback.'
        self.player.set_state(Gst.State.NULL)
        self.playing = False
        self.cached_time = None

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        'Start a new thread for the player.\n\n        Call this function before trying to play any music with\n        play_file() or play().\n        '

        def start():
            if False:
                return 10
            loop = GLib.MainLoop()
            loop.run()
        _thread.start_new_thread(start, ())

    def time(self):
        if False:
            return 10
        'Returns a tuple containing (position, length) where both\n        values are integers in seconds. If no stream is available,\n        returns (0, 0).\n        '
        fmt = Gst.Format(Gst.Format.TIME)
        try:
            posq = self.player.query_position(fmt)
            if not posq[0]:
                raise QueryError('query_position failed')
            pos = posq[1] / 10 ** 9
            lengthq = self.player.query_duration(fmt)
            if not lengthq[0]:
                raise QueryError('query_duration failed')
            length = lengthq[1] / 10 ** 9
            self.cached_time = (pos, length)
            return (pos, length)
        except QueryError:
            if self.playing and self.cached_time:
                return self.cached_time
            else:
                return (0, 0)

    def seek(self, position):
        if False:
            print('Hello World!')
        'Seeks to position (in seconds).'
        (cur_pos, cur_len) = self.time()
        if position > cur_len:
            self.stop()
            return
        fmt = Gst.Format(Gst.Format.TIME)
        ns = position * 10 ** 9
        self.player.seek_simple(fmt, Gst.SeekFlags.FLUSH, ns)
        self.cached_time = (position, cur_len)

    def block(self):
        if False:
            print('Hello World!')
        'Block until playing finishes.'
        while self.playing:
            time.sleep(1)

    def get_decoders(self):
        if False:
            i = 10
            return i + 15
        return get_decoders()

def get_decoders():
    if False:
        for i in range(10):
            print('nop')
    'Get supported audio decoders from GStreamer.\n    Returns a dict mapping decoder element names to the associated media types\n    and file extensions.\n    '
    filt = Gst.ELEMENT_FACTORY_TYPE_DEPAYLOADER | Gst.ELEMENT_FACTORY_TYPE_DEMUXER | Gst.ELEMENT_FACTORY_TYPE_PARSER | Gst.ELEMENT_FACTORY_TYPE_DECODER | Gst.ELEMENT_FACTORY_TYPE_MEDIA_AUDIO
    decoders = {}
    mime_types = set()
    for f in Gst.ElementFactory.list_get_elements(filt, Gst.Rank.NONE):
        for pad in f.get_static_pad_templates():
            if pad.direction == Gst.PadDirection.SINK:
                caps = pad.static_caps.get()
                mimes = set()
                for i in range(caps.get_size()):
                    struct = caps.get_structure(i)
                    mime = struct.get_name()
                    if mime == 'unknown/unknown':
                        continue
                    mimes.add(mime)
                    mime_types.add(mime)
                if mimes:
                    decoders[f.get_name()] = (mimes, set())
    mime_extensions = {mime: set() for mime in mime_types}
    for feat in Gst.Registry.get().get_feature_list(Gst.TypeFindFactory):
        caps = feat.get_caps()
        if caps:
            for i in range(caps.get_size()):
                struct = caps.get_structure(i)
                mime = struct.get_name()
                if mime in mime_types:
                    mime_extensions[mime].update(feat.get_extensions())
    for (name, (mimes, exts)) in decoders.items():
        for mime in mimes:
            exts.update(mime_extensions[mime])
    return decoders

def play_simple(paths):
    if False:
        return 10
    "Play the files in paths in a straightforward way, without\n    using the player's callback function.\n    "
    p = GstPlayer()
    p.run()
    for path in paths:
        p.play_file(path)
        p.block()

def play_complicated(paths):
    if False:
        i = 10
        return i + 15
    'Play the files in the path one after the other by using the\n    callback function to advance to the next song.\n    '
    my_paths = copy.copy(paths)

    def next_song():
        if False:
            return 10
        my_paths.pop(0)
        p.play_file(my_paths[0])
    p = GstPlayer(next_song)
    p.run()
    p.play_file(my_paths[0])
    while my_paths:
        time.sleep(1)
if __name__ == '__main__':
    paths = [os.path.abspath(os.path.expanduser(p)) for p in sys.argv[1:]]
    play_complicated(paths)