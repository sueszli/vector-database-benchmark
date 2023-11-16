"""
AudioPygame: implementation of Sound with Pygame

.. warning::

    Pygame has been deprecated and will be removed in the release after Kivy
    1.11.0.
"""
__all__ = ('SoundPygame',)
from kivy.clock import Clock
from kivy.utils import platform, deprecated
from kivy.core.audio import Sound, SoundLoader
_platform = platform
try:
    if _platform == 'android':
        try:
            import android.mixer as mixer
        except ImportError:
            import android_mixer as mixer
    else:
        from pygame import mixer
except:
    raise
mixer.pre_init(44100, -16, 2, 1024)
mixer.init()
mixer.set_num_channels(32)

class SoundPygame(Sound):
    _check_play_ev = None

    @staticmethod
    def extensions():
        if False:
            print('Hello World!')
        if _platform == 'android':
            return ('wav', 'ogg', 'mp3', 'm4a')
        return ('wav', 'ogg')

    @deprecated(msg='Pygame has been deprecated and will be removed after 1.11.0')
    def __init__(self, **kwargs):
        if False:
            return 10
        self._data = None
        self._channel = None
        super(SoundPygame, self).__init__(**kwargs)

    def _check_play(self, dt):
        if False:
            for i in range(10):
                print('nop')
        if self._channel is None:
            return False
        if self._channel.get_busy():
            return
        if self.loop:

            def do_loop(dt):
                if False:
                    for i in range(10):
                        print('nop')
                self.play()
            Clock.schedule_once(do_loop)
        else:
            self.stop()
        return False

    def play(self):
        if False:
            i = 10
            return i + 15
        if not self._data:
            return
        self._data.set_volume(self.volume)
        self._channel = self._data.play()
        self.start_time = Clock.time()
        self._check_play_ev = Clock.schedule_interval(self._check_play, 0.1)
        super(SoundPygame, self).play()

    def stop(self):
        if False:
            while True:
                i = 10
        if not self._data:
            return
        self._data.stop()
        if self._check_play_ev is not None:
            self._check_play_ev.cancel()
            self._check_play_ev = None
        self._channel = None
        super(SoundPygame, self).stop()

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        self.unload()
        if self.source is None:
            return
        self._data = mixer.Sound(self.source)

    def unload(self):
        if False:
            for i in range(10):
                print('nop')
        self.stop()
        self._data = None

    def seek(self, position):
        if False:
            while True:
                i = 10
        if not self._data:
            return
        if _platform == 'android' and self._channel:
            self._channel.seek(position)

    def get_pos(self):
        if False:
            while True:
                i = 10
        if self._data is not None and self._channel:
            if _platform == 'android':
                return self._channel.get_pos()
            return Clock.time() - self.start_time
        return 0

    def on_volume(self, instance, volume):
        if False:
            for i in range(10):
                print('nop')
        if self._data is not None:
            self._data.set_volume(volume)

    def _get_length(self):
        if False:
            return 10
        if _platform == 'android' and self._channel:
            return self._channel.get_length()
        if self._data is not None:
            return self._data.get_length()
        return super(SoundPygame, self)._get_length()
SoundLoader.register(SoundPygame)