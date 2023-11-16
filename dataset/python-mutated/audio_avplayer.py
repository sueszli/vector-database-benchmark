"""
AudioAvplayer: implementation of Sound using pyobjus / AVFoundation.
Works on iOS / OSX.
"""
__all__ = ('SoundAvplayer',)
from kivy.core.audio import Sound, SoundLoader
from pyobjus import autoclass
from pyobjus.dylib_manager import load_framework, INCLUDE
load_framework(INCLUDE.AVFoundation)
AVAudioPlayer = autoclass('AVAudioPlayer')
NSURL = autoclass('NSURL')
NSString = autoclass('NSString')

class SoundAvplayer(Sound):

    @staticmethod
    def extensions():
        if False:
            print('Hello World!')
        return ('aac', 'adts', 'aif', 'aiff', 'aifc', 'caf', 'mp3', 'mp4', 'm4a', 'snd', 'au', 'sd2', 'wav')

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self._avplayer = None
        super(SoundAvplayer, self).__init__(**kwargs)

    def load(self):
        if False:
            while True:
                i = 10
        self.unload()
        fn = NSString.alloc().initWithUTF8String_(self.source)
        url = NSURL.alloc().initFileURLWithPath_(fn)
        self._avplayer = AVAudioPlayer.alloc().initWithContentsOfURL_error_(url, None)

    def unload(self):
        if False:
            i = 10
            return i + 15
        self.stop()
        self._avplayer = None

    def play(self):
        if False:
            print('Hello World!')
        if not self._avplayer:
            return
        self._avplayer.play()
        super(SoundAvplayer, self).play()

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._avplayer:
            return
        self._avplayer.stop()
        super(SoundAvplayer, self).stop()

    def seek(self, position):
        if False:
            print('Hello World!')
        if not self._avplayer:
            return
        self._avplayer.playAtTime_(float(position))

    def get_pos(self):
        if False:
            return 10
        if self._avplayer:
            return self._avplayer.currentTime
        return super(SoundAvplayer, self).get_pos()

    def on_volume(self, instance, volume):
        if False:
            for i in range(10):
                print('nop')
        if self._avplayer:
            self._avplayer.volume = float(volume)

    def _get_length(self):
        if False:
            for i in range(10):
                print('nop')
        if self._avplayer:
            return self._avplayer.duration
        return super(SoundAvplayer, self)._get_length()
SoundLoader.register(SoundAvplayer)