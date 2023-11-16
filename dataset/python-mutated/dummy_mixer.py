import pykka
from mopidy import mixer

def create_proxy(config=None):
    if False:
        i = 10
        return i + 15
    return DummyMixer.start(config=None).proxy()

class DummyMixer(pykka.ThreadingActor, mixer.Mixer):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._volume = None
        self._mute = None

    def get_volume(self):
        if False:
            i = 10
            return i + 15
        return self._volume

    def set_volume(self, volume):
        if False:
            i = 10
            return i + 15
        self._volume = volume
        self.trigger_volume_changed(volume=volume)
        return True

    def get_mute(self):
        if False:
            while True:
                i = 10
        return self._mute

    def set_mute(self, mute):
        if False:
            print('Hello World!')
        self._mute = mute
        self.trigger_mute_changed(mute=mute)
        return True