import os
from typing import List, Callable
from tests.support.fake_is_mount import FakeIsMount
from trashcli.fstab.volume_of import VolumeOf, VolumeOfImpl

def fake_volume_of(volumes):
    if False:
        while True:
            i = 10
    return VolumeOfImpl(FakeIsMount(volumes), os.path.normpath)

def volume_of_stub(func=lambda x: 'volume_of %s' % x):
    if False:
        for i in range(10):
            print('nop')
    return _FakeVolumeOf(func)

class _FakeVolumeOf(VolumeOf):

    def __init__(self, volume_of_impl):
        if False:
            i = 10
            return i + 15
        self.volume_of_impl = volume_of_impl

    def volume_of(self, path):
        if False:
            print('Hello World!')
        return self.volume_of_impl(path)