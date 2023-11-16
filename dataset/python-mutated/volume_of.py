import os
from abc import ABCMeta, abstractmethod
import six
from trashcli.fstab.volume_listing import RealIsMount

@six.add_metaclass(ABCMeta)
class VolumeOf:

    @abstractmethod
    def volume_of(self, path):
        if False:
            print('Hello World!')
        raise NotImplementedError()

class RealVolumeOf(VolumeOf):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.impl = VolumeOfImpl(RealIsMount(), os.path.abspath)

    def volume_of(self, path):
        if False:
            for i in range(10):
                print('nop')
        return self.impl.volume_of(path)

class VolumeOfImpl(VolumeOf):

    def __init__(self, ismount, abspath):
        if False:
            print('Hello World!')
        self.ismount = ismount
        self.abspath = abspath

    def volume_of(self, path):
        if False:
            return 10
        path = self.abspath(path)
        while path != os.path.dirname(path):
            if self.ismount.is_mount(path):
                break
            path = os.path.dirname(path)
        return path