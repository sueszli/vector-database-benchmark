from trashcli.fstab.volume_of import VolumeOf
from trashcli.put.fs.parent_realpath import ParentRealpathFs

class VolumeOfParent:

    def __init__(self, volumes, parent_realpath_fs):
        if False:
            return 10
        self.volumes = volumes
        self.parent_realpath_fs = parent_realpath_fs

    def volume_of_parent(self, path):
        if False:
            for i in range(10):
                print('nop')
        parent_realpath = self.parent_realpath_fs.parent_realpath(path)
        return self.volumes.volume_of(parent_realpath)