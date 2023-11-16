import os
from trashcli.fstab.volume_of import VolumeOf
from trashcli.put.fs.fs import RealPathFs

class TrashDirVolumeReader:

    def __init__(self, volumes, fs):
        if False:
            print('Hello World!')
        self.volumes = volumes
        self.fs = fs

    def volume_of_trash_dir(self, trash_dir_path):
        if False:
            i = 10
            return i + 15
        norm_trash_dir_path = os.path.normpath(trash_dir_path)
        return self.volumes.volume_of(self.fs.realpath(norm_trash_dir_path))