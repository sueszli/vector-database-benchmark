from typing import NamedTuple
from trashcli.fstab.volume_of import VolumeOf
from trashcli.lib.environ import Environ
from trashcli.put.core.candidate import Candidate
from trashcli.put.core.either import Either, Right, Left
from trashcli.put.core.failure_reason import FailureReason, LogContext
from trashcli.put.core.trashee import Trashee
from trashcli.put.fs.fs import Fs
from trashcli.put.gate import Gate
from trashcli.put.trash_dir_volume_reader import TrashDirVolumeReader

class DifferentVolumes(NamedTuple('DifferentVolumes', [('trash_dir_volume', str), ('file_volume', str)]), FailureReason):

    def log_entries(self, context):
        if False:
            for i in range(10):
                print('nop')
        return 'trash dir and file to be trashed are not in the same volume, trash-dir volume: %s, file volume: %s' % (self.trash_dir_volume, self.file_volume)

class HomeFallBackNotEnabled(FailureReason):

    def log_entries(self, context):
        if False:
            while True:
                i = 10
        return 'home fallback not enabled'

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, HomeFallBackNotEnabled)
GateCheckResult = Either[None, FailureReason]

class TrashDirChecker:

    def __init__(self, fs, volumes):
        if False:
            while True:
                i = 10
        self.fs = fs
        self.volumes = volumes

    def file_could_be_trashed_in(self, trashee, candidate, environ):
        if False:
            i = 10
            return i + 15
        if candidate.gate is Gate.HomeFallback:
            return self._can_be_trashed_in_home_trash_dir(environ)
        elif candidate.gate is Gate.SameVolume:
            return SameVolumeGateImpl(self.volumes, self.fs).can_trash_in(trashee, candidate)
        else:
            raise ValueError('Unknown gate: %s' % candidate.gate)

    @staticmethod
    def _can_be_trashed_in_home_trash_dir(environ):
        if False:
            return 10
        if environ.get('TRASH_ENABLE_HOME_FALLBACK', None) == '1':
            return make_ok()
        return Left(HomeFallBackNotEnabled())

def make_ok():
    if False:
        return 10
    return Right(None)

class SameVolumeGateImpl:

    def __init__(self, volumes, fs):
        if False:
            for i in range(10):
                print('nop')
        self.volumes = volumes
        self.fs = fs

    def can_trash_in(self, trashee, candidate):
        if False:
            print('Hello World!')
        trash_dir_volume = self._volume_of_trash_dir(candidate)
        same_volume = trash_dir_volume == trashee.volume
        if not same_volume:
            return Left(DifferentVolumes(trash_dir_volume, trashee.volume))
        return make_ok()

    def _volume_of_trash_dir(self, candidate):
        if False:
            print('Hello World!')
        return TrashDirVolumeReader(self.volumes, self.fs).volume_of_trash_dir(candidate.trash_dir_path)