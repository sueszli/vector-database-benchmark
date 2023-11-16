from typing import NamedTuple
from trashcli.put.core.candidate import Candidate
from trashcli.put.core.either import Either, Right, Left
from trashcli.put.core.failure_reason import FailureReason, LogContext
from trashcli.put.dir_maker import DirMaker
from trashcli.put.fs.fs import Fs

class TrashDirCannotBeCreated(NamedTuple('TrashDirCannotBeCreated', [('error', Exception)]), FailureReason):

    def log_entries(self, context):
        if False:
            print('Hello World!')
        return 'error during directory creation: %s' % self.error

class TrashDirCreator:

    def __init__(self, fs):
        if False:
            i = 10
            return i + 15
        self.dir_maker = DirMaker(fs)

    def make_candidate_dirs(self, candidate):
        if False:
            print('Hello World!')
        try:
            self.dir_maker.mkdir_p(candidate.trash_dir_path, 448)
            self.dir_maker.mkdir_p(candidate.files_dir(), 448)
            self.dir_maker.mkdir_p(candidate.info_dir(), 448)
            return Right(None)
        except (IOError, OSError) as error:
            return Left(TrashDirCannotBeCreated(error))