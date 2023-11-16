from trashcli.put.core.candidate import Candidate
from trashcli.put.core.check_type import NoCheck, TopTrashDirCheck
from trashcli.put.core.either import Either, Right, Left
from trashcli.put.core.failure_reason import FailureReason, LogContext

class SecurityCheck:

    def __init__(self, fs):
        if False:
            for i in range(10):
                print('nop')
        self.fs = fs

    def check_trash_dir_is_secure(self, candidate):
        if False:
            while True:
                i = 10
        if candidate.check_type == NoCheck:
            return Right(None)
        if candidate.check_type == TopTrashDirCheck:
            parent = candidate.parent_dir()
            if not self.fs.lexists(parent):
                return Left(TrashDirDoesNotHaveParent())
            if not self.fs.isdir(parent):
                return Left(TrashDirCannotBeCreatedBecauseParentIsFile())
            if self.fs.islink(parent):
                return Left(TrashDirIsNotSecureBecauseSymLink())
            if not self.fs.has_sticky_bit(parent):
                return Left(TrashDirIsNotSecureBecauseNotSticky())
            return Right(None)
        raise Exception('Unknown check type: %s' % candidate.check_type)

class TrashDirDoesNotHaveParent(FailureReason):

    def log_entries(self, context):
        if False:
            while True:
                i = 10
        return trash_dir_parent_problem(context, 'trash dir cannot be created because its parent does not exists')

class TrashDirCannotBeCreatedBecauseParentIsFile(FailureReason):

    def log_entries(self, context):
        if False:
            return 10
        return trash_dir_parent_problem(context, 'trash dir cannot be created as its parent is a file instead of being a directory')

class TrashDirIsNotSecureBecauseSymLink(FailureReason):

    def log_entries(self, context):
        if False:
            for i in range(10):
                print('nop')
        return trash_dir_parent_problem(context, 'trash dir is insecure, its parent should not be a symlink')

class TrashDirIsNotSecureBecauseNotSticky(FailureReason):

    def log_entries(self, context):
        if False:
            return 10
        return trash_dir_parent_problem(context, 'trash dir is insecure, its parent should be sticky')

def trash_dir_parent_problem(context, message):
    if False:
        return 10
    return '%s, trash-dir: %s, parent: %s' % (message, context.trash_dir_norm_path(), context.candidate.parent_dir())