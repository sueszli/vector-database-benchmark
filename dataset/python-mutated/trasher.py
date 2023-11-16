import os
from trashcli.lib.environ import Environ
from trashcli.put.core.trash_result import TrashResult
from trashcli.put.file_trasher import FileTrasher
from trashcli.put.fs.fs import Fs
from trashcli.put.my_logger import LogData
from trashcli.put.parser import mode_force
from trashcli.put.parser import mode_interactive
from trashcli.put.reporter import TrashPutReporter
from trashcli.put.user import User
from trashcli.put.user import user_replied_no

class Trasher:

    def __init__(self, file_trasher, user, reporter, fs):
        if False:
            i = 10
            return i + 15
        self.file_trasher = file_trasher
        self.user = user
        self.reporter = reporter
        self.fs = fs

    def trash_single(self, path, user_trash_dir, mode, forced_volume, home_fallback, program_name, log_data, environ, uid):
        if False:
            while True:
                i = 10
        '\n        Trash a file in the appropriate trash directory.\n        If the file belong to the same volume of the trash home directory it\n        will be trashed in the home trash directory.\n        Otherwise it will be trashed in one of the relevant volume trash\n        directories.\n\n        Each volume can have two trash directories, they are\n            - $volume/.Trash/$uid\n            - $volume/.Trash-$uid\n\n        Firstly the software attempt to trash the file in the first directory\n        then try to trash in the second trash directory.\n        '
        if self._should_skipped_by_specs(path):
            self.reporter.unable_to_trash_dot_entries(path, program_name)
            return TrashResult.Failure
        if not self.fs.lexists(path):
            if mode == mode_force:
                return TrashResult.Success
            else:
                self.reporter.unable_to_trash_file_non_existent(path, log_data)
                return TrashResult.Failure
        if mode == mode_interactive and self.fs.is_accessible(path):
            reply = self.user.ask_user_about_deleting_file(program_name, path)
            if reply == user_replied_no:
                return TrashResult.Success
        return self.file_trasher.trash_file(path, forced_volume, user_trash_dir, home_fallback, environ, uid, log_data)

    @staticmethod
    def _should_skipped_by_specs(path):
        if False:
            for i in range(10):
                print('nop')
        basename = os.path.basename(path)
        return basename == '.' or basename == '..'