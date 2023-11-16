from trashcli.rm.file_remover import FileRemover
from trashcli.lib.path_of_backup_copy import path_of_backup_copy

class CleanableTrashcan:

    def __init__(self, file_remover):
        if False:
            for i in range(10):
                print('nop')
        self._file_remover = file_remover

    def delete_trash_info_and_backup_copy(self, trash_info_path):
        if False:
            i = 10
            return i + 15
        backup_copy = path_of_backup_copy(trash_info_path)
        self._file_remover.remove_file_if_exists(backup_copy)
        self._file_remover.remove_file2(trash_info_path)