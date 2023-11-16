from typing import NamedTuple, Iterable, Optional
from trashcli.restore.info_files import InfoFiles
from trashcli.restore.trash_directories import TrashDirectories

class InfoDirSearcher:

    def __init__(self, trash_directories, info_files):
        if False:
            for i in range(10):
                print('nop')
        self.trash_directories = trash_directories
        self.info_files = info_files

    def all_file_in_info_dir(self, trash_dir_from_cli):
        if False:
            print('Hello World!')
        for (trash_dir_path, volume) in self.trash_directories.list_trash_dirs(trash_dir_from_cli):
            for (type, path) in self.info_files.all_info_files(trash_dir_path):
                yield FileFound(type, path, volume)

class FileFound(NamedTuple('Info', [('type', 'str'), ('path', 'str'), ('volume', 'str')])):
    pass