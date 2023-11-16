import os
from trashcli.restore.file_system import RestoreWriteFileSystem, RestoreReadFileSystem
from trashcli.restore.trashed_file import TrashedFile

class Restorer:

    def __init__(self, read_fs, write_fs):
        if False:
            print('Hello World!')
        self.read_fs = read_fs
        self.write_fs = write_fs

    def restore_trashed_file(self, trashed_file, overwrite):
        if False:
            while True:
                i = 10
        '\n        If overwrite is enabled, then the restore functionality will overwrite an existing file\n        '
        if not overwrite and self.read_fs.path_exists(trashed_file.original_location):
            raise IOError('Refusing to overwrite existing file "%s".' % os.path.basename(trashed_file.original_location))
        else:
            parent = os.path.dirname(trashed_file.original_location)
            self.write_fs.mkdirs(parent)
        self.write_fs.move(trashed_file.original_file, trashed_file.original_location)
        self.write_fs.remove_file(trashed_file.info_file)