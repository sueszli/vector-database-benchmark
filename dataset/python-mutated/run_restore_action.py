import os
from abc import ABCMeta, abstractmethod
import six
from typing import Optional, Iterable
from trashcli.restore.args import RunRestoreArgs
from trashcli.restore.sort_method import sort_files
from trashcli.restore.trashed_file import TrashedFile
from trashcli.restore.trashed_files import TrashedFiles

class RunRestoreAction:

    def __init__(self, handler, trashed_files):
        if False:
            return 10
        self.handler = handler
        self.trashed_files = trashed_files

    def run_action(self, args):
        if False:
            return 10
        trashed_files = self.all_files_trashed_from_path(args.path, args.trash_dir)
        trashed_files = sort_files(args.sort, trashed_files)
        self.handler.handle_trashed_files(trashed_files, args.overwrite)

    def all_files_trashed_from_path(self, path, trash_dir_from_cli):
        if False:
            print('Hello World!')
        for trashed_file in self.trashed_files.all_trashed_files(trash_dir_from_cli):
            if trashed_file.original_location_matches_path(path):
                yield trashed_file

@six.add_metaclass(ABCMeta)
class Handler:

    @abstractmethod
    def handle_trashed_files(self, trashed_files, overwrite):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

def original_location_matches_path(trashed_file_original_location, path):
    if False:
        for i in range(10):
            print('nop')
    if path == os.path.sep:
        return True
    if trashed_file_original_location.startswith(path + os.path.sep):
        return True
    return trashed_file_original_location == path