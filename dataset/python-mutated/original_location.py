import os
from trashcli.put.core.path_maker_type import PathMakerType
from trashcli.put.fs.fs import Fs

class OriginalLocation:

    def __init__(self, fs):
        if False:
            i = 10
            return i + 15
        self.fs = fs

    def for_file(self, path, path_maker_type, volume_top_dir):
        if False:
            i = 10
            return i + 15
        normalized_path = os.path.normpath(path)
        basename = os.path.basename(normalized_path)
        parent = self.fs.parent_realpath2(normalized_path)
        parent = self._calc_parent_path(parent, volume_top_dir, path_maker_type)
        return os.path.join(parent, basename)

    @staticmethod
    def _calc_parent_path(parent, volume_top_dir, path_maker_type):
        if False:
            return 10
        if path_maker_type == PathMakerType.AbsolutePaths:
            return parent
        if path_maker_type == PathMakerType.RelativePaths:
            if parent == volume_top_dir or parent.startswith(volume_top_dir + os.path.sep):
                parent = parent[len(volume_top_dir + os.path.sep):]
            return parent