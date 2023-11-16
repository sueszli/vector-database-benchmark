from typing import Dict, List
from .glob_group import GlobGroup, GlobPattern
__all__ = ['Directory']

class Directory:
    """A file structure representation. Organized as Directory nodes that have lists of
    their Directory children. Directories for a package are created by calling
    :meth:`PackageImporter.file_structure`."""

    def __init__(self, name: str, is_dir: bool):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.is_dir = is_dir
        self.children: Dict[str, Directory] = {}

    def _get_dir(self, dirs: List[str]) -> 'Directory':
        if False:
            i = 10
            return i + 15
        'Builds path of Directories if not yet built and returns last directory\n        in list.\n\n        Args:\n            dirs (List[str]): List of directory names that are treated like a path.\n\n        Returns:\n            :class:`Directory`: The last Directory specified in the dirs list.\n        '
        if len(dirs) == 0:
            return self
        dir_name = dirs[0]
        if dir_name not in self.children:
            self.children[dir_name] = Directory(dir_name, True)
        return self.children[dir_name]._get_dir(dirs[1:])

    def _add_file(self, file_path: str):
        if False:
            print('Hello World!')
        'Adds a file to a Directory.\n\n        Args:\n            file_path (str): Path of file to add. Last element is added as a file while\n                other paths items are added as directories.\n        '
        (*dirs, file) = file_path.split('/')
        dir = self._get_dir(dirs)
        dir.children[file] = Directory(file, False)

    def has_file(self, filename: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if a file is present in a :class:`Directory`.\n\n        Args:\n            filename (str): Path of file to search for.\n        Returns:\n            bool: If a :class:`Directory` contains the specified file.\n        '
        lineage = filename.split('/', maxsplit=1)
        child = lineage[0]
        grandchildren = lineage[1] if len(lineage) > 1 else None
        if child in self.children.keys():
            if grandchildren is None:
                return True
            else:
                return self.children[child].has_file(grandchildren)
        return False

    def __str__(self):
        if False:
            while True:
                i = 10
        str_list: List[str] = []
        self._stringify_tree(str_list)
        return ''.join(str_list)

    def _stringify_tree(self, str_list: List[str], preamble: str='', dir_ptr: str='─── '):
        if False:
            return 10
        'Recursive method to generate print-friendly version of a Directory.'
        space = '    '
        branch = '│   '
        tee = '├── '
        last = '└── '
        str_list.append(f'{preamble}{dir_ptr}{self.name}\n')
        if dir_ptr == tee:
            preamble = preamble + branch
        else:
            preamble = preamble + space
        file_keys: List[str] = []
        dir_keys: List[str] = []
        for (key, val) in self.children.items():
            if val.is_dir:
                dir_keys.append(key)
            else:
                file_keys.append(key)
        for (index, key) in enumerate(sorted(dir_keys)):
            if index == len(dir_keys) - 1 and len(file_keys) == 0:
                self.children[key]._stringify_tree(str_list, preamble, last)
            else:
                self.children[key]._stringify_tree(str_list, preamble, tee)
        for (index, file) in enumerate(sorted(file_keys)):
            pointer = last if index == len(file_keys) - 1 else tee
            str_list.append(f'{preamble}{pointer}{file}\n')

def _create_directory_from_file_list(filename: str, file_list: List[str], include: 'GlobPattern'='**', exclude: 'GlobPattern'=()) -> Directory:
    if False:
        return 10
    'Return a :class:`Directory` file structure representation created from a list of files.\n\n    Args:\n        filename (str): The name given to the top-level directory that will be the\n            relative root for all file paths found in the file_list.\n\n        file_list (List[str]): List of files to add to the top-level directory.\n\n        include (Union[List[str], str]): An optional pattern that limits what is included from the file_list to\n            files whose name matches the pattern.\n\n        exclude (Union[List[str], str]): An optional pattern that excludes files whose name match the pattern.\n\n    Returns:\n            :class:`Directory`: a :class:`Directory` file structure representation created from a list of files.\n    '
    glob_pattern = GlobGroup(include, exclude=exclude, separator='/')
    top_dir = Directory(filename, True)
    for file in file_list:
        if glob_pattern.matches(file):
            top_dir._add_file(file)
    return top_dir