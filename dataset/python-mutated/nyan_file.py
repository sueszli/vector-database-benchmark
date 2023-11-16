"""
Nyan file struct that stores a bunch of objects and
manages imports.
"""
from __future__ import annotations
import typing
from .....nyan.nyan_structs import NyanObject
from .....util.ordered_set import OrderedSet
from ..data_definition import DataDefinition
if typing.TYPE_CHECKING:
    from openage.nyan.import_tree import ImportTree
FILE_VERSION = '0.2.0'

class NyanFile(DataDefinition):
    """
    Groups nyan objects into files. Contains methods for creating imports
    and dumping all objects into a human-readable .nyan file.
    """

    def __init__(self, targetdir: str, filename: str, modpack_name: str, nyan_objects: typing.Collection=None):
        if False:
            i = 10
            return i + 15
        super().__init__(targetdir, filename)
        self.modpack_name = modpack_name
        self.nyan_objects = OrderedSet()
        if nyan_objects:
            for nyan_object in nyan_objects:
                self.add_nyan_object(nyan_object)
        self.import_tree = None
        if len(targetdir) == 0 or targetdir == '/':
            self.fqon = (self.modpack_name, self.filename.split('.')[0])
        else:
            self.fqon = (self.modpack_name, *self.targetdir.replace('/', '.')[:-1].split('.'), self.filename.split('.')[0])

    def add_nyan_object(self, new_object: NyanObject) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Adds a nyan object to the file.\n        '
        if not isinstance(new_object, NyanObject):
            raise TypeError(f'nyan file cannot contain non-nyan object {new_object}')
        self.nyan_objects.add(new_object)
        new_fqon = (*self.fqon, new_object.get_name())
        new_object.set_fqon(new_fqon)

    def dump(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Returns the string that represents the nyan file.\n        '
        fileinfo_str = '# NYAN FILE\n'
        fileinfo_str += f'!version {FILE_VERSION}\n\n'
        import_str = ''
        objects_str = ''
        for nyan_object in self.nyan_objects:
            objects_str += nyan_object.dump(import_tree=self.import_tree)
        objects_str = objects_str[:-1]
        import_aliases = self.import_tree.get_alias_dict()
        import_files = self.import_tree.get_import_list()
        self.import_tree.clear_marks()
        if len(import_files) > 0:
            for fqon in import_files:
                import_str += 'import '
                import_str += '.'.join(fqon)
                import_str += '\n'
            import_str += '\n'
        if len(import_aliases) > 0:
            for (alias, fqon) in import_aliases.items():
                import_str += 'import '
                import_str += '.'.join(fqon)
                if len(alias) > 0:
                    import_str += f' as {alias}'
                import_str += '\n'
            import_str += '\n'
        output_str = fileinfo_str + import_str + objects_str
        return output_str

    def get_fqon(self) -> str:
        if False:
            return 10
        '\n        Return the fqon of the nyan file\n        '
        return self.fqon

    def get_relative_file_path(self) -> str:
        if False:
            return 10
        '\n        Relative path of the nyan file in the modpack.\n        '
        return f'{self.modpack_name}/{self.targetdir}{self.filename}'

    def set_import_tree(self, import_tree: ImportTree) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Sets the import tree of the file.\n        '
        self.import_tree = import_tree

    def set_filename(self, filename: str):
        if False:
            i = 10
            return i + 15
        super().set_filename(filename)
        self._reset_fqons()

    def set_modpack_name(self, modpack_name: str) -> None:
        if False:
            return 10
        '\n        Set the name of the modpack, the file is contained in.\n        '
        self.modpack_name = modpack_name

    def set_targetdir(self, targetdir: str) -> None:
        if False:
            while True:
                i = 10
        super().set_targetdir(targetdir)
        self._reset_fqons()

    def _reset_fqons(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Resets fqons, depending on the modpack name,\n        target directory and filename.\n        '
        for nyan_object in self.nyan_objects:
            new_fqon = (*self.fqon, nyan_object.get_name())
            nyan_object.set_fqon(new_fqon)
        self.fqon = (self.modpack_name, *self.targetdir.replace('/', '.')[:-1].split('.'), self.filename.split('.')[0])