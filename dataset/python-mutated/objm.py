import os
import pickle
import tempfile
from pathlib import Path
from qlib.config import C

class ObjManager:

    def save_obj(self, obj: object, name: str):
        if False:
            i = 10
            return i + 15
        '\n        save obj as name\n\n        Parameters\n        ----------\n        obj : object\n            object to be saved\n        name : str\n            name of the object\n        '
        raise NotImplementedError(f'Please implement `save_obj`')

    def save_objs(self, obj_name_l):
        if False:
            return 10
        '\n        save objects\n\n        Parameters\n        ----------\n        obj_name_l : list of <obj, name>\n        '
        raise NotImplementedError(f'Please implement the `save_objs` method')

    def load_obj(self, name: str) -> object:
        if False:
            for i in range(10):
                print('nop')
        '\n        load object by name\n\n        Parameters\n        ----------\n        name : str\n            the name of the object\n\n        Returns\n        -------\n        object:\n            loaded object\n        '
        raise NotImplementedError(f'Please implement the `load_obj` method')

    def exists(self, name: str) -> bool:
        if False:
            print('Hello World!')
        '\n        if the object named `name` exists\n\n        Parameters\n        ----------\n        name : str\n            name of the objecT\n\n        Returns\n        -------\n        bool:\n            If the object exists\n        '
        raise NotImplementedError(f'Please implement the `exists` method')

    def list(self) -> list:
        if False:
            return 10
        '\n        list the objects\n\n        Returns\n        -------\n        list:\n            the list of returned objects\n        '
        raise NotImplementedError(f'Please implement the `list` method')

    def remove(self, fname=None):
        if False:
            i = 10
            return i + 15
        'remove.\n\n        Parameters\n        ----------\n        fname :\n            if file name is provided. specific file is removed\n            otherwise, The all the objects will be removed.\n        '
        raise NotImplementedError(f'Please implement the `remove` method')

class FileManager(ObjManager):
    """
    Use file system to manage objects
    """

    def __init__(self, path=None):
        if False:
            i = 10
            return i + 15
        if path is None:
            self.path = Path(self.create_path())
        else:
            self.path = Path(path).resolve()

    def create_path(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        try:
            return tempfile.mkdtemp(prefix=str(C['file_manager_path']) + os.sep)
        except AttributeError as attribute_e:
            raise NotImplementedError(f'If path is not given, the `create_path` function should be implemented') from attribute_e

    def save_obj(self, obj, name):
        if False:
            return 10
        with (self.path / name).open('wb') as f:
            pickle.dump(obj, f, protocol=C.dump_protocol_version)

    def save_objs(self, obj_name_l):
        if False:
            print('Hello World!')
        for (obj, name) in obj_name_l:
            self.save_obj(obj, name)

    def load_obj(self, name):
        if False:
            while True:
                i = 10
        with (self.path / name).open('rb') as f:
            return pickle.load(f)

    def exists(self, name):
        if False:
            print('Hello World!')
        return (self.path / name).exists()

    def list(self):
        if False:
            for i in range(10):
                print('nop')
        return list(self.path.iterdir())

    def remove(self, fname=None):
        if False:
            for i in range(10):
                print('nop')
        if fname is None:
            for fp in self.path.glob('*'):
                fp.unlink()
            self.path.rmdir()
        else:
            (self.path / fname).unlink()