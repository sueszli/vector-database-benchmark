import os.path
from glob import glob
from typing import cast
import torch
from torch.types import Storage
__serialization_id_record_name__ = '.data/serialization_id'

class _HasStorage:

    def __init__(self, storage):
        if False:
            while True:
                i = 10
        self._storage = storage

    def storage(self):
        if False:
            i = 10
            return i + 15
        return self._storage

class DirectoryReader:
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the internal PyTorchFileReader class (which is used for
    accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """

    def __init__(self, directory):
        if False:
            for i in range(10):
                print('nop')
        self.directory = directory

    def get_record(self, name):
        if False:
            i = 10
            return i + 15
        filename = f'{self.directory}/{name}'
        with open(filename, 'rb') as f:
            return f.read()

    def get_storage_from_record(self, name, numel, dtype):
        if False:
            for i in range(10):
                print('nop')
        filename = f'{self.directory}/{name}'
        nbytes = torch._utils._element_size(dtype) * numel
        storage = cast(Storage, torch.UntypedStorage)
        return _HasStorage(storage.from_file(filename=filename, nbytes=nbytes))

    def has_record(self, path):
        if False:
            while True:
                i = 10
        full_path = os.path.join(self.directory, path)
        return os.path.isfile(full_path)

    def get_all_records(self):
        if False:
            return 10
        files = []
        for filename in glob(f'{self.directory}/**', recursive=True):
            if not os.path.isdir(filename):
                files.append(filename[len(self.directory) + 1:])
        return files

    def serialization_id(self):
        if False:
            while True:
                i = 10
        if self.has_record(__serialization_id_record_name__):
            return self.get_record(__serialization_id_record_name__)
        else:
            return ''