import torch
from torch.serialization import StorageType
import pickle
import zipfile
import io
from typing import Dict, IO, Any, Callable, List
from dataclasses import dataclass
from .common import invalidInputError
item_size = {torch.bfloat16: 2, torch.float16: 2, torch.int: 4, torch.float: 4, torch.float32: 4, torch.int8: 1}

@dataclass
class LazyStorage:
    load: Callable[[int, int], torch.Tensor]
    kind: StorageType
    description: str

@dataclass
class LazyTensor:
    _load: Callable[[], torch.Tensor]
    shape: List[int]
    data_type: torch.dtype
    description: str

    def load(self) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        ret = self._load()
        return ret

    def to(self, data_type):
        if False:
            return 10

        def load() -> torch.Tensor:
            if False:
                print('Hello World!')
            print(f'to {data_type}')
            return self.load().to(data_type)
        return LazyTensor(load, self.shape, data_type, f'convert({data_type}) {self.description}')

def _load(pickle_fp, map_location, picklemoudle, pickle_file='data.pkl', zip_file=None):
    if False:
        return 10
    load_module_mapping: Dict[str, str] = {'torch.tensor': 'torch._tensor'}

    class LazyUnpickler(picklemoudle.Unpickler):

        def __init__(self, fp: IO[bytes], data_base_path: str, zip_file: zipfile.ZipFile):
            if False:
                while True:
                    i = 10
            super().__init__(fp)
            self.data_base_path = data_base_path
            self.zip_file = zip_file

        def persistent_load(self, pid):
            if False:
                return 10
            data_type = pid[1].dtype
            filename_stem = pid[2]
            filename = f'{self.data_base_path}/{filename_stem}'
            info = self.zip_file.getinfo(filename)

            def load(offset: int, elm_count: int):
                if False:
                    i = 10
                    return i + 15
                dtype = data_type
                fp = self.zip_file.open(info)
                fp.seek(offset * item_size[dtype])
                size = elm_count * item_size[dtype]
                data = fp.read(size)
                return torch.frombuffer(bytearray(data), dtype=dtype)
            description = f'storage data_type={data_type} path-in-zip={{filename}} path={{self.zip_file.filename}}'
            return LazyStorage(load=load, kind=pid[1], description=description)

        @staticmethod
        def lazy_rebuild_tensor_v2(storage: Any, storage_offset: Any, size: Any, stride: Any, requires_grad: Any, backward_hooks: Any, metadata: Any=None) -> LazyTensor:
            if False:
                print('Hello World!')
            invalidInputError(isinstance(storage, LazyStorage), f'storage should be an instance of class `LazyStorage`, but get {type(storage)}.')

            def load() -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                elm_count = stride[0] * size[0]
                return storage.load(storage_offset, elm_count).reshape(size)
            description = f'pickled storage_offset={storage_offset} in {storage.description}'
            return LazyTensor(load, list(size), storage.kind.dtype, description)

        @staticmethod
        def rebuild_from_type_v2(func, new_type, args, state):
            if False:
                while True:
                    i = 10
            return func(*args)
        CLASSES: dict[tuple[str, str], Any] = {('torch._tensor', '_rebuild_from_type_v2'): getattr(rebuild_from_type_v2, '__func__'), ('torch._utils', '_rebuild_tensor_v2'): getattr(lazy_rebuild_tensor_v2, '__func__'), ('torch', 'Tensor'): LazyTensor}

        def find_class(self, mod_name, name):
            if False:
                i = 10
                return i + 15
            if (mod_name, name) in self.CLASSES:
                return self.CLASSES[mod_name, name]
            if type(name) is str and 'Storage' in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            mod_name = load_module_mapping.get(mod_name, mod_name)
            return super().find_class(mod_name, name)
    unpickler = LazyUnpickler(pickle_fp, data_base_path=pickle_file, zip_file=zip_file)
    result = unpickler.load()
    return result

def lazyload(f, *args, **kwargs):
    if False:
        return 10
    if isinstance(f, io.BufferedIOBase):
        fp = f
    else:
        fp = open(f, 'rb')
    zf = zipfile.ZipFile(fp)
    pickle_paths = [name for name in zf.namelist() if name.endswith('.pkl')]
    invalidInputError(len(pickle_paths) == 1, f'There should be only one pickle_paths found, but get {pickle_paths}. ')
    pickle_fp = zf.open(pickle_paths[0], 'r')
    state_dict = _load(pickle_fp, None, pickle, pickle_file=pickle_paths[0][:-4], zip_file=zf)
    fp.close()
    return state_dict

class LazyLoadTensors:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.torch_load = torch.load

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        torch.load = lazyload

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            while True:
                i = 10
        torch.load = self.torch_load