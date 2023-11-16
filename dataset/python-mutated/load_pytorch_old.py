import pickle
import os
import io
import shutil
from zipfile import ZipFile
import jittor as jt
import numpy as np
from typing import Any, BinaryIO, cast, Dict, Optional, Type, Tuple, Union, IO, List
loaded_storages = {}
deserialized_objects = {}

def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
    if False:
        while True:
            i = 10
    if isinstance(bytes_str, bytes):
        return bytes_str.decode('ascii')
    return bytes_str

def load_tensor(contents, dtype, numel, key, location):
    if False:
        return 10
    name = os.path.join(prefix, 'data', str(key))
    loaded_storages[key] = contents.read_var(name, dtype)

def get_dtype_size(dtype):
    if False:
        print('Hello World!')
    dtype = dtype.__str__()
    if dtype == 'float32' or dtype == 'int32':
        return 4
    if dtype == 'float64' or dtype == 'int64':
        return 8
    if dtype == 'float16' or dtype == 'int16':
        return 2
    return 1

def persistent_load(saved_id):
    if False:
        while True:
            i = 10
    global contents
    assert isinstance(saved_id, tuple)
    typename = _maybe_decode_ascii(saved_id[0])
    data = saved_id[1:]
    assert typename == 'storage', f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
    (storage_type, key, location, numel) = data
    dtype = storage_type.dtype
    if key not in loaded_storages:
        nbytes = numel
        load_tensor(contents, dtype, nbytes, key, _maybe_decode_ascii(location))
    return loaded_storages[key]

def _dtype_to_storage_type_map():
    if False:
        i = 10
        return i + 15
    return {np.float16: 'HalfStorage', np.float32: 'FloatStorage', np.int64: 'LongStorage', np.int32: 'IntStorage', np.int16: 'ShortStorage', np.int8: 'CharStorage'}

def _storage_type_to_dtype_map():
    if False:
        for i in range(10):
            print('nop')
    dtype_map = {val: key for (key, val) in _dtype_to_storage_type_map().items()}
    return dtype_map

def _get_dtype_from_pickle_storage_type(pickle_storage_type: str):
    if False:
        for i in range(10):
            print('nop')
    try:
        return _storage_type_to_dtype_map()[pickle_storage_type]
    except KeyError:
        raise KeyError(f'pickle storage type "{pickle_storage_type}" is not recognized')

class StorageType:

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.dtype = _get_dtype_from_pickle_storage_type(name)

    def __str__(self):
        if False:
            return 10
        return f'StorageType(dtype={self.dtype})'

def jittor_rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    if False:
        for i in range(10):
            print('nop')
    if len(size) == 0:
        return jt.array(storage)
    record_size = np.prod(size)
    return jt.array(storage[:record_size]).reshape(size)

def jittor_rebuild_var(data, requires_grad, backward_hooks):
    if False:
        print('Hello World!')
    v = jt.array(data)
    v.requires_grad = requires_grad
    return v

class UnpicklerWrapper(pickle.Unpickler):

    def find_class(self, mod_name, name):
        if False:
            print('Hello World!')
        if type(name) is str and 'Storage' in name:
            try:
                return StorageType(name)
            except KeyError:
                pass
        if type(name) is str and '_rebuild_tensor_v2' in name:
            return super().find_class('jittor_utils.load_pytorch', 'jittor_rebuild')
        if type(name) is str and '_rebuild_parameter' in name:
            return super().find_class('jittor_utils.load_pytorch', 'jittor_rebuild_var')
        return super().find_class(mod_name, name)

class ArrayWrapper:

    def __init__(self, storage, stride=None, size=None, requires_grad=None):
        if False:
            i = 10
            return i + 15
        self.requires_grad = requires_grad
        self.size = size
        self.storage = storage
        self.stride = stride

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.storage.__str__()

def jittor_rebuild_direct(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    if False:
        for i in range(10):
            print('nop')
    if len(size) == 0:
        return ArrayWrapper(storage, stride=stride, size=size)
    storage.reshape(size)
    return ArrayWrapper(storage, stride=stride, size=size)

def jittor_rebuild_var_direct(data, requires_grad, backward_hooks):
    if False:
        while True:
            i = 10
    v = ArrayWrapper(storage, requires_grad=requires_grad)
    return v

class DirectUnpicklerWrapper(pickle.Unpickler):

    def find_class(self, mod_name, name):
        if False:
            i = 10
            return i + 15
        if type(name) is str and 'Storage' in name:
            try:
                return StorageType(name)
            except KeyError:
                pass
        if type(name) is str and '_rebuild_tensor_v2' in name:
            return super().find_class('jittor_utils.load_pytorch', 'jittor_rebuild_direct')
        if type(name) is str and '_rebuild_parameter' in name:
            return super().find_class('jittor_utils.load_pytorch', 'jittor_rebuild_var_direct')
        return super().find_class(mod_name, name)

def _check_seekable(f) -> bool:
    if False:
        i = 10
        return i + 15

    def raise_err_msg(patterns, e):
        if False:
            return 10
        for p in patterns:
            if p in str(e):
                msg = str(e) + '. You can only load from a file that is seekable.' + ' Please pre-load the data into a buffer like io.BytesIO and' + ' try to load from it instead.'
                raise type(e)(msg)
        raise e
    try:
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        raise_err_msg(['seek', 'tell'], e)
    return False

def extract_zip(input_zip):
    if False:
        print('Hello World!')
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def _is_compressed_file(f):
    if False:
        while True:
            i = 10
    compress_modules = ['gzip']
    try:
        return f.__module__ in compress_modules
    except AttributeError:
        return False

def _should_read_directly(f):
    if False:
        i = 10
        return i + 15
    if _is_compressed_file(f):
        return False
    try:
        return f.fileno() >= 0
    except io.UnsupportedOperation:
        return False
    except AttributeError:
        return False

def persistent_load_direct(saved_id):
    if False:
        print('Hello World!')
    global deserialized_objects
    assert isinstance(saved_id, tuple)
    typename = _maybe_decode_ascii(saved_id[0])
    data = saved_id[1:]
    if typename == 'module':
        return data[0]
    elif typename == 'storage':
        (data_type, root_key, location, size, view_metadata) = data
        location = _maybe_decode_ascii(location)
        if root_key not in deserialized_objects:
            deserialized_objects[root_key] = np.zeros(size, dtype=data_type)
        storage = deserialized_objects[root_key]
        if view_metadata is not None:
            (view_key, offset, view_size) = view_metadata
            if view_key not in deserialized_objects:
                deserialized_objects[view_key] = storage[offset:offset + view_size]
            return deserialized_objects[view_key]
        else:
            return storage
    else:
        raise RuntimeError('Unknown saved id type: %s' % saved_id[0])

def load_pytorch(fn_name):
    if False:
        i = 10
        return i + 15
    import jittor as jt
    global contents, deserialized_objects, loaded_storages, prefix
    loaded_storages = {}
    deserialized_objects = {}
    if not (fn_name.endswith('.pth') or fn_name.endswith('.pt') or fn_name.endswith('.bin')):
        print('This function is designed to load pytorch pth format files.')
        return None
    else:
        contents = jt.ZipFile(fn_name)
        if contents.valid():
            loaded_storages = {}
            deserialized_objects = {}
            for name in contents.list():
                if 'data.pkl' in name:
                    prefix = name[:-8]
                    break
            else:
                raise RuntimeError(f'zipfile <{fn_name}> format error, data.pkl not found')
            with jt.flag_scope(use_cuda=0):
                print('load??', fn_name)
                data_file = contents.read_var(prefix + 'data.pkl').data.tobytes()
                data_file = io.BytesIO(data_file)
                pickle_load_args = {'encoding': 'utf-8'}
                unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
                unpickler.persistent_load = persistent_load
                result = unpickler.load()
        else:
            deserialized_objects = {}
            f = open(fn_name, 'rb')
            f_should_read_directly = _should_read_directly(f)
            MAGIC_NUMBER = 119547037146038801333356
            PROTOCOL_VERSION = 1001
            pickle_load_args = {'encoding': 'utf-8'}
            magic_number = pickle.load(f, **pickle_load_args)
            if magic_number != MAGIC_NUMBER:
                raise RuntimeError('Invalid magic number; corrupt file?')
            protocol_version = pickle.load(f, **pickle_load_args)
            if PROTOCOL_VERSION != protocol_version:
                raise RuntimeError('Invalid protocal version.')
            _sys_info = pickle.load(f, **pickle_load_args)
            unpickler = DirectUnpicklerWrapper(f, **pickle_load_args)
            unpickler.persistent_load = persistent_load_direct
            result = unpickler.load()
            offset = f.tell() if f_should_read_directly else None
            deserialized_storage_keys = pickle.load(f, **pickle_load_args)
            f.read(8)
            for key in deserialized_storage_keys:
                assert key in deserialized_objects
                dtype = deserialized_objects[key].dtype
                size = deserialized_objects[key].size * get_dtype_size(dtype)
                byte_data = f.read(size)
                deserialized_objects[key][:] = np.frombuffer(byte_data, dtype).copy()
                f.read(8)
                if offset is not None:
                    offset = f.tell()
            for (key, params) in result.items():
                requires_grad = params.requires_grad
                shape = params.size
                result[key] = jt.array(params.storage)
                if shape is not None and len(shape) > 0:
                    if len(params.stride) > 1:
                        eval_list = []
                        for idx in range(len(params.stride)):
                            eval_list.append(f'@e0({idx}) * i{idx}')
                        evals = '+'.join(eval_list)
                        result[key] = result[key].reindex(params.size, [evals], extras=[jt.array(params.stride)])
                    else:
                        result[key] = result[key].reshape(shape)
                if requires_grad is not None:
                    result[key].requires_grad = requires_grad
        return result
if __name__ == '__main__':
    result = load_pytorch('van_base.pth')
    for (key, val) in result.items():
        print(key, val.shape)