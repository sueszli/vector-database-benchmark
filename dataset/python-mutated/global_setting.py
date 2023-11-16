from ctypes import *
import numpy as np
from .base import _Ctensor, _lib, _LiteCObjBase
from .network import *
from .struct import LiteDataType, LiteDeviceType, LiteIOType, Structure
from .tensor import *
LiteDecryptionFunc = CFUNCTYPE(c_size_t, c_void_p, c_size_t, POINTER(c_uint8), c_size_t, c_void_p)

class _GlobalAPI(_LiteCObjBase):
    """
    Get APIs from the lib
    """
    _api_ = [('LITE_get_device_count', [c_int, POINTER(c_size_t)]), ('LITE_try_coalesce_all_free_memory', []), ('LITE_register_decryption_and_key', [c_char_p, LiteDecryptionFunc, POINTER(c_uint8), c_size_t]), ('LITE_update_decryption_or_key', [c_char_p, c_void_p, POINTER(c_uint8), c_size_t]), ('LITE_set_loader_lib_path', [c_char_p]), ('LITE_set_persistent_cache', [c_char_p, c_int]), ('LITE_dump_persistent_cache', [c_char_p]), ('LITE_dump_tensor_rt_cache', [c_char_p]), ('LITE_register_memory_pair', [c_void_p, c_void_p, c_size_t, c_int, c_int]), ('LITE_clear_memory_pair', [c_void_p, c_void_p, c_int, c_int]), ('LITE_lookup_physic_ptr', [c_void_p, POINTER(c_void_p), c_int, c_int])]

def decryption_func(func):
    if False:
        for i in range(10):
            print('nop')
    'the decryption function decorator.\n    \n    .. note::\n\n       The function accept three array: ``in_arr``, ``key_arr`` and ``out_arr``.\n       If ``out_arr`` is None, just query the out array length in byte.\n    '

    @CFUNCTYPE(c_size_t, c_void_p, c_size_t, POINTER(c_uint8), c_size_t, c_void_p)
    def wrapper(c_in_data, in_length, c_key_data, key_length, c_out_data):
        if False:
            while True:
                i = 10
        in_arr = np.frombuffer(c_in_data, dtype=np.uint8, count=in_length)
        key_arr = np.frombuffer(c_key_data, dtype=np.uint8, count=key_length)
        if c_out_data:
            out_length = func(in_arr, None)
            out_arr = np.frombuffer(c_out_data, dtype=np.uint8, count=out_length)
            return func(in_arr, key_arr, out_arr)
        else:
            return func(in_arr, key_arr, None)
    return wrapper

class LiteGlobal(object):
    """
    Some global config in lite
    """
    _api = _GlobalAPI()._lib

    @staticmethod
    def register_decryption_and_key(decryption_name, decryption_func, key):
        if False:
            while True:
                i = 10
        'Register a custom decryption method and key to lite\n\n        Args:\n            decryption_name: the name of the decryption, which will act as the hash \n                key to find the decryption method.\n            decryption_func: the decryption function, which will decrypt the model with\n                the registered key, then return the decrypted model.\n                See :py:func:`~.decryption_func` for more details.\n            key: the decryption key of the method.\n        '
        c_name = c_char_p(decryption_name.encode('utf-8'))
        key_length = len(key)
        c_key = (c_uint8 * key_length)(*key)
        LiteGlobal._api.LITE_register_decryption_and_key(c_name, decryption_func, c_key, key_length)

    @staticmethod
    def update_decryption_key(decryption_name, key):
        if False:
            while True:
                i = 10
        'Update decryption key of a custom decryption method.\n\n        Args:\n            decrypt_name:  the name of the decryption, \n                which will act as the hash key to find the decryption method.\n            key:  the decryption key of the method,\n                if the length of key is zero, the key will not be updated.\n        '
        c_name = c_char_p(decryption_name.encode('utf-8'))
        key_length = len(key)
        c_key = (c_uint8 * key_length)(*key)
        LiteGlobal._api.LITE_update_decryption_or_key(c_name, None, c_key, key_length)

    @staticmethod
    def set_loader_lib_path(path):
        if False:
            return 10
        'Set the loader path to be used in lite.\n\n        Args:\n            path: the file path which store the loader library.\n        '
        c_path = c_char_p(path.encode('utf-8'))
        LiteGlobal._api.LITE_set_loader_lib_path(c_path)

    @staticmethod
    def set_persistent_cache(path, always_sync=False):
        if False:
            i = 10
            return i + 15
        'Set the algo policy cache file for CPU/CUDA,\n        the algo policy cache is produced by MegEngine fast-run.\n        \n        Args:\n            path: the file path which store the cache.\n            always_sync: always update the cache file when model runs.\n        '
        c_path = c_char_p(path.encode('utf-8'))
        LiteGlobal._api.LITE_set_persistent_cache(c_path, always_sync)

    @staticmethod
    def set_tensorrt_cache(path):
        if False:
            for i in range(10):
                print('nop')
        'Set the TensorRT engine cache path for serialized prebuilt ICudaEngine.\n\n        Args:\n            path: the cache file path to set\n        '
        c_path = c_char_p(path.encode('utf-8'))
        LiteGlobal._api.LITE_set_tensorrt_cache(c_path)

    @staticmethod
    def dump_persistent_cache(path):
        if False:
            i = 10
            return i + 15
        'Dump the PersistentCache policy cache to the specific file.\n        If the network is set to profile when forward, \n        though this the algo policy will dump to file.\n\n        Args:\n            path: the cache file path to be dumped.\n        '
        c_path = c_char_p(path.encode('utf-8'))
        LiteGlobal._api.LITE_dump_persistent_cache(c_path)

    @staticmethod
    def dump_tensorrt_cache():
        if False:
            for i in range(10):
                print('nop')
        'Dump the TensorRT cache to the file set in :py:func:`~.set_tensorrt_cache`.'
        LiteGlobal._api.LITE_dump_tensorrt_cache()

    @staticmethod
    def get_device_count(device_type):
        if False:
            i = 10
            return i + 15
        'Get the number of device of the given device type in current context.\n\n        Args:\n            device_type: the device type to be counted.\n\n        Returns:\n            the number of device.\n        '
        count = c_size_t()
        LiteGlobal._api.LITE_get_device_count(device_type, byref(count))
        return count.value

    @staticmethod
    def try_coalesce_all_free_memory():
        if False:
            for i in range(10):
                print('nop')
        'Try to coalesce all free memory in MegEngine.\n        When call it MegEnine Lite will try to free all the unused memory\n        thus decrease the runtime memory usage.\n        '
        LiteGlobal._api.LITE_try_coalesce_all_free_memory()

    @staticmethod
    def register_memory_pair(vir_ptr, phy_ptr, length, device, backend=LiteBackend.LITE_DEFAULT):
        if False:
            i = 10
            return i + 15
        'Register the physical and virtual address pair to the MegEngine,\n        some device need the map from physical to virtual.\n\n        Args:\n            vir_ptr: the virtual ptr to set to MegEngine.\n            phy_ptr: the physical ptr to set to MegEngine.\n            length: the length of bytes to set pair memory.\n            device: the the device to set the pair memory.\n            backend: the backend to set the pair memory\n\n        Return:\n            Whether the register operation is successful.\n        '
        assert isinstance(vir_ptr, c_void_p) and isinstance(phy_ptr, c_void_p), 'clear memory pair only accept c_void_p type.'
        LiteGlobal._api.LITE_register_memory_pair(vir_ptr, phy_ptr, length, device, backend)

    @staticmethod
    def clear_memory_pair(vir_ptr, phy_ptr, device, backend=LiteBackend.LITE_DEFAULT):
        if False:
            i = 10
            return i + 15
        'Clear the physical and virtual address pair in MegEngine.\n\n        Args:\n            vir_ptr: the virtual ptr to set to MegEngine.\n            phy_ptr: the physical ptr to set to MegEngine.\n            device: the the device to set the pair memory.\n            backend: the backend to set the pair memory.\n\n        Return:\n            Whether the clear is operation successful.\n        '
        assert isinstance(vir_ptr, c_void_p) and isinstance(phy_ptr, c_void_p), 'clear memory pair only accept c_void_p type.'
        LiteGlobal._api.LITE_clear_memory_pair(vir_ptr, phy_ptr, device, backend)

    @staticmethod
    def lookup_physic_ptr(vir_ptr, device, backend=LiteBackend.LITE_DEFAULT):
        if False:
            while True:
                i = 10
        'Get the physic address by the virtual address in MegEngine.\n\n        Args:\n            vir_ptr: the virtual ptr to set to MegEngine.\n            device: the the device to set the pair memory.\n            backend: the backend to set the pair memory.\n\n        Return:\n            The physic address to lookup.\n        '
        assert isinstance(vir_ptr, c_void_p), 'lookup physic ptr only accept c_void_p type.'
        mem = c_void_p()
        LiteGlobal._api.LITE_lookup_physic_ptr(vir_ptr, byref(mem), device, backend)
        return mem