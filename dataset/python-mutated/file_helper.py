import io
from ditk import logging
import os
import pickle
import time
from functools import lru_cache
from typing import Union
import torch
from .import_helper import try_import_ceph, try_import_redis, try_import_rediscluster, try_import_mc
from .lock_helper import get_file_lock
_memcached = None
_redis_cluster = None
if os.environ.get('DI_STORE', 'off').lower() == 'on':
    print('Enable DI-store')
    from di_store import Client
    di_store_config_path = os.environ.get('DI_STORE_CONFIG_PATH', './di_store.yaml')
    di_store_client = Client(di_store_config_path)

    def save_to_di_store(data):
        if False:
            while True:
                i = 10
        return di_store_client.put(data)

    def read_from_di_store(object_ref):
        if False:
            print('Hello World!')
        data = di_store_client.get(object_ref)
        di_store_client.delete(object_ref)
        return data
else:
    save_to_di_store = read_from_di_store = None

@lru_cache()
def get_ceph_package():
    if False:
        while True:
            i = 10
    return try_import_ceph()

@lru_cache()
def get_redis_package():
    if False:
        for i in range(10):
            print('nop')
    return try_import_redis()

@lru_cache()
def get_rediscluster_package():
    if False:
        return 10
    return try_import_rediscluster()

@lru_cache()
def get_mc_package():
    if False:
        return 10
    return try_import_mc()

def read_from_ceph(path: str) -> object:
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Read file from ceph\n    Arguments:\n        - path (:obj:`str`): File path in ceph, start with ``"s3://"``\n    Returns:\n        - (:obj:`data`): Deserialized data\n    '
    value = get_ceph_package().Get(path)
    if not value:
        raise FileNotFoundError("File({}) doesn't exist in ceph".format(path))
    return pickle.loads(value)

@lru_cache()
def _get_redis(host='localhost', port=6379):
    if False:
        return 10
    '\n    Overview:\n        Ensures redis usage\n    Arguments:\n        - host (:obj:`str`): Host string\n        - port (:obj:`int`): Port number\n    Returns:\n        - (:obj:`Redis(object)`): Redis object with given ``host``, ``port``, and ``db=0``\n    '
    return get_redis_package().StrictRedis(host=host, port=port, db=0)

def read_from_redis(path: str) -> object:
    if False:
        print('Hello World!')
    '\n    Overview:\n        Read file from redis\n    Arguments:\n        - path (:obj:`str`): Dile path in redis, could be a string key\n    Returns:\n        - (:obj:`data`): Deserialized data\n    '
    return pickle.loads(_get_redis().get(path))

def _ensure_rediscluster(startup_nodes=[{'host': '127.0.0.1', 'port': '7000'}]):
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Ensures redis usage\n    Arguments:\n        - List of startup nodes (:obj:`dict`) of\n            - host (:obj:`str`): Host string\n            - port (:obj:`int`): Port number\n    Returns:\n        - (:obj:`RedisCluster(object)`): RedisCluster object with given ``host``, ``port``,             and ``False`` for ``decode_responses`` in default.\n    '
    global _redis_cluster
    if _redis_cluster is None:
        _redis_cluster = get_rediscluster_package().RedisCluster(startup_nodes=startup_nodes, decode_responses=False)
    return

def read_from_rediscluster(path: str) -> object:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        Read file from rediscluster\n    Arguments:\n        - path (:obj:`str`): Dile path in rediscluster, could be a string key\n    Returns:\n        - (:obj:`data`): Deserialized data\n    '
    _ensure_rediscluster()
    value_bytes = _redis_cluster.get(path)
    value = pickle.loads(value_bytes)
    return value

def read_from_file(path: str) -> object:
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Read file from local file system\n    Arguments:\n        - path (:obj:`str`): File path in local file system\n    Returns:\n        - (:obj:`data`): Deserialized data\n    '
    with open(path, 'rb') as f:
        value = pickle.load(f)
    return value

def _ensure_memcached():
    if False:
        while True:
            i = 10
    "\n    Overview:\n        Ensures memcache usage\n    Returns:\n        - (:obj:`MemcachedClient instance`): MemcachedClient's class instance built with current             memcached_client's ``server_list.conf`` and ``client.conf`` files\n    "
    global _memcached
    if _memcached is None:
        server_list_config_file = '/mnt/lustre/share/memcached_client/server_list.conf'
        client_config_file = '/mnt/lustre/share/memcached_client/client.conf'
        _memcached = get_mc_package().MemcachedClient.GetInstance(server_list_config_file, client_config_file)
    return

def read_from_mc(path: str, flush=False) -> object:
    if False:
        return 10
    '\n    Overview:\n        Read file from memcache, file must be saved by `torch.save()`\n    Arguments:\n        - path (:obj:`str`): File path in local system\n    Returns:\n        - (:obj:`data`): Deserialized data\n    '
    _ensure_memcached()
    while True:
        try:
            value = get_mc_package().pyvector()
            if flush:
                _memcached.Get(path, value, get_mc_package().MC_READ_THROUGH)
                return
            else:
                _memcached.Get(path, value)
            value_buf = get_mc_package().ConvertBuffer(value)
            value_str = io.BytesIO(value_buf)
            value_str = torch.load(value_str, map_location='cpu')
            return value_str
        except Exception:
            print('read mc failed, retry...')
            time.sleep(0.01)

def read_from_path(path: str):
    if False:
        return 10
    '\n    Overview:\n        Read file from ceph\n    Arguments:\n        - path (:obj:`str`): File path in ceph, start with ``"s3://"``, or use local file system\n    Returns:\n        - (:obj:`data`): Deserialized data\n    '
    if get_ceph_package() is None:
        logging.info('You do not have ceph installed! Loading local file! If you are not testing locally, something is wrong!')
        return read_from_file(path)
    else:
        return read_from_ceph(path)

def save_file_ceph(path, data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Save pickle dumped data file to ceph\n    Arguments:\n        - path (:obj:`str`): File path in ceph, start with ``"s3://"``, use file system when not\n        - data (:obj:`Any`): Could be dict, list or tensor etc.\n    '
    data = pickle.dumps(data)
    save_path = os.path.dirname(path)
    file_name = os.path.basename(path)
    ceph = get_ceph_package()
    if ceph is not None:
        if hasattr(ceph, 'save_from_string'):
            ceph.save_from_string(save_path, file_name, data)
        elif hasattr(ceph, 'put'):
            ceph.put(os.path.join(save_path, file_name), data)
        else:
            raise RuntimeError('ceph can not save file, check your ceph installation')
    else:
        size = len(data)
        if save_path == 'do_not_save':
            logging.info('You do not have ceph installed! ignored file {} of size {}!'.format(file_name, size) + ' If you are not testing locally, something is wrong!')
            return
        p = os.path.join(save_path, file_name)
        with open(p, 'wb') as f:
            logging.info('You do not have ceph installed! Saving as local file at {} of size {}!'.format(p, size) + ' If you are not testing locally, something is wrong!')
            f.write(data)

def save_file_redis(path, data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Save pickle dumped data file to redis\n    Arguments:\n        - path (:obj:`str`): File path (could be a string key) in redis\n        - data (:obj:`Any`): Could be dict, list or tensor etc.\n    '
    _get_redis().set(path, pickle.dumps(data))

def save_file_rediscluster(path, data):
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Save pickle dumped data file to rediscluster\n    Arguments:\n        - path (:obj:`str`): File path (could be a string key) in redis\n        - data (:obj:`Any`): Could be dict, list or tensor etc.\n    '
    _ensure_rediscluster()
    data = pickle.dumps(data)
    _redis_cluster.set(path, data)
    return

def read_file(path: str, fs_type: Union[None, str]=None, use_lock: bool=False) -> object:
    if False:
        print('Hello World!')
    "\n    Overview:\n        Read file from path\n    Arguments:\n        - path (:obj:`str`): The path of file to read\n        - fs_type (:obj:`str` or :obj:`None`): The file system type, support ``{'normal', 'ceph'}``\n        - use_lock (:obj:`bool`): Whether ``use_lock`` is in local normal file system\n    "
    if fs_type is None:
        if path.lower().startswith('s3'):
            fs_type = 'ceph'
        elif get_mc_package() is not None:
            fs_type = 'mc'
        else:
            fs_type = 'normal'
    assert fs_type in ['normal', 'ceph', 'mc']
    if fs_type == 'ceph':
        data = read_from_path(path)
    elif fs_type == 'normal':
        if use_lock:
            with get_file_lock(path, 'read'):
                data = torch.load(path, map_location='cpu')
        else:
            data = torch.load(path, map_location='cpu')
    elif fs_type == 'mc':
        data = read_from_mc(path)
    return data

def save_file(path: str, data: object, fs_type: Union[None, str]=None, use_lock: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    "\n    Overview:\n        Save data to file of path\n    Arguments:\n        - path (:obj:`str`): The path of file to save to\n        - data (:obj:`object`): The data to save\n        - fs_type (:obj:`str` or :obj:`None`): The file system type, support ``{'normal', 'ceph'}``\n        - use_lock (:obj:`bool`): Whether ``use_lock`` is in local normal file system\n    "
    if fs_type is None:
        if path.lower().startswith('s3'):
            fs_type = 'ceph'
        elif get_mc_package() is not None:
            fs_type = 'mc'
        else:
            fs_type = 'normal'
    assert fs_type in ['normal', 'ceph', 'mc']
    if fs_type == 'ceph':
        save_file_ceph(path, data)
    elif fs_type == 'normal':
        if use_lock:
            with get_file_lock(path, 'write'):
                torch.save(data, path)
        else:
            torch.save(data, path)
    elif fs_type == 'mc':
        torch.save(data, path)
        read_from_mc(path, flush=True)

def remove_file(path: str, fs_type: Union[None, str]=None) -> None:
    if False:
        i = 10
        return i + 15
    "\n    Overview:\n        Remove file\n    Arguments:\n        - path (:obj:`str`): The path of file you want to remove\n        - fs_type (:obj:`str` or :obj:`None`): The file system type, support ``{'normal', 'ceph'}``\n    "
    if fs_type is None:
        fs_type = 'ceph' if path.lower().startswith('s3') else 'normal'
    assert fs_type in ['normal', 'ceph']
    if fs_type == 'ceph':
        os.popen('aws s3 rm --recursive {}'.format(path))
    elif fs_type == 'normal':
        os.popen('rm -rf {}'.format(path))