import importlib
from typing import List
import ding
from .default_helper import one_time_warning

def try_import_ceph():
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Try import ceph module, if failed, return ``None``\n\n    Returns:\n        - (:obj:`Module`): Imported module, or ``None`` when ceph not found\n    '
    try:
        import ceph
        client = ceph.S3Client()
        return client
    except ModuleNotFoundError as e:
        try:
            from petrel_client.client import Client
            client = Client(conf_path='~/petreloss.conf')
            return client
        except ModuleNotFoundError as e:
            one_time_warning('You have not installed ceph package! DI-engine has changed to some alternatives.')
            ceph = None
            return ceph

def try_import_mc():
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Try import mc module, if failed, return ``None``\n\n    Returns:\n        - (:obj:`Module`): Imported module, or ``None`` when mc not found\n    '
    try:
        import mc
    except ModuleNotFoundError as e:
        mc = None
    return mc

def try_import_redis():
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Try import redis module, if failed, return ``None``\n\n    Returns:\n        - (:obj:`Module`): Imported module, or ``None`` when redis not found\n    '
    try:
        import redis
    except ModuleNotFoundError as e:
        one_time_warning('You have not installed redis package! DI-engine has changed to some alternatives.')
        redis = None
    return redis

def try_import_rediscluster():
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Try import rediscluster module, if failed, return ``None``\n\n    Returns:\n        - (:obj:`Module`): Imported module, or ``None`` when rediscluster not found\n    '
    try:
        import rediscluster
    except ModuleNotFoundError as e:
        one_time_warning('You have not installed rediscluster package! DI-engine has changed to some alternatives.')
        rediscluster = None
    return rediscluster

def try_import_link():
    if False:
        return 10
    '\n    Overview:\n        Try import linklink module, if failed, import ding.tests.fake_linklink instead\n\n    Returns:\n        - (:obj:`Module`): Imported module (may be ``fake_linklink``)\n    '
    if ding.enable_linklink:
        try:
            import linklink as link
        except ModuleNotFoundError as e:
            one_time_warning('You have not installed linklink package! DI-engine has changed to some alternatives.')
            from .fake_linklink import link
    else:
        from .fake_linklink import link
    return link

def import_module(modules: List[str]) -> None:
    if False:
        return 10
    '\n    Overview:\n        Import several module as a list\n    Arguments:\n        - (:obj:`str list`): List of module names\n    '
    for name in modules:
        importlib.import_module(name)