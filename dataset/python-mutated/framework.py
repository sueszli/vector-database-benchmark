__all__ = ['set_default_framework', 'get_default_framework', 'shortcut_module', 'shortcut_framework']
import importlib
import os
import sys
from typing import Optional, cast
from typing_extensions import Literal
framework_type = Literal['pytorch', 'tensorflow', 'mxnet', 'none']
'Supported framework types.'
ENV_NNI_FRAMEWORK = 'NNI_FRAMEWORK'

def framework_from_env() -> framework_type:
    if False:
        return 10
    framework = os.getenv(ENV_NNI_FRAMEWORK, 'pytorch')
    if framework not in framework_type.__args__:
        raise ValueError(f'{framework} does not belong to {framework_type.__args__}')
    return cast(framework_type, framework)
DEFAULT_FRAMEWORK = framework_from_env()

def set_default_framework(framework: framework_type) -> None:
    if False:
        i = 10
        return i + 15
    'Set default deep learning framework to simplify imports.\n\n    Some functionalities in NNI (e.g., NAS / Compression), relies on an underlying DL framework.\n    For different DL frameworks, the implementation of NNI can be very different.\n    Thus, users need import things tailored for their own framework. For example: ::\n\n        from nni.nas.xxx.pytorch import yyy\n\n    rather than: ::\n\n        from nni.nas.xxx import yyy\n\n    By setting a default framework, shortcuts will be made. As such ``nni.nas.xxx`` will be equivalent to ``nni.nas.xxx.pytorch``.\n\n    Another way to setting it is through environment variable ``NNI_FRAMEWORK``,\n    which needs to be set before the whole process starts.\n\n    If you set the framework with :func:`set_default_framework`,\n    it should be done before all imports (except nni itself) happen,\n    because it will affect other import\'s behaviors.\n    And the behavior is undefined if the framework is "re"-set in the middle.\n\n    The supported frameworks here are listed below.\n    It doesn\'t mean that they are fully supported by NAS / Compression in NNI.\n\n    * ``pytorch`` (default)\n    * ``tensorflow``\n    * ``mxnet``\n    * ``none`` (to disable the shortcut-import behavior).\n\n    Examples\n    --------\n    >>> import nni\n    >>> nni.set_default_framework(\'tensorflow\')\n    >>> # then other imports\n    >>> from nni.nas.xxx import yyy\n    '
    if framework is None:
        framework = 'none'
    global DEFAULT_FRAMEWORK
    DEFAULT_FRAMEWORK = framework

def get_default_framework() -> framework_type:
    if False:
        return 10
    'Retrieve default deep learning framework set either with env variables or manually.'
    return DEFAULT_FRAMEWORK

def shortcut_module(current: str, target: str, package: Optional[str]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Make ``current`` module an alias of ``target`` module in ``package``.'
    mod = importlib.import_module(target, package)
    thismod = sys.modules[current]
    for (api, obj) in mod.__dict__.items():
        setattr(thismod, api, obj)

def shortcut_framework(current: str) -> None:
    if False:
        print('Hello World!')
    'Make ``current`` a shortcut of ``current.framework``.'
    if get_default_framework() != 'none':
        shortcut_module(current, '.' + get_default_framework(), current)