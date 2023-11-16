"""
dyld emulation
"""
import os
from ctypes.macholib.framework import framework_info
from ctypes.macholib.dylib import dylib_info
from itertools import *
try:
    from _ctypes import _dyld_shared_cache_contains_path
except ImportError:

    def _dyld_shared_cache_contains_path(*args):
        if False:
            return 10
        raise NotImplementedError
__all__ = ['dyld_find', 'framework_find', 'framework_info', 'dylib_info']
DEFAULT_FRAMEWORK_FALLBACK = [os.path.expanduser('~/Library/Frameworks'), '/Library/Frameworks', '/Network/Library/Frameworks', '/System/Library/Frameworks']
DEFAULT_LIBRARY_FALLBACK = [os.path.expanduser('~/lib'), '/usr/local/lib', '/lib', '/usr/lib']

def dyld_env(env, var):
    if False:
        i = 10
        return i + 15
    if env is None:
        env = os.environ
    rval = env.get(var)
    if rval is None:
        return []
    return rval.split(':')

def dyld_image_suffix(env=None):
    if False:
        i = 10
        return i + 15
    if env is None:
        env = os.environ
    return env.get('DYLD_IMAGE_SUFFIX')

def dyld_framework_path(env=None):
    if False:
        print('Hello World!')
    return dyld_env(env, 'DYLD_FRAMEWORK_PATH')

def dyld_library_path(env=None):
    if False:
        while True:
            i = 10
    return dyld_env(env, 'DYLD_LIBRARY_PATH')

def dyld_fallback_framework_path(env=None):
    if False:
        i = 10
        return i + 15
    return dyld_env(env, 'DYLD_FALLBACK_FRAMEWORK_PATH')

def dyld_fallback_library_path(env=None):
    if False:
        for i in range(10):
            print('nop')
    return dyld_env(env, 'DYLD_FALLBACK_LIBRARY_PATH')

def dyld_image_suffix_search(iterator, env=None):
    if False:
        for i in range(10):
            print('nop')
    'For a potential path iterator, add DYLD_IMAGE_SUFFIX semantics'
    suffix = dyld_image_suffix(env)
    if suffix is None:
        return iterator

    def _inject(iterator=iterator, suffix=suffix):
        if False:
            print('Hello World!')
        for path in iterator:
            if path.endswith('.dylib'):
                yield (path[:-len('.dylib')] + suffix + '.dylib')
            else:
                yield (path + suffix)
            yield path
    return _inject()

def dyld_override_search(name, env=None):
    if False:
        while True:
            i = 10
    framework = framework_info(name)
    if framework is not None:
        for path in dyld_framework_path(env):
            yield os.path.join(path, framework['name'])
    for path in dyld_library_path(env):
        yield os.path.join(path, os.path.basename(name))

def dyld_executable_path_search(name, executable_path=None):
    if False:
        i = 10
        return i + 15
    if name.startswith('@executable_path/') and executable_path is not None:
        yield os.path.join(executable_path, name[len('@executable_path/'):])

def dyld_default_search(name, env=None):
    if False:
        print('Hello World!')
    yield name
    framework = framework_info(name)
    if framework is not None:
        fallback_framework_path = dyld_fallback_framework_path(env)
        for path in fallback_framework_path:
            yield os.path.join(path, framework['name'])
    fallback_library_path = dyld_fallback_library_path(env)
    for path in fallback_library_path:
        yield os.path.join(path, os.path.basename(name))
    if framework is not None and (not fallback_framework_path):
        for path in DEFAULT_FRAMEWORK_FALLBACK:
            yield os.path.join(path, framework['name'])
    if not fallback_library_path:
        for path in DEFAULT_LIBRARY_FALLBACK:
            yield os.path.join(path, os.path.basename(name))

def dyld_find(name, executable_path=None, env=None):
    if False:
        print('Hello World!')
    '\n    Find a library or framework using dyld semantics\n    '
    for path in dyld_image_suffix_search(chain(dyld_override_search(name, env), dyld_executable_path_search(name, executable_path), dyld_default_search(name, env)), env):
        if os.path.isfile(path):
            return path
        try:
            if _dyld_shared_cache_contains_path(path):
                return path
        except NotImplementedError:
            pass
    raise ValueError('dylib %s could not be found' % (name,))

def framework_find(fn, executable_path=None, env=None):
    if False:
        print('Hello World!')
    '\n    Find a framework using dyld semantics in a very loose manner.\n\n    Will take input such as:\n        Python\n        Python.framework\n        Python.framework/Versions/Current\n    '
    error = None
    try:
        return dyld_find(fn, executable_path=executable_path, env=env)
    except ValueError as e:
        error = e
    fmwk_index = fn.rfind('.framework')
    if fmwk_index == -1:
        fmwk_index = len(fn)
        fn += '.framework'
    fn = os.path.join(fn, os.path.basename(fn[:fmwk_index]))
    try:
        return dyld_find(fn, executable_path=executable_path, env=env)
    except ValueError:
        raise error
    finally:
        error = None

def test_dyld_find():
    if False:
        i = 10
        return i + 15
    env = {}
    assert dyld_find('libSystem.dylib') == '/usr/lib/libSystem.dylib'
    assert dyld_find('System.framework/System') == '/System/Library/Frameworks/System.framework/System'
if __name__ == '__main__':
    test_dyld_find()