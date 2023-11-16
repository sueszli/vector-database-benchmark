class Override(object):

    def __init__(self, obj_mapping, wrapped_function):
        if False:
            print('Hello World!')
        self._obj_mapping = obj_mapping
        self._wrapped = wrapped_function

    @property
    def obj_mapping(self):
        if False:
            i = 10
            return i + 15
        return self._obj_mapping

    @property
    def func(self):
        if False:
            i = 10
            return i + 15
        return self._wrapped

class AttrOverride(Override):

    def __init__(self, is_setattr, obj_mapping, wrapped_function):
        if False:
            while True:
                i = 10
        super(AttrOverride, self).__init__(obj_mapping, wrapped_function)
        self._is_setattr = is_setattr

    @property
    def is_setattr(self):
        if False:
            print('Hello World!')
        return self._is_setattr

class LocalOverride(Override):
    pass

class LocalAttrOverride(AttrOverride):
    pass

class RemoteOverride(Override):
    pass

class RemoteAttrOverride(AttrOverride):
    pass

def local_override(obj_mapping):
    if False:
        while True:
            i = 10
    if not isinstance(obj_mapping, dict):
        raise ValueError('@local_override takes a dictionary: <class name> -> [<overridden method>]')

    def _wrapped(func):
        if False:
            print('Hello World!')
        return LocalOverride(obj_mapping, func)
    return _wrapped

def local_getattr_override(obj_mapping):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(obj_mapping, dict):
        raise ValueError('@local_getattr_override takes a dictionary: <class name> -> [<overridden attribute>]')

    def _wrapped(func):
        if False:
            for i in range(10):
                print('nop')
        return LocalAttrOverride(False, obj_mapping, func)
    return _wrapped

def local_setattr_override(obj_mapping):
    if False:
        return 10
    if not isinstance(obj_mapping, dict):
        raise ValueError('@local_setattr_override takes a dictionary: <class name> -> [<overridden attribute>]')

    def _wrapped(func):
        if False:
            return 10
        return LocalAttrOverride(True, obj_mapping, func)
    return _wrapped

def remote_override(obj_mapping):
    if False:
        while True:
            i = 10
    if not isinstance(obj_mapping, dict):
        raise ValueError('@remote_override takes a dictionary: <class name> -> [<overridden method>]')

    def _wrapped(func):
        if False:
            while True:
                i = 10
        return RemoteOverride(obj_mapping, func)
    return _wrapped

def remote_getattr_override(obj_mapping):
    if False:
        return 10
    if not isinstance(obj_mapping, dict):
        raise ValueError('@remote_getattr_override takes a dictionary: <class name> -> [<overridden attribute>]')

    def _wrapped(func):
        if False:
            i = 10
            return i + 15
        return RemoteAttrOverride(False, obj_mapping, func)
    return _wrapped

def remote_setattr_override(obj_mapping):
    if False:
        return 10
    if not isinstance(obj_mapping, dict):
        raise ValueError('@remote_setattr_override takes a dictionary: <class name> -> [<overridden attribute>]')

    def _wrapped(func):
        if False:
            for i in range(10):
                print('nop')
        return RemoteAttrOverride(True, obj_mapping, func)
    return _wrapped

class LocalException(object):

    def __init__(self, class_path, wrapped_class):
        if False:
            return 10
        self._class_path = class_path
        self._class = wrapped_class

    @property
    def class_path(self):
        if False:
            i = 10
            return i + 15
        return self._class_path

    @property
    def wrapped_class(self):
        if False:
            i = 10
            return i + 15
        return self._class

class RemoteExceptionSerializer(object):

    def __init__(self, class_path, serializer):
        if False:
            for i in range(10):
                print('nop')
        self._class_path = class_path
        self._serializer = serializer

    @property
    def class_path(self):
        if False:
            i = 10
            return i + 15
        return self._class_path

    @property
    def serializer(self):
        if False:
            i = 10
            return i + 15
        return self._serializer

def local_exception(class_path):
    if False:
        while True:
            i = 10

    def _wrapped(cls):
        if False:
            while True:
                i = 10
        return LocalException(class_path, cls)
    return _wrapped

def remote_exception_serialize(class_path):
    if False:
        print('Hello World!')

    def _wrapped(func):
        if False:
            print('Hello World!')
        return RemoteExceptionSerializer(class_path, func)
    return _wrapped