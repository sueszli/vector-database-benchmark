import copy

class Configurable:
    global_defaults = {}

    def __init__(self, **config):
        if False:
            return 10
        self._variable_defaults = {}
        self._user_config = config

    def add_defaults(self, defaults):
        if False:
            for i in range(10):
                print('nop')
        'Add defaults to this object, overwriting any which already exist'
        self._variable_defaults.update(((d[0], copy.copy(d[1])) for d in defaults))

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        if name == '_variable_defaults':
            raise AttributeError
        (found, value) = self._find_default(name)
        if found:
            setattr(self, name, value)
            return value
        else:
            cname = self.__class__.__name__
            raise AttributeError('%s has no attribute: %s' % (cname, name))

    def _find_default(self, name):
        if False:
            return 10
        'Returns a tuple (found, value)'
        defaults = self._variable_defaults.copy()
        defaults.update(self.global_defaults)
        defaults.update(self._user_config)
        if name in defaults:
            return (True, defaults[name])
        else:
            return (False, None)

class ExtraFallback:
    """Adds another layer of fallback to attributes

    Used to look up a different attribute name
    """

    def __init__(self, name, fallback):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.hidden_attribute = '_' + name
        self.fallback = fallback

    def __get__(self, instance, owner=None):
        if False:
            i = 10
            return i + 15
        retval = getattr(instance, self.hidden_attribute, None)
        if retval is None:
            (_found, retval) = Configurable._find_default(instance, self.name)
        if retval is None:
            retval = getattr(instance, self.fallback, None)
        return retval

    def __set__(self, instance, value):
        if False:
            while True:
                i = 10
        'Set own value to a hidden attribute of the object'
        setattr(instance, self.hidden_attribute, value)