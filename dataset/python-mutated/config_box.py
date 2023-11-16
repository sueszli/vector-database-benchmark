from dynaconf.vendor.box.box import Box

class ConfigBox(Box):
    """
    Modified box object to add object transforms.

    Allows for build in transforms like:

    cns = ConfigBox(my_bool='yes', my_int='5', my_list='5,4,3,3,2')

    cns.bool('my_bool') # True
    cns.int('my_int') # 5
    cns.list('my_list', mod=lambda x: int(x)) # [5, 4, 3, 3, 2]
    """
    _protected_keys = dir(Box) + ['bool', 'int', 'float', 'list', 'getboolean', 'getfloat', 'getint']

    def __getattr__(self, item):
        if False:
            i = 10
            return i + 15
        '\n        Config file keys are stored in lower case, be a little more\n        loosey goosey\n        '
        try:
            return super().__getattr__(item)
        except AttributeError:
            return super().__getattr__(item.lower())

    def __dir__(self):
        if False:
            print('Hello World!')
        return super().__dir__() + ['bool', 'int', 'float', 'list', 'getboolean', 'getfloat', 'getint']

    def bool(self, item, default=None):
        if False:
            return 10
        '\n        Return value of key as a boolean\n\n        :param item: key of value to transform\n        :param default: value to return if item does not exist\n        :return: approximated bool of value\n        '
        try:
            item = self.__getattr__(item)
        except AttributeError as err:
            if default is not None:
                return default
            raise err
        if isinstance(item, (bool, int)):
            return bool(item)
        if isinstance(item, str) and item.lower() in ('n', 'no', 'false', 'f', '0'):
            return False
        return True if item else False

    def int(self, item, default=None):
        if False:
            while True:
                i = 10
        '\n        Return value of key as an int\n\n        :param item: key of value to transform\n        :param default: value to return if item does not exist\n        :return: int of value\n        '
        try:
            item = self.__getattr__(item)
        except AttributeError as err:
            if default is not None:
                return default
            raise err
        return int(item)

    def float(self, item, default=None):
        if False:
            return 10
        '\n        Return value of key as a float\n\n        :param item: key of value to transform\n        :param default: value to return if item does not exist\n        :return: float of value\n        '
        try:
            item = self.__getattr__(item)
        except AttributeError as err:
            if default is not None:
                return default
            raise err
        return float(item)

    def list(self, item, default=None, spliter=',', strip=True, mod=None):
        if False:
            print('Hello World!')
        '\n        Return value of key as a list\n\n        :param item: key of value to transform\n        :param mod: function to map against list\n        :param default: value to return if item does not exist\n        :param spliter: character to split str on\n        :param strip: clean the list with the `strip`\n        :return: list of items\n        '
        try:
            item = self.__getattr__(item)
        except AttributeError as err:
            if default is not None:
                return default
            raise err
        if strip:
            item = item.lstrip('[').rstrip(']')
        out = [x.strip() if strip else x for x in item.split(spliter)]
        if mod:
            return list(map(mod, out))
        return out

    def getboolean(self, item, default=None):
        if False:
            i = 10
            return i + 15
        return self.bool(item, default)

    def getint(self, item, default=None):
        if False:
            return 10
        return self.int(item, default)

    def getfloat(self, item, default=None):
        if False:
            return 10
        return self.float(item, default)

    def __repr__(self):
        if False:
            return 10
        return '<ConfigBox: {0}>'.format(str(self.to_dict()))

    def copy(self):
        if False:
            print('Hello World!')
        return ConfigBox(super().copy())

    def __copy__(self):
        if False:
            print('Hello World!')
        return ConfigBox(super().copy())