"""
AutoId module
"""
import inspect
import uuid

class AutoId:
    """
    Generates unique ids.
    """

    def __init__(self, method=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a unique id generator.\n\n        Args:\n            method: generation method - supports int sequence (default) or UUID function\n        '
        (self.method, self.function, self.value) = (None, None, None)
        if not method or isinstance(method, int):
            self.method = self.sequence
            self.value = method if method else 0
        else:
            self.method = self.uuid
            self.function = getattr(uuid, method)
        args = inspect.getfullargspec(self.function).args if self.function else []
        self.deterministic = 'namespace' in args

    def __call__(self, data=None):
        if False:
            i = 10
            return i + 15
        '\n        Generates a unique id.\n\n        Args:\n            data: optional data to use for deterministic algorithms (i.e. uuid3, uuid5)\n\n        Returns:\n            unique id\n        '
        return self.method(data)

    def sequence(self, data):
        if False:
            print('Hello World!')
        '\n        Gets and increments sequence.\n\n        Args:\n            data: not used\n\n        Returns:\n            current sequence value\n        '
        value = self.value
        self.value += 1
        return value

    def uuid(self, data):
        if False:
            print('Hello World!')
        '\n        Generates a UUID and return as a string.\n\n        Args:\n            data: used with determistic algorithms (uuid3, uuid5)\n\n        Returns:\n            UUID string\n        '
        uid = self.function(uuid.NAMESPACE_DNS, str(data)) if self.deterministic else self.function()
        return str(uid)

    def current(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the current sequence value. Only applicable for sequence ids, will be None for UUID methods.\n\n        Returns:\n            current sequence value\n        '
        return self.value