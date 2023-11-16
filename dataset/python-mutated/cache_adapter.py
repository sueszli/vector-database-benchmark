from abc import ABCMeta, abstractmethod

class CacheAdapter:
    """
    CacheAdapter Abstract Base Class
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get(self, public_id, type, resource_type, transformation, format):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets value specified by parameters\n\n        :param public_id:       The public ID of the resource\n        :param type:            The storage type\n        :param resource_type:   The type of the resource\n        :param transformation:  The transformation string\n        :param format:          The format of the resource\n\n        :return: None|mixed value, None if not found\n        '
        raise NotImplementedError

    @abstractmethod
    def set(self, public_id, type, resource_type, transformation, format, value):
        if False:
            i = 10
            return i + 15
        '\n        Sets value specified by parameters\n\n        :param public_id:       The public ID of the resource\n        :param type:            The storage type\n        :param resource_type:   The type of the resource\n        :param transformation:  The transformation string\n        :param format:          The format of the resource\n        :param value:           The value to set\n\n        :return: bool True on success or False on failure\n        '
        raise NotImplementedError

    @abstractmethod
    def delete(self, public_id, type, resource_type, transformation, format):
        if False:
            while True:
                i = 10
        '\n        Deletes entry specified by parameters\n\n        :param public_id:       The public ID of the resource\n        :param type:            The storage type\n        :param resource_type:   The type of the resource\n        :param transformation:  The transformation string\n        :param format:          The format of the resource\n\n        :return: bool True on success or False on failure\n        '
        raise NotImplementedError

    @abstractmethod
    def flush_all(self):
        if False:
            while True:
                i = 10
        '\n        Flushes all entries from cache\n\n        :return: bool True on success or False on failure\n        '
        raise NotImplementedError