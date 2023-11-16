from abc import ABCMeta, abstractmethod

class KeyValueStorage:
    """
    A simple key-value storage abstract base class
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get(self, key):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a value identified by the given key\n\n        :param key: The unique identifier\n\n        :return: The value identified by key or None if no value was found\n        '
        raise NotImplementedError

    @abstractmethod
    def set(self, key, value):
        if False:
            return 10
        '\n        Store the value identified by the key\n\n        :param key: The unique identifier\n        :param value: Value to store\n\n        :return: bool True on success or False on failure\n        '
        raise NotImplementedError

    @abstractmethod
    def delete(self, key):
        if False:
            print('Hello World!')
        '\n        Deletes item by key\n\n        :param key: The unique identifier\n\n        :return: bool True on success or False on failure\n        '
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clears all entries\n\n        :return: bool True on success or False on failure\n        '
        raise NotImplementedError