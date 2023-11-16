class FileSystem(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.__lookup = {'': -1}

    def create(self, path, value):
        if False:
            print('Hello World!')
        '\n        :type path: str\n        :type value: int\n        :rtype: bool\n        '
        if path[:path.rfind('/')] not in self.__lookup:
            return False
        self.__lookup[path] = value
        return True

    def get(self, path):
        if False:
            return 10
        '\n        :type path: str\n        :rtype: int\n        '
        if path not in self.__lookup:
            return -1
        return self.__lookup[path]