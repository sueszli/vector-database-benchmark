class OrderedStream(object):

    def __init__(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        '
        self.__i = 0
        self.__values = [None] * n

    def insert(self, id, value):
        if False:
            return 10
        '\n        :type id: int\n        :type value: str\n        :rtype: List[str]\n        '
        id -= 1
        self.__values[id] = value
        result = []
        if self.__i != id:
            return result
        while self.__i < len(self.__values) and self.__values[self.__i]:
            result.append(self.__values[self.__i])
            self.__i += 1
        return result