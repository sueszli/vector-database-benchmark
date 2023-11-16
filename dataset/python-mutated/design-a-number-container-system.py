from sortedcontainers import SortedList

class NumberContainers(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__idx_to_num = {}
        self.__num_to_idxs = collections.defaultdict(SortedList)

    def change(self, index, number):
        if False:
            while True:
                i = 10
        '\n        :type index: int\n        :type number: int\n        :rtype: None\n        '
        if index in self.__idx_to_num:
            self.__num_to_idxs[self.__idx_to_num[index]].remove(index)
            if not self.__num_to_idxs[self.__idx_to_num[index]]:
                del self.__num_to_idxs[self.__idx_to_num[index]]
        self.__idx_to_num[index] = number
        self.__num_to_idxs[number].add(index)

    def find(self, number):
        if False:
            return 10
        '\n        :type number: int\n        :rtype: int\n        '
        return self.__num_to_idxs[number][0] if number in self.__num_to_idxs else -1