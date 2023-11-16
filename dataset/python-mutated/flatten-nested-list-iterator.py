class NestedIterator(object):

    def __init__(self, nestedList):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize your data structure here.\n        :type nestedList: List[NestedInteger]\n        '
        self.__depth = [[nestedList, 0]]

    def next(self):
        if False:
            while True:
                i = 10
        '\n        :rtype: int\n        '
        (nestedList, i) = self.__depth[-1]
        self.__depth[-1][1] += 1
        return nestedList[i].getInteger()

    def hasNext(self):
        if False:
            print('Hello World!')
        '\n        :rtype: bool\n        '
        while self.__depth:
            (nestedList, i) = self.__depth[-1]
            if i == len(nestedList):
                self.__depth.pop()
            elif nestedList[i].isInteger():
                return True
            else:
                self.__depth[-1][1] += 1
                self.__depth.append([nestedList[i].getList(), 0])
        return False