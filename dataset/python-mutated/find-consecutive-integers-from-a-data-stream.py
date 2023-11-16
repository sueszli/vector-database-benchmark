class DataStream(object):

    def __init__(self, value, k):
        if False:
            i = 10
            return i + 15
        '\n        :type value: int\n        :type k: int\n        '
        self.__value = value
        self.__k = k
        self.__cnt = 0

    def consec(self, num):
        if False:
            return 10
        '\n        :type num: int\n        :rtype: bool\n        '
        if num == self.__value:
            self.__cnt += 1
        else:
            self.__cnt = 0
        return self.__cnt >= self.__k