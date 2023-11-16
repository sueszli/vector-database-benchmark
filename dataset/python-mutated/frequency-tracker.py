class FrequencyTracker(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__cnt = collections.Counter()
        self.__freq = collections.Counter()

    def add(self, number):
        if False:
            print('Hello World!')
        '\n        :type number: int\n        :rtype: None\n        '
        self.__freq[self.__cnt[number]] -= 1
        if self.__freq[self.__cnt[number]] == 0:
            del self.__freq[self.__cnt[number]]
        self.__cnt[number] += 1
        self.__freq[self.__cnt[number]] += 1

    def deleteOne(self, number):
        if False:
            return 10
        '\n        :type number: int\n        :rtype: None\n        '
        if self.__cnt[number] == 0:
            return
        self.__freq[self.__cnt[number]] -= 1
        if self.__freq[self.__cnt[number]] == 0:
            del self.__freq[self.__cnt[number]]
        self.__cnt[number] -= 1
        self.__freq[self.__cnt[number]] += 1
        if self.__cnt[number] == 0:
            del self.__cnt[number]

    def hasFrequency(self, frequency):
        if False:
            while True:
                i = 10
        '\n        :type frequency: int\n        :rtype: bool\n        '
        return frequency in self.__freq