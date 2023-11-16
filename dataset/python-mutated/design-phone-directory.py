class PhoneDirectory(object):

    def __init__(self, maxNumbers):
        if False:
            print('Hello World!')
        '\n        Initialize your data structure here\n        @param maxNumbers - The maximum numbers that can be stored in the phone directory.\n        :type maxNumbers: int\n        '
        self.__curr = 0
        self.__numbers = range(maxNumbers)
        self.__used = [False] * maxNumbers

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Provide a number which is not assigned to anyone.\n        @return - Return an available number. Return -1 if none is available.\n        :rtype: int\n        '
        if self.__curr == len(self.__numbers):
            return -1
        number = self.__numbers[self.__curr]
        self.__curr += 1
        self.__used[number] = True
        return number

    def check(self, number):
        if False:
            while True:
                i = 10
        '\n        Check if a number is available or not.\n        :type number: int\n        :rtype: bool\n        '
        return 0 <= number < len(self.__numbers) and (not self.__used[number])

    def release(self, number):
        if False:
            while True:
                i = 10
        '\n        Recycle or release a number.\n        :type number: int\n        :rtype: void\n        '
        if not 0 <= number < len(self.__numbers) or not self.__used[number]:
            return
        self.__used[number] = False
        self.__curr -= 1
        self.__numbers[self.__curr] = number