from collections import defaultdict

class TwoSum(object):

    def __init__(self):
        if False:
            return 10
        '\n        initialize your data structure here\n        '
        self.lookup = defaultdict(int)

    def add(self, number):
        if False:
            while True:
                i = 10
        '\n        Add the number to an internal data structure.\n        :rtype: nothing\n        '
        self.lookup[number] += 1

    def find(self, value):
        if False:
            while True:
                i = 10
        '\n        Find if there exists any pair of numbers which sum is equal to the value.\n        :type value: int\n        :rtype: bool\n        '
        for key in self.lookup:
            num = value - key
            if num in self.lookup and (num != key or self.lookup[key] > 1):
                return True
        return False