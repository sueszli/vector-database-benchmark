class Solution(object):
    lookup = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}
    cache = {}

    def strobogrammaticInRange(self, low, high):
        if False:
            print('Hello World!')
        count = self.countStrobogrammaticUntil(high, False) - self.countStrobogrammaticUntil(low, False) + self.isStrobogrammatic(low)
        return count if count >= 0 else 0

    def countStrobogrammaticUntil(self, num, can_start_with_0):
        if False:
            i = 10
            return i + 15
        if can_start_with_0 and num in self.cache:
            return self.cache[num]
        count = 0
        if len(num) == 1:
            for c in ['0', '1', '8']:
                if num[0] >= c:
                    count += 1
            self.cache[num] = count
            return count
        for (key, val) in self.lookup.iteritems():
            if can_start_with_0 or key != '0':
                if num[0] > key:
                    if len(num) == 2:
                        count += 1
                    else:
                        count += self.countStrobogrammaticUntil('9' * (len(num) - 2), True)
                elif num[0] == key:
                    if len(num) == 2:
                        if num[-1] >= val:
                            count += 1
                    elif num[-1] >= val:
                        count += self.countStrobogrammaticUntil(self.getMid(num), True)
                    elif self.getMid(num) != '0' * (len(num) - 2):
                        count += self.countStrobogrammaticUntil(self.getMid(num), True) - self.isStrobogrammatic(self.getMid(num))
        if not can_start_with_0:
            for i in xrange(len(num) - 1, 0, -1):
                count += self.countStrobogrammaticByLength(i)
        else:
            self.cache[num] = count
        return count

    def getMid(self, num):
        if False:
            i = 10
            return i + 15
        return num[1:len(num) - 1]

    def countStrobogrammaticByLength(self, n):
        if False:
            i = 10
            return i + 15
        if n == 1:
            return 3
        elif n == 2:
            return 4
        elif n == 3:
            return 4 * 3
        else:
            return 5 * self.countStrobogrammaticByLength(n - 2)

    def isStrobogrammatic(self, num):
        if False:
            while True:
                i = 10
        n = len(num)
        for i in xrange((n + 1) / 2):
            if num[n - 1 - i] not in self.lookup or num[i] != self.lookup[num[n - 1 - i]]:
                return False
        return True