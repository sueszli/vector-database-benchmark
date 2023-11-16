class Street:

    def closeDoor(self):
        if False:
            return 10
        pass

    def isDoorOpen(self):
        if False:
            i = 10
            return i + 15
        pass

    def moveRight(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class Solution(object):

    def houseCount(self, street, k):
        if False:
            return 10
        '\n        :type street: Street\n        :type k: int\n        :rtype: int\n        '
        while not street.isDoorOpen():
            street.moveRight()
        result = 0
        for i in xrange(k + 1):
            if i and street.isDoorOpen():
                street.closeDoor()
                result = i
            street.moveRight()
        return result