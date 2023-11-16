class Street:

    def openDoor(self):
        if False:
            while True:
                i = 10
        pass

    def closeDoor(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def isDoorOpen(self):
        if False:
            i = 10
            return i + 15
        pass

    def moveRight(self):
        if False:
            i = 10
            return i + 15
        pass

    def moveLeft(self):
        if False:
            while True:
                i = 10
        pass

class Solution(object):

    def houseCount(self, street, k):
        if False:
            print('Hello World!')
        '\n        :type street: Street\n        :type k: int\n        :rtype: int\n        '
        for _ in xrange(k):
            street.closeDoor()
            street.moveRight()
        for result in xrange(k + 1):
            if street.isDoorOpen():
                break
            street.openDoor()
            street.moveRight()
        return result