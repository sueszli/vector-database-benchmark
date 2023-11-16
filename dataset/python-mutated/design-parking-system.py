class ParkingSystem(object):

    def __init__(self, big, medium, small):
        if False:
            i = 10
            return i + 15
        '\n        :type big: int\n        :type medium: int\n        :type small: int\n        '
        self.__space = [0, big, medium, small]

    def addCar(self, carType):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type carType: int\n        :rtype: bool\n        '
        if self.__space[carType] > 0:
            self.__space[carType] -= 1
            return True
        return False