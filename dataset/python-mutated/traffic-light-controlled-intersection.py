import threading

class TrafficLight(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__l = threading.Lock()
        self.__light = 1

    def carArrived(self, carId, roadId, direction, turnGreen, crossCar):
        if False:
            while True:
                i = 10
        '\n        :type roadId: int --> // ID of the car\n        :type carId: int --> // ID of the road the car travels on. Can be 1 (road A) or 2 (road B)\n        :type direction: int --> // Direction of the car\n        :type turnGreen: method --> // Use turnGreen() to turn light to green on current road\n        :type crossCar: method --> // Use crossCar() to make car cross the intersection\n        :rtype: void\n        '
        with self.__l:
            if self.__light != roadId:
                self.__light = roadId
                turnGreen()
            crossCar()