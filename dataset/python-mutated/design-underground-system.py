import collections

class UndergroundSystem(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__live = {}
        self.__statistics = collections.defaultdict(lambda : [0, 0])

    def checkIn(self, id, stationName, t):
        if False:
            i = 10
            return i + 15
        '\n        :type id: int\n        :type stationName: str\n        :type t: int\n        :rtype: None\n        '
        self.__live[id] = (stationName, t)

    def checkOut(self, id, stationName, t):
        if False:
            return 10
        '\n        :type id: int\n        :type stationName: str\n        :type t: int\n        :rtype: None\n        '
        (startStation, startTime) = self.__live.pop(id)
        self.__statistics[startStation, stationName][0] += t - startTime
        self.__statistics[startStation, stationName][1] += 1

    def getAverageTime(self, startStation, endStation):
        if False:
            i = 10
            return i + 15
        '\n        :type startStation: str\n        :type endStation: str\n        :rtype: float\n        '
        (total_time, cnt) = self.__statistics[startStation, endStation]
        return float(total_time) / cnt