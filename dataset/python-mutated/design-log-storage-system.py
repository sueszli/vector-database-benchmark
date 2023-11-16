class LogSystem(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.__logs = []
        self.__granularity = {'Year': 4, 'Month': 7, 'Day': 10, 'Hour': 13, 'Minute': 16, 'Second': 19}

    def put(self, id, timestamp):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type id: int\n        :type timestamp: str\n        :rtype: void\n        '
        self.__logs.append((id, timestamp))

    def retrieve(self, s, e, gra):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type e: str\n        :type gra: str\n        :rtype: List[int]\n        '
        i = self.__granularity[gra]
        begin = s[:i]
        end = e[:i]
        return sorted((id for (id, timestamp) in self.__logs if begin <= timestamp[:i] <= end))