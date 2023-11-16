class BidirMap(object):

    def __init__(self, **map):
        if False:
            while True:
                i = 10
        self.k2v = {}
        self.v2k = {}
        for key in map:
            self.__setitem__(key, map[key])

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        if value in self.v2k:
            if self.v2k[value] != key:
                raise KeyError("Value '" + str(value) + "' already in use with key '" + str(self.v2k[value]) + "'")
        try:
            del self.v2k[self.k2v[key]]
        except KeyError:
            pass
        self.k2v[key] = value
        self.v2k[value] = key

    def __getitem__(self, key):
        if False:
            return 10
        return self.k2v[key]

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.v2k.__str__()

    def getkey(self, value):
        if False:
            while True:
                i = 10
        return self.v2k[value]

    def getvalue(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self.k2v[key]

    def keys(self):
        if False:
            return 10
        return [key for key in self.k2v]

    def values(self):
        if False:
            print('Hello World!')
        return [value for value in self.v2k]