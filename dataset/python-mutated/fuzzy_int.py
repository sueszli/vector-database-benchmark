class FuzzyInt(int):

    def __new__(cls, lowest, highest):
        if False:
            i = 10
            return i + 15
        obj = super(FuzzyInt, cls).__new__(cls, highest)
        obj.lowest = lowest
        obj.highest = highest
        return obj

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return other >= self.lowest and other <= self.highest

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '[%d..%d]' % (self.lowest, self.highest)