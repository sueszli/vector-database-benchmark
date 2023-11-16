class XY(object):
    """A class to represent 2-tuple value"""

    def __init__(self, x, y, z=None):
        if False:
            i = 10
            return i + 15
        "\n\t\tYou can use the constructor in many ways:\n\t\tXY(0, 1) - passing two arguments\n\t\tXY(x=0, y=1) - passing keywords arguments\n\t\tXY(**{'x': 0, 'y': 1}) - unpacking a dictionary\n\t\tXY(*[0, 1]) - unpacking a list or a tuple (or a generic iterable)\n\t\t"
        if z is None:
            self.data = [x, y]
        else:
            self.data = [x, y, z]

    def __str__(self):
        if False:
            return 10
        if self.z is not None:
            return '(%s, %s, %s)' % (self.x, self.y, self.z)
        else:
            return '(%s, %s)' % (self.x, self.y)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__str__()

    def __getitem__(self, item):
        if False:
            return 10
        return self.data[item]

    def __setitem__(self, idx, value):
        if False:
            i = 10
            return i + 15
        self.data[idx] = value

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.data)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.data)

    @property
    def x(self):
        if False:
            i = 10
            return i + 15
        return self.data[0]

    @property
    def y(self):
        if False:
            return 10
        return self.data[1]

    @property
    def z(self):
        if False:
            while True:
                i = 10
        try:
            return self.data[2]
        except IndexError:
            return None

    @property
    def xy(self):
        if False:
            while True:
                i = 10
        return self.data[:2]

    @property
    def xyz(self):
        if False:
            for i in range(10):
                print('nop')
        return self.data