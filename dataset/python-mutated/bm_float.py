from math import sin, cos, sqrt

class Point(object):
    __slots__ = ('x', 'y', 'z')

    def __init__(self, i):
        if False:
            i = 10
            return i + 15
        self.x = x = sin(i)
        self.y = cos(i) * 3
        self.z = x * x / 2

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<Point: x=%s, y=%s, z=%s>' % (self.x, self.y, self.z)

    def normalize(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.x
        y = self.y
        z = self.z
        norm = sqrt(x * x + y * y + z * z)
        self.x /= norm
        self.y /= norm
        self.z /= norm

    def maximize(self, other):
        if False:
            i = 10
            return i + 15
        self.x = self.x if self.x > other.x else other.x
        self.y = self.y if self.y > other.y else other.y
        self.z = self.z if self.z > other.z else other.z
        return self

def maximize(points):
    if False:
        for i in range(10):
            print('nop')
    next = points[0]
    for p in points[1:]:
        next = next.maximize(p)
    return next

def benchmark(n):
    if False:
        i = 10
        return i + 15
    points = [None] * n
    for i in range(n):
        points[i] = Point(i)
    for p in points:
        p.normalize()
    return maximize(points)
bm_params = {(50, 25): (1, 150), (100, 100): (1, 250), (1000, 1000): (10, 1500), (5000, 1000): (20, 3000)}

def bm_setup(params):
    if False:
        while True:
            i = 10
    state = None

    def run():
        if False:
            return 10
        nonlocal state
        for _ in range(params[0]):
            state = benchmark(params[1])

    def result():
        if False:
            for i in range(10):
                print('nop')
        return (params[0] * params[1], 'Point(%.4f, %.4f, %.4f)' % (state.x, state.y, state.z))
    return (run, result)