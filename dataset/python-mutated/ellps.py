import math

class Ellps:
    """ellipsoid"""

    def __init__(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        self.a = a
        self.b = b
        self.f = (self.a - self.b) / self.a
        self.perimeter = 2 * math.pi * self.a
GRS80 = Ellps(6378137, 6356752.314245)

def dd2meters(dst):
    if False:
        for i in range(10):
            print('nop')
    '\n\tBasic function to approximaly convert a short distance in decimal degrees to meters\n\tOnly true at equator and along horizontal axis\n\t'
    k = GRS80.perimeter / 360
    return dst * k

def meters2dd(dst):
    if False:
        for i in range(10):
            print('nop')
    k = GRS80.perimeter / 360
    return dst / k