from collections import namedtuple
from math import sqrt

class Rectangle:

    def __init__(self, left=0, top=0, width=0, height=0):
        if False:
            for i in range(10):
                print('nop')
        self.left = left
        self.top = top
        self.right = left + width
        self.bottom = top + height

    @classmethod
    def from_w_h(cls, width, height):
        if False:
            i = 10
            return i + 15
        return cls(0, 0, width, height)

    @classmethod
    def from_tuple_l_t_w_h(cls, l_t_w_h=None):
        if False:
            for i in range(10):
                print('nop')
        if l_t_w_h is None:
            l_t_w_h = (0, 0, 0, 0)
        return cls(*l_t_w_h)

    @classmethod
    def from_tuple_w_h(cls, w_h):
        if False:
            while True:
                i = 10
        return cls.from_w_h(*w_h)

    @classmethod
    def from_struct_l_t_r_b(cls, struct):
        if False:
            while True:
                i = 10
        return cls.from_l_t_r_b(struct.left, struct.top, struct.right, struct.bottom)

    @classmethod
    def from_l_t_r_b(cls, left, top, right, bottom):
        if False:
            while True:
                i = 10
        return cls(left, top, right - left, bottom - top)

    @property
    def width(self):
        if False:
            i = 10
            return i + 15
        return self.right - self.left

    @property
    def height(self):
        if False:
            return 10
        return self.bottom - self.top

    @property
    def center(self):
        if False:
            for i in range(10):
                print('nop')
        return Point(self.left + self.width / 2, self.top + self.height / 2)

    @property
    def east(self):
        if False:
            print('Hello World!')
        return self.clip(Point(self.right - 1, self.center.y))

    @property
    def west(self):
        if False:
            for i in range(10):
                print('nop')
        return Point(self.left, self.center.y)

    @property
    def north(self):
        if False:
            return 10
        return Point(self.center.x, self.top)

    @property
    def south(self):
        if False:
            i = 10
            return i + 15
        return self.clip(Point(self.center.x, self.bottom - 1))

    @property
    def northeast(self):
        if False:
            return 10
        return Point(self.east.x, self.north.y)

    @property
    def southeast(self):
        if False:
            return 10
        return Point(self.east.x, self.south.y)

    @property
    def southwest(self):
        if False:
            while True:
                i = 10
        return Point(self.west.x, self.south.y)

    @property
    def northwest(self):
        if False:
            while True:
                i = 10
        return Point(self.west.x, self.north.y)

    @property
    def area(self):
        if False:
            i = 10
            return i + 15
        if not self:
            return 0
        return self.width * self.height

    def __contains__(self, point):
        if False:
            while True:
                i = 10
        return self.left <= point.x < self.right and self.top <= point.y < self.bottom

    def translate(self, dx, dy):
        if False:
            for i in range(10):
                print('nop')
        self.left += dx
        self.right += dx
        self.top += dy
        self.bottom += dy
        return self

    def clip(self, point):
        if False:
            while True:
                i = 10
        return Point(min(max(point[0], self.left), max(self.left, self.right - 1)), min(max(point[1], self.top), max(self.top, self.bottom - 1)))

    def intersect(self, rectangle):
        if False:
            i = 10
            return i + 15
        left = max(self.left, rectangle.left)
        top = max(self.top, rectangle.top)
        right = min(self.right, rectangle.right)
        bottom = min(self.bottom, rectangle.bottom)
        return self.from_l_t_r_b(left, top, right, bottom) or Rectangle()

    def intersects(self, rectangle):
        if False:
            print('Hello World!')
        return bool(self.intersect(rectangle))

    def as_numpy_slice(self):
        if False:
            i = 10
            return i + 15
        return (slice(self.top, self.bottom), slice(self.left, self.right))

    def is_to_left_of(self, other):
        if False:
            for i in range(10):
                print('nop')
        self_starts_to_left_of_other = self.left < other.left
        self_overlaps_other_top = self.top <= other.top < self.bottom
        other_overlaps_self_top = other.top <= self.top < other.bottom
        return self_starts_to_left_of_other and (self_overlaps_other_top or other_overlaps_self_top)

    def is_to_right_of(self, other):
        if False:
            return 10
        return other.is_to_left_of(self)

    def is_above(self, other):
        if False:
            return 10
        self_starts_above_other = self.top < other.top
        self_overlaps_other_left = self.left <= other.left < self.right
        other_overlaps_self_left = other.left <= self.left < other.right
        return self_starts_above_other and (self_overlaps_other_left or other_overlaps_self_left)

    def is_below(self, other):
        if False:
            for i in range(10):
                print('nop')
        return other.is_above(self)

    def is_in_direction(self, in_direction, of_other):
        if False:
            i = 10
            return i + 15
        return getattr(self, 'is_' + in_direction)(of_other)

    def distance_to(self, other):
        if False:
            return 10
        leftmost = self if self.left < other.left else other
        rightmost = self if leftmost == other else other
        distance_x = max(0, rightmost.left - leftmost.right)
        topmost = self if self.top < other.top else other
        bottommost = self if topmost == other else other
        distance_y = max(0, bottommost.top - topmost.bottom)
        return sqrt(distance_x ** 2 + distance_y ** 2)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, Rectangle):
            return False
        return self.left == other.left and self.top == other.top and (self.right == other.right) and (self.bottom == other.bottom)

    def __ne__(self, other):
        if False:
            return 10
        return not self.__eq__(other)

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.width > 0 and self.height > 0)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return type(self).__name__ + '(left=%d, top=%d, width=%d, height=%d)' % (self.left, self.top, self.width, self.height)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return self.left + 7 * self.top + 11 * self.right + 13 * self.bottom

class Point(namedtuple('Point', ['x', 'y'])):

    def __new__(cls, x=0, y=0):
        if False:
            return 10
        return cls.__bases__[0].__new__(cls, x, y)

    def __init__(self, x=0, y=0):
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    def from_tuple(cls, tpl):
        if False:
            while True:
                i = 10
        return cls(*tpl)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return (self.x, self.y) == other

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self == other

    def __add__(self, other):
        if False:
            while True:
                i = 10
        (dx, dy) = other
        return Point(self.x + dx, self.y + dy)

    def __radd__(self, other):
        if False:
            return 10
        return self.__add__(other)

    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        (dx, dy) = other
        return Point(self.x - dx, self.y - dy)

    def __rsub__(self, other):
        if False:
            return 10
        (x, y) = other
        (dx, dy) = self
        return Point(x - dx, y - dy)

    def __mul__(self, scalar):
        if False:
            return 10
        if isinstance(scalar, (int, float)):
            return Point(self.x * scalar, self.y * scalar)
        else:
            raise ValueError('Invalid argument')

    def __rmul__(self, scalar):
        if False:
            print('Hello World!')
        return self.__mul__(scalar)

    def __div__(self, scalar):
        if False:
            print('Hello World!')
        if isinstance(scalar, (int, float)):
            return Point(self.x / scalar, self.y / scalar)
        else:
            raise ValueError('Invalid argument')

    def __bool__(self):
        if False:
            print('Hello World!')
        return bool(self.x) or bool(self.y)

class Direction:

    def __init__(self, unit_vector):
        if False:
            i = 10
            return i + 15
        self.unit_vector = unit_vector

    def iterate_points_starting_at(self, point, offsets):
        if False:
            for i in range(10):
                print('nop')
        for offset in offsets:
            yield (point + offset * self.unit_vector)

    def is_horizontal(self):
        if False:
            i = 10
            return i + 15
        return bool(self.unit_vector.x)

    def is_vertical(self):
        if False:
            return 10
        return not self.is_horizontal()

    @property
    def orthog_vector(self):
        if False:
            return 10
        return Point(-self.unit_vector[1], self.unit_vector[0])

    def __eq__(self, other):
        if False:
            return 10
        return self.unit_vector == other.unit_vector

    def __repr__(self):
        if False:
            print('Hello World!')
        for module_element in dir(self.__module__):
            if self == getattr(self.__module__, module_element):
                return module_element
NORTH = Direction(Point(0, -1))
EAST = Direction(Point(1, 0))
SOUTH = Direction(Point(0, 1))
WEST = Direction(Point(-1, 0))