import math

class Vector:

    def __init__(self, x, y):
        if False:
            i = 10
            return i + 15
        self.x = x
        self.y = y

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter((self.x, self.y))

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return (self.x, self.y)[key]

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return tuple(self) == tuple(other)

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'Vector(x: %d, y: %d)' % (self.x, self.y)

class Rect:

    def __init__(self, left, top, right, bottom):
        if False:
            i = 10
            return i + 15
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def _get_size(self):
        if False:
            print('Hello World!')
        return Vector(self.right - self.left, self.bottom - self.top)

    def _set_size(self, new_size):
        if False:
            return 10
        centroid = self.centroid
        self.left = centroid[0] - new_size[0] / 2
        self.right = centroid[0] + new_size[0] / 2
        self.top = centroid[1] - new_size[1] / 2
        self.bottom = centroid[1] + new_size[1] / 2
    size = property(_get_size, _set_size)

    @property
    def width(self):
        if False:
            for i in range(10):
                print('nop')
        return self.size.x

    @property
    def height(self):
        if False:
            return 10
        return self.size.y

    def _get_centroid(self):
        if False:
            for i in range(10):
                print('nop')
        return Vector((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def _set_centroid(self, new_centroid):
        if False:
            while True:
                i = 10
        size = self.size
        self.left = new_centroid[0] - size[0] / 2
        self.right = new_centroid[0] + size[0] / 2
        self.top = new_centroid[1] - size[1] / 2
        self.bottom = new_centroid[1] + size[1] / 2
    centroid = property(_get_centroid, _set_centroid)

    @property
    def x(self):
        if False:
            while True:
                i = 10
        return self.centroid.x

    @property
    def y(self):
        if False:
            i = 10
            return i + 15
        return self.centroid.y

    @property
    def centroid_x(self):
        if False:
            while True:
                i = 10
        return self.centroid.x

    @property
    def centroid_y(self):
        if False:
            print('Hello World!')
        return self.centroid.y

    def as_tuple(self):
        if False:
            while True:
                i = 10
        return (self.left, self.top, self.right, self.bottom)

    def clone(self):
        if False:
            i = 10
            return i + 15
        return type(self)(self.left, self.top, self.right, self.bottom)

    def round(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a new rect with all attributes rounded to integers\n        '
        clone = self.clone()
        clone.left = int(math.floor(clone.left))
        clone.top = int(math.floor(clone.top))
        clone.right = int(math.ceil(clone.right))
        clone.bottom = int(math.ceil(clone.bottom))
        return clone

    def move_to_clamp(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Moves this rect so it is completely covered by the rect in "other" and\n        returns a new Rect instance.\n        '
        other = Rect(*other)
        clone = self.clone()
        if clone.left < other.left:
            clone.right -= clone.left - other.left
            clone.left = other.left
        if clone.top < other.top:
            clone.bottom -= clone.top - other.top
            clone.top = other.top
        if clone.right > other.right:
            clone.left -= clone.right - other.right
            clone.right = other.right
        if clone.bottom > other.bottom:
            clone.top -= clone.bottom - other.bottom
            clone.bottom = other.bottom
        return clone

    def move_to_cover(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Moves this rect so it completely covers the rect specified in the\n        "other" parameter and returns a new Rect instance.\n        '
        other = Rect(*other)
        clone = self.clone()
        if clone.left > other.left:
            clone.right -= clone.left - other.left
            clone.left = other.left
        if clone.top > other.top:
            clone.bottom -= clone.top - other.top
            clone.top = other.top
        if clone.right < other.right:
            clone.left += other.right - clone.right
            clone.right = other.right
        if clone.bottom < other.bottom:
            clone.top += other.bottom - clone.bottom
            clone.bottom = other.bottom
        return clone

    def transform(self, transform):
        if False:
            return 10
        tl_transformed = transform.transform_vector(Vector(self.left, self.top))
        tr_transformed = transform.transform_vector(Vector(self.right, self.top))
        bl_transformed = transform.transform_vector(Vector(self.left, self.bottom))
        br_transformed = transform.transform_vector(Vector(self.right, self.bottom))
        left = min([tl_transformed.x, tr_transformed.x, bl_transformed.x, br_transformed.x])
        right = max([tl_transformed.x, tr_transformed.x, bl_transformed.x, br_transformed.x])
        top = min([tl_transformed.y, tr_transformed.y, bl_transformed.y, br_transformed.y])
        bottom = max([tl_transformed.y, tr_transformed.y, bl_transformed.y, br_transformed.y])
        return Rect(left, top, right, bottom)

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter((self.left, self.top, self.right, self.bottom))

    def __getitem__(self, key):
        if False:
            return 10
        return (self.left, self.top, self.right, self.bottom)[key]

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return tuple(self) == tuple(other)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Rect(left: %d, top: %d, right: %d, bottom: %d)' % (self.left, self.top, self.right, self.bottom)

    @classmethod
    def from_point(cls, x, y, width, height):
        if False:
            for i in range(10):
                print('nop')
        return cls(x - width / 2, y - height / 2, x + width / 2, y + height / 2)