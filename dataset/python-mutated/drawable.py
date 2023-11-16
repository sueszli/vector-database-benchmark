"""
Copyright 2007, 2008, 2009, 2016 Free Software Foundation, Inc.
This file is part of GNU Radio

SPDX-License-Identifier: GPL-2.0-or-later

"""
from ..Constants import LINE_SELECT_SENSITIVITY

class Drawable(object):
    """
    GraphicalElement is the base class for all graphical elements.
    It contains an X,Y coordinate, a list of rectangular areas that the element occupies,
    and methods to detect selection of those areas.
    """

    @classmethod
    def make_cls_with_base(cls, super_cls):
        if False:
            while True:
                i = 10
        name = super_cls.__name__
        bases = (super_cls,) + cls.__bases__[1:]
        namespace = cls.__dict__.copy()
        return type(name, bases, namespace)

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Make a new list of rectangular areas and lines, and set the coordinate and the rotation.\n        '
        self.coordinate = (0, 0)
        self.rotation = 0
        self.highlighted = False
        self._bounding_rects = []
        self._bounding_points = []

    def is_horizontal(self, rotation=None):
        if False:
            i = 10
            return i + 15
        "\n        Is this element horizontal?\n        If rotation is None, use this element's rotation.\n\n        Args:\n            rotation: the optional rotation\n\n        Returns:\n            true if rotation is horizontal\n        "
        rotation = rotation or self.rotation
        return rotation in (0, 180)

    def is_vertical(self, rotation=None):
        if False:
            print('Hello World!')
        "\n        Is this element vertical?\n        If rotation is None, use this element's rotation.\n\n        Args:\n            rotation: the optional rotation\n\n        Returns:\n            true if rotation is vertical\n        "
        rotation = rotation or self.rotation
        return rotation in (90, 270)

    def rotate(self, rotation):
        if False:
            return 10
        '\n        Rotate all of the areas by 90 degrees.\n\n        Args:\n            rotation: multiple of 90 degrees\n        '
        self.rotation = (self.rotation + rotation) % 360

    def move(self, delta_coor):
        if False:
            for i in range(10):
                print('nop')
        '\n        Move the element by adding the delta_coor to the current coordinate.\n\n        Args:\n            delta_coor: (delta_x,delta_y) tuple\n        '
        (x, y) = self.coordinate
        (dx, dy) = delta_coor
        self.coordinate = (x + dx, y + dy)

    def create_labels(self, cr=None):
        if False:
            return 10
        '\n        Create labels (if applicable) and call on all children.\n        Call this base method before creating labels in the element.\n        '

    def create_shapes(self):
        if False:
            while True:
                i = 10
        '\n        Create shapes (if applicable) and call on all children.\n        Call this base method before creating shapes in the element.\n        '

    def draw(self, cr):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def bounds_from_area(self, area):
        if False:
            i = 10
            return i + 15
        (x1, y1, w, h) = area
        x2 = x1 + w
        y2 = y1 + h
        self._bounding_rects = [(x1, y1, x2, y2)]
        self._bounding_points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    def bounds_from_line(self, line):
        if False:
            print('Hello World!')
        self._bounding_rects = rects = []
        self._bounding_points = list(line)
        last_point = line[0]
        for (x2, y2) in line[1:]:
            ((x1, y1), last_point) = (last_point, (x2, y2))
            if x1 == x2:
                (x1, x2) = (x1 - LINE_SELECT_SENSITIVITY, x2 + LINE_SELECT_SENSITIVITY)
                if y2 < y1:
                    (y1, y2) = (y2, y1)
            elif y1 == y2:
                (y1, y2) = (y1 - LINE_SELECT_SENSITIVITY, y2 + LINE_SELECT_SENSITIVITY)
                if x2 < x1:
                    (x1, x2) = (x2, x1)
            rects.append((x1, y1, x2, y2))

    def what_is_selected(self, coor, coor_m=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        One coordinate specified:\n            Is this element selected at given coordinate?\n            ie: is the coordinate encompassed by one of the areas or lines?\n        Both coordinates specified:\n            Is this element within the rectangular region defined by both coordinates?\n            ie: do any area corners or line endpoints fall within the region?\n\n        Args:\n            coor: the selection coordinate, tuple x, y\n            coor_m: an additional selection coordinate.\n\n        Returns:\n            self if one of the areas/lines encompasses coor, else None.\n        '
        (x, y) = [a - b for (a, b) in zip(coor, self.coordinate)]
        if not coor_m:
            for (x1, y1, x2, y2) in self._bounding_rects:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return self
        else:
            (x_m, y_m) = [a - b for (a, b) in zip(coor_m, self.coordinate)]
            if y_m < y:
                (y, y_m) = (y_m, y)
            if x_m < x:
                (x, x_m) = (x_m, x)
            for (x1, y1) in self._bounding_points:
                if x <= x1 <= x_m and y <= y1 <= y_m:
                    return self

    def get_extents(self):
        if False:
            return 10
        (x_min, y_min) = (x_max, y_max) = self.coordinate
        x_min += min((x for (x, y) in self._bounding_points))
        y_min += min((y for (x, y) in self._bounding_points))
        x_max += max((x for (x, y) in self._bounding_points))
        y_max += max((y for (x, y) in self._bounding_points))
        return (x_min, y_min, x_max, y_max)

    def mouse_over(self):
        if False:
            while True:
                i = 10
        pass

    def mouse_out(self):
        if False:
            for i in range(10):
                print('nop')
        pass