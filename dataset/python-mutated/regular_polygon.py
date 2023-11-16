"""RegularPolygonVisual visual based on EllipseVisual"""
from __future__ import division
from .ellipse import EllipseVisual

class RegularPolygonVisual(EllipseVisual):
    """
    Displays a regular polygon

    Parameters
    ----------
    center : array-like (x, y)
        Center of the regular polygon
    color : str | tuple | list of colors
        Fill color of the polygon
    border_color : str | tuple | list of colors
        Border color of the polygon
    border_width: float
        The width of the border in pixels
    radius : float
        Radius of the regular polygon
        Defaults to  0.1
    sides : int
        Number of sides of the regular polygon
    """

    def __init__(self, center=None, color='black', border_color=None, border_width=1, radius=0.1, sides=4, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        EllipseVisual.__init__(self, center=center, radius=radius, color=color, border_color=border_color, border_width=border_width, num_segments=sides, **kwargs)

    @property
    def sides(self):
        if False:
            print('Hello World!')
        'The number of sides in the regular polygon.'
        return self.num_segments

    @sides.setter
    def sides(self, sides):
        if False:
            while True:
                i = 10
        if sides < 3:
            raise ValueError('PolygonVisual must have at least 3 sides, not %s' % sides)
        self.num_segments = sides