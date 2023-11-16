"""
Provides classes to style the axis lines.
"""
import math
import numpy as np
import matplotlib as mpl
from matplotlib.patches import _Style, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform

class _FancyAxislineStyle:

    class SimpleArrow(FancyArrowPatch):
        """The artist class that will be returned for SimpleArrow style."""
        _ARROW_STYLE = '->'

        def __init__(self, axis_artist, line_path, transform, line_mutation_scale):
            if False:
                return 10
            self._axis_artist = axis_artist
            self._line_transform = transform
            self._line_path = line_path
            self._line_mutation_scale = line_mutation_scale
            FancyArrowPatch.__init__(self, path=self._line_path, arrowstyle=self._ARROW_STYLE, patchA=None, patchB=None, shrinkA=0.0, shrinkB=0.0, mutation_scale=line_mutation_scale, mutation_aspect=None, transform=IdentityTransform())

        def set_line_mutation_scale(self, scale):
            if False:
                for i in range(10):
                    print('nop')
            self.set_mutation_scale(scale * self._line_mutation_scale)

        def _extend_path(self, path, mutation_size=10):
            if False:
                print('Hello World!')
            '\n            Extend the path to make a room for drawing arrow.\n            '
            ((x0, y0), (x1, y1)) = path.vertices[-2:]
            theta = math.atan2(y1 - y0, x1 - x0)
            x2 = x1 + math.cos(theta) * mutation_size
            y2 = y1 + math.sin(theta) * mutation_size
            if path.codes is None:
                return Path(np.concatenate([path.vertices, [[x2, y2]]]))
            else:
                return Path(np.concatenate([path.vertices, [[x2, y2]]]), np.concatenate([path.codes, [Path.LINETO]]))

        def set_path(self, path):
            if False:
                print('Hello World!')
            self._line_path = path

        def draw(self, renderer):
            if False:
                print('Hello World!')
            '\n            Draw the axis line.\n             1) Transform the path to the display coordinate.\n             2) Extend the path to make a room for arrow.\n             3) Update the path of the FancyArrowPatch.\n             4) Draw.\n            '
            path_in_disp = self._line_transform.transform_path(self._line_path)
            mutation_size = self.get_mutation_scale()
            extended_path = self._extend_path(path_in_disp, mutation_size=mutation_size)
            self._path_original = extended_path
            FancyArrowPatch.draw(self, renderer)

        def get_window_extent(self, renderer=None):
            if False:
                while True:
                    i = 10
            path_in_disp = self._line_transform.transform_path(self._line_path)
            mutation_size = self.get_mutation_scale()
            extended_path = self._extend_path(path_in_disp, mutation_size=mutation_size)
            self._path_original = extended_path
            return FancyArrowPatch.get_window_extent(self, renderer)

    class FilledArrow(SimpleArrow):
        """The artist class that will be returned for FilledArrow style."""
        _ARROW_STYLE = '-|>'

        def __init__(self, axis_artist, line_path, transform, line_mutation_scale, facecolor):
            if False:
                i = 10
                return i + 15
            super().__init__(axis_artist, line_path, transform, line_mutation_scale)
            self.set_facecolor(facecolor)

class AxislineStyle(_Style):
    """
    A container class which defines style classes for AxisArtists.

    An instance of any axisline style class is a callable object,
    whose call signature is ::

       __call__(self, axis_artist, path, transform)

    When called, this should return an `.Artist` with the following methods::

      def set_path(self, path):
          # set the path for axisline.

      def set_line_mutation_scale(self, scale):
          # set the scale

      def draw(self, renderer):
          # draw
    """
    _style_list = {}

    class _Base:

        def __init__(self):
            if False:
                while True:
                    i = 10
            '\n            initialization.\n            '
            super().__init__()

        def __call__(self, axis_artist, transform):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Given the AxisArtist instance, and transform for the path (set_path\n            method), return the Matplotlib artist for drawing the axis line.\n            '
            return self.new_line(axis_artist, transform)

    class SimpleArrow(_Base):
        """
        A simple arrow.
        """
        ArrowAxisClass = _FancyAxislineStyle.SimpleArrow

        def __init__(self, size=1):
            if False:
                i = 10
                return i + 15
            '\n            Parameters\n            ----------\n            size : float\n                Size of the arrow as a fraction of the ticklabel size.\n            '
            self.size = size
            super().__init__()

        def new_line(self, axis_artist, transform):
            if False:
                i = 10
                return i + 15
            linepath = Path([(0, 0), (0, 1)])
            axisline = self.ArrowAxisClass(axis_artist, linepath, transform, line_mutation_scale=self.size)
            return axisline
    _style_list['->'] = SimpleArrow

    class FilledArrow(SimpleArrow):
        """
        An arrow with a filled head.
        """
        ArrowAxisClass = _FancyAxislineStyle.FilledArrow

        def __init__(self, size=1, facecolor=None):
            if False:
                i = 10
                return i + 15
            '\n            Parameters\n            ----------\n            size : float\n                Size of the arrow as a fraction of the ticklabel size.\n            facecolor : color, default: :rc:`axes.edgecolor`\n                Fill color.\n\n                .. versionadded:: 3.7\n            '
            if facecolor is None:
                facecolor = mpl.rcParams['axes.edgecolor']
            self.size = size
            self._facecolor = facecolor
            super().__init__(size=size)

        def new_line(self, axis_artist, transform):
            if False:
                return 10
            linepath = Path([(0, 0), (0, 1)])
            axisline = self.ArrowAxisClass(axis_artist, linepath, transform, line_mutation_scale=self.size, facecolor=self._facecolor)
            return axisline
    _style_list['-|>'] = FilledArrow