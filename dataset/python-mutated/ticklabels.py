import warnings
from collections import defaultdict
import numpy as np
from matplotlib import rcParams
from matplotlib.artist import allow_rasterization
from matplotlib.text import Text
from astropy.utils.decorators import deprecated_renamed_argument
from astropy.utils.exceptions import AstropyDeprecationWarning
from .frame import RectangularFrame

def sort_using(X, Y):
    if False:
        print('Hello World!')
    return [x for (y, x) in sorted(zip(Y, X))]

class TickLabels(Text):

    def __init__(self, frame, *args, **kwargs):
        if False:
            print('Hello World!')
        self.clear()
        self._frame = frame
        super().__init__(*args, **kwargs)
        self.set_clip_on(True)
        self.set_visible_axes('all')
        self.set_pad(rcParams['xtick.major.pad'])
        self._exclude_overlapping = False
        self._axis_bboxes = defaultdict(list)
        self._stale = True
        if 'color' not in kwargs:
            self.set_color(rcParams['xtick.color'])
        if 'size' not in kwargs:
            self.set_size(rcParams['xtick.labelsize'])

    def clear(self):
        if False:
            return 10
        self.world = defaultdict(list)
        self.data = defaultdict(list)
        self.angle = defaultdict(list)
        self.text = defaultdict(list)
        self.disp = defaultdict(list)

    def add(self, axis=None, world=None, pixel=None, angle=None, text=None, axis_displacement=None, data=None):
        if False:
            print('Hello World!')
        '\n        Add a label.\n\n        Parameters\n        ----------\n        axis : str\n            Axis to add label to.\n        world : Quantity\n            Coordinate value along this axis.\n        pixel : [float, float]\n            Pixel coordinates of the label. Deprecated and no longer used.\n        angle : float\n            Angle of the label.\n        text : str\n            Label text.\n        axis_displacement : float\n            Displacement from axis.\n        data : [float, float]\n            Data coordinates of the label.\n        '
        required_args = ['axis', 'world', 'angle', 'text', 'axis_displacement', 'data']
        if pixel is not None:
            warnings.warn(f'Setting the pixel coordinates of a label does nothing and is deprecated, as these can only be accurately calculated when Matplotlib is drawing a figure. To prevent this warning pass the following arguments as keyword arguments: {required_args}', AstropyDeprecationWarning)
        if axis is None or world is None or angle is None or (text is None) or (axis_displacement is None) or (data is None):
            raise TypeError(f'All of the following arguments must be provided: {required_args}')
        self.world[axis].append(world)
        self.data[axis].append(data)
        self.angle[axis].append(angle)
        self.text[axis].append(text)
        self.disp[axis].append(axis_displacement)
        self._stale = True

    def sort(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sort by axis displacement, which allows us to figure out which parts\n        of labels to not repeat.\n        '
        for axis in self.world:
            self.world[axis] = sort_using(self.world[axis], self.disp[axis])
            self.data[axis] = sort_using(self.data[axis], self.disp[axis])
            self.angle[axis] = sort_using(self.angle[axis], self.disp[axis])
            self.text[axis] = sort_using(self.text[axis], self.disp[axis])
            self.disp[axis] = sort_using(self.disp[axis], self.disp[axis])
        self._stale = True

    def simplify_labels(self):
        if False:
            i = 10
            return i + 15
        '\n        Figure out which parts of labels can be dropped to avoid repetition.\n        '
        self.sort()
        for axis in self.world:
            t1 = self.text[axis][0]
            for i in range(1, len(self.world[axis])):
                t2 = self.text[axis][i]
                if len(t1) != len(t2):
                    t1 = self.text[axis][i]
                    continue
                start = 0
                for j in range(len(t1) - 1):
                    if t1[j] != t2[j]:
                        break
                    if t1[j] not in '-0123456789.':
                        start = j + 1
                t1 = self.text[axis][i]
                if start != 0:
                    starts_dollar = self.text[axis][i].startswith('$')
                    self.text[axis][i] = self.text[axis][i][start:]
                    if starts_dollar:
                        self.text[axis][i] = '$' + self.text[axis][i]
                if self.text[axis][i] == '$$':
                    self.text[axis][i] = ''
        self._stale = True

    def set_pad(self, value):
        if False:
            while True:
                i = 10
        self._pad = value
        self._stale = True

    def get_pad(self):
        if False:
            for i in range(10):
                print('nop')
        return self._pad

    def set_visible_axes(self, visible_axes):
        if False:
            for i in range(10):
                print('nop')
        self._visible_axes = visible_axes
        self._stale = True

    def get_visible_axes(self):
        if False:
            return 10
        if self._visible_axes == 'all':
            return self.world.keys()
        else:
            return [x for x in self._visible_axes if x in self.world]

    def set_exclude_overlapping(self, exclude_overlapping):
        if False:
            while True:
                i = 10
        self._exclude_overlapping = exclude_overlapping

    def _set_xy_alignments(self, renderer):
        if False:
            return 10
        '\n        Compute and set the x, y positions and the horizontal/vertical alignment of\n        each label.\n        '
        if not self._stale:
            return
        self.simplify_labels()
        text_size = renderer.points_to_pixels(self.get_size())
        visible_axes = self.get_visible_axes()
        self.xy = {axis: {} for axis in visible_axes}
        self.ha = {axis: {} for axis in visible_axes}
        self.va = {axis: {} for axis in visible_axes}
        for axis in visible_axes:
            for i in range(len(self.world[axis])):
                if self.text[axis][i] == '':
                    continue
                (x, y) = self._frame.parent_axes.transData.transform(self.data[axis][i])
                pad = renderer.points_to_pixels(self.get_pad() + self._tick_out_size)
                if isinstance(self._frame, RectangularFrame):
                    if np.abs(self.angle[axis][i]) < 45.0:
                        ha = 'right'
                        va = 'bottom'
                        dx = -pad
                        dy = -text_size * 0.5
                    elif np.abs(self.angle[axis][i] - 90.0) < 45:
                        ha = 'center'
                        va = 'bottom'
                        dx = 0
                        dy = -text_size - pad
                    elif np.abs(self.angle[axis][i] - 180.0) < 45:
                        ha = 'left'
                        va = 'bottom'
                        dx = pad
                        dy = -text_size * 0.5
                    else:
                        ha = 'center'
                        va = 'bottom'
                        dx = 0
                        dy = pad
                    x = x + dx
                    y = y + dy
                else:
                    self.set_text(self.text[axis][i])
                    self.set_position((x, y))
                    bb = super().get_window_extent(renderer)
                    width = bb.width
                    height = bb.height
                    ax = np.cos(np.radians(self.angle[axis][i]))
                    ay = np.sin(np.radians(self.angle[axis][i]))
                    if np.abs(self.angle[axis][i]) < 45.0:
                        dx = width
                        dy = ay * height
                    elif np.abs(self.angle[axis][i] - 90.0) < 45:
                        dx = ax * width
                        dy = height
                    elif np.abs(self.angle[axis][i] - 180.0) < 45:
                        dx = -width
                        dy = ay * height
                    else:
                        dx = ax * width
                        dy = -height
                    dx *= 0.5
                    dy *= 0.5
                    dist = np.hypot(dx, dy)
                    ddx = dx / dist
                    ddy = dy / dist
                    dx += ddx * pad
                    dy += ddy * pad
                    x = x - dx
                    y = y - dy
                    ha = 'center'
                    va = 'center'
                self.xy[axis][i] = (x, y)
                self.ha[axis][i] = ha
                self.va[axis][i] = va
        self._stale = False

    def _get_bb(self, axis, i, renderer):
        if False:
            return 10
        '\n        Get the bounding box of an individual label. n.b. _set_xy_alignment()\n        must be called before this method.\n        '
        if self.text[axis][i] == '':
            return
        self.set_text(self.text[axis][i])
        self.set_position(self.xy[axis][i])
        self.set_ha(self.ha[axis][i])
        self.set_va(self.va[axis][i])
        return super().get_window_extent(renderer)

    @property
    def _all_bboxes(self):
        if False:
            while True:
                i = 10
        ret = []
        for axis in self._axis_bboxes:
            ret += self._axis_bboxes[axis]
        return ret

    def _set_existing_bboxes(self, bboxes):
        if False:
            while True:
                i = 10
        self._existing_bboxes = bboxes

    @allow_rasterization
    @deprecated_renamed_argument(old_name='bboxes', new_name=None, since='6.0')
    @deprecated_renamed_argument(old_name='ticklabels_bbox', new_name=None, since='6.0')
    @deprecated_renamed_argument(old_name='tick_out_size', new_name=None, since='6.0')
    def draw(self, renderer, bboxes=None, ticklabels_bbox=None, tick_out_size=None):
        if False:
            return 10
        self._axis_bboxes = defaultdict(list)
        if not self.get_visible():
            return
        self._set_xy_alignments(renderer)
        for axis in self.get_visible_axes():
            for i in range(len(self.world[axis])):
                bb = self._get_bb(axis, i, renderer)
                if bb is None:
                    continue
                if not self._exclude_overlapping or bb.count_overlaps(self._all_bboxes + self._existing_bboxes) == 0:
                    super().draw(renderer)
                    self._axis_bboxes[axis].append(bb)