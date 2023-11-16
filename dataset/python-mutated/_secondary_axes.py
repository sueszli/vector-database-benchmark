import numbers
import numpy as np
from matplotlib import _api, _docstring
import matplotlib.ticker as mticker
from matplotlib.axes._base import _AxesBase, _TransformedBoundsLocator
from matplotlib.axis import Axis

class SecondaryAxis(_AxesBase):
    """
    General class to hold a Secondary_X/Yaxis.
    """

    def __init__(self, parent, orientation, location, functions, **kwargs):
        if False:
            return 10
        '\n        See `.secondary_xaxis` and `.secondary_yaxis` for the doc string.\n        While there is no need for this to be private, it should really be\n        called by those higher level functions.\n        '
        _api.check_in_list(['x', 'y'], orientation=orientation)
        self._functions = functions
        self._parent = parent
        self._orientation = orientation
        self._ticks_set = False
        if self._orientation == 'x':
            super().__init__(self._parent.figure, [0, 1.0, 1, 0.0001], **kwargs)
            self._axis = self.xaxis
            self._locstrings = ['top', 'bottom']
            self._otherstrings = ['left', 'right']
        else:
            super().__init__(self._parent.figure, [0, 1.0, 0.0001, 1], **kwargs)
            self._axis = self.yaxis
            self._locstrings = ['right', 'left']
            self._otherstrings = ['top', 'bottom']
        self._parentscale = None
        self.set_location(location)
        self.set_functions(functions)
        otheraxis = self.yaxis if self._orientation == 'x' else self.xaxis
        otheraxis.set_major_locator(mticker.NullLocator())
        otheraxis.set_ticks_position('none')
        self.spines[self._otherstrings].set_visible(False)
        self.spines[self._locstrings].set_visible(True)
        if self._pos < 0.5:
            self._locstrings = self._locstrings[::-1]
        self.set_alignment(self._locstrings[0])

    def set_alignment(self, align):
        if False:
            i = 10
            return i + 15
        "\n        Set if axes spine and labels are drawn at top or bottom (or left/right)\n        of the axes.\n\n        Parameters\n        ----------\n        align : {'top', 'bottom', 'left', 'right'}\n            Either 'top' or 'bottom' for orientation='x' or\n            'left' or 'right' for orientation='y' axis.\n        "
        _api.check_in_list(self._locstrings, align=align)
        if align == self._locstrings[1]:
            self._locstrings = self._locstrings[::-1]
        self.spines[self._locstrings[0]].set_visible(True)
        self.spines[self._locstrings[1]].set_visible(False)
        self._axis.set_ticks_position(align)
        self._axis.set_label_position(align)

    def set_location(self, location):
        if False:
            print('Hello World!')
        "\n        Set the vertical or horizontal location of the axes in\n        parent-normalized coordinates.\n\n        Parameters\n        ----------\n        location : {'top', 'bottom', 'left', 'right'} or float\n            The position to put the secondary axis.  Strings can be 'top' or\n            'bottom' for orientation='x' and 'right' or 'left' for\n            orientation='y'. A float indicates the relative position on the\n            parent axes to put the new axes, 0.0 being the bottom (or left)\n            and 1.0 being the top (or right).\n        "
        if isinstance(location, str):
            _api.check_in_list(self._locstrings, location=location)
            self._pos = 1.0 if location in ('top', 'right') else 0.0
        elif isinstance(location, numbers.Real):
            self._pos = location
        else:
            raise ValueError(f'location must be {self._locstrings[0]!r}, {self._locstrings[1]!r}, or a float, not {location!r}')
        self._loc = location
        if self._orientation == 'x':
            bounds = [0, self._pos, 1.0, 1e-10]
        else:
            bounds = [self._pos, 0, 1e-10, 1]
        self.set_axes_locator(_TransformedBoundsLocator(bounds, self._parent.transAxes))

    def apply_aspect(self, position=None):
        if False:
            for i in range(10):
                print('nop')
        self._set_lims()
        super().apply_aspect(position)

    @_docstring.copy(Axis.set_ticks)
    def set_ticks(self, ticks, labels=None, *, minor=False, **kwargs):
        if False:
            print('Hello World!')
        ret = self._axis.set_ticks(ticks, labels, minor=minor, **kwargs)
        self.stale = True
        self._ticks_set = True
        return ret

    def set_functions(self, functions):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set how the secondary axis converts limits from the parent axes.\n\n        Parameters\n        ----------\n        functions : 2-tuple of func, or `Transform` with an inverse.\n            Transform between the parent axis values and the secondary axis\n            values.\n\n            If supplied as a 2-tuple of functions, the first function is\n            the forward transform function and the second is the inverse\n            transform.\n\n            If a transform is supplied, then the transform must have an\n            inverse.\n        '
        if isinstance(functions, tuple) and len(functions) == 2 and callable(functions[0]) and callable(functions[1]):
            self._functions = functions
        elif functions is None:
            self._functions = (lambda x: x, lambda x: x)
        else:
            raise ValueError('functions argument of secondary axes must be a two-tuple of callable functions with the first function being the transform and the second being the inverse')
        self._set_scale()

    def draw(self, renderer):
        if False:
            for i in range(10):
                print('nop')
        '\n        Draw the secondary axes.\n\n        Consults the parent axes for its limits and converts them\n        using the converter specified by\n        `~.axes._secondary_axes.set_functions` (or *functions*\n        parameter when axes initialized.)\n        '
        self._set_lims()
        self._set_scale()
        super().draw(renderer)

    def _set_scale(self):
        if False:
            i = 10
            return i + 15
        '\n        Check if parent has set its scale\n        '
        if self._orientation == 'x':
            pscale = self._parent.xaxis.get_scale()
            set_scale = self.set_xscale
        else:
            pscale = self._parent.yaxis.get_scale()
            set_scale = self.set_yscale
        if pscale == self._parentscale:
            return
        if self._ticks_set:
            ticks = self._axis.get_ticklocs()
        set_scale('functionlog' if pscale == 'log' else 'function', functions=self._functions[::-1])
        if self._ticks_set:
            self._axis.set_major_locator(mticker.FixedLocator(ticks))
        self._parentscale = pscale

    def _set_lims(self):
        if False:
            while True:
                i = 10
        '\n        Set the limits based on parent limits and the convert method\n        between the parent and this secondary axes.\n        '
        if self._orientation == 'x':
            lims = self._parent.get_xlim()
            set_lim = self.set_xlim
        else:
            lims = self._parent.get_ylim()
            set_lim = self.set_ylim
        order = lims[0] < lims[1]
        lims = self._functions[0](np.array(lims))
        neworder = lims[0] < lims[1]
        if neworder != order:
            lims = lims[::-1]
        set_lim(lims)

    def set_aspect(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Secondary axes cannot set the aspect ratio, so calling this just\n        sets a warning.\n        '
        _api.warn_external("Secondary axes can't set the aspect ratio")

    def set_color(self, color):
        if False:
            print('Hello World!')
        '\n        Change the color of the secondary axes and all decorators.\n\n        Parameters\n        ----------\n        color : color; see :ref:`colors_def`\n        '
        axis = self._axis_map[self._orientation]
        axis.set_tick_params(colors=color)
        for spine in self.spines.values():
            if spine.axis is axis:
                spine.set_color(color)
        axis.label.set_color(color)
_secax_docstring = "\nWarnings\n--------\nThis method is experimental as of 3.1, and the API may change.\n\nParameters\n----------\nlocation : {'top', 'bottom', 'left', 'right'} or float\n    The position to put the secondary axis.  Strings can be 'top' or\n    'bottom' for orientation='x' and 'right' or 'left' for\n    orientation='y'. A float indicates the relative position on the\n    parent axes to put the new axes, 0.0 being the bottom (or left)\n    and 1.0 being the top (or right).\n\nfunctions : 2-tuple of func, or Transform with an inverse\n\n    If a 2-tuple of functions, the user specifies the transform\n    function and its inverse.  i.e.\n    ``functions=(lambda x: 2 / x, lambda x: 2 / x)`` would be an\n    reciprocal transform with a factor of 2. Both functions must accept\n    numpy arrays as input.\n\n    The user can also directly supply a subclass of\n    `.transforms.Transform` so long as it has an inverse.\n\n    See :doc:`/gallery/subplots_axes_and_figures/secondary_axis`\n    for examples of making these conversions.\n\nReturns\n-------\nax : axes._secondary_axes.SecondaryAxis\n\nOther Parameters\n----------------\n**kwargs : `~matplotlib.axes.Axes` properties.\n    Other miscellaneous axes parameters.\n"
_docstring.interpd.update(_secax_docstring=_secax_docstring)