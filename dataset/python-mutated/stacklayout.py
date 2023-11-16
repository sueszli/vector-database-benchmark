"""
Stack Layout
============

.. only:: html

    .. image:: images/stacklayout.gif
        :align: right

.. only:: latex

    .. image:: images/stacklayout.png
        :align: right

.. versionadded:: 1.0.5

The :class:`StackLayout` arranges children vertically or horizontally, as many
as the layout can fit. The size of the individual children widgets do not
have to be uniform.

For example, to display widgets that get progressively larger in width::

    root = StackLayout()
    for i in range(25):
        btn = Button(text=str(i), width=40 + i * 5, size_hint=(None, 0.15))
        root.add_widget(btn)

.. image:: images/stacklayout_sizing.png
    :align: left
"""
__all__ = ('StackLayout',)
from kivy.uix.layout import Layout
from kivy.properties import NumericProperty, OptionProperty, ReferenceListProperty, VariableListProperty

def _compute_size(c, available_size, idx):
    if False:
        for i in range(10):
            print('nop')
    sh_min = c.size_hint_min[idx]
    sh_max = c.size_hint_max[idx]
    val = c.size_hint[idx] * available_size
    if sh_min is not None:
        if sh_max is not None:
            return max(min(sh_max, val), sh_min)
        return max(val, sh_min)
    if sh_max is not None:
        return min(sh_max, val)
    return val

class StackLayout(Layout):
    """Stack layout class. See module documentation for more information.
    """
    spacing = VariableListProperty([0, 0], length=2)
    'Spacing between children: [spacing_horizontal, spacing_vertical].\n\n    spacing also accepts a single argument form [spacing].\n\n    :attr:`spacing` is a\n    :class:`~kivy.properties.VariableListProperty` and defaults to [0, 0].\n\n    '
    padding = VariableListProperty([0, 0, 0, 0])
    "Padding between the layout box and it's children: [padding_left,\n    padding_top, padding_right, padding_bottom].\n\n    padding also accepts a two argument form [padding_horizontal,\n    padding_vertical] and a single argument form [padding].\n\n    .. versionchanged:: 1.7.0\n        Replaced the NumericProperty with a VariableListProperty.\n\n    :attr:`padding` is a\n    :class:`~kivy.properties.VariableListProperty` and defaults to\n    [0, 0, 0, 0].\n\n    "
    orientation = OptionProperty('lr-tb', options=('lr-tb', 'tb-lr', 'rl-tb', 'tb-rl', 'lr-bt', 'bt-lr', 'rl-bt', 'bt-rl'))
    "Orientation of the layout.\n\n    :attr:`orientation` is an :class:`~kivy.properties.OptionProperty` and\n    defaults to 'lr-tb'.\n\n    Valid orientations are 'lr-tb', 'tb-lr', 'rl-tb', 'tb-rl', 'lr-bt',\n    'bt-lr', 'rl-bt' and 'bt-rl'.\n\n    .. versionchanged:: 1.5.0\n        :attr:`orientation` now correctly handles all valid combinations of\n        'lr','rl','tb','bt'. Before this version only 'lr-tb' and\n        'tb-lr' were supported, and 'tb-lr' was misnamed and placed\n        widgets from bottom to top and from right to left (reversed compared\n        to what was expected).\n\n    .. note::\n\n        'lr' means Left to Right.\n        'rl' means Right to Left.\n        'tb' means Top to Bottom.\n        'bt' means Bottom to Top.\n    "
    minimum_width = NumericProperty(0)
    'Minimum width needed to contain all children. It is automatically set\n    by the layout.\n\n    .. versionadded:: 1.0.8\n\n    :attr:`minimum_width` is a :class:`kivy.properties.NumericProperty` and\n    defaults to 0.\n    '
    minimum_height = NumericProperty(0)
    'Minimum height needed to contain all children. It is automatically set\n    by the layout.\n\n    .. versionadded:: 1.0.8\n\n    :attr:`minimum_height` is a :class:`kivy.properties.NumericProperty` and\n    defaults to 0.\n    '
    minimum_size = ReferenceListProperty(minimum_width, minimum_height)
    'Minimum size needed to contain all children. It is automatically set\n    by the layout.\n\n    .. versionadded:: 1.0.8\n\n    :attr:`minimum_size` is a\n    :class:`~kivy.properties.ReferenceListProperty` of\n    (:attr:`minimum_width`, :attr:`minimum_height`) properties.\n    '

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(StackLayout, self).__init__(**kwargs)
        trigger = self._trigger_layout
        fbind = self.fbind
        fbind('padding', trigger)
        fbind('spacing', trigger)
        fbind('children', trigger)
        fbind('orientation', trigger)
        fbind('size', trigger)
        fbind('pos', trigger)

    def do_layout(self, *largs):
        if False:
            return 10
        if not self.children:
            self.minimum_size = (0.0, 0.0)
            return
        selfpos = self.pos
        selfsize = self.size
        orientation = self.orientation.split('-')
        padding_left = self.padding[0]
        padding_top = self.padding[1]
        padding_right = self.padding[2]
        padding_bottom = self.padding[3]
        padding_x = padding_left + padding_right
        padding_y = padding_top + padding_bottom
        (spacing_x, spacing_y) = self.spacing
        posattr = [0] * 2
        posdelta = [0] * 2
        posstart = [0] * 2
        for i in (0, 1):
            posattr[i] = 1 * (orientation[i] in ('tb', 'bt'))
            k = posattr[i]
            if orientation[i] == 'lr':
                posdelta[i] = 1
                posstart[i] = selfpos[k] + padding_left
            elif orientation[i] == 'bt':
                posdelta[i] = 1
                posstart[i] = selfpos[k] + padding_bottom
            elif orientation[i] == 'rl':
                posdelta[i] = -1
                posstart[i] = selfpos[k] + selfsize[k] - padding_right
            else:
                posdelta[i] = -1
                posstart[i] = selfpos[k] + selfsize[k] - padding_top
        (innerattr, outerattr) = posattr
        (ustart, vstart) = posstart
        (deltau, deltav) = posdelta
        del posattr, posdelta, posstart
        u = ustart
        v = vstart
        if orientation[0] in ('lr', 'rl'):
            sv = padding_y
            su = padding_x
            spacing_u = spacing_x
            spacing_v = spacing_y
            padding_u = padding_x
            padding_v = padding_y
        else:
            sv = padding_x
            su = padding_y
            spacing_u = spacing_y
            spacing_v = spacing_x
            padding_u = padding_y
            padding_v = padding_x
        lv = 0
        urev = deltau < 0
        vrev = deltav < 0
        firstchild = self.children[0]
        sizes = []
        lc = []
        for c in reversed(self.children):
            if c.size_hint[outerattr] is not None:
                c.size[outerattr] = max(1, _compute_size(c, selfsize[outerattr] - padding_v, outerattr))
            ccount = len(lc)
            totalsize = availsize = max(0, selfsize[innerattr] - padding_u - spacing_u * ccount)
            if not lc:
                if c.size_hint[innerattr] is not None:
                    childsize = max(1, _compute_size(c, totalsize, innerattr))
                else:
                    childsize = max(0, c.size[innerattr])
                availsize = selfsize[innerattr] - padding_u - childsize
                testsizes = [childsize]
            else:
                testsizes = [0] * (ccount + 1)
                for (i, child) in enumerate(lc):
                    if availsize <= 0:
                        availsize = -1
                        break
                    if child.size_hint[innerattr] is not None:
                        testsizes[i] = childsize = max(1, _compute_size(child, totalsize, innerattr))
                    else:
                        childsize = max(0, child.size[innerattr])
                        testsizes[i] = childsize
                    availsize -= childsize
                if c.size_hint[innerattr] is not None:
                    testsizes[-1] = max(1, _compute_size(c, totalsize, innerattr))
                else:
                    testsizes[-1] = max(0, c.size[innerattr])
                availsize -= testsizes[-1]
            if availsize + 1e-10 >= 0 or not lc:
                lc.append(c)
                sizes = testsizes
                lv = max(lv, c.size[outerattr])
                continue
            for (i, child) in enumerate(lc):
                if child.size_hint[innerattr] is not None:
                    child.size[innerattr] = sizes[i]
            sv += lv + spacing_v
            for c2 in lc:
                if urev:
                    u -= c2.size[innerattr]
                c2.pos[innerattr] = u
                pos_outer = v
                if vrev:
                    pos_outer -= c2.size[outerattr]
                c2.pos[outerattr] = pos_outer
                if urev:
                    u -= spacing_u
                else:
                    u += c2.size[innerattr] + spacing_u
            v += deltav * lv
            v += deltav * spacing_v
            lc = [c]
            lv = c.size[outerattr]
            if c.size_hint[innerattr] is not None:
                sizes = [max(1, _compute_size(c, selfsize[innerattr] - padding_u, innerattr))]
            else:
                sizes = [max(0, c.size[innerattr])]
            u = ustart
        if lc:
            for (i, child) in enumerate(lc):
                if child.size_hint[innerattr] is not None:
                    child.size[innerattr] = sizes[i]
            sv += lv + spacing_v
            for c2 in lc:
                if urev:
                    u -= c2.size[innerattr]
                c2.pos[innerattr] = u
                pos_outer = v
                if vrev:
                    pos_outer -= c2.size[outerattr]
                c2.pos[outerattr] = pos_outer
                if urev:
                    u -= spacing_u
                else:
                    u += c2.size[innerattr] + spacing_u
        self.minimum_size[outerattr] = sv