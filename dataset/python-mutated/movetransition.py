from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy

def position(d):
    if False:
        while True:
            i = 10
    (xpos, ypos, xanchor, yanchor, _xoffset, _yoffset, _subpixel) = d.get_placement()
    if xpos is None:
        xpos = 0
    if ypos is None:
        ypos = 0
    if xanchor is None:
        xanchor = 0
    if yanchor is None:
        yanchor = 0
    return (xpos, ypos, xanchor, yanchor)

def offsets(d):
    if False:
        print('Hello World!')
    (_xpos, _ypos, _xanchor, _yanchor, xoffset, yoffset, _subpixel) = d.get_placement()
    if renpy.config.movetransition_respects_offsets:
        return {'xoffset': xoffset, 'yoffset': yoffset}
    else:
        return {}

def MoveFactory(pos1, pos2, delay, d, **kwargs):
    if False:
        while True:
            i = 10
    if pos1 == pos2:
        return d
    return renpy.display.motion.Move(pos1, pos2, delay, d, **kwargs)

def default_enter_factory(pos, delay, d, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return d

def default_leave_factory(pos, delay, d, **kwargs):
    if False:
        print('Hello World!')
    return None

def MoveIn(pos, pos1, delay, d, **kwargs):
    if False:
        for i in range(10):
            print('nop')

    def aorb(a, b):
        if False:
            print('Hello World!')
        if a is None:
            return b
        return a
    pos = tuple((aorb(a, b) for (a, b) in zip(pos, pos1)))
    return renpy.display.motion.Move(pos, pos1, delay, d, **kwargs)

def MoveOut(pos, pos1, delay, d, **kwargs):
    if False:
        for i in range(10):
            print('nop')

    def aorb(a, b):
        if False:
            print('Hello World!')
        if a is None:
            return b
        return a
    pos = tuple((aorb(a, b) for (a, b) in zip(pos, pos1)))
    return renpy.display.motion.Move(pos1, pos, delay, d, **kwargs)

def ZoomInOut(start, end, pos, delay, d, **kwargs):
    if False:
        i = 10
        return i + 15
    (xpos, ypos, xanchor, yanchor) = pos
    FactorZoom = renpy.display.motion.FactorZoom
    if end == 1.0:
        return FactorZoom(start, end, delay, d, after_child=d, opaque=False, xpos=xpos, ypos=ypos, xanchor=xanchor, yanchor=yanchor, **kwargs)
    else:
        return FactorZoom(start, end, delay, d, opaque=False, xpos=xpos, ypos=ypos, xanchor=xanchor, yanchor=yanchor, **kwargs)

def RevolveInOut(start, end, pos, delay, d, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return renpy.display.motion.Revolve(start, end, delay, d, pos=pos, **kwargs)

def OldMoveTransition(delay, old_widget=None, new_widget=None, factory=None, enter_factory=None, leave_factory=None, old=False, layers=['master']):
    if False:
        return 10
    '\n    Returns a transition that attempts to find images that have changed\n    position, and moves them from the old position to the new transition, taking\n    delay seconds to complete the move.\n\n    If `factory` is given, it is expected to be a function that takes as\n    arguments: an old position, a new position, the delay, and a\n    displayable, and to return a displayable as an argument. If not\n    given, the default behavior is to move the displayable from the\n    starting to the ending positions. Positions are always given as\n    (xpos, ypos, xanchor, yanchor) tuples.\n\n    If `enter_factory` or `leave_factory` are given, they are expected\n    to be functions that take as arguments a position, a delay, and a\n    displayable, and return a displayable. They are applied to\n    displayables that are entering or leaving the scene,\n    respectively. The default is to show in place displayables that\n    are entering, and not to show those that are leaving.\n\n    If `old` is True, then factory moves the old displayable with the\n    given tag. Otherwise, it moves the new displayable with that\n    tag.\n\n    `layers` is a list of layers that the transition will be applied\n    to.\n\n    Images are considered to be the same if they have the same tag, in\n    the same way that the tag is used to determine which image to\n    replace or to hide. They are also considered to be the same if\n    they have no tag, but use the same displayable.\n\n    Computing the order in which images are displayed is a three-step\n    process. The first step is to create a list of images that\n    preserves the relative ordering of entering and moving images. The\n    second step is to insert the leaving images such that each leaving\n    image is at the lowest position that is still above all images\n    that were below it in the original scene. Finally, the list\n    is sorted by zorder, to ensure no zorder violations occur.\n\n    If you use this transition to slide an image off the side of the\n    screen, remember to hide it when you are done. (Or just use\n    a leave_factory.)\n    '
    if factory is None:
        factory = MoveFactory
    if enter_factory is None:
        enter_factory = default_enter_factory
    if leave_factory is None:
        leave_factory = default_leave_factory
    use_old = old

    def merge_slide(old, new):
        if False:
            while True:
                i = 10
        if not isinstance(new, renpy.display.layout.MultiBox) or (new.layers is None and new.layer_name is None):
            if use_old:
                child = old
            else:
                child = new
            old_pos = position(old)
            new_pos = position(new)
            if old_pos != new_pos:
                return factory(old_pos, new_pos, delay, child, **offsets(child))
            else:
                return child
        if new.layers:
            rv = renpy.display.layout.MultiBox(layout='fixed')
            rv.layers = {}
            for layer in renpy.config.layers:
                f = new.layers[layer]
                if isinstance(f, renpy.display.layout.MultiBox) and layer in layers and (f.scene_list is not None):
                    f = merge_slide(old.layers[layer], new.layers[layer])
                rv.layers[layer] = f
                rv.add(f)
            return rv

        def wrap(sle):
            if False:
                for i in range(10):
                    print('nop')
            return renpy.display.layout.AdjustTimes(sle.displayable, sle.show_time, sle.animation_time)

        def tag(sle):
            if False:
                i = 10
                return i + 15
            return sle.tag or sle.displayable

        def merge(sle, d):
            if False:
                return 10
            rv = sle.copy()
            rv.show_time = None
            rv.displayable = d
            return rv

        def entering(sle):
            if False:
                for i in range(10):
                    print('nop')
            new_d = wrap(sle)
            move = enter_factory(position(new_d), delay, new_d, **offsets(new_d))
            if move is None:
                return
            rv_sl.append(merge(sle, move))

        def leaving(sle):
            if False:
                i = 10
                return i + 15
            old_d = wrap(sle)
            move = leave_factory(position(old_d), delay, old_d, **offsets(old_d))
            if move is None:
                return
            move = renpy.display.layout.IgnoresEvents(move)
            rv_sl.append(merge(sle, move))

        def moving(old_sle, new_sle):
            if False:
                for i in range(10):
                    print('nop')
            old_d = wrap(old_sle)
            new_d = wrap(new_sle)
            if use_old:
                child = old_d
            else:
                child = new_d
            move = factory(position(old_d), position(new_d), delay, child, **offsets(child))
            if move is None:
                return
            rv_sl.append(merge(new_sle, move))
        old_sl = old.scene_list[:]
        new_sl = new.scene_list[:]
        rv_sl = []
        old_map = dict(((tag(i), i) for i in old_sl if i is not None))
        new_tags = set((tag(i) for i in new_sl if i is not None))
        rv_tags = set()
        while old_sl or new_sl:
            if old_sl:
                old_sle = old_sl[0]
                old_tag = tag(old_sle)
                if old_tag in rv_tags:
                    old_sl.pop(0)
                    continue
                if old_tag not in new_tags:
                    leaving(old_sle)
                    rv_tags.add(old_tag)
                    old_sl.pop(0)
                    continue
            new_sle = new_sl.pop(0)
            new_tag = tag(new_sle)
            if new_tag in old_map:
                old_sle = old_map[new_tag]
                moving(old_sle, new_sle)
                rv_tags.add(new_tag)
                continue
            else:
                entering(new_sle)
                rv_tags.add(new_tag)
                continue
        rv_sl.sort(key=lambda a: a.zorder)
        layer = new.layer_name
        rv = renpy.display.layout.MultiBox(layout='fixed', focus=layer, **renpy.game.interface.layer_properties[layer])
        rv.append_scene_list(rv_sl)
        rv.layer_name = layer
        return rv
    rv = merge_slide(old_widget, new_widget)
    rv.delay = delay
    return rv

class MoveInterpolate(renpy.display.displayable.Displayable):
    """
    This displayable has two children. It interpolates between the positions
    of its two children to place them on the screen.
    """

    def __init__(self, delay, old, new, use_old, time_warp):
        if False:
            while True:
                i = 10
        super(MoveInterpolate, self).__init__()
        self.old = old
        self.new = new
        self.use_old = use_old
        self.time_warp = time_warp
        self.screen_width = 0
        self.screen_height = 0
        self.child_width = 0
        self.child_height = 0
        self.delay = delay
        self.st = 0

    def render(self, width, height, st, at):
        if False:
            for i in range(10):
                print('nop')
        self.screen_width = width
        self.screen_height = height
        old_r = renpy.display.render.render(self.old, width, height, st, at)
        new_r = renpy.display.render.render(self.new, width, height, st, at)
        if self.use_old:
            cr = old_r
        else:
            cr = new_r
        (self.child_width, self.child_height) = cr.get_size()
        self.st = st
        if self.st < self.delay:
            renpy.display.render.redraw(self, 0)
        return cr

    def child_placement(self, child):
        if False:
            for i in range(10):
                print('nop')

        def based(v, base):
            if False:
                for i in range(10):
                    print('nop')
            if v is None:
                return 0
            elif isinstance(v, int):
                return v
            elif isinstance(v, renpy.display.core.absolute):
                return v
            else:
                return v * base
        (xpos, ypos, xanchor, yanchor, xoffset, yoffset, subpixel) = child.get_placement()
        xpos = based(xpos, self.screen_width)
        ypos = based(ypos, self.screen_height)
        xanchor = based(xanchor, self.child_width)
        yanchor = based(yanchor, self.child_height)
        return (xpos, ypos, xanchor, yanchor, xoffset, yoffset, subpixel)

    def get_placement(self):
        if False:
            while True:
                i = 10
        if self.st > self.delay:
            done = 1.0
        else:
            done = self.st / self.delay
        if self.time_warp is not None:
            done = self.time_warp(done)
        absolute = renpy.display.core.absolute

        def I(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return absolute(a + done * (b - a))
        (old_xpos, old_ypos, old_xanchor, old_yanchor, old_xoffset, old_yoffset, old_subpixel) = self.child_placement(self.old)
        (new_xpos, new_ypos, new_xanchor, new_yanchor, new_xoffset, new_yoffset, new_subpixel) = self.child_placement(self.new)
        xpos = I(old_xpos, new_xpos)
        ypos = I(old_ypos, new_ypos)
        xanchor = I(old_xanchor, new_xanchor)
        yanchor = I(old_yanchor, new_yanchor)
        xoffset = I(old_xoffset, new_xoffset)
        yoffset = I(old_yoffset, new_yoffset)
        subpixel = old_subpixel or new_subpixel
        return (xpos, ypos, xanchor, yanchor, xoffset, yoffset, subpixel)

def MoveTransition(delay, old_widget=None, new_widget=None, enter=None, leave=None, old=False, layers=['master'], time_warp=None, enter_time_warp=None, leave_time_warp=None):
    if False:
        print('Hello World!')
    "\n    :doc: transition function\n    :args: (delay, *, enter=None, leave=None, old=False, layers=['master'], time_warp=None, enter_time_warp=None, leave_time_warp=None)\n    :name: MoveTransition\n\n    Returns a transition that interpolates the position of images (with the\n    same tag) in the old and new scenes.\n\n    As only layers have tags, MoveTransitions can only be applied to a single\n    layer or all layers at once, using the :ref:`with statement <with-statement>`.\n    It will not work in other contexts such as :ref:`ATL <expression-atl-statement>`,\n    :func:`ComposeTransition`, or other ways of applying transitions.\n\n    `delay`\n        The time it takes for the interpolation to finish.\n\n    `enter`\n        If not None, images entering the scene will also be moved. The value\n        of `enter` should be a transform that is applied to the image to\n        get it in its starting position.\n\n    `leave`\n        If not None, images leaving the scene will also be moved. The value\n        of `leave` should be a transform that is applied to the image to\n        get it in its ending position.\n\n    `old`\n        If true, when a tag gets its image changed during the transition,\n        the old image will be used in preference to the new one. Otherwise,\n        the new images will be used.\n\n    `layers`\n        A list of layers that moves are applied to.\n\n    `time_warp`\n        A :ref:`time warp function <warpers>` that's applied to the interpolation. This\n        takes a number between 0.0 and 1.0, and should return a number in\n        the same range.\n\n    `enter_time_warp`\n        A time warp function that's applied to images entering the scene.\n\n    `leave_time_warp`\n        A time warp function that's applied to images leaving the scene.\n    "
    if renpy.config.developer:
        for widget in (old_widget, new_widget):
            if not (hasattr(widget, 'scene_list') or hasattr(widget, 'layers')):
                raise Exception('MoveTransition can only be applied to one or all layers, not %s.' % type(widget).__name__)
    use_old = old

    def merge_slide(old, new, merge_slide):
        if False:
            i = 10
            return i + 15
        if not isinstance(new, renpy.display.layout.MultiBox) or (new.layers is None and new.layer_name is None):
            if old is new:
                return new
            else:
                return MoveInterpolate(delay, old, new, use_old, time_warp)
        if new.layers:
            rv = renpy.display.layout.MultiBox(layout='fixed')
            rv.layers = {}
            for layer in renpy.config.layers:
                d = merge_slide(old.layers[layer], new.layers[layer], merge_slide)
                rv.layers[layer] = d
                rv.add(d, True, True)
            return rv
        old = old.untransformed_layer or old
        if new.untransformed_layer:
            rv = new
            new = new.untransformed_layer
            layer = new.layer_name
            if isinstance(new, renpy.display.layout.MultiBox) and layer in layers and (new.scene_list is not None):
                d = merge_slide(old, new, merge_slide)
                adjust = renpy.display.layout.AdjustTimes(d, None, None)
                rv = renpy.game.context().scene_lists.transform_layer(layer, adjust)
                if rv is adjust:
                    rv = d
                else:
                    rv = renpy.display.layout.MatchTimes(rv, adjust)
            return rv

        def wrap(sle):
            if False:
                for i in range(10):
                    print('nop')
            return renpy.display.layout.AdjustTimes(sle.displayable, sle.show_time, sle.animation_time)

        def tag(sle):
            if False:
                return 10
            return sle.tag or sle.displayable

        def merge(sle, d):
            if False:
                return 10
            rv = sle.copy()
            rv.show_time = 0
            rv.displayable = d
            return rv

        def entering(sle):
            if False:
                return 10
            if not enter:
                return
            new_d = wrap(sle)
            move = MoveInterpolate(delay, renpy.store.At(new_d, enter), new_d, False, enter_time_warp)
            rv_sl.append(merge(sle, move))

        def leaving(sle):
            if False:
                while True:
                    i = 10
            if not leave:
                return
            old_d = wrap(sle)
            move = MoveInterpolate(delay, old_d, renpy.store.At(old_d, leave), True, leave_time_warp)
            move = renpy.display.layout.IgnoresEvents(move)
            rv_sl.append(merge(sle, move))

        def moving(old_sle, new_sle):
            if False:
                while True:
                    i = 10
            if old_sle.displayable is new_sle.displayable:
                rv_sl.append(new_sle)
                return
            old_d = wrap(old_sle)
            new_d = wrap(new_sle)
            move = MoveInterpolate(delay, old_d, new_d, use_old, time_warp)
            rv_sl.append(merge(new_sle, move))
        old_sl = old.scene_list[:]
        new_sl = new.scene_list[:]
        rv_sl = []
        old_map = dict(((tag(i), i) for i in old_sl if i is not None))
        new_tags = set((tag(i) for i in new_sl if i is not None))
        rv_tags = set()
        while old_sl or new_sl:
            if old_sl:
                old_sle = old_sl[0]
                old_tag = tag(old_sle)
                if old_tag in rv_tags:
                    old_sl.pop(0)
                    continue
                if old_tag not in new_tags:
                    leaving(old_sle)
                    rv_tags.add(old_tag)
                    old_sl.pop(0)
                    continue
            new_sle = new_sl.pop(0)
            new_tag = tag(new_sle)
            if new_tag in old_map:
                old_sle = old_map[new_tag]
                moving(old_sle, new_sle)
                rv_tags.add(new_tag)
                continue
            else:
                entering(new_sle)
                rv_tags.add(new_tag)
                continue
        rv_sl.sort(key=lambda a: a.zorder)
        layer = new.layer_name
        rv = renpy.display.layout.MultiBox(layout='fixed', focus=layer, **renpy.game.interface.layer_properties[layer])
        rv.append_scene_list(rv_sl)
        return rv
    rv = merge_slide(old_widget, new_widget, merge_slide)
    rv.delay = delay
    return rv