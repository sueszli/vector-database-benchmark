from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import math
import renpy
from renpy.display.render import render
from renpy.display.layout import Container
from renpy.display.transform import Transform, Proxy, TransformState, ATLTransform, null

class Motion(Container):
    """
    This is used to move a child displayable around the screen. It
    works by supplying a time value to a user-supplied function,
    which is in turn expected to return a pair giving the x and y
    location of the upper-left-hand corner of the child, or a
    4-tuple giving that and the xanchor and yanchor of the child.

    The time value is a floating point number that ranges from 0 to
    1. If repeat is True, then the motion repeats every period
    sections. (Otherwise, it stops.) If bounce is true, the
    time value varies from 0 to 1 to 0 again.

    The function supplied needs to be pickleable, which means it needs
    to be defined as a name in an init block. It cannot be a lambda or
    anonymous inner function. If you can get away with using Pan or
    Move, use them instead.

    Please note that floats and ints are interpreted as for xpos and
    ypos, with floats being considered fractions of the screen.
    """

    def __init__(self, function, period, child=None, new_widget=None, old_widget=None, repeat=False, bounce=False, delay=None, anim_timebase=False, tag_start=None, time_warp=None, add_sizes=False, style='motion', **properties):
        if False:
            print('Hello World!')
        '\n        @param child: The child displayable.\n\n        @param new_widget: If child is None, it is set to new_widget,\n        so that we can speak the transition protocol.\n\n        @param old_widget: Ignored, for compatibility with the transition protocol.\n\n        @param function: A function that takes a floating point value and returns\n        an xpos, ypos tuple.\n\n        @param period: The amount of time it takes to go through one cycle, in seconds.\n\n        @param repeat: Should we repeat after a period is up?\n\n        @param bounce: Should we bounce?\n\n        @param delay: How long this motion should take. If repeat is None, defaults to period.\n\n        @param anim_timebase: If True, use the animation timebase rather than the shown timebase.\n\n        @param time_warp: If not None, this is a function that takes a\n        fraction of the period (between 0.0 and 1.0), and returns a\n        new fraction of the period. Use this to warp time, applying\n        acceleration and deceleration to motions.\n\n        This can also be used as a transition. When used as a\n        transition, the motion is applied to the new_widget for delay\n        seconds.\n        '
        if child is None:
            child = new_widget
        if delay is None and (not repeat):
            delay = period
        super(Motion, self).__init__(style=style, **properties)
        if child is not None:
            self.add(child)
        self.function = function
        self.period = period
        self.repeat = repeat
        self.bounce = bounce
        self.delay = delay
        self.anim_timebase = anim_timebase
        self.time_warp = time_warp
        self.add_sizes = add_sizes
        self.position = None

    def update_position(self, t, sizes):
        if False:
            i = 10
            return i + 15
        if renpy.game.less_updates:
            if self.delay:
                t = self.delay
                if self.repeat:
                    t = t % self.period
            else:
                t = self.period
        elif self.delay and t >= self.delay:
            t = self.delay
            if self.repeat:
                t = t % self.period
        elif self.repeat:
            t = t % self.period
            renpy.display.render.redraw(self, 0)
        elif t > self.period:
            t = self.period
        else:
            renpy.display.render.redraw(self, 0)
        if self.period > 0:
            t /= self.period
        else:
            t = 1
        if self.time_warp:
            t = self.time_warp(t)
        if self.bounce:
            t = t * 2
            if t > 1.0:
                t = 2.0 - t
        if self.add_sizes:
            res = self.function(t, sizes)
        else:
            res = self.function(t)
        res = tuple(res)
        if len(res) == 2:
            self.position = res + (self.style.xanchor or 0, self.style.yanchor or 0)
        else:
            self.position = res

    def get_placement(self):
        if False:
            return 10
        if self.position is None:
            if self.add_sizes:
                return super(Motion, self).get_placement()
            else:
                self.update_position(0.0, None)
        return self.position + (self.style.xoffset, self.style.yoffset, self.style.subpixel)

    def render(self, width, height, st, at):
        if False:
            print('Hello World!')
        if self.anim_timebase:
            t = at
        else:
            t = st
        child = render(self.child, width, height, st, at)
        (cw, ch) = child.get_size()
        self.update_position(t, (width, height, cw, ch))
        rv = renpy.display.render.Render(cw, ch)
        rv.blit(child, (0, 0))
        self.offsets = [(0, 0)]
        return rv

class Interpolate(object):
    anchors = {'top': 0.0, 'center': 0.5, 'bottom': 1.0, 'left': 0.0, 'right': 1.0}

    def __init__(self, start, end):
        if False:
            i = 10
            return i + 15
        if len(start) != len(end):
            raise Exception('The start and end must have the same number of arguments.')
        self.start = [self.anchors.get(i, i) for i in start]
        self.end = [self.anchors.get(i, i) for i in end]

    def __call__(self, t, sizes=(None, None, None, None)):
        if False:
            return 10
        types = (renpy.atl.position,) * len(self.start)
        return renpy.atl.interpolate(t, tuple(self.start), tuple(self.end), types)

def Pan(startpos, endpos, time, child=None, repeat=False, bounce=False, anim_timebase=False, style='motion', time_warp=None, **properties):
    if False:
        for i in range(10):
            print('nop')
    "\n    This is used to pan over a child displayable, which is almost\n    always an image. It works by interpolating the placement of the\n    upper-left corner of the screen, over time. It's only really\n    suitable for use with images that are larger than the screen,\n    and we don't do any cropping on the image.\n\n    @param startpos: The initial coordinates of the upper-left\n    corner of the screen, relative to the image.\n\n    @param endpos: The coordinates of the upper-left corner of the\n    screen, relative to the image, after time has elapsed.\n\n    @param time: The time it takes to pan from startpos to endpos.\n\n    @param child: The child displayable.\n\n    @param repeat: True if we should repeat this forever.\n\n    @param bounce: True if we should bounce from the start to the end\n    to the start.\n\n    @param anim_timebase: True if we use the animation timebase, False to use the\n    displayable timebase.\n\n    @param time_warp: If not None, this is a function that takes a\n    fraction of the period (between 0.0 and 1.0), and returns a\n    new fraction of the period. Use this to warp time, applying\n    acceleration and deceleration to motions.\n\n    This can be used as a transition. See Motion for details.\n    "
    (x0, y0) = startpos
    (x1, y1) = endpos
    return Motion(Interpolate((-x0, -y0), (-x1, -y1)), time, child, repeat=repeat, bounce=bounce, style=style, anim_timebase=anim_timebase, time_warp=time_warp, **properties)

def Move(startpos, endpos, time, child=None, repeat=False, bounce=False, anim_timebase=False, style='motion', time_warp=None, **properties):
    if False:
        while True:
            i = 10
    '\n    This is used to pan over a child displayable relative to\n    the containing area. It works by interpolating the placement of the\n    the child, over time.\n\n    @param startpos: The initial coordinates of the child\n    relative to the containing area.\n\n    @param endpos: The coordinates of the child at the end of the\n    move.\n\n    @param time: The time it takes to move from startpos to endpos.\n\n    @param child: The child displayable.\n\n    @param repeat: True if we should repeat this forever.\n\n    @param bounce: True if we should bounce from the start to the end\n    to the start.\n\n    @param anim_timebase: True if we use the animation timebase, False to use the\n    displayable timebase.\n\n    @param time_warp: If not None, this is a function that takes a\n    fraction of the period (between 0.0 and 1.0), and returns a\n    new fraction of the period. Use this to warp time, applying\n    acceleration and deceleration to motions.\n\n    This can be used as a transition. See Motion for details.\n    '
    return Motion(Interpolate(startpos, endpos), time, child, repeat=repeat, bounce=bounce, anim_timebase=anim_timebase, style=style, time_warp=time_warp, **properties)

class Revolver(object):

    def __init__(self, start, end, child, around=(0.5, 0.5), cor=(0.5, 0.5), pos=None):
        if False:
            i = 10
            return i + 15
        self.start = start
        self.end = end
        self.around = around
        self.cor = cor
        self.pos = pos
        self.child = child

    def __call__(self, t, rect):
        if False:
            i = 10
            return i + 15
        absolute = renpy.display.core.absolute
        (w, h, cw, ch) = rect

        def fti(x, r):
            if False:
                return 10
            if x is None:
                x = 0
            return absolute.compute_raw(x, r)
        if self.pos is None:
            pos = self.child.get_placement()
        else:
            pos = self.pos
        (xpos, ypos, xanchor, yanchor, _xoffset, _yoffset, _subpixel) = pos
        xpos = fti(xpos, w)
        ypos = fti(ypos, h)
        xanchor = fti(xanchor, cw)
        yanchor = fti(yanchor, ch)
        (xaround, yaround) = self.around
        xaround = fti(xaround, w)
        yaround = fti(yaround, h)
        (xcor, ycor) = self.cor
        xcor = fti(xcor, cw)
        ycor = fti(ycor, ch)
        angle = self.start + (self.end - self.start) * t
        angle *= math.pi / 180
        x = xpos - xanchor + xcor - xaround
        y = ypos - yanchor + ycor - yaround
        nx = x * math.cos(angle) - y * math.sin(angle)
        ny = x * math.sin(angle) + y * math.cos(angle)
        nx = nx - xcor + xaround
        ny = ny - ycor + yaround
        return (absolute(nx), absolute(ny), 0, 0)

def Revolve(start, end, time, child, around=(0.5, 0.5), cor=(0.5, 0.5), pos=None, **properties):
    if False:
        i = 10
        return i + 15
    return Motion(Revolver(start, end, child, around=around, cor=cor, pos=pos), time, child, add_sizes=True, **properties)

def zoom_render(crend, x, y, w, h, zw, zh, bilinear):
    if False:
        for i in range(10):
            print('nop')
    '\n    This creates a render that zooms its child.\n\n    `crend` - The render of the child.\n    `x`, `y`, `w`, `h` - A rectangle inside the child.\n    `zw`, `zh` - The size the rectangle is rendered to.\n    `bilinear` - Should we be rendering in bilinear mode?\n    '
    rv = renpy.display.render.Render(zw, zh)
    if zw == 0 or zh == 0 or w == 0 or (h == 0):
        return rv
    rv.forward = renpy.display.matrix.Matrix2D(w / zw, 0, 0, h / zh)
    rv.reverse = renpy.display.matrix.Matrix2D(zw / w, 0, 0, zh / h)
    rv.xclipping = True
    rv.yclipping = True
    rv.blit(crend, rv.reverse.transform(-x, -y))
    return rv

class ZoomCommon(renpy.display.displayable.Displayable):

    def __init__(self, time, child, end_identity=False, after_child=None, time_warp=None, bilinear=True, opaque=True, anim_timebase=False, repeat=False, style='motion', **properties):
        if False:
            i = 10
            return i + 15
        '\n        @param time: The amount of time it will take to\n        interpolate from the start to the end rectange.\n\n        @param child: The child displayable.\n\n        @param after_child: If present, a second child\n        widget. This displayable will be rendered after the zoom\n        completes. Use this to snap to a sharp displayable after\n        the zoom is done.\n\n        @param time_warp: If not None, this is a function that takes a\n        fraction of the period (between 0.0 and 1.0), and returns a\n        new fraction of the period. Use this to warp time, applying\n        acceleration and deceleration to motions.\n        '
        super(ZoomCommon, self).__init__(style=style, **properties)
        child = renpy.easy.displayable(child)
        self.time = time
        self.child = child
        self.repeat = repeat
        if after_child:
            self.after_child = renpy.easy.displayable(after_child)
        elif end_identity:
            self.after_child = child
        else:
            self.after_child = None
        self.time_warp = time_warp
        self.bilinear = bilinear
        self.opaque = opaque
        self.anim_timebase = anim_timebase

    def visit(self):
        if False:
            return 10
        return [self.child, self.after_child]

    def zoom_rectangle(self, done, width, height):
        if False:
            print('Hello World!')
        raise Exception('Zoom rectangle not implemented.')

    def render(self, width, height, st, at):
        if False:
            i = 10
            return i + 15
        if self.anim_timebase:
            t = at
        else:
            t = st
        if self.time:
            done = min(t / self.time, 1.0)
        else:
            done = 1.0
        if self.repeat:
            done = done % 1.0
        if renpy.game.less_updates:
            done = 1.0
        self.done = done
        if self.after_child and done == 1.0:
            return renpy.display.render.render(self.after_child, width, height, st, at)
        if self.time_warp:
            done = self.time_warp(done)
        rend = renpy.display.render.render(self.child, width, height, st, at)
        (rx, ry, rw, rh, zw, zh) = self.zoom_rectangle(done, rend.width, rend.height)
        if rx < 0 or ry < 0 or rx + rw > rend.width or (ry + rh > rend.height):
            raise Exception('Zoom rectangle %r falls outside of %dx%d parent surface.' % ((rx, ry, rw, rh), rend.width, rend.height))
        rv = zoom_render(rend, rx, ry, rw, rh, zw, zh, self.bilinear)
        if self.done < 1.0:
            renpy.display.render.redraw(self, 0)
        return rv

    def event(self, ev, x, y, st):
        if False:
            i = 10
            return i + 15
        if not self.time:
            done = 1.0
        else:
            done = min(st / self.time, 1.0)
        if done == 1.0 and self.after_child:
            return self.after_child.event(ev, x, y, st)
        else:
            return None

class Zoom(ZoomCommon):

    def __init__(self, size, start, end, time, child, **properties):
        if False:
            print('Hello World!')
        end_identity = end == (0.0, 0.0) + size
        super(Zoom, self).__init__(time, child, end_identity=end_identity, **properties)
        self.size = size
        self.start = start
        self.end = end

    def zoom_rectangle(self, done, width, height):
        if False:
            return 10
        (rx, ry, rw, rh) = [a + (b - a) * done for (a, b) in zip(self.start, self.end)]
        return (rx, ry, rw, rh, self.size[0], self.size[1])

class FactorZoom(ZoomCommon):

    def __init__(self, start, end, time, child, **properties):
        if False:
            return 10
        end_identity = end == 1.0
        super(FactorZoom, self).__init__(time, child, end_identity=end_identity, **properties)
        self.start = start
        self.end = end

    def zoom_rectangle(self, done, width, height):
        if False:
            return 10
        factor = self.start + (self.end - self.start) * done
        return (0, 0, width, height, factor * width, factor * height)

class SizeZoom(ZoomCommon):

    def __init__(self, start, end, time, child, **properties):
        if False:
            for i in range(10):
                print('nop')
        end_identity = False
        super(SizeZoom, self).__init__(time, child, end_identity=end_identity, **properties)
        self.start = start
        self.end = end

    def zoom_rectangle(self, done, width, height):
        if False:
            print('Hello World!')
        (sw, sh) = self.start
        (ew, eh) = self.end
        zw = sw + (ew - sw) * done
        zh = sh + (eh - sh) * done
        return (0, 0, width, height, zw, zh)

class RotoZoom(renpy.display.displayable.Displayable):
    transform = None

    def __init__(self, rot_start, rot_end, rot_delay, zoom_start, zoom_end, zoom_delay, child, rot_repeat=False, zoom_repeat=False, rot_bounce=False, zoom_bounce=False, rot_anim_timebase=False, zoom_anim_timebase=False, rot_time_warp=None, zoom_time_warp=None, opaque=False, style='motion', **properties):
        if False:
            return 10
        super(RotoZoom, self).__init__(style=style, **properties)
        self.rot_start = rot_start
        self.rot_end = rot_end
        self.rot_delay = rot_delay
        self.zoom_start = zoom_start
        self.zoom_end = zoom_end
        self.zoom_delay = zoom_delay
        self.child = renpy.easy.displayable(child)
        self.rot_repeat = rot_repeat
        self.zoom_repeat = zoom_repeat
        self.rot_bounce = rot_bounce
        self.zoom_bounce = zoom_bounce
        self.rot_anim_timebase = rot_anim_timebase
        self.zoom_anim_timebase = zoom_anim_timebase
        self.rot_time_warp = rot_time_warp
        self.zoom_time_warp = zoom_time_warp
        self.opaque = opaque

    def visit(self):
        if False:
            print('Hello World!')
        return [self.child]

    def render(self, width, height, st, at):
        if False:
            print('Hello World!')
        if self.rot_anim_timebase:
            rot_time = at
        else:
            rot_time = st
        if self.zoom_anim_timebase:
            zoom_time = at
        else:
            zoom_time = st
        if self.rot_delay == 0:
            rot_time = 1.0
        else:
            rot_time /= self.rot_delay
        if self.zoom_delay == 0:
            zoom_time = 1.0
        else:
            zoom_time /= self.zoom_delay
        if self.rot_repeat:
            rot_time %= 1.0
        if self.zoom_repeat:
            zoom_time %= 1.0
        if self.rot_bounce:
            rot_time *= 2
            rot_time = min(rot_time, 2.0 - rot_time)
        if self.zoom_bounce:
            zoom_time *= 2
            zoom_time = min(zoom_time, 2.0 - zoom_time)
        if renpy.game.less_updates:
            rot_time = 1.0
            zoom_time = 1.0
        rot_time = min(rot_time, 1.0)
        zoom_time = min(zoom_time, 1.0)
        if self.rot_time_warp:
            rot_time = self.rot_time_warp(rot_time)
        if self.zoom_time_warp:
            zoom_time = self.zoom_time_warp(zoom_time)
        angle = self.rot_start + (1.0 * self.rot_end - self.rot_start) * rot_time
        zoom = self.zoom_start + (1.0 * self.zoom_end - self.zoom_start) * zoom_time
        zoom = max(zoom, 0.001)
        if self.transform is None:
            self.transform = Transform(self.child)
        self.transform.rotate = angle
        self.transform.zoom = zoom
        rv = renpy.display.render.render(self.transform, width, height, st, at)
        if rot_time <= 1.0 or zoom_time <= 1.0:
            renpy.display.render.redraw(self.transform, 0)
        return rv
renpy.display.layout.Transform = Transform
renpy.display.layout.RotoZoom = RotoZoom
renpy.display.layout.SizeZoom = SizeZoom
renpy.display.layout.FactorZoom = FactorZoom
renpy.display.layout.Zoom = Zoom
renpy.display.layout.Revolver = Revolver
renpy.display.layout.Motion = Motion
renpy.display.layout.Interpolate = Interpolate
renpy.display.layout.Revolve = Revolve
renpy.display.layout.Move = Move
renpy.display.layout.Pan = Pan