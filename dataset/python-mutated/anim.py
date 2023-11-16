from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import random
import renpy

class State(object):
    """
    This creates a state that can be used in a SMAnimation.
    """

    def __init__(self, name, image, *atlist, **properties):
        if False:
            print('Hello World!')
        '\n        @param name: A string giving the name of this state.\n\n        @param image: The displayable that is shown to the user while\n        we are in (entering) this state. For convenience, this can\n        also be a string or tuple, which is interpreted with Image.\n\n        image should be None when this State is used with motion,\n        to indicate that the image will be replaced with the child of\n        the motion.\n\n        @param atlist: A list of functions to call on the image. (In\n        general, if something can be used in an at clause, it can be\n        used here as well.)\n\n        If any keyword arguments are given, they are used to construct a\n        Position object, that modifies the position of the image.\n        '
        if image and (not isinstance(image, renpy.display.displayable.Displayable)):
            image = renpy.easy.displayable(image)
        self.name = name
        self.image = image
        self.atlist = atlist
        self.properties = properties

    def add(self, sma):
        if False:
            print('Hello World!')
        sma.states[self.name] = self

    def get_image(self):
        if False:
            print('Hello World!')
        rv = self.image
        for i in self.atlist:
            rv = i(rv)
        if self.properties:
            rv = renpy.display.layout.Position(rv, **self.properties)
        return rv

    def motion_copy(self, child):
        if False:
            while True:
                i = 10
        if self.image is not None:
            child = self.image
        return State(self.name, child, *self.atlist)

class Edge(object):
    """
    This creates an edge that can be used with a SMAnimation.
    """

    def __init__(self, old, delay, new, trans=None, prob=1):
        if False:
            while True:
                i = 10
        "\n        @param old: The name (a string) of the state that this transition is from.\n\n        @param delay: The number of seconds that this transition takes.\n\n        @param new: The name (a string) of the state that this transition is to.\n\n        @param trans: The transition that will be used to show the\n        image found in the new state. If None, the image is show\n        immediately.\n\n        When used with an SMMotion, the transition should probably be\n        move.\n\n        @param prob: The number of times this edge is added. This can\n        be used to make a transition more probable then others. For\n        example, if one transition out of a state has prob=5, and the\n        other has prob=1, then the one with prob=5 will execute 5/6 of\n        the time, while the one with prob=1 will only occur 1/6 of the\n        time. (Don't make this too large, as memory use is proportional to\n        this value.)\n        "
        self.old = old
        self.delay = delay
        self.new = new
        self.trans = trans
        self.prob = prob

    def add(self, sma):
        if False:
            return 10
        for _i in range(0, self.prob):
            sma.edges.setdefault(self.old, []).append(self)

class SMAnimation(renpy.display.displayable.Displayable):
    """
    This creates a state-machine animation. Such an animation is
    created by randomly traversing the edges between states in a
    defined state machine. Each state corresponds to an image shown to
    the user, with the edges corresponding to the amount of time an
    image is shown, and the transition it is shown with.

    Images are shown, perhaps with a transition, when we are
    transitioning into a state containing that image.
    """

    def __init__(self, initial, *args, **properties):
        if False:
            print('Hello World!')
        '\n        @param initial: The name (a string) of the initial state we\n        start in.\n\n        @param showold: If the keyword parameter showold is True, then\n        the old image is shown instead of the new image when in an\n        edge.\n\n        @param anim_timebase: If True, we use the animation\n        timebase. If False, we use the displayable timebase.\n\n        This accepts as additional arguments the anim.State and\n        anim.Edge objects that are used to make up this state\n        machine.\n        '
        if 'delay' in properties:
            self.delay = properties['delay']
            del properties['delay']
        else:
            self.delay = None
        if 'showold' in properties:
            self.showold = properties['showold']
            del properties['showold']
        else:
            self.showold = False
        if 'anim_timebase' in properties:
            self.anim_timebase = properties['anim_timebase']
            del properties['anim_timebase']
        else:
            self.anim_timebase = True
        super(SMAnimation, self).__init__(**properties)
        self.properties = properties
        self.initial = initial
        self.states = {}
        self.edges = {}
        for i in args:
            i.add(self)
        self.edge_start = None
        self.edge_cache = None
        self.edge = None
        self.state = None

    def visit(self):
        if False:
            i = 10
            return i + 15
        return [i.image for i in self.states.values()]

    def pick_edge(self, state):
        if False:
            return 10
        '\n        This randomly picks an edge out of the given state, if\n        one exists. It updates self.edge if a transition has\n        been selected, or returns None if none can be found. It also\n        updates self.image to be the new image on the selected edge.\n        '
        if state not in self.edges:
            self.edge = None
            return
        edges = self.edges[state]
        self.edge = random.choice(edges)
        self.state = self.edge.new

    def update_cache(self):
        if False:
            while True:
                i = 10
        '\n        Places the correct Displayable into the edge cache, based on\n        what is contained in the given edge. This takes into account\n        the old and new states, and any transition that is present.\n        '
        if self.edge.trans:
            im = self.edge.trans(old_widget=self.states[self.edge.old].get_image(), new_widget=self.states[self.edge.new].get_image())
        elif self.showold:
            im = self.states[self.edge.old].get_image()
        else:
            im = self.states[self.edge.new].get_image()
        self.edge_cache = im

    def get_placement(self):
        if False:
            return 10
        if self.edge_cache:
            return self.edge_cache.get_placement()
        if self.state:
            return self.states[self.state].get_image().get_placement()
        return super(SMAnimation, self).get_placement()

    def render(self, width, height, st, at):
        if False:
            i = 10
            return i + 15
        if self.anim_timebase:
            t = at
        else:
            t = st
        if self.edge_start is None or t < self.edge_start:
            self.edge_start = t
            self.edge_cache = None
            self.pick_edge(self.initial)
        while self.edge and t > self.edge_start + self.edge.delay:
            self.edge_start += self.edge.delay
            self.edge_cache = None
            self.pick_edge(self.edge.new)
        if not self.edge:
            im = renpy.display.render.render(self.states[self.state].get_image(), width, height, st - self.edge_start, at)
        else:
            if not self.edge_cache:
                self.update_cache()
            im = renpy.display.render.render(self.edge_cache, width, height, t - self.edge_start, at)
            if not renpy.game.less_updates:
                renpy.display.render.redraw(self.edge_cache, self.edge.delay - (t - self.edge_start))
        (iw, ih) = im.get_size()
        rv = renpy.display.render.Render(iw, ih)
        rv.blit(im, (0, 0))
        return rv

    def __call__(self, child=None, new_widget=None, old_widget=None):
        if False:
            return 10
        '\n        Used when this SMAnimation is used as a SMMotion. This creates\n        a duplicate of the animation, with all states containing None\n        as the image having that None replaced with the image that is provided here.\n        '
        if child is None:
            child = new_widget
        args = []
        for state in self.states.values():
            args.append(state.motion_copy(child))
        for edges in self.edges.values():
            args.extend(edges)
        return SMAnimation(self.initial, *args, delay=self.delay, **self.properties)

def Animation(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    newargs = []
    for (i, a) in enumerate(args):
        newargs.append(a)
        if i % 2 == 1:
            newargs.append(None)
    return TransitionAnimation(*newargs, **kwargs)

class TransitionAnimation(renpy.display.displayable.Displayable):
    """
    A displayable that draws an animation with each frame separated
    by a transition.
    """

    def __init__(self, *args, **properties):
        if False:
            while True:
                i = 10
        '\n        This takes arguments such that the 1st, 4th, 7th, ...\n        arguments are displayables, the 2nd, 5th, 8th, ... on arguments\n        are times, and the 3rd, 6th, 9th, ... are transitions.\n\n        This displays the first displayable for the given time, then\n        transitions to the second displayable using the given\n        transition, and shows it for the given time (the time of the\n        transition is taken out of the time the frame is shown), and\n        so on.\n\n        The last argument may be a displayable (in which case that\n        displayable is used to transition back to the first frame), or\n        a displayable (which is shown forever).\n\n        There is one keyword argument, apart from the style properties:\n\n        @param anim_timebase: If True, the default, use the animation\n        timebase. Otherwise, use the displayable timebase.\n        '
        properties.setdefault('style', 'animation')
        self.anim_timebase = properties.pop('anim_timebase', True)
        super(TransitionAnimation, self).__init__(**properties)
        images = []
        delays = []
        transitions = []
        for (i, arg) in enumerate(args):
            if i % 3 == 0:
                images.append(renpy.easy.displayable(arg))
            elif i % 3 == 1:
                delays.append(arg)
            else:
                transitions.append(arg)
        if len(images) > len(delays):
            delays.append(365.25 * 86400.0)
        if len(images) > len(transitions):
            transitions.append(None)
        self.images = images
        self.prev_images = [images[-1]] + images[:-1]
        self.delays = delays
        self.transitions = [transitions[-1]] + transitions[:-1]

    def render(self, width, height, st, at):
        if False:
            print('Hello World!')
        if self.anim_timebase:
            orig_t = at
        else:
            orig_t = st
        t = orig_t % sum(self.delays)
        for (image, prev, delay, trans) in zip(self.images, self.prev_images, self.delays, self.transitions):
            if t < delay:
                if not renpy.game.less_updates:
                    renpy.display.render.redraw(self, delay - t)
                if trans and orig_t >= self.delays[0]:
                    image = trans(old_widget=prev, new_widget=image)
                im = renpy.display.render.render(image, width, height, t, at)
                (width, height) = im.get_size()
                rv = renpy.display.render.Render(width, height)
                rv.blit(im, (0, 0))
                return rv
            else:
                t = t - delay

    def visit(self):
        if False:
            while True:
                i = 10
        return self.images

class Blink(renpy.display.displayable.Displayable):

    def __init__(self, image, on=0.5, off=0.5, rise=0.5, set=0.5, high=1.0, low=0.0, offset=0.0, anim_timebase=False, **properties):
        if False:
            print('Hello World!')
        '\n        This takes as an argument an image or widget, and blinks that image\n        by varying its alpha. The sequence of phases is\n        on - set - off - rise - on - ... All times are given in seconds, all\n        alphas are fractions between 0 and 1.\n\n        @param image: The image or widget that will be blinked.\n\n        @param on: The amount of time the widget spends on, at high alpha.\n\n        @param off: The amount of time the widget spends off, at low alpha.\n\n        @param rise: The amount time the widget takes to ramp from low to high alpha.\n\n        @param set: The amount of time the widget takes to ram from high to low.\n\n        @param high: The high alpha.\n\n        @param low: The low alpha.\n\n        @param offset: A time offset, in seconds. Use this to have a\n        blink that does not start at the start of the on phase.\n\n        @param anim_timebase: If True, use the animation timebase, if false, the displayable timebase.\n        '
        super(Blink, self).__init__(**properties)
        self.image = renpy.easy.displayable(image)
        self.on = on
        self.off = off
        self.rise = rise
        self.set = set
        self.high = high
        self.low = low
        self.offset = offset
        self.anim_timebase = anim_timebase
        self.cycle = on + set + off + rise

    def visit(self):
        if False:
            i = 10
            return i + 15
        return [self.image]

    def render(self, height, width, st, at):
        if False:
            while True:
                i = 10
        delay = 0
        if self.anim_timebase:
            t = at
        else:
            t = st
        time = (self.offset + t) % self.cycle
        alpha = self.high
        if 0 <= time < self.on:
            delay = self.on - time
            alpha = self.high
        time -= self.on
        if 0 <= time < self.set:
            delay = 0
            frac = time / self.set
            alpha = self.low * frac + self.high * (1.0 - frac)
        time -= self.set
        if 0 <= time < self.off:
            delay = self.off - time
            alpha = self.low
        time -= self.off
        if 0 <= time < self.rise:
            delay = 0
            frac = time / self.rise
            alpha = self.high * frac + self.low * (1.0 - frac)
        rend = renpy.display.render.render(self.image, height, width, st, at)
        (w, h) = rend.get_size()
        rv = renpy.display.render.Render(w, h)
        rv.blit(rend, (0, 0))
        rv.alpha = alpha
        rv.add_shader('renpy.alpha')
        rv.add_uniform('u_renpy_alpha', alpha)
        rv.add_uniform('u_renpy_over', 1.0)
        if not renpy.game.less_updates:
            renpy.display.render.redraw(self, delay)
        return rv

def Filmstrip(image, framesize, gridsize, delay, frames=None, loop=True, **properties):
    if False:
        return 10
    '\n    This creates an animation from a single image. This image\n    must consist of a grid of frames, with the number of columns and\n    rows in the grid being taken from gridsize, and the size of each\n    frame in the grid being taken from framesize. This takes frames\n    and sticks them into an Animation, with the given delay between\n    each frame. The frames are taken by going from left-to-right\n    across the first row, left-to-right across the second row, and\n    so on until all frames are consumed, or a specified number of\n    frames are taken.\n\n    @param image: The image that the frames must be taken from.\n\n    @param framesize: A (width, height) tuple giving the size of\n    each of the frames in the animation.\n\n    @param gridsize: A (columns, rows) tuple giving the number of\n    columns and rows in the grid.\n\n    @param delay: The delay, in seconds, between frames.\n\n    @param frames: The number of frames in this animation. If None,\n    then this defaults to colums * rows frames, that is, taking\n    every frame in the grid.\n\n    @param loop: If True, loop at the end of the animation. If False,\n    this performs the animation once, and then stops.\n\n    Other keyword arguments are as for anim.SMAnimation.\n    '
    (width, height) = framesize
    (cols, rows) = gridsize
    if frames is None:
        frames = cols * rows
    i = 0
    args = []
    for r in range(0, rows):
        for c in range(0, cols):
            x = c * width
            y = r * height
            args.append(renpy.display.transform.Transform(image, crop=(x, y, width, height)))
            args.append(delay)
            i += 1
            if i == frames:
                break
        if i == frames:
            break
    if not loop:
        args.pop()
    return Animation(*args, **properties)