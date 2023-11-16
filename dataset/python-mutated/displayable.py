from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import copy
import renpy

def place(width, height, sw, sh, placement):
    if False:
        i = 10
        return i + 15
    "\n    Performs the Ren'Py placement algorithm.\n\n    `width`, `height`\n        The width and height of the area the image will be\n        placed in.\n\n    `sw`, `sh`\n        The size of the image to be placed.\n\n    `placement`\n        The tuple returned by Displayable.get_placement().\n    "
    (xpos, ypos, xanchor, yanchor, xoffset, yoffset, _subpixel) = placement
    compute_raw = renpy.display.core.absolute.compute_raw
    if xpos is None:
        xpos = 0
    if ypos is None:
        ypos = 0
    if xanchor is None:
        xanchor = 0
    if yanchor is None:
        yanchor = 0
    if xoffset is None:
        xoffset = 0
    if yoffset is None:
        yoffset = 0
    xpos = compute_raw(xpos, width)
    xanchor = compute_raw(xanchor, sw)
    x = xpos + xoffset - xanchor
    ypos = compute_raw(ypos, height)
    yanchor = compute_raw(yanchor, sh)
    y = ypos + yoffset - yanchor
    return (x, y)

class DisplayableArguments(renpy.object.Object):
    """
    Represents a set of arguments that can be passed to a duplicated
    displayable.
    """
    name = ()
    args = ()
    prefix = None
    lint = False

    def copy(self, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns a copy of this object with the various fields set to the\n        values they were given in kwargs.\n        '
        rv = DisplayableArguments()
        rv.__dict__.update(self.__dict__)
        rv.__dict__.update(kwargs)
        return rv

    def extraneous(self):
        if False:
            return 10
        if renpy.config.developer and renpy.config.report_extraneous_attributes:
            raise Exception("Image '{}' does not accept attributes '{}'.".format(' '.join(self.name), ' '.join(self.args)))
default_style = renpy.style.Style('default')

class Displayable(renpy.object.Object):
    """
    The base class for every object in Ren'Py that can be
    displayed to the screen.

    Drawables will be serialized to a savegame file. Therefore, they
    shouldn't store non-serializable things (like pygame surfaces) in
    their fields.
    """
    focusable = None
    full_focus_name = None
    role = ''
    transform_event = None
    transform_event_responder = False
    _main = None
    _composite_parts = []
    _location = None
    _uses_scope = False
    _args = DisplayableArguments()
    _duplicatable = False
    _clipping = False
    _tooltip = None
    _box_skip = False
    _offer_size = None
    _draggable = False
    delay = None

    def __ne__(self, o):
        if False:
            return 10
        return not self == o

    def __init__(self, focus=None, default=False, style='default', _args=None, tooltip=None, default_focus=False, **properties):
        if False:
            for i in range(10):
                print('nop')
        global default_style
        if style == 'default' and (not properties):
            self.style = default_style
        else:
            self.style = renpy.style.Style(style, properties)
        self.focus_name = focus
        self.default = default or default_focus
        self._tooltip = tooltip
        if _args is not None:
            self._args = _args

    def _copy(self, args=None):
        if False:
            i = 10
            return i + 15
        '\n        Makes a shallow copy of the displayable. If `args` is provided,\n        replaces the arguments with the stored copy.\n        '
        rv = copy.copy(self)
        if args is not None:
            rv._args = args
        return rv

    def _duplicate(self, args):
        if False:
            print('Hello World!')
        '\n        Makes a duplicate copy of the following kids of displayables:\n\n        * Displayables that can accept arguments.\n        * Displayables that maintain state that should be reset before being\n          shown to the user.\n        * Containers that contain (including transitively) one of the other\n          kinds of displayables.\n\n        Displayables that contain state that can be manipulated by the user\n        are never copied.\n\n        This should call _unique on children that have been copied before\n        setting its own _duplicatable flag.\n        '
        if args and args.args:
            args.extraneous()
        return self

    def _get_tooltip(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the tooltip of this displayable.\n        '
        return self._tooltip

    def _in_current_store(self):
        if False:
            return 10
        '\n        Returns a version of this displayable that will not change as it is\n        rendered.\n        '
        return self

    def _unique(self):
        if False:
            i = 10
            return i + 15
        '\n        This is called when a displayable is "unique", meaning there will\n        only be one reference to it, ever, from the tree of displayables.\n        '
        self._duplicatable = False
        return

    def parameterize(self, name, parameters):
        if False:
            i = 10
            return i + 15
        '\n        Obsolete alias for _duplicate.\n        '
        a = self._args.copy(name=name, args=parameters)
        return self._duplicate(a)

    def _equals(self, o):
        if False:
            print('Hello World!')
        "\n        This is a utility method that can be called by a Displayable's\n        __eq__ method, to compare displayables for type and displayable\n        component equality.\n        "
        if type(self) is not type(o):
            return False
        if self.focus_name != o.focus_name:
            return False
        if self.style != o.style:
            return False
        if self.default != o.default:
            return False
        return True

    def _repr_info(self):
        if False:
            return 10
        return None

    def __repr__(self):
        if False:
            while True:
                i = 10
        rep = object.__repr__(self)
        reprinfo = self._repr_info()
        if reprinfo is None:
            return rep
        if reprinfo and (not (reprinfo[0] == '(' and reprinfo[-1] == ')')):
            reprinfo = ''.join(('(', reprinfo, ')'))
        parto = rep.rpartition(' at ')
        return ' '.join((parto[0], reprinfo, 'at', parto[2]))

    def find_focusable(self, callback, focus_name):
        if False:
            return 10
        focus_name = self.focus_name or focus_name
        if self.focusable:
            callback(self, focus_name)
        elif self.focusable is not None:
            callback(None, focus_name)
        for i in self.visit():
            if i is None:
                continue
            i.find_focusable(callback, focus_name)

    def focus(self, default=False):
        if False:
            i = 10
            return i + 15
        '\n        Called to indicate that this widget has the focus.\n        '
        self.set_style_prefix(self.role + 'hover_', True)
        if not default:
            renpy.exports.play(self.style.hover_sound)

    def unfocus(self, default=False):
        if False:
            while True:
                i = 10
        '\n        Called to indicate that this widget has become unfocused.\n        '
        self.set_style_prefix(self.role + 'idle_', True)

    def is_focused(self):
        if False:
            print('Hello World!')
        if renpy.display.focus.grab and renpy.display.focus.grab is not self:
            return
        return renpy.game.context().scene_lists.focused is self

    def set_style_prefix(self, prefix, root):
        if False:
            i = 10
            return i + 15
        '\n        Called to set the style prefix of this widget and its child\n        widgets, if any.\n\n        `root` - True if this is the root of a style tree, False if this\n        has been passed on to a child.\n        '
        if prefix == self.style.prefix:
            return
        self.style.set_prefix(prefix)
        renpy.display.render.redraw(self, 0)

    def render(self, width, height, st, at):
        if False:
            i = 10
            return i + 15
        "\n        Called to display this displayable. This is called with width\n        and height parameters, which give the largest width and height\n        that this drawable can be drawn to without overflowing some\n        bounding box. It's also given two times. It returns a Surface\n        that is the current image of this drawable.\n\n        @param st: The time since this widget was first shown, in seconds.\n        @param at: The time since a similarly named widget was first shown,\n        in seconds.\n        "
        raise Exception('Render not implemented.')

    def event(self, ev, x, y, st):
        if False:
            print('Hello World!')
        '\n        Called to report than an event has occured. Ev is the raw\n        pygame event object representing that event. If the event\n        involves the mouse, x and y are the translation of the event\n        into the coordinates of this displayable. st is the time this\n        widget has been shown for.\n\n        @returns A value that should be returned from Interact, or None if\n        no value is appropriate.\n        '
        return None

    def get_placement(self):
        if False:
            while True:
                i = 10
        '\n        Returns a style object containing placement information for\n        this Displayable. Children are expected to overload this\n        to return something more sensible.\n        '
        return self.style.get_placement()

    def visit_all(self, callback, seen=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls the callback on this displayable, and then on all children\n        of this displayable.\n        '
        if seen is None:
            seen = set()
        for d in self.visit():
            if d is None:
                continue
            id_d = id(d)
            if id_d in seen:
                continue
            seen.add(id_d)
            d.visit_all(callback, seen)
        callback(self)

    def visit(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called to ask the displayable to return a list of its children\n        (including children taken from styles). For convenience, this\n        list may also include None values.\n        '
        return []

    def per_interact(self):
        if False:
            i = 10
            return i + 15
        '\n        Called once per widget per interaction.\n        '
        return None

    def predict_one(self):
        if False:
            while True:
                i = 10
        '\n        Called to ask this displayable to call the callback with all\n        the images it may want to load.\n        '
        return

    def predict_one_action(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called to ask this displayable to cause image prediction\n        to occur for images that may be loaded by its actions.\n        '
        return

    def place(self, dest, x, y, width, height, surf, main=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        This places a render (which must be of this displayable)\n        within a bounding area. Returns an (x, y) tuple giving the location\n        the displayable was placed at.\n\n        `dest`\n            If not None, the `surf` will be blitted to `dest` at the\n            computed coordinates.\n\n        `x`, `y`, `width`, `height`\n            The bounding area.\n\n        `surf`\n            The render to place.\n\n        `main`\n            This is passed to Render.blit().\n        '
        placement = self.get_placement()
        subpixel = placement[6]
        (xpos, ypos) = place(width, height, surf.width, surf.height, placement)
        xpos += x
        ypos += y
        pos = (xpos, ypos)
        if dest is not None:
            if subpixel:
                dest.subpixel_blit(surf, pos, main, main, None)
            else:
                dest.blit(surf, pos, main, main, None)
        return pos

    def set_transform_event(self, event):
        if False:
            while True:
                i = 10
        '\n        Sets the transform event of this displayable to event.\n        '
        if event == self.transform_event:
            return
        self.transform_event = event
        if self.transform_event_responder:
            renpy.display.render.redraw(self, 0)

    def _handles_event(self, event):
        if False:
            print('Hello World!')
        '\n        Returns True if the displayable handles event, False otherwise.\n        '
        return False

    def _hide(self, st, at, kind):
        if False:
            print('Hello World!')
        '\n        Returns None if this displayable is ready to be hidden, or\n        a replacement displayable if it doesn\'t want to be hidden\n        quite yet.\n\n        Kind may be "hide", "replace", or "cancel", with the latter\n        being called when the hide is being hidden itself because\n        another displayable is shown.\n        '
        return None

    def _show(self):
        if False:
            print('Hello World!')
        '\n        No longer used.\n        '

    def _target(self):
        if False:
            return 10
        '\n        If this displayable is part of a chain of one or more references,\n        returns the ultimate target of those references. Otherwise, returns\n        the displayable.\n        '
        return self

    def _change_transform_child(self, child):
        if False:
            while True:
                i = 10
        '\n        If this is a transform, makes a copy of the transform and sets\n        the child of the innermost transform to this. Otherwise,\n        simply returns child.\n        '
        return child

    def _clear(self):
        if False:
            while True:
                i = 10
        '\n        Clears out the children of this displayable, if any.\n        '
        return

    def _tts_common(self, default_alt=None, reverse=False):
        if False:
            for i in range(10):
                print('nop')
        rv = []
        if reverse:
            order = -1
        else:
            order = 1
        speech = ''
        for i in self.visit()[::order]:
            if i is not None:
                speech = i._tts()
                if speech.strip():
                    if isinstance(speech, renpy.display.tts.TTSDone):
                        rv = [speech]
                    else:
                        rv.append(speech)
        rv = ': '.join(rv)
        rv = rv.replace('::', ':')
        rv = rv.replace(': :', ':')
        alt = self.style.alt
        if alt is None:
            alt = default_alt
        if alt is not None:
            rv = renpy.substitutions.substitute(alt, scope={'text': rv})[0]
        rv = type(speech)(rv)
        return rv

    def _tts(self):
        if False:
            print('Hello World!')
        '\n        Returns the self-voicing text of this displayable and all of its\n        children that cannot take focus. If the displayable can take focus,\n        returns the empty string.\n        '
        return self._tts_common()

    def _tts_all(self):
        if False:
            return 10
        '\n        Returns the self-voicing text of this displayable and all of its\n        children that cannot take focus.\n        '
        return self._tts_common()