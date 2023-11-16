import unittest
import tkinter
from tkinter.test.support import AbstractTkTest, tcl_version, requires_tcl, get_tk_patchlevel, pixels_conv, tcl_obj_eq
import test.support
noconv = False
if get_tk_patchlevel() < (8, 5, 11):
    noconv = str
pixels_round = round
if get_tk_patchlevel()[:3] == (8, 5, 11):
    pixels_round = int
_sentinel = object()

class AbstractWidgetTest(AbstractTkTest):
    _conv_pixels = staticmethod(pixels_round)
    _conv_pad_pixels = None
    _stringify = False

    @property
    def scaling(self):
        if False:
            return 10
        try:
            return self._scaling
        except AttributeError:
            self._scaling = float(self.root.call('tk', 'scaling'))
            return self._scaling

    def _str(self, value):
        if False:
            for i in range(10):
                print('nop')
        if not self._stringify and self.wantobjects and (tcl_version >= (8, 6)):
            return value
        if isinstance(value, tuple):
            return ' '.join(map(self._str, value))
        return str(value)

    def assertEqual2(self, actual, expected, msg=None, eq=object.__eq__):
        if False:
            return 10
        if eq(actual, expected):
            return
        self.assertEqual(actual, expected, msg)

    def checkParam(self, widget, name, value, *, expected=_sentinel, conv=False, eq=None):
        if False:
            for i in range(10):
                print('nop')
        widget[name] = value
        if expected is _sentinel:
            expected = value
        if conv:
            expected = conv(expected)
        if self._stringify or not self.wantobjects:
            if isinstance(expected, tuple):
                expected = tkinter._join(expected)
            else:
                expected = str(expected)
        if eq is None:
            eq = tcl_obj_eq
        self.assertEqual2(widget[name], expected, eq=eq)
        self.assertEqual2(widget.cget(name), expected, eq=eq)
        t = widget.configure(name)
        self.assertEqual(len(t), 5)
        self.assertEqual2(t[4], expected, eq=eq)

    def checkInvalidParam(self, widget, name, value, errmsg=None, *, keep_orig=True):
        if False:
            for i in range(10):
                print('nop')
        orig = widget[name]
        if errmsg is not None:
            errmsg = errmsg.format(value)
        with self.assertRaises(tkinter.TclError) as cm:
            widget[name] = value
        if errmsg is not None:
            self.assertEqual(str(cm.exception), errmsg)
        if keep_orig:
            self.assertEqual(widget[name], orig)
        else:
            widget[name] = orig
        with self.assertRaises(tkinter.TclError) as cm:
            widget.configure({name: value})
        if errmsg is not None:
            self.assertEqual(str(cm.exception), errmsg)
        if keep_orig:
            self.assertEqual(widget[name], orig)
        else:
            widget[name] = orig

    def checkParams(self, widget, name, *values, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        for value in values:
            self.checkParam(widget, name, value, **kwargs)

    def checkIntegerParam(self, widget, name, *values, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.checkParams(widget, name, *values, **kwargs)
        self.checkInvalidParam(widget, name, '', errmsg='expected integer but got ""')
        self.checkInvalidParam(widget, name, '10p', errmsg='expected integer but got "10p"')
        self.checkInvalidParam(widget, name, 3.2, errmsg='expected integer but got "3.2"')

    def checkFloatParam(self, widget, name, *values, conv=float, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        for value in values:
            self.checkParam(widget, name, value, conv=conv, **kwargs)
        self.checkInvalidParam(widget, name, '', errmsg='expected floating-point number but got ""')
        self.checkInvalidParam(widget, name, 'spam', errmsg='expected floating-point number but got "spam"')

    def checkBooleanParam(self, widget, name):
        if False:
            i = 10
            return i + 15
        for value in (False, 0, 'false', 'no', 'off'):
            self.checkParam(widget, name, value, expected=0)
        for value in (True, 1, 'true', 'yes', 'on'):
            self.checkParam(widget, name, value, expected=1)
        self.checkInvalidParam(widget, name, '', errmsg='expected boolean value but got ""')
        self.checkInvalidParam(widget, name, 'spam', errmsg='expected boolean value but got "spam"')

    def checkColorParam(self, widget, name, *, allow_empty=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.checkParams(widget, name, '#ff0000', '#00ff00', '#0000ff', '#123456', 'red', 'green', 'blue', 'white', 'black', 'grey', **kwargs)
        self.checkInvalidParam(widget, name, 'spam', errmsg='unknown color name "spam"')

    def checkCursorParam(self, widget, name, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.checkParams(widget, name, 'arrow', 'watch', 'cross', '', **kwargs)
        if tcl_version >= (8, 5):
            self.checkParam(widget, name, 'none')
        self.checkInvalidParam(widget, name, 'spam', errmsg='bad cursor spec "spam"')

    def checkCommandParam(self, widget, name):
        if False:
            return 10

        def command(*args):
            if False:
                return 10
            pass
        widget[name] = command
        self.assertTrue(widget[name])
        self.checkParams(widget, name, '')

    def checkEnumParam(self, widget, name, *values, errmsg=None, **kwargs):
        if False:
            i = 10
            return i + 15
        self.checkParams(widget, name, *values, **kwargs)
        if errmsg is None:
            errmsg2 = ' %s "{}": must be %s%s or %s' % (name, ', '.join(values[:-1]), ',' if len(values) > 2 else '', values[-1])
            self.checkInvalidParam(widget, name, '', errmsg='ambiguous' + errmsg2)
            errmsg = 'bad' + errmsg2
        self.checkInvalidParam(widget, name, 'spam', errmsg=errmsg)

    def checkPixelsParam(self, widget, name, *values, conv=None, keep_orig=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if conv is None:
            conv = self._conv_pixels
        for value in values:
            expected = _sentinel
            conv1 = conv
            if isinstance(value, str):
                if conv1 and conv1 is not str:
                    expected = pixels_conv(value) * self.scaling
                    conv1 = round
            self.checkParam(widget, name, value, expected=expected, conv=conv1, **kwargs)
        self.checkInvalidParam(widget, name, '6x', errmsg='bad screen distance "6x"', keep_orig=keep_orig)
        self.checkInvalidParam(widget, name, 'spam', errmsg='bad screen distance "spam"', keep_orig=keep_orig)

    def checkReliefParam(self, widget, name):
        if False:
            for i in range(10):
                print('nop')
        self.checkParams(widget, name, 'flat', 'groove', 'raised', 'ridge', 'solid', 'sunken')
        errmsg = 'bad relief "spam": must be flat, groove, raised, ridge, solid, or sunken'
        if tcl_version < (8, 6):
            errmsg = None
        self.checkInvalidParam(widget, name, 'spam', errmsg=errmsg)

    def checkImageParam(self, widget, name):
        if False:
            i = 10
            return i + 15
        image = tkinter.PhotoImage(master=self.root, name='image1')
        self.checkParam(widget, name, image, conv=str)
        self.checkInvalidParam(widget, name, 'spam', errmsg='image "spam" doesn\'t exist')
        widget[name] = ''

    def checkVariableParam(self, widget, name, var):
        if False:
            print('Hello World!')
        self.checkParam(widget, name, var, conv=str)

    def assertIsBoundingBox(self, bbox):
        if False:
            return 10
        self.assertIsNotNone(bbox)
        self.assertIsInstance(bbox, tuple)
        if len(bbox) != 4:
            self.fail('Invalid bounding box: %r' % (bbox,))
        for item in bbox:
            if not isinstance(item, int):
                self.fail('Invalid bounding box: %r' % (bbox,))
                break

    def test_keys(self):
        if False:
            print('Hello World!')
        widget = self.create()
        keys = widget.keys()
        self.assertEqual(sorted(keys), sorted(widget.configure()))
        for k in keys:
            widget[k]
        if test.support.verbose:
            aliases = {'bd': 'borderwidth', 'bg': 'background', 'fg': 'foreground', 'invcmd': 'invalidcommand', 'vcmd': 'validatecommand'}
            keys = set(keys)
            expected = set(self.OPTIONS)
            for k in sorted(keys - expected):
                if not (k in aliases and aliases[k] in keys and (aliases[k] in expected)):
                    print('%s.OPTIONS doesn\'t contain "%s"' % (self.__class__.__name__, k))

class StandardOptionsTests:
    STANDARD_OPTIONS = ('activebackground', 'activeborderwidth', 'activeforeground', 'anchor', 'background', 'bitmap', 'borderwidth', 'compound', 'cursor', 'disabledforeground', 'exportselection', 'font', 'foreground', 'highlightbackground', 'highlightcolor', 'highlightthickness', 'image', 'insertbackground', 'insertborderwidth', 'insertofftime', 'insertontime', 'insertwidth', 'jump', 'justify', 'orient', 'padx', 'pady', 'relief', 'repeatdelay', 'repeatinterval', 'selectbackground', 'selectborderwidth', 'selectforeground', 'setgrid', 'takefocus', 'text', 'textvariable', 'troughcolor', 'underline', 'wraplength', 'xscrollcommand', 'yscrollcommand')

    def test_configure_activebackground(self):
        if False:
            print('Hello World!')
        widget = self.create()
        self.checkColorParam(widget, 'activebackground')

    def test_configure_activeborderwidth(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkPixelsParam(widget, 'activeborderwidth', 0, 1.3, 2.9, 6, -2, '10p')

    def test_configure_activeforeground(self):
        if False:
            while True:
                i = 10
        widget = self.create()
        self.checkColorParam(widget, 'activeforeground')

    def test_configure_anchor(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkEnumParam(widget, 'anchor', 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw', 'center')

    def test_configure_background(self):
        if False:
            return 10
        widget = self.create()
        self.checkColorParam(widget, 'background')
        if 'bg' in self.OPTIONS:
            self.checkColorParam(widget, 'bg')

    def test_configure_bitmap(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.create()
        self.checkParam(widget, 'bitmap', 'questhead')
        self.checkParam(widget, 'bitmap', 'gray50')
        filename = test.support.findfile('python.xbm', subdir='imghdrdata')
        self.checkParam(widget, 'bitmap', '@' + filename)
        if not ('aqua' in self.root.tk.call('tk', 'windowingsystem') and 'AppKit' in self.root.winfo_server()):
            self.checkInvalidParam(widget, 'bitmap', 'spam', errmsg='bitmap "spam" not defined')

    def test_configure_borderwidth(self):
        if False:
            print('Hello World!')
        widget = self.create()
        self.checkPixelsParam(widget, 'borderwidth', 0, 1.3, 2.6, 6, -2, '10p')
        if 'bd' in self.OPTIONS:
            self.checkPixelsParam(widget, 'bd', 0, 1.3, 2.6, 6, -2, '10p')

    def test_configure_compound(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkEnumParam(widget, 'compound', 'bottom', 'center', 'left', 'none', 'right', 'top')

    def test_configure_cursor(self):
        if False:
            return 10
        widget = self.create()
        self.checkCursorParam(widget, 'cursor')

    def test_configure_disabledforeground(self):
        if False:
            while True:
                i = 10
        widget = self.create()
        self.checkColorParam(widget, 'disabledforeground')

    def test_configure_exportselection(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkBooleanParam(widget, 'exportselection')

    def test_configure_font(self):
        if False:
            print('Hello World!')
        widget = self.create()
        self.checkParam(widget, 'font', '-Adobe-Helvetica-Medium-R-Normal--*-120-*-*-*-*-*-*')
        self.checkInvalidParam(widget, 'font', '', errmsg='font "" doesn\'t exist')

    def test_configure_foreground(self):
        if False:
            print('Hello World!')
        widget = self.create()
        self.checkColorParam(widget, 'foreground')
        if 'fg' in self.OPTIONS:
            self.checkColorParam(widget, 'fg')

    def test_configure_highlightbackground(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.create()
        self.checkColorParam(widget, 'highlightbackground')

    def test_configure_highlightcolor(self):
        if False:
            print('Hello World!')
        widget = self.create()
        self.checkColorParam(widget, 'highlightcolor')

    def test_configure_highlightthickness(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkPixelsParam(widget, 'highlightthickness', 0, 1.3, 2.6, 6, '10p')
        self.checkParam(widget, 'highlightthickness', -2, expected=0, conv=self._conv_pixels)

    def test_configure_image(self):
        if False:
            while True:
                i = 10
        widget = self.create()
        self.checkImageParam(widget, 'image')

    def test_configure_insertbackground(self):
        if False:
            print('Hello World!')
        widget = self.create()
        self.checkColorParam(widget, 'insertbackground')

    def test_configure_insertborderwidth(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.create()
        self.checkPixelsParam(widget, 'insertborderwidth', 0, 1.3, 2.6, 6, -2, '10p')

    def test_configure_insertofftime(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.create()
        self.checkIntegerParam(widget, 'insertofftime', 100)

    def test_configure_insertontime(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkIntegerParam(widget, 'insertontime', 100)

    def test_configure_insertwidth(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkPixelsParam(widget, 'insertwidth', 1.3, 2.6, -2, '10p')

    def test_configure_jump(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkBooleanParam(widget, 'jump')

    def test_configure_justify(self):
        if False:
            print('Hello World!')
        widget = self.create()
        self.checkEnumParam(widget, 'justify', 'left', 'right', 'center', errmsg='bad justification "{}": must be left, right, or center')
        self.checkInvalidParam(widget, 'justify', '', errmsg='ambiguous justification "": must be left, right, or center')

    def test_configure_orient(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.assertEqual(str(widget['orient']), self.default_orient)
        self.checkEnumParam(widget, 'orient', 'horizontal', 'vertical')

    def test_configure_padx(self):
        if False:
            while True:
                i = 10
        widget = self.create()
        self.checkPixelsParam(widget, 'padx', 3, 4.4, 5.6, -2, '12m', conv=self._conv_pad_pixels)

    def test_configure_pady(self):
        if False:
            while True:
                i = 10
        widget = self.create()
        self.checkPixelsParam(widget, 'pady', 3, 4.4, 5.6, -2, '12m', conv=self._conv_pad_pixels)

    def test_configure_relief(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.create()
        self.checkReliefParam(widget, 'relief')

    def test_configure_repeatdelay(self):
        if False:
            return 10
        widget = self.create()
        self.checkIntegerParam(widget, 'repeatdelay', -500, 500)

    def test_configure_repeatinterval(self):
        if False:
            print('Hello World!')
        widget = self.create()
        self.checkIntegerParam(widget, 'repeatinterval', -500, 500)

    def test_configure_selectbackground(self):
        if False:
            while True:
                i = 10
        widget = self.create()
        self.checkColorParam(widget, 'selectbackground')

    def test_configure_selectborderwidth(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.create()
        self.checkPixelsParam(widget, 'selectborderwidth', 1.3, 2.6, -2, '10p')

    def test_configure_selectforeground(self):
        if False:
            print('Hello World!')
        widget = self.create()
        self.checkColorParam(widget, 'selectforeground')

    def test_configure_setgrid(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkBooleanParam(widget, 'setgrid')

    def test_configure_state(self):
        if False:
            while True:
                i = 10
        widget = self.create()
        self.checkEnumParam(widget, 'state', 'active', 'disabled', 'normal')

    def test_configure_takefocus(self):
        if False:
            print('Hello World!')
        widget = self.create()
        self.checkParams(widget, 'takefocus', '0', '1', '')

    def test_configure_text(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkParams(widget, 'text', '', 'any string')

    def test_configure_textvariable(self):
        if False:
            return 10
        widget = self.create()
        var = tkinter.StringVar(self.root)
        self.checkVariableParam(widget, 'textvariable', var)

    def test_configure_troughcolor(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkColorParam(widget, 'troughcolor')

    def test_configure_underline(self):
        if False:
            return 10
        widget = self.create()
        self.checkIntegerParam(widget, 'underline', 0, 1, 10)

    def test_configure_wraplength(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkPixelsParam(widget, 'wraplength', 100)

    def test_configure_xscrollcommand(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkCommandParam(widget, 'xscrollcommand')

    def test_configure_yscrollcommand(self):
        if False:
            while True:
                i = 10
        widget = self.create()
        self.checkCommandParam(widget, 'yscrollcommand')

    def test_configure_command(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.create()
        self.checkCommandParam(widget, 'command')

    def test_configure_indicatoron(self):
        if False:
            while True:
                i = 10
        widget = self.create()
        self.checkBooleanParam(widget, 'indicatoron')

    def test_configure_offrelief(self):
        if False:
            return 10
        widget = self.create()
        self.checkReliefParam(widget, 'offrelief')

    def test_configure_overrelief(self):
        if False:
            return 10
        widget = self.create()
        self.checkReliefParam(widget, 'overrelief')

    def test_configure_selectcolor(self):
        if False:
            print('Hello World!')
        widget = self.create()
        self.checkColorParam(widget, 'selectcolor')

    def test_configure_selectimage(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkImageParam(widget, 'selectimage')

    @requires_tcl(8, 5)
    def test_configure_tristateimage(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.create()
        self.checkImageParam(widget, 'tristateimage')

    @requires_tcl(8, 5)
    def test_configure_tristatevalue(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkParam(widget, 'tristatevalue', 'unknowable')

    def test_configure_variable(self):
        if False:
            while True:
                i = 10
        widget = self.create()
        var = tkinter.DoubleVar(self.root)
        self.checkVariableParam(widget, 'variable', var)

class IntegerSizeTests:

    def test_configure_height(self):
        if False:
            while True:
                i = 10
        widget = self.create()
        self.checkIntegerParam(widget, 'height', 100, -100, 0)

    def test_configure_width(self):
        if False:
            print('Hello World!')
        widget = self.create()
        self.checkIntegerParam(widget, 'width', 402, -402, 0)

class PixelSizeTests:

    def test_configure_height(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.create()
        self.checkPixelsParam(widget, 'height', 100, 101.2, 102.6, -100, 0, '3c')

    def test_configure_width(self):
        if False:
            i = 10
            return i + 15
        widget = self.create()
        self.checkPixelsParam(widget, 'width', 402, 403.4, 404.6, -402, 0, '5i')

def add_standard_options(*source_classes):
    if False:
        while True:
            i = 10

    def decorator(cls):
        if False:
            for i in range(10):
                print('nop')
        for option in cls.OPTIONS:
            methodname = 'test_configure_' + option
            if not hasattr(cls, methodname):
                for source_class in source_classes:
                    if hasattr(source_class, methodname):
                        setattr(cls, methodname, getattr(source_class, methodname))
                        break
                else:

                    def test(self, option=option):
                        if False:
                            print('Hello World!')
                        widget = self.create()
                        widget[option]
                        raise AssertionError('Option "%s" is not tested in %s' % (option, cls.__name__))
                    test.__name__ = methodname
                    setattr(cls, methodname, test)
        return cls
    return decorator

def setUpModule():
    if False:
        while True:
            i = 10
    if test.support.verbose:
        tcl = tkinter.Tcl()
        print('patchlevel =', tcl.call('info', 'patchlevel'))