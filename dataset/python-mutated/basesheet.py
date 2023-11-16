import os
import visidata
from visidata import Extensible, VisiData, vd, EscapeException, cleanName, MissingAttrFormatter, AttrDict
UNLOADED = tuple()
vd.beforeExecHooks = []

class LazyChainMap:
    """provides a lazy mapping to obj attributes.  useful when some attributes are expensive properties."""

    def __init__(self, *objs, locals=None):
        if False:
            while True:
                i = 10
        self.locals = {} if locals is None else locals
        self.objs = {}
        for obj in objs:
            for k in dir(obj):
                if k not in self.objs:
                    self.objs[k] = obj

    def __contains__(self, k):
        if False:
            for i in range(10):
                print('nop')
        return k in self.objs

    def keys(self):
        if False:
            print('Hello World!')
        return list(self.objs.keys())

    def get(self, key, default=None):
        if False:
            for i in range(10):
                print('nop')
        if key in self.locals:
            return self.locals[key]
        return self.objs.get(key, default)

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.locals.clear()

    def __getitem__(self, k):
        if False:
            print('Hello World!')
        obj = self.objs.get(k, None)
        if obj:
            return getattr(obj, k)
        return self.locals[k]

    def __setitem__(self, k, v):
        if False:
            print('Hello World!')
        obj = self.objs.get(k, None)
        if obj:
            return setattr(obj, k, v)
        self.locals[k] = v

class DrawablePane(Extensible):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.__dict__.update(kwargs)
    'Base class for all interaction owners that can be drawn in a window.'

    def draw(self, scr):
        if False:
            while True:
                i = 10
        'Draw on the terminal window *scr*.  Should be overridden.'
        vd.error('no draw')

    @property
    def windowHeight(self):
        if False:
            print('Hello World!')
        'Height of the current sheet window, in terminal lines.'
        return self._scr.getmaxyx()[0] if self._scr else 25

    @property
    def windowWidth(self):
        if False:
            while True:
                i = 10
        'Width of the current sheet window, in single-width characters.'
        return self._scr.getmaxyx()[1] if self._scr else 80

    def execCommand2(self, cmd, vdglobals=None):
        if False:
            return 10
        "Execute `cmd` with `vdglobals` as globals and this sheet's attributes as locals.  Return True if user cancelled."
        try:
            self.sheet = self
            code = compile(cmd.execstr, cmd.longname, 'exec')
            exec(code, vdglobals, LazyChainMap(vd, self))
            return False
        except EscapeException as e:
            vd.warning(str(e))
            return True

class _dualproperty:
    """Return *obj_method* or *cls_method* depending on whether property is on instance or class."""

    def __init__(self, obj_method, cls_method):
        if False:
            print('Hello World!')
        self._obj_method = obj_method
        self._cls_method = cls_method

    def __get__(self, obj, objtype=None):
        if False:
            for i in range(10):
                print('nop')
        if obj is None:
            return self._cls_method(objtype)
        else:
            return self._obj_method(obj)

class BaseSheet(DrawablePane):
    """Base class for all sheet types."""
    _rowtype = object
    _coltype = None
    rowtype = 'objects'
    precious = True
    defer = False
    help = ''

    def _obj_options(self):
        if False:
            print('Hello World!')
        return vd.OptionsObject(vd._options, obj=self)

    def _class_options(cls):
        if False:
            i = 10
            return i + 15
        return vd.OptionsObject(vd._options, obj=cls)
    class_options = options = _dualproperty(_obj_options, _class_options)

    def __init__(self, *names, rows=UNLOADED, **kwargs):
        if False:
            i = 10
            return i + 15
        self._name = None
        self.loading = False
        self.names = list(names)
        self.name = self.options.name_joiner.join((str(x) for x in self.names if x))
        self.source = None
        self.rows = rows
        self._scr = None
        self.hasBeenModified = False
        super().__init__(**kwargs)
        self._sidebar = ''

    def setModified(self):
        if False:
            print('Hello World!')
        if not self.hasBeenModified:
            vd.addUndo(setattr, self, 'hasBeenModified', self.hasBeenModified)
            self.hasBeenModified = True

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if self.name != other.name:
            return self.name < other.name
        else:
            return id(self) < id(other)

    def __copy__(self):
        if False:
            return 10
        'Return shallow copy of sheet.'
        cls = self.__class__
        ret = cls.__new__(cls)
        ret.__dict__.update(self.__dict__)
        ret.precious = True
        ret.hasBeenModified = False
        return ret

    def __bool__(self):
        if False:
            print('Hello World!')
        'an instantiated Sheet always tests true'
        return True

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Number of elements on this sheet.'
        return self.nRows

    def __str__(self):
        if False:
            return 10
        return self.name

    @property
    def rows(self):
        if False:
            return 10
        return self._rows

    @rows.setter
    def rows(self, rows):
        if False:
            while True:
                i = 10
        self._rows = rows

    @property
    def nRows(self):
        if False:
            i = 10
            return i + 15
        'Number of rows on this sheet.  Override in subclass.'
        return 0

    def __contains__(self, vs):
        if False:
            i = 10
            return i + 15
        if self.source is vs:
            return True
        if isinstance(self.source, BaseSheet):
            return vs in self.source
        return False

    @property
    def displaySource(self):
        if False:
            while True:
                i = 10
        if isinstance(self.source, BaseSheet):
            return f'the *{self.source[0]}* sheet'
        if isinstance(self.source, (list, tuple)):
            if len(self.source) == 1:
                return f'the **{self.source[0]}** sheet'
            return f'{len(self.source)} sheets'
        return f'**{self.source}**'

    def execCommand(self, longname, vdglobals=None, keystrokes=None):
        if False:
            return 10
        if ' ' in longname:
            (cmd, arg) = longname.split(' ', maxsplit=1)
            vd.injectInput(arg)
        cmd = self.getCommand(longname or keystrokes)
        if not cmd:
            vd.warning('no command for %s' % (longname or keystrokes))
            return False
        escaped = False
        err = ''
        if vdglobals is None:
            vdglobals = vd.getGlobals()
        vd.cmdlog
        try:
            for hookfunc in vd.beforeExecHooks:
                hookfunc(self, cmd, '', keystrokes)
            escaped = super().execCommand2(cmd, vdglobals=vdglobals)
        except Exception as e:
            vd.debug(cmd.execstr)
            err = vd.exceptionCaught(e)
            escaped = True
        try:
            if vd.cmdlog:
                vd.cmdlog.afterExecSheet(vd.activeSheet, escaped, err)
        except Exception as e:
            vd.exceptionCaught(e)
        self.checkCursorNoExceptions()
        vd.clearCaches()
        for t in self.currentThreads:
            if not hasattr(t, 'lastCommand'):
                t.lastCommand = True
        return escaped

    @property
    def lastCommandThreads(self):
        if False:
            for i in range(10):
                print('nop')
        return [t for t in self.currentThreads if getattr(t, 'lastCommand', None)]

    @property
    def names(self):
        if False:
            print('Hello World!')
        return self._names

    @names.setter
    def names(self, names):
        if False:
            i = 10
            return i + 15
        self._names = names
        self.name = self.options.name_joiner.join((self.maybeClean(str(x)) for x in self._names))

    @property
    def name(self):
        if False:
            print('Hello World!')
        'Name of this sheet.'
        return self._name

    @name.setter
    def name(self, name):
        if False:
            return 10
        'Set name without spaces.'
        if self._names:
            vd.addUndo(setattr, self, '_names', self._names)
        self._name = self.maybeClean(str(name))

    def maybeClean(self, s):
        if False:
            return 10
        if self.options.clean_names:
            s = cleanName(s)
        return s

    def recalc(self):
        if False:
            i = 10
            return i + 15
        'Clear any calculated value caches.'
        pass

    def refresh(self):
        if False:
            print('Hello World!')
        'Recalculate any internal state needed for `draw()`.  Overridable.'
        pass

    def ensureLoaded(self):
        if False:
            i = 10
            return i + 15
        'Call ``reload()`` if not already loaded.'
        if self.rows is UNLOADED:
            self.rows = []
            return self.reload()

    def reload(self):
        if False:
            for i in range(10):
                print('nop')
        'Load sheet from *self.source*.  Override in subclass.'
        vd.error('no reload')

    @property
    def cursorRow(self):
        if False:
            for i in range(10):
                print('nop')
        'The row object at the row cursor.  Overridable.'
        return None

    def checkCursor(self):
        if False:
            for i in range(10):
                print('nop')
        'Check cursor and fix if out-of-bounds.  Overridable.'
        pass

    def checkCursorNoExceptions(self):
        if False:
            return 10
        try:
            return self.checkCursor()
        except Exception as e:
            vd.exceptionCaught(e)

    def evalExpr(self, expr, **kwargs):
        if False:
            return 10
        'Evaluate Python expression *expr* in the context of *kwargs* (may vary by sheet type).'
        return eval(expr, vd.getGlobals(), None)

    def formatString(self, fmt):
        if False:
            return 10
        'Return formatted string with *sheet* and *vd* accessible to expressions.  Missing expressions return empty strings instead of error.'
        return MissingAttrFormatter().format(fmt, sheet=self, vd=vd)

@VisiData.api
def redraw(vd):
    if False:
        while True:
            i = 10
    'Clear the terminal screen and let the next draw cycle recreate the windows and redraw everything.'
    for vs in vd.sheets:
        vs._scr = None
    if vd.win1:
        vd.win1.clear()
    if vd.win2:
        vd.win2.clear()
    if vd.scrFull:
        vd.scrFull.clear()
        vd.setWindows(vd.scrFull)

@VisiData.property
def sheet(self):
    if False:
        print('Hello World!')
    return self.activeSheet

@VisiData.api
def isLongname(self, ks):
    if False:
        i = 10
        return i + 15
    'Return True if *ks* is a longname.'
    return '-' in ks and ks[-1] != '-' or (len(ks) > 3 and ks.islower())

@VisiData.api
def getSheet(vd, sheetname):
    if False:
        while True:
            i = 10
    'Return Sheet from the sheet stack.  *sheetname* can be a sheet name or a sheet number indexing directly into ``vd.sheets``.'
    if isinstance(sheetname, BaseSheet):
        return sheetname
    matchingSheets = [x for x in vd.sheets if x.name == sheetname]
    if matchingSheets:
        if len(matchingSheets) > 1:
            vd.warning('more than one sheet named "%s"' % sheetname)
        return matchingSheets[0]
    try:
        sheetidx = int(sheetname)
        return vd.sheets[sheetidx]
    except ValueError:
        pass
    if sheetname == 'options':
        vs = vd.globalOptionsSheet
        vs.reload()
        vs.vd = vd
        return vs