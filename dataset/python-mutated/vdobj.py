from functools import wraps
import curses
import visidata
__all__ = ['ENTER', 'ALT', 'ESC', 'asyncthread', 'VisiData']
ENTER = 'Enter'
ALT = ESC = '^['

def asyncthread(func):
    if False:
        print('Hello World!')
    'Function decorator, to make calls to `func()` spawn a separate thread if available.'

    @wraps(func)
    def _execAsync(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if args and isinstance(args[0], visidata.BaseSheet):
            if 'sheet' not in kwargs:
                kwargs['sheet'] = args[0]
        return visidata.vd.execAsync(func, *args, **kwargs)
    return _execAsync

class VisiData(visidata.Extensible):
    allPrefixes = ['g', 'z', 'Alt+']

    @classmethod
    def global_api(cls, func):
        if False:
            for i in range(10):
                print('nop')
        'Make global func() and identical vd.func()'

        def _vdfunc(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            return getattr(visidata.vd, func.__name__)(*args, **kwargs)
        visidata.vd.addGlobals({func.__name__: func})
        setattr(cls, func.__name__, func)
        return wraps(func)(_vdfunc)

    def __init__(self):
        if False:
            return 10
        self.sheets = []
        self.allSheets = []
        self.lastErrors = []
        self.pendingKeys = []
        self.keystrokes = ''
        self.scrFull = None
        self._cmdlog = None
        self.currentReplay = None
        self.contexts = [self]
        self.importingModule = None
        self.importedModules = []

    @property
    def cursesEnabled(self):
        if False:
            i = 10
            return i + 15
        return bool(self.scrFull)

    def sheetstack(self, pane=0):
        if False:
            i = 10
            return i + 15
        'Return list of sheets in given *pane*. pane=0 is the active pane.  pane=-1 is the inactive pane.'
        if pane == -1:
            return list((vs for vs in self.sheets if vs.pane and vs.pane != self.activePane))
        else:
            return list((vs for vs in self.sheets if vs.pane == (pane or self.activePane)))

    @property
    def stackedSheets(self):
        if False:
            while True:
                i = 10
        return list((vs for vs in self.sheets if vs.pane))

    @property
    def activeSheet(self):
        if False:
            i = 10
            return i + 15
        'Return top sheet on sheets stack, or cmdlog if no sheets.'
        for vs in self.sheets:
            if vs.pane and vs.pane == self.activePane:
                return vs
        for vs in self.sheets:
            if vs.pane and vs.pane != self.activePane:
                return vs
        return self._cmdlog

    @property
    def activeStack(self):
        if False:
            print('Hello World!')
        return self.sheetstack() or self.sheetstack(-1)

    def __copy__(self):
        if False:
            for i in range(10):
                print('nop')
        'Dummy method for Extensible.init()'
        pass

    def finalInit(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize members specified in other modules with init()'
        pass

    @classmethod
    def init(cls, membername, initfunc, **kwargs):
        if False:
            print('Hello World!')
        'Overload Extensible.init() to call finalInit instead of __init__'
        oldinit = cls.finalInit

        def newinit(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            oldinit(self, *args, **kwargs)
            setattr(self, membername, initfunc())
        cls.finalInit = newinit
        super().init(membername, lambda : None, **kwargs)

    def clearCaches(self):
        if False:
            i = 10
            return i + 15
        'Invalidate internal caches between command inputs.'
        visidata.Extensible.clear_all_caches()

    def resetVisiData(self):
        if False:
            print('Hello World!')
        self.clearCaches()
        vd = visidata.vd
        vd.cmdlog.rows = []
        vd.sheets = []
        vd.allSheets = []
        return vd

    def get_wch(self, scr):
        if False:
            return 10
        try:
            return scr.get_wch()
        except AttributeError:
            k = scr.getch()
            if k == -1:
                raise curses.error('no char ready')
            return k

    def drainPendingKeys(self, scr):
        if False:
            i = 10
            return i + 15
        'Call scr.get_wch() until no more keypresses are available.  Return True if any keypresses are pending.'
        scr.timeout(0)
        try:
            while True:
                k = self.get_wch(scr)
                if k:
                    self.pendingKeys.append(k)
                else:
                    break
        except curses.error:
            pass
        finally:
            scr.timeout(self.curses_timeout)
        return bool(self.pendingKeys)

    def getkeystroke(self, scr, vs=None):
        if False:
            print('Hello World!')
        'Get keystroke and display it on status bar.'
        self.drainPendingKeys(scr)
        k = None
        if self.pendingKeys:
            k = self.pendingKeys.pop(0)
        else:
            curses.reset_prog_mode()
            try:
                scr.refresh()
                k = self.get_wch(scr)
                vs = vs or self.activeSheet
                if vs:
                    self.drawRightStatus(vs._scr, vs)
            except curses.error:
                return ''
        if isinstance(k, str):
            if ord(k) >= 32 and ord(k) != 127:
                return k
            k = ord(k)
        return curses.keyname(k).decode('utf-8')

    @property
    def screenHeight(self):
        if False:
            print('Hello World!')
        return self.scrFull.getmaxyx()[0] if self.scrFull else 25

    @property
    def screenWidth(self):
        if False:
            return 10
        return self.scrFull.getmaxyx()[1] if self.scrFull else 80