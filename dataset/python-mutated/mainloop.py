import builtins
import contextlib
import os
import curses
import signal
import threading
import time
from visidata import vd, VisiData, colors, ESC, options, BaseSheet, AttrDict
__all__ = ['ReturnValue', 'run']
vd.curses_timeout = 100
vd.timeouts_before_idle = 10
vd.min_draw_ms = 100
vd._lastDrawTime = 0

class ReturnValue(BaseException):
    """raise ReturnValue(ret) to exit from an inner runresult() with its result."""
    pass

@VisiData.api
def drawSheet(self, scr, sheet):
    if False:
        for i in range(10):
            print('nop')
    'Erase *scr* and draw *sheet* on it, including status bars and sidebar.'
    sheet.ensureLoaded()
    scr.erase()
    scr.bkgd(' ', colors.color_default.attr)
    sheet._scr = scr
    try:
        sheet.draw(scr)
    except Exception as e:
        self.exceptionCaught(e)
    self.drawLeftStatus(scr, sheet)
    self.drawRightStatus(scr, sheet)
vd.windowConfig = dict(pct=0, n=0, h=0, w=0)
vd.winTop = None
vd.scrMenu = None
vd.scrFull = None

@VisiData.api
def setWindows(vd, scr, pct=None):
    if False:
        while True:
            i = 10
    'Assign winTop, winBottom, win1 and win2 according to options.disp_splitwin_pct.'
    if pct is None:
        pct = options.disp_splitwin_pct
    disp_menu = getattr(vd, 'menuRunning', None) or vd.options.disp_menu
    topmenulines = 1 if disp_menu else 0
    (h, w) = scr.getmaxyx()
    n = 0
    if pct:
        n = abs(pct) * h // 100
        n = min(n, h - topmenulines - 3)
        n = max(3, n)
    desiredConfig = dict(pct=pct, n=n, h=h - topmenulines, w=w)
    if vd.scrFull is not scr or vd.windowConfig != desiredConfig:
        if not topmenulines:
            vd.scrMenu = None
        elif not vd.scrMenu:
            vd.scrMenu = vd.subwindow(scr, 0, 0, w, h)
            vd.scrMenu.keypad(1)
        vd.winTop = vd.subwindow(scr, 0, topmenulines, w, n)
        vd.winTop.keypad(1)
        vd.winBottom = vd.subwindow(scr, 0, n + topmenulines, w, h - n - topmenulines)
        vd.winBottom.keypad(1)
        if pct == 0 or pct >= 100:
            vd.win1 = vd.winBottom
            vd.win2 = None
        elif pct > 0:
            vd.win1 = vd.winTop
            vd.win2 = vd.winBottom
        elif pct < 0:
            vd.win1 = vd.winBottom
            vd.win2 = vd.winTop
        for vs in vd.sheetstack(1)[0:1] + vd.sheetstack(2)[0:1]:
            vs.refresh()
        vd.windowConfig = desiredConfig
        vd.scrFull = scr
        return True

@VisiData.api
def draw_all(vd):
    if False:
        for i in range(10):
            print('nop')
    'Draw all sheets in all windows.'
    vd.clearCaches()
    ss1 = vd.sheetstack(1)
    ss2 = vd.sheetstack(2)
    if ss1 and (not ss2):
        vd.activePane = 1
        vd.setWindows(vd.scrFull)
        vd.drawSheet(vd.win1, ss1[0])
        if vd.win2:
            vd.win2.erase()
    elif not ss1 and ss2:
        vd.activePane = 2
        vd.setWindows(vd.scrFull)
        vd.drawSheet(vd.win2, ss2[0])
        if vd.win1:
            vd.win1.erase()
    elif ss1 and ss2 and vd.win2:
        vd.drawSheet(vd.win1, ss1[0])
        vd.drawSheet(vd.win2, ss2[0])
    elif ss1 and ss2 and (not vd.win2):
        vd.drawSheet(vd.win1, vd.sheetstack(vd.activePane)[0])
        vd.setWindows(vd.scrFull)
    if vd.scrMenu:
        vd.drawMenu(vd.scrMenu, vd.activeSheet)
    vd.drawSidebar(vd.scrFull, vd.activeSheet)
    if vd.win1:
        vd.win1.refresh()
    if vd.win2:
        vd.win2.refresh()
    if vd.scrMenu:
        vd.scrMenu.refresh()

@VisiData.api
def runresult(vd):
    if False:
        i = 10
        return i + 15
    try:
        err = vd.mainloop(vd.scrFull)
        if err:
            raise Exception(err)
    except ReturnValue as e:
        return e.args[0]

@VisiData.api
def mainloop(self, scr):
    if False:
        print('Hello World!')
    'Manage execution of keystrokes and subsequent redrawing of screen.'
    nonidle_timeout = vd.curses_timeout
    scr.timeout(vd.curses_timeout)
    with contextlib.suppress(curses.error):
        curses.curs_set(0)
    numTimeouts = 0
    prefixWaiting = False
    vd.scrFull = scr
    self.keystrokes = ''
    while True:
        if not self.stackedSheets and self.currentReplay is None:
            return
        sheet = self.activeSheet
        if not sheet:
            continue
        threading.current_thread().sheet = sheet
        vd.drawThread = threading.current_thread()
        vd.setWindows(vd.scrFull)
        if not self.drainPendingKeys(scr) or time.time() - self._lastDrawTime > self.min_draw_ms / 1000:
            self.draw_all()
            self._lastDrawTime = time.time()
        keystroke = self.getkeystroke(scr, sheet)
        if not keystroke and prefixWaiting and ('Alt+' in self.keystrokes):
            self.keystrokes = ''
        if keystroke:
            numTimeouts = 0
            if not prefixWaiting:
                self.keystrokes = ''
            self.statuses.clear()
            if keystroke == 'KEY_MOUSE':
                try:
                    keystroke = vd.handleMouse(sheet)
                except Exception as e:
                    self.exceptionCaught(e)
            if keystroke and keystroke in vd.allPrefixes and (keystroke in vd.keystrokes[:-1]):
                vd.warning('duplicate prefix: ' + keystroke)
                self.keystrokes = ''
            else:
                keystroke = self.prettykeys(keystroke)
                self.keystrokes += keystroke
        self.drawRightStatus(sheet._scr, sheet)
        if not keystroke:
            pass
        elif keystroke == 'Ctrl+Q':
            return self.lastErrors and '\n'.join(self.lastErrors[-1])
        elif vd.bindkeys._get(self.keystrokes):
            sheet.execCommand(self.keystrokes, keystrokes=self.keystrokes)
            prefixWaiting = False
        elif keystroke in self.allPrefixes:
            prefixWaiting = True
        else:
            vd.status('no command for "%s"' % self.keystrokes)
            prefixWaiting = False
        if self._nextCommands:
            cmd = self._nextCommands.pop(0)
            if isinstance(cmd, (dict, list)):
                if self.replayOne(cmd):
                    self.replay_cancel()
            else:
                sheet.execCommand(cmd, keystrokes=self.keystrokes)
        if not self._nextCommands:
            if self.currentReplay:
                self.currentReplayRow = None
                self.currentReplay = None
        self.checkForFinishedThreads()
        sheet.checkCursorNoExceptions()
        time.sleep(0)
        if vd._nextCommands:
            vd.curses_timeout = int(vd.options.replay_wait * 1000)
        elif vd.unfinishedThreads:
            vd.curses_timeout = nonidle_timeout
        else:
            numTimeouts += 1
            if vd.timeouts_before_idle >= 0 and numTimeouts > vd.timeouts_before_idle:
                vd.curses_timeout = -1
            else:
                vd.curses_timeout = nonidle_timeout
        scr.timeout(vd.curses_timeout)

@VisiData.api
def initCurses(vd):
    if False:
        for i in range(10):
            print('nop')
    os.putenv('ESCDELAY', '25')
    curses.use_env(True)
    scr = curses.initscr()
    curses.start_color()
    colors.setup()
    curses.noecho()
    curses.raw()
    curses.meta(1)
    scr.keypad(1)
    curses.def_prog_mode()
    vd.drainPendingKeys(scr)
    if '\x1b' in vd.pendingKeys:
        vd.pendingKeys.clear()
        curses.flushinp()
    return scr

def wrapper(f, *args, **kwargs):
    if False:
        while True:
            i = 10
    try:
        scr = vd.initCurses()
        return f(scr, *args, **kwargs)
    finally:
        curses.endwin()

@VisiData.global_api
def run(vd, *sheetlist):
    if False:
        return 10
    'Main entry point; launches vdtui with the given sheets already pushed (last one is visible)'
    scr = None
    try:
        for vs in sheetlist:
            vd.push(vs, load=False)
        scr = vd.initCurses()
        ret = vd.mainloop(scr)
    except curses.error as e:
        if vd.options.debug:
            raise
        vd.fail(str(e))
    finally:
        if scr:
            curses.endwin()
    vd.cancelThread(*[t for t in vd.unfinishedThreads if not t.name.startswith('save_')])
    if ret:
        builtins.print(ret)
    return ret

@VisiData.api
def addCommand(vd, *args, **kwargs):
    if False:
        return 10
    return BaseSheet.addCommand(*args, **kwargs)
import sys
vd.addGlobals({k: getattr(sys.modules[__name__], k) for k in __all__})