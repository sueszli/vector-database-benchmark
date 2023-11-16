from contextlib import suppress
import curses
import visidata
from visidata import EscapeException, ExpectedException, clipdraw, Sheet, VisiData, BaseSheet
from visidata import vd, options, colors, dispwidth, ColorAttr
from visidata import AttrDict
vd.theme_option('color_edit_unfocused', '238 on 110', 'display color for unfocused input in form')
vd.theme_option('color_edit_cell', '233 on 110', 'cell color to use when editing cell')
vd.theme_option('disp_edit_fill', '_', 'edit field fill character')
vd.theme_option('disp_unprintable', '·', 'substitute character for unprintables')
vd.theme_option('mouse_interval', 1, 'max time between press/release for click (ms)', sheettype=None)
vd.option('input_history', '', 'basename of file to store persistent input history')
vd.disp_help = 1

class AcceptInput(Exception):
    """*args[0]* is the input to be accepted"""
vd._injectedInput = None

@VisiData.api
def injectInput(vd, x):
    if False:
        for i in range(10):
            print('nop')
    'Use *x* as input to next command.'
    assert vd._injectedInput is None, vd._injectedInput
    vd._injectedInput = x

@VisiData.api
def getCommandInput(vd):
    if False:
        for i in range(10):
            print('nop')
    if vd._injectedInput is not None:
        r = vd._injectedInput
        vd._injectedInput = None
        return r
    return vd.getLastArgs()

@BaseSheet.after
def execCommand(sheet, longname, *args, **kwargs):
    if False:
        while True:
            i = 10
    if vd._injectedInput is not None:
        vd.debug(f'{longname} did not consume input "{vd._injectedInput}"')
        vd._injectedInput = None

def acceptThenFunc(*longnames):
    if False:
        print('Hello World!')

    def _acceptthen(v, i):
        if False:
            while True:
                i = 10
        for longname in longnames:
            vd.queueCommand(longname)
        raise AcceptInput(v)
    return _acceptthen

class EnableCursor:

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        with suppress(curses.error):
            curses.mousemask(0)
            curses.curs_set(1)

    def __exit__(self, exc_type, exc_val, tb):
        if False:
            while True:
                i = 10
        with suppress(curses.error):
            curses.curs_set(0)
            if options.mouse_interval:
                curses.mousemask(curses.MOUSE_ALL if hasattr(curses, 'MOUSE_ALL') else 4294967295)
            else:
                curses.mousemask(0)

def until_get_wch(scr):
    if False:
        for i in range(10):
            print('nop')
    'Ignores get_wch timeouts'
    ret = None
    while not ret:
        try:
            ret = vd.get_wch(scr)
        except curses.error:
            pass
    if isinstance(ret, int):
        return chr(ret)
    return ret

def splice(v: str, i: int, s: str):
    if False:
        return 10
    'Insert `s` into string `v` at `i` (such that v[i] == s[0]).'
    return v if i < 0 else v[:i] + s + v[i:]

class HelpCycler:

    def __init__(self, scr=None, help=''):
        if False:
            i = 10
            return i + 15
        self.help = help
        self.scr = scr

    def __enter__(self):
        if False:
            print('Hello World!')
        if self.scr:
            vd.drawInputHelp(self.scr, self.help)
        return self

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        pass

    def cycle(self):
        if False:
            return 10
        vd.disp_help = (vd.disp_help - 1) % (vd.options.disp_help + 1)
        if self.scr:
            vd.drawInputHelp(self.scr, self.help)

@VisiData.api
def drawInputHelp(vd, scr, help: str=''):
    if False:
        return 10
    if not scr or not vd.cursesEnabled:
        return
    sheet = vd.activeSheet
    if not sheet:
        return
    vd.drawSheet(scr, sheet)
    curhelp = ''
    if vd.disp_help == 0:
        vd.drawSidebar(scr, sheet)
    elif vd.disp_help == 1:
        curhelp = help
        sheet.drawSidebarText(scr, curhelp)
    elif vd.disp_help >= 2:
        curhelp = vd.getHelpPane('input', module='visidata')
        sheet.drawSidebarText(scr, curhelp, title='Input Keystrokes Help')

def clean_printable(s):
    if False:
        print('Hello World!')
    'Escape unprintable characters.'
    return ''.join((c if c.isprintable() else options.disp_unprintable for c in str(s)))

def delchar(s, i, remove=1):
    if False:
        print('Hello World!')
    'Delete `remove` characters from str `s` beginning at position `i`.'
    return s if i < 0 else s[:i] + s[i + remove:]

class CompleteState:

    def __init__(self, completer_func):
        if False:
            while True:
                i = 10
        self.comps_idx = -1
        self.completer_func = completer_func
        self.former_i = None
        self.just_completed = False

    def complete(self, v, i, state_incr):
        if False:
            i = 10
            return i + 15
        self.just_completed = True
        self.comps_idx += state_incr
        if self.former_i is None:
            self.former_i = i
        try:
            r = self.completer_func(v[:self.former_i], self.comps_idx)
        except Exception as e:
            return (v, i)
        if not r:
            return (v, i)
        v = r + v[i:]
        return (v, len(v))

    def reset(self):
        if False:
            i = 10
            return i + 15
        if self.just_completed:
            self.just_completed = False
        else:
            self.former_i = None
            self.comps_idx = -1

class HistoryState:

    def __init__(self, history):
        if False:
            for i in range(10):
                print('nop')
        self.history = history
        self.hist_idx = None
        self.prev_val = None

    def up(self, v, i):
        if False:
            while True:
                i = 10
        if self.hist_idx is None:
            self.hist_idx = len(self.history)
            self.prev_val = v
        if self.hist_idx > 0:
            self.hist_idx -= 1
            v = self.history[self.hist_idx]
        i = len(str(v))
        return (v, i)

    def down(self, v, i):
        if False:
            return 10
        if self.hist_idx is None:
            return (v, i)
        elif self.hist_idx < len(self.history) - 1:
            self.hist_idx += 1
            v = self.history[self.hist_idx]
        else:
            v = self.prev_val
            self.hist_idx = None
        i = len(str(v))
        return (v, i)

@VisiData.api
def editline(vd, scr, y, x, w, i=0, attr=ColorAttr(), value='', fillchar=' ', truncchar='-', unprintablechar='.', completer=lambda text, idx: None, history=[], display=True, updater=lambda val: None, bindings={}, help='', clear=True):
    if False:
        return 10
    'A better curses line editing widget.\n  If *clear* is True, clear whole editing area before displaying.\n  '
    with EnableCursor():
        with HelpCycler(scr, help) as disp_help:
            ESC = '^['
            TAB = '^I'
            history_state = HistoryState(history)
            complete_state = CompleteState(completer)
            insert_mode = True
            first_action = True
            v = str(value)
            if i != 0:
                first_action = False
            left_truncchar = right_truncchar = truncchar

            def find_nonword(s, a, b, incr):
                if False:
                    for i in range(10):
                        print('nop')
                if not s:
                    return 0
                a = min(max(a, 0), len(s) - 1)
                b = min(max(b, 0), len(s) - 1)
                if incr < 0:
                    while not s[b].isalnum() and b >= a:
                        b += incr
                    while s[b].isalnum() and b >= a:
                        b += incr
                    return min(max(b, -1), len(s))
                else:
                    while not s[a].isalnum() and a < b:
                        a += incr
                    while s[a].isalnum() and a < b:
                        a += incr
                    return min(max(a, 0), len(s))
            while True:
                updater(v)
                if display:
                    dispval = clean_printable(v)
                else:
                    dispval = '*' * len(v)
                dispi = i
                if len(dispval) < w:
                    dispval += fillchar * (w - len(dispval) - 1)
                elif i == len(dispval):
                    dispi = w - 1
                    dispval = left_truncchar + dispval[len(dispval) - w + 2:] + fillchar
                elif i >= len(dispval) - w // 2:
                    dispi = w - (len(dispval) - i)
                    dispval = left_truncchar + dispval[len(dispval) - w + 1:]
                elif i <= w // 2:
                    dispval = dispval[:w - 1] + right_truncchar
                else:
                    dispi = w // 2
                    k = 1 if w % 2 == 0 else 0
                    dispval = left_truncchar + dispval[i - w // 2 + 1:i + w // 2 - k] + right_truncchar
                prew = clipdraw(scr, y, x, dispval[:dispi], attr, w, clear=clear, literal=True)
                clipdraw(scr, y, x + prew, dispval[dispi:], attr, w - prew + 1, clear=clear, literal=True)
                if scr:
                    scr.move(y, x + prew)
                ch = vd.getkeystroke(scr)
                if ch == '':
                    continue
                elif ch in bindings:
                    (v, i) = bindings[ch](v, i)
                elif ch == 'KEY_IC':
                    insert_mode = not insert_mode
                elif ch == '^A' or ch == 'KEY_HOME':
                    i = 0
                elif ch == '^B' or ch == 'KEY_LEFT':
                    i -= 1
                elif ch in ('^C', '^Q', ESC):
                    raise EscapeException(ch)
                elif ch == '^D' or ch == 'KEY_DC':
                    v = delchar(v, i)
                elif ch == '^E' or ch == 'KEY_END':
                    i = len(v)
                elif ch == '^F' or ch == 'KEY_RIGHT':
                    i += 1
                elif ch == '^G':
                    disp_help.cycle()
                    continue
                elif ch in ('^H', 'KEY_BACKSPACE', '^?'):
                    i -= 1
                    v = delchar(v, i)
                elif ch == TAB:
                    (v, i) = complete_state.complete(v, i, +1)
                elif ch == 'KEY_BTAB':
                    (v, i) = complete_state.complete(v, i, -1)
                elif ch in ['^J', '^M']:
                    break
                elif ch == '^K':
                    v = v[:i]
                elif ch == '^N':
                    c = ''
                    while not c:
                        c = vd.getkeystroke(scr)
                    c = vd.prettykeys(c)
                    i += len(c)
                    v += c
                elif ch == '^O':
                    v = vd.launchExternalEditor(v)
                    break
                elif ch == '^R':
                    v = str(value)
                elif ch == '^T':
                    v = delchar(splice(v, i - 2, v[i - 1:i]), i)
                elif ch == '^U':
                    v = v[i:]
                    i = 0
                elif ch == '^V':
                    v = splice(v, i, until_get_wch(scr))
                    i += 1
                elif ch == '^W':
                    j = find_nonword(v, 0, i - 1, -1)
                    v = v[:j + 1] + v[i:]
                    i = j + 1
                elif ch == '^Y':
                    v = splice(v, i, str(vd.memory.clipval))
                elif ch == '^Z':
                    vd.suspend()
                elif ch == 'kLFT5':
                    i = find_nonword(v, 0, i - 1, -1) + 1
                elif ch == 'kRIT5':
                    i = find_nonword(v, i + 1, len(v) - 1, +1) + 1
                elif ch == 'kUP5':
                    pass
                elif ch == 'kDN5':
                    pass
                elif history and ch == 'KEY_UP':
                    (v, i) = history_state.up(v, i)
                elif history and ch == 'KEY_DOWN':
                    (v, i) = history_state.down(v, i)
                elif len(ch) > 1:
                    pass
                else:
                    if first_action:
                        v = ''
                    if insert_mode:
                        v = splice(v, i, ch)
                    else:
                        v = v[:i] + ch + v[i + 1:]
                    i += 1
                if i < 0:
                    i = 0
                v = str(v)
                if i > len(v):
                    i = len(v)
                first_action = False
                complete_state.reset()
            return v

@VisiData.api
def editText(vd, y, x, w, record=True, display=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Invoke modal single-line editor at (*y*, *x*) for *w* terminal chars. Use *display* is False for sensitive input like passphrases.  If *record* is True, get input from the cmdlog in batch mode, and save input to the cmdlog if *display* is also True. Return new value as string.'
    v = None
    if record and vd.cmdlog:
        v = vd.getCommandInput()
    if v is None:
        try:
            v = vd.editline(vd.activeSheet._scr, y, x, w, display=display, **kwargs)
        except AcceptInput as e:
            v = e.args[0]
        if vd.cursesEnabled:
            curses.flushinp()
    if display:
        if record and vd.cmdlog:
            vd.setLastArgs(v)
    if 'value' in kwargs:
        starting_value = kwargs['value']
        if isinstance(starting_value, (int, float)) and v[-1] == '%':
            pct = float(v[:-1])
            v = pct * starting_value / 100
        v = type(starting_value)(v)
    return v

@VisiData.api
def inputsingle(vd, prompt, record=True):
    if False:
        while True:
            i = 10
    'Display prompt and return single character of user input.'
    sheet = vd.activeSheet
    v = None
    if record and vd.cmdlog:
        v = vd.getCommandInput()
    if v is not None:
        return v
    y = sheet.windowHeight - 1
    w = sheet.windowWidth
    rstatuslen = vd.drawRightStatus(sheet._scr, sheet)
    promptlen = clipdraw(sheet._scr, y, 0, prompt, 0, w=w - rstatuslen - 1)
    sheet._scr.move(y, w - promptlen - rstatuslen - 2)
    while not v:
        v = vd.getkeystroke(sheet._scr)
    if record and vd.cmdlog:
        vd.setLastArgs(v)
    return v

@VisiData.api
def inputMultiple(vd, updater=lambda val: None, **kwargs):
    if False:
        while True:
            i = 10
    'A simple form, where each input is an entry in `kwargs`, with the key being the key in the returned dict, and the value being a dictionary of kwargs to the singular input().'
    sheet = vd.activeSheet
    scr = sheet._scr
    y = sheet.windowHeight - 1
    maxw = sheet.windowWidth // 2
    attr = colors.color_edit_unfocused
    keys = list(kwargs.keys())
    cur_input_key = keys[0]
    if scr:
        scr.erase()
    for (i, (k, v)) in enumerate(kwargs.items()):
        v['dy'] = i
        v['w'] = maxw - dispwidth(v.get('prompt'))

    class ChangeInput(Exception):
        pass

    def change_input(offset):
        if False:
            return 10

        def _throw(v, i):
            if False:
                i = 10
                return i + 15
            if scr:
                scr.erase()
            raise ChangeInput(v, offset)
        return _throw

    def _drawPrompt(val):
        if False:
            for i in range(10):
                print('nop')
        for (k, v) in kwargs.items():
            maxw = min(sheet.windowWidth - 1, max(dispwidth(v.get('prompt')), dispwidth(str(v.get('value', '')))))
            promptlen = clipdraw(scr, y - v.get('dy'), 0, v.get('prompt'), attr, w=maxw)
            promptlen = clipdraw(scr, y - v.get('dy'), promptlen, v.get('value', ''), attr, w=maxw)
        return updater(val)
    with HelpCycler() as disp_help:
        while True:
            try:
                input_kwargs = kwargs[cur_input_key]
                input_kwargs['value'] = vd.input(**input_kwargs, attr=colors.color_edit_cell, updater=_drawPrompt, bindings={'KEY_BTAB': change_input(-1), '^I': change_input(+1), 'KEY_SR': change_input(-1), 'KEY_SF': change_input(+1), 'kUP': change_input(-1), 'kDN': change_input(+1)})
                break
            except ChangeInput as e:
                vd.lastInputsSheet.appendRow(AttrDict(type=input_kwargs.get('type', ''), input=e.args[0]))
                input_kwargs['value'] = e.args[0]
                offset = e.args[1]
                i = keys.index(cur_input_key)
                cur_input_key = keys[(i + offset) % len(keys)]
    return {k: v.get('value', '') for (k, v) in kwargs.items()}

@VisiData.api
def input(self, prompt, type=None, defaultLast=False, history=[], dy=0, attr=None, updater=lambda v: None, **kwargs):
    if False:
        while True:
            i = 10
    'Display *prompt* and return line of user input.\n\n        - *type*: string indicating the type of input to use for history.\n        - *history*: list of strings to use for input history.\n        - *defaultLast*:  on empty input, if True, return last history item.\n        - *display*: pass False to not display input (for sensitive input, e.g. a password).\n        - *record*: pass False to not record input on cmdlog (for sensitive or inconsequential input).\n        - *completer*: ``completer(val, idx)`` is called on TAB to get next completed value.\n        - *updater*: ``updater(val)`` is called every keypress or timeout.\n        - *bindings*: dict of keystroke to func(v, i) that returns updated (v, i)\n        - *dy*: number of lines from bottom of pane\n        - *attr*: curses attribute for prompt\n        - *help*: string to include in help\n    '
    if attr is None:
        attr = ColorAttr()
    sheet = self.activeSheet
    if not vd.cursesEnabled:
        if kwargs.get('record', True) and vd.cmdlog:
            return vd.getCommandInput()
        if kwargs.get('display', True):
            import builtins
            return builtins.input(prompt)
        else:
            import getpass
            return getpass.getpass(prompt)
    history = self.lastInputsSheet.history(type)
    y = sheet.windowHeight - dy - 1
    promptlen = dispwidth(prompt)

    def _drawPrompt(val=''):
        if False:
            print('Hello World!')
        rstatuslen = vd.drawRightStatus(sheet._scr, sheet)
        clipdraw(sheet._scr, y, 0, prompt, attr, w=sheet.windowWidth - rstatuslen - 1)
        updater(val)
        return sheet.windowWidth - promptlen - rstatuslen - 2
    w = kwargs.pop('w', _drawPrompt())
    ret = self.editText(y, promptlen, w=w, attr=colors.color_edit_cell, unprintablechar=options.disp_unprintable, truncchar=options.disp_truncator, history=history, updater=_drawPrompt, **kwargs)
    if ret:
        self.lastInputsSheet.appendRow(AttrDict(type=type, input=ret))
    elif defaultLast:
        history or vd.fail('no previous input')
        ret = history[-1]
    return ret

@VisiData.api
def confirm(vd, prompt, exc=EscapeException):
    if False:
        return 10
    'Display *prompt* on status line and demand input that starts with "Y" or "y" to proceed.  Raise *exc* otherwise.  Return True.'
    if options.batch and (not options.interactive):
        return vd.fail('cannot confirm in batch mode: ' + prompt)
    yn = vd.input(prompt, value='no', record=False)[:1]
    if not yn or yn not in 'Yy':
        msg = 'disconfirmed: ' + prompt
        if exc:
            raise exc(msg)
        vd.warning(msg)
        return False
    return True

class CompleteKey:

    def __init__(self, items):
        if False:
            print('Hello World!')
        self.items = items

    def __call__(self, val, state):
        if False:
            i = 10
            return i + 15
        opts = [x for x in self.items if x.startswith(val)]
        return opts[state % len(opts)] if opts else val

@Sheet.api
def editCell(self, vcolidx=None, rowidx=None, value=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'Call vd.editText for the cell at (*rowidx*, *vcolidx*).  Return the new value, properly typed.\n\n       - *rowidx*: numeric index into ``self.rows``.  If negative, indicates the column name in the header.\n       - *value*: if given, the starting input; otherwise the starting input is the cell value or column name as appropriate.\n       - *kwargs*: passthrough args to ``vd.editText``.\n       '
    if vcolidx is None:
        vcolidx = self.cursorVisibleColIndex
    (x, w) = self._visibleColLayout.get(vcolidx, (0, 0))
    col = self.visibleCols[vcolidx]
    if rowidx is None:
        rowidx = self.cursorRowIndex
    if rowidx < 0:
        y = 0
        value = value or col.name
    else:
        (y, h) = self._rowLayout.get(rowidx, (0, 0))
        value = value or col.getDisplayValue(self.rows[self.cursorRowIndex])
    bindings = {'kUP': acceptThenFunc('go-up', 'rename-col' if rowidx < 0 else 'edit-cell'), 'KEY_SR': acceptThenFunc('go-up', 'rename-col' if rowidx < 0 else 'edit-cell'), 'kDN': acceptThenFunc('go-down', 'rename-col' if rowidx < 0 else 'edit-cell'), 'KEY_SF': acceptThenFunc('go-down', 'rename-col' if rowidx < 0 else 'edit-cell'), 'KEY_SRIGHT': acceptThenFunc('go-right', 'rename-col' if rowidx < 0 else 'edit-cell'), 'KEY_SLEFT': acceptThenFunc('go-left', 'rename-col' if rowidx < 0 else 'edit-cell')}
    bindings.update(kwargs.get('bindings', {}))
    kwargs['bindings'] = bindings
    editargs = dict(value=value, fillchar=self.options.disp_edit_fill, truncchar=self.options.disp_truncator)
    editargs.update(kwargs)
    r = vd.editText(y, x, w, attr=colors.color_edit_cell, **editargs)
    if rowidx >= 0:
        r = col.type(r)
    return r
vd.addGlobals({'CompleteKey': CompleteKey, 'AcceptInput': AcceptInput})