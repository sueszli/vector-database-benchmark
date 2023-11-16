import win32api
import win32con
import win32ui
MAPVK_VK_TO_CHAR = 2
key_name_to_vk = {}
key_code_to_name = {}
_better_names = {'escape': 'esc', 'return': 'enter', 'back': 'pgup', 'next': 'pgdn'}

def _fillvkmap():
    if False:
        print('Hello World!')
    names = [entry for entry in win32con.__dict__ if entry.startswith('VK_')]
    for name in names:
        code = getattr(win32con, name)
        n = name[3:].lower()
        key_name_to_vk[n] = code
        if n in _better_names:
            n = _better_names[n]
            key_name_to_vk[n] = code
        key_code_to_name[code] = n
_fillvkmap()

def get_vk(chardesc):
    if False:
        i = 10
        return i + 15
    if len(chardesc) == 1:
        info = win32api.VkKeyScan(chardesc)
        if info == -1:
            return (0, 0)
        vk = win32api.LOBYTE(info)
        state = win32api.HIBYTE(info)
        modifiers = 0
        if state & 1:
            modifiers |= win32con.SHIFT_PRESSED
        if state & 2:
            modifiers |= win32con.LEFT_CTRL_PRESSED | win32con.RIGHT_CTRL_PRESSED
        if state & 4:
            modifiers |= win32con.LEFT_ALT_PRESSED | win32con.RIGHT_ALT_PRESSED
        return (vk, modifiers)
    return (key_name_to_vk.get(chardesc.lower()), 0)
modifiers = {'alt': win32con.LEFT_ALT_PRESSED | win32con.RIGHT_ALT_PRESSED, 'lalt': win32con.LEFT_ALT_PRESSED, 'ralt': win32con.RIGHT_ALT_PRESSED, 'ctrl': win32con.LEFT_CTRL_PRESSED | win32con.RIGHT_CTRL_PRESSED, 'ctl': win32con.LEFT_CTRL_PRESSED | win32con.RIGHT_CTRL_PRESSED, 'control': win32con.LEFT_CTRL_PRESSED | win32con.RIGHT_CTRL_PRESSED, 'lctrl': win32con.LEFT_CTRL_PRESSED, 'lctl': win32con.LEFT_CTRL_PRESSED, 'rctrl': win32con.RIGHT_CTRL_PRESSED, 'rctl': win32con.RIGHT_CTRL_PRESSED, 'shift': win32con.SHIFT_PRESSED, 'key': 0}

def parse_key_name(name):
    if False:
        i = 10
        return i + 15
    name = name + '-'
    start = pos = 0
    max = len(name)
    toks = []
    while pos < max:
        if name[pos] in '+-':
            tok = name[start:pos]
            toks.append(tok.lower())
            pos += 1
            start = pos
        pos += 1
    flags = 0
    for tok in toks[:-1]:
        mod = modifiers.get(tok.lower())
        if mod is not None:
            flags |= mod
    (vk, this_flags) = get_vk(toks[-1])
    return (vk, flags | this_flags)
_checks = [[('Shift', win32con.SHIFT_PRESSED)], [('Ctrl', win32con.LEFT_CTRL_PRESSED | win32con.RIGHT_CTRL_PRESSED), ('LCtrl', win32con.LEFT_CTRL_PRESSED), ('RCtrl', win32con.RIGHT_CTRL_PRESSED)], [('Alt', win32con.LEFT_ALT_PRESSED | win32con.RIGHT_ALT_PRESSED), ('LAlt', win32con.LEFT_ALT_PRESSED), ('RAlt', win32con.RIGHT_ALT_PRESSED)]]

def make_key_name(vk, flags):
    if False:
        i = 10
        return i + 15
    flags_done = 0
    parts = []
    for moddata in _checks:
        for (name, checkflag) in moddata:
            if flags & checkflag:
                parts.append(name)
                flags_done = flags_done & checkflag
                break
    if flags_done & flags:
        parts.append(hex(flags & ~flags_done))
    if vk is None:
        parts.append('<Unknown scan code>')
    else:
        try:
            parts.append(key_code_to_name[vk])
        except KeyError:
            scancode = win32api.MapVirtualKey(vk, MAPVK_VK_TO_CHAR)
            parts.append(chr(scancode))
    sep = '+'
    if sep in parts:
        sep = '-'
    return sep.join([p.capitalize() for p in parts])

def _psc(char):
    if False:
        i = 10
        return i + 15
    (sc, mods) = get_vk(char)
    print('Char %s -> %d -> %s' % (repr(char), sc, key_code_to_name.get(sc)))

def test1():
    if False:
        for i in range(10):
            print('nop')
    for ch in 'aA0/?[{}];:\'"`~_-+=\\|,<.>/?':
        _psc(ch)
    for code in ['Home', 'End', 'Left', 'Right', 'Up', 'Down', 'Menu', 'Next']:
        _psc(code)

def _pkn(n):
    if False:
        print('Hello World!')
    (vk, flags) = parse_key_name(n)
    print(f'{n} -> {vk},{flags} -> {make_key_name(vk, flags)}')

def test2():
    if False:
        return 10
    _pkn('ctrl+alt-shift+x')
    _pkn('ctrl-home')
    _pkn('Shift-+')
    _pkn('Shift--')
    _pkn('Shift+-')
    _pkn('Shift++')
    _pkn('LShift-+')
    _pkn('ctl+home')
    _pkn('ctl+enter')
    _pkn('alt+return')
    _pkn('Alt+/')
    _pkn('Alt+BadKeyName')
    _pkn('A')
    _pkn('a')
    _pkn('Shift-A')
    _pkn('Shift-a')
    _pkn('a')
    _pkn('(')
    _pkn('Ctrl+(')
    _pkn('Ctrl+Shift-8')
    _pkn('Ctrl+*')
    _pkn('{')
    _pkn('!')
    _pkn('.')
if __name__ == '__main__':
    test2()