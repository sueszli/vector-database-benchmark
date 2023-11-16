"""Set of cursor resources available for use. These cursors come
in a sequence of values that are needed as the arguments for
pygame.mouse.set_cursor(). To dereference the sequence in place
and create the cursor in one step, call like this:
    pygame.mouse.set_cursor(*pygame.cursors.arrow).

Here is a list of available cursors:
    arrow, diamond, ball, broken_x, tri_left, tri_right

There is also a sample string cursor named 'thickarrow_strings'.
The compile() function can convert these string cursors into cursor byte data that can be used to
create Cursor objects.

Alternately, you can also create Cursor objects using surfaces or cursors constants,
such as pygame.SYSTEM_CURSOR_ARROW.
"""
import pygame
_cursor_id_table = {pygame.SYSTEM_CURSOR_ARROW: 'SYSTEM_CURSOR_ARROW', pygame.SYSTEM_CURSOR_IBEAM: 'SYSTEM_CURSOR_IBEAM', pygame.SYSTEM_CURSOR_WAIT: 'SYSTEM_CURSOR_WAIT', pygame.SYSTEM_CURSOR_CROSSHAIR: 'SYSTEM_CURSOR_CROSSHAIR', pygame.SYSTEM_CURSOR_WAITARROW: 'SYSTEM_CURSOR_WAITARROW', pygame.SYSTEM_CURSOR_SIZENWSE: 'SYSTEM_CURSOR_SIZENWSE', pygame.SYSTEM_CURSOR_SIZENESW: 'SYSTEM_CURSOR_SIZENESW', pygame.SYSTEM_CURSOR_SIZEWE: 'SYSTEM_CURSOR_SIZEWE', pygame.SYSTEM_CURSOR_SIZENS: 'SYSTEM_CURSOR_SIZENS', pygame.SYSTEM_CURSOR_SIZEALL: 'SYSTEM_CURSOR_SIZEALL', pygame.SYSTEM_CURSOR_NO: 'SYSTEM_CURSOR_NO', pygame.SYSTEM_CURSOR_HAND: 'SYSTEM_CURSOR_HAND'}

class Cursor:

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Cursor(size, hotspot, xormasks, andmasks) -> Cursor\n        Cursor(hotspot, Surface) -> Cursor\n        Cursor(constant) -> Cursor\n        Cursor(Cursor) -> copies the Cursor object passed as an argument\n        Cursor() -> Cursor\n\n        pygame object for representing cursors\n\n        You can initialize a cursor from a system cursor or use the\n        constructor on an existing Cursor object, which will copy it.\n        Providing a Surface instance will render the cursor displayed\n        as that Surface when used.\n\n        These Surfaces may use other colors than black and white.'
        if len(args) == 0:
            self.type = 'system'
            self.data = (pygame.SYSTEM_CURSOR_ARROW,)
        elif len(args) == 1 and args[0] in _cursor_id_table:
            self.type = 'system'
            self.data = (args[0],)
        elif len(args) == 1 and isinstance(args[0], Cursor):
            self.type = args[0].type
            self.data = args[0].data
        elif len(args) == 2 and len(args[0]) == 2 and isinstance(args[1], pygame.Surface):
            self.type = 'color'
            self.data = tuple(args)
        elif len(args) == 4 and len(args[0]) == 2 and (len(args[1]) == 2):
            self.type = 'bitmap'
            self.data = tuple((tuple(arg) for arg in args))
        else:
            raise TypeError('Arguments must match a cursor specification')

    def __len__(self):
        if False:
            return 10
        return len(self.data)

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.data)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        return self.data[index]

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, Cursor) and self.data == other.data

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self.__eq__(other)

    def __copy__(self):
        if False:
            i = 10
            return i + 15
        'Clone the current Cursor object.\n        You can do the same thing by doing Cursor(Cursor).'
        return self.__class__(self)
    copy = __copy__

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(tuple([self.type] + list(self.data)))

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if self.type == 'system':
            id_string = _cursor_id_table.get(self.data[0], 'constant lookup error')
            return f'<Cursor(type: system, constant: {id_string})>'
        if self.type == 'bitmap':
            size = f'size: {self.data[0]}'
            hotspot = f'hotspot: {self.data[1]}'
            return f'<Cursor(type: bitmap, {size}, {hotspot})>'
        if self.type == 'color':
            hotspot = f'hotspot: {self.data[0]}'
            surf = repr(self.data[1])
            return f'<Cursor(type: color, {hotspot}, surf: {surf})>'
        raise TypeError('Invalid Cursor')

def set_cursor(*args):
    if False:
        i = 10
        return i + 15
    'set_cursor(pygame.cursors.Cursor OR args for a pygame.cursors.Cursor) -> None\n    set the mouse cursor to a new cursor'
    cursor = Cursor(*args)
    pygame.mouse._set_cursor(**{cursor.type: cursor.data})
pygame.mouse.set_cursor = set_cursor
del set_cursor

def get_cursor():
    if False:
        while True:
            i = 10
    'get_cursor() -> pygame.cursors.Cursor\n    get the current mouse cursor'
    return Cursor(*pygame.mouse._get_cursor())
pygame.mouse.get_cursor = get_cursor
del get_cursor
arrow = Cursor((16, 16), (0, 0), (0, 0, 64, 0, 96, 0, 112, 0, 120, 0, 124, 0, 126, 0, 127, 0, 127, 128, 124, 0, 108, 0, 70, 0, 6, 0, 3, 0, 3, 0, 0, 0), (64, 0, 224, 0, 240, 0, 248, 0, 252, 0, 254, 0, 255, 0, 255, 128, 255, 192, 255, 128, 254, 0, 239, 0, 79, 0, 7, 128, 7, 128, 3, 0))
diamond = Cursor((16, 16), (7, 7), (0, 0, 1, 0, 3, 128, 7, 192, 14, 224, 28, 112, 56, 56, 112, 28, 56, 56, 28, 112, 14, 224, 7, 192, 3, 128, 1, 0, 0, 0, 0, 0), (1, 0, 3, 128, 7, 192, 15, 224, 31, 240, 62, 248, 124, 124, 248, 62, 124, 124, 62, 248, 31, 240, 15, 224, 7, 192, 3, 128, 1, 0, 0, 0))
ball = Cursor((16, 16), (7, 7), (0, 0, 3, 192, 15, 240, 24, 248, 51, 252, 55, 252, 127, 254, 127, 254, 127, 254, 127, 254, 63, 252, 63, 252, 31, 248, 15, 240, 3, 192, 0, 0), (3, 192, 15, 240, 31, 248, 63, 252, 127, 254, 127, 254, 255, 255, 255, 255, 255, 255, 255, 255, 127, 254, 127, 254, 63, 252, 31, 248, 15, 240, 3, 192))
broken_x = Cursor((16, 16), (7, 7), (0, 0, 96, 6, 112, 14, 56, 28, 28, 56, 12, 48, 0, 0, 0, 0, 0, 0, 0, 0, 12, 48, 28, 56, 56, 28, 112, 14, 96, 6, 0, 0), (224, 7, 240, 15, 248, 31, 124, 62, 62, 124, 30, 120, 14, 112, 0, 0, 0, 0, 14, 112, 30, 120, 62, 124, 124, 62, 248, 31, 240, 15, 224, 7))
tri_left = Cursor((16, 16), (1, 1), (0, 0, 96, 0, 120, 0, 62, 0, 63, 128, 31, 224, 31, 248, 15, 254, 15, 254, 7, 128, 7, 128, 3, 128, 3, 128, 1, 128, 1, 128, 0, 0), (224, 0, 248, 0, 254, 0, 127, 128, 127, 224, 63, 248, 63, 254, 31, 255, 31, 255, 15, 254, 15, 192, 7, 192, 7, 192, 3, 192, 3, 192, 1, 128))
tri_right = Cursor((16, 16), (14, 1), (0, 0, 0, 6, 0, 30, 0, 124, 1, 252, 7, 248, 31, 248, 127, 240, 127, 240, 1, 224, 1, 224, 1, 192, 1, 192, 1, 128, 1, 128, 0, 0), (0, 7, 0, 31, 0, 127, 1, 254, 7, 254, 31, 252, 127, 252, 255, 248, 255, 248, 127, 240, 3, 240, 3, 224, 3, 224, 3, 192, 3, 192, 1, 128))
thickarrow_strings = ('XX                      ', 'XXX                     ', 'XXXX                    ', 'XX.XX                   ', 'XX..XX                  ', 'XX...XX                 ', 'XX....XX                ', 'XX.....XX               ', 'XX......XX              ', 'XX.......XX             ', 'XX........XX            ', 'XX........XXX           ', 'XX......XXXXX           ', 'XX.XXX..XX              ', 'XXXX XX..XX             ', 'XX   XX..XX             ', '     XX..XX             ', '      XX..XX            ', '      XX..XX            ', '       XXXX             ', '       XX               ', '                        ', '                        ', '                        ')
sizer_x_strings = ('     X      X           ', '    XX      XX          ', '   X.X      X.X         ', '  X..X      X..X        ', ' X...XXXXXXXX...X       ', 'X................X      ', ' X...XXXXXXXX...X       ', '  X..X      X..X        ', '   X.X      X.X         ', '    XX      XX          ', '     X      X           ', '                        ', '                        ', '                        ', '                        ', '                        ')
sizer_y_strings = ('     X          ', '    X.X         ', '   X...X        ', '  X.....X       ', ' X.......X      ', 'XXXXX.XXXXX     ', '    X.X         ', '    X.X         ', '    X.X         ', '    X.X         ', '    X.X         ', '    X.X         ', '    X.X         ', 'XXXXX.XXXXX     ', ' X.......X      ', '  X.....X       ', '   X...X        ', '    X.X         ', '     X          ', '                ', '                ', '                ', '                ', '                ')
sizer_xy_strings = ('XXXXXXXX                ', 'X.....X                 ', 'X....X                  ', 'X...X                   ', 'X..X.X                  ', 'X.X X.X                 ', 'XX   X.X    X           ', 'X     X.X  XX           ', '       X.XX.X           ', '        X...X           ', '        X...X           ', '       X....X           ', '      X.....X           ', '     XXXXXXXX           ', '                        ', '                        ')
textmarker_strings = ('ooo ooo ', '   o    ', '   o    ', '   o    ', '   o    ', '   o    ', '   o    ', '   o    ', '   o    ', '   o    ', '   o    ', 'ooo ooo ', '        ', '        ', '        ', '        ')

def compile(strings, black='X', white='.', xor='o'):
    if False:
        return 10
    'pygame.cursors.compile(strings, black, white, xor) -> data, mask\n    compile cursor strings into cursor data\n\n    This takes a set of strings with equal length and computes\n    the binary data for that cursor. The string widths must be\n    divisible by 8.\n\n    The black and white arguments are single letter strings that\n    tells which characters will represent black pixels, and which\n    characters represent white pixels. All other characters are\n    considered clear.\n\n    Some systems allow you to set a special toggle color for the\n    system color, this is also called the xor color. If the system\n    does not support xor cursors, that color will simply be black.\n\n    This returns a tuple containing the cursor data and cursor mask\n    data. Both these arguments are used when setting a cursor with\n    pygame.mouse.set_cursor().\n    '
    size = (len(strings[0]), len(strings))
    if size[0] % 8 or size[1] % 8:
        raise ValueError(f'cursor string sizes must be divisible by 8 {size}')
    for s in strings[1:]:
        if len(s) != size[0]:
            raise ValueError('Cursor strings are inconsistent lengths')
    maskdata = []
    filldata = []
    maskitem = fillitem = 0
    step = 8
    for s in strings:
        for c in s:
            maskitem = maskitem << 1
            fillitem = fillitem << 1
            step = step - 1
            if c == black:
                maskitem = maskitem | 1
                fillitem = fillitem | 1
            elif c == white:
                maskitem = maskitem | 1
            elif c == xor:
                fillitem = fillitem | 1
            if not step:
                maskdata.append(maskitem)
                filldata.append(fillitem)
                maskitem = fillitem = 0
                step = 8
    return (tuple(filldata), tuple(maskdata))

def load_xbm(curs, mask):
    if False:
        for i in range(10):
            print('nop')
    'pygame.cursors.load_xbm(cursorfile, maskfile) -> cursor_args\n    reads a pair of XBM files into set_cursor arguments\n\n    Arguments can either be filenames or filelike objects\n    with the readlines method. Not largely tested, but\n    should work with typical XBM files.\n    '

    def bitswap(num):
        if False:
            while True:
                i = 10
        val = 0
        for x in range(8):
            b = num & 1 << x != 0
            val = val << 1 | b
        return val
    if hasattr(curs, 'readlines'):
        curs = curs.readlines()
    else:
        with open(curs, encoding='ascii') as cursor_f:
            curs = cursor_f.readlines()
    if hasattr(mask, 'readlines'):
        mask = mask.readlines()
    else:
        with open(mask, encoding='ascii') as mask_f:
            mask = mask_f.readlines()
    for (i, line) in enumerate(curs):
        if line.startswith('#define'):
            curs = curs[i:]
            break
    for (i, line) in enumerate(mask):
        if line.startswith('#define'):
            mask = mask[i:]
            break
    width = int(curs[0].split()[-1])
    height = int(curs[1].split()[-1])
    if curs[2].startswith('#define'):
        hotx = int(curs[2].split()[-1])
        hoty = int(curs[3].split()[-1])
    else:
        hotx = hoty = 0
    info = (width, height, hotx, hoty)
    possible_starts = ('static char', 'static unsigned char')
    for (i, line) in enumerate(curs):
        if line.startswith(possible_starts):
            break
    data = ' '.join(curs[i + 1:]).replace('};', '').replace(',', ' ')
    cursdata = []
    for x in data.split():
        cursdata.append(bitswap(int(x, 16)))
    cursdata = tuple(cursdata)
    for (i, line) in enumerate(mask):
        if line.startswith(possible_starts):
            break
    data = ' '.join(mask[i + 1:]).replace('};', '').replace(',', ' ')
    maskdata = []
    for x in data.split():
        maskdata.append(bitswap(int(x, 16)))
    maskdata = tuple(maskdata)
    return (info[:2], info[2:], cursdata, maskdata)