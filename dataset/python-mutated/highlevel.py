"""
Highlevel scripting API, requires xautomation to be installed
"""
import time
import os
import subprocess
import tempfile
import imghdr
import struct

class PatternNotFound(Exception):
    """Exception raised by functions"""
    pass
LEFT = 1
'Left mouse button'
MIDDLE = 2
'Middle mouse button'
RIGHT = 3
'Right mouse button'

def visgrep(scr: str, pat: str, tolerance: int=0) -> int:
    if False:
        return 10
    '\n    Usage: C{visgrep(scr: str, pat: str, tolerance: int = 0) -> int}\n\n    Visual grep of scr for pattern pat.\n\n    Requires xautomation (http://hoopajoo.net/projects/xautomation.html).\n\n    Usage: C{visgrep("screen.png", "pat.png")}\n\n    \n\n    @param scr: path of PNG image to be grepped.\n    @param pat: path of pattern image (PNG) to look for in scr.\n    @param tolerance: An integer ≥ 0 to specify the level of tolerance for \'fuzzy\' matches.\n    @raise ValueError: Raised if tolerance is negative or not convertable to int\n    @raise PatternNotFound: Raised if C{pat} not found.\n    @raise FileNotFoundError: Raised if either file is not found\n    @returns: Coordinates of the topleft point of the match, if any. Raises L{PatternNotFound} exception otherwise.\n    '
    tol = int(tolerance)
    if tol < 0:
        raise ValueError('tolerance must be ≥ 0.')
    with open(scr), open(pat):
        pass
    with tempfile.NamedTemporaryFile() as f:
        subprocess.call(['png2pat', pat], stdout=f)
        f.flush()
        os.fsync(f.fileno())
        vg = subprocess.Popen(['visgrep', '-t' + str(tol), scr, f.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = vg.communicate()
    coord_str = out[0].decode().split(' ')[0].split(',')
    try:
        coord = [int(coord_str[0]), int(coord_str[1])]
    except (ValueError, IndexError) as e:
        raise PatternNotFound(str([x.decode() for x in out]) + '\n\t' + repr(e))
    return coord

def get_png_dim(filepath: str) -> int:
    if False:
        print('Hello World!')
    '\n    Usage: C{get_png_dim(filepath:str) -> (int)}\n\n    Finds the dimension of a PNG.\n    @param filepath: file path of the PNG.\n    @returns: (width, height).\n    @raise Exception: Raised if the file is not a png\n    '
    if not imghdr.what(filepath) == 'png':
        raise Exception('not PNG')
    head = open(filepath, 'rb').read(24)
    return struct.unpack('!II', head[16:24])

def mouse_move(x: int, y: int, display: str=''):
    if False:
        i = 10
        return i + 15
    '\n    Moves the mouse using xte C{mousemove} from xautomation\n\n    @param x: x location to move the mouse to\n    @param y: y location to move the mouse to\n    @param display: X display to pass to C{xte}\n    '
    subprocess.call(['xte', '-x', display, 'mousemove {} {}'.format(int(x), int(y))])

def mouse_rmove(x: int, y: int, display: str=''):
    if False:
        print('Hello World!')
    '\n    Moves the mouse using xte C{mousermove} command from xautomation\n\n    @param x: x location to move the mouse to\n    @param y: y location to move the mouse to\n    @param display: X display to pass to C{xte}\n    '
    subprocess.call(['xte', '-x', display, 'mousermove {} {}'.format(int(x), int(y))])

def mouse_click(button: int, display: str=''):
    if False:
        return 10
    '\n    Clicks the mouse in the current location using xte C{mouseclick} from xautomation\n\n    @param button: Which button signal to send from the mouse\n    @param display: X display to pass to C{xte}\n    '
    subprocess.call(['xte', '-x', display, 'mouseclick {}'.format(int(button))])

def mouse_pos():
    if False:
        return 10
    '\n    Returns the current location of the mouse.\n\n    @returns: Returns the mouse location in a C{list}\n    '
    tmp = subprocess.check_output('xmousepos').decode().split()
    return list(map(int, tmp))[:2]

def click_on_pat(pat: str, mousebutton: int=1, offset: (float, float)=None, tolerance: int=0, restore_pos: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    "\n    Requires C{imagemagick}, C{xautomation}, C{xwd}.\n\n    Click on a pattern at a specified offset (x,y) in percent of the pattern dimension. x is the horizontal distance from the top left corner, y is the vertical distance from the top left corner. By default, the offset is (50,50), which means that the center of the pattern will be clicked at.\n\n    @param pat: path of pattern image (PNG) to click on.\n    @param mousebutton: mouse button number used for the click\n    @param offset: offset from the top left point of the match. (float,float)\n    @param tolerance: An integer ≥ 0 to specify the level of tolerance for 'fuzzy' matches. If negative or not convertible to int, raises ValueError.\n    @param restore_pos: return to the initial mouse position after the click.\n    @raises: L{PatternNotFound}: Raised when the pattern is not found on the screen\n    "
    (x0, y0) = mouse_pos()
    move_to_pat(pat, offset, tolerance)
    mouse_click(mousebutton)
    if restore_pos:
        mouse_move(x0, y0)

def move_to_pat(pat: str, offset: (float, float)=None, tolerance: int=0) -> None:
    if False:
        for i in range(10):
            print('nop')
    'See L{click_on_pat}'
    with tempfile.NamedTemporaryFile() as f:
        subprocess.call('\n        xwd -root -silent -display :0 | \n        convert xwd:- png:' + f.name, shell=True)
        loc = visgrep(f.name, pat, tolerance)
    pat_size = get_png_dim(pat)
    if offset is None:
        (x, y) = [l + ps // 2 for (l, ps) in zip(loc, pat_size)]
    else:
        (x, y) = [l + ps * (off / 100) for (off, l, ps) in zip(offset, loc, pat_size)]
    mouse_move(x, y)

def acknowledge_gnome_notification():
    if False:
        return 10
    '\n    Moves mouse pointer to the bottom center of the screen and clicks on it.\n    '
    (x0, y0) = mouse_pos()
    mouse_move(10000, 10000)
    (x, y) = mouse_pos()
    mouse_rmove(-x / 2, 0)
    mouse_click(LEFT)
    time.sleep(0.2)
    mouse_move(x0, y0)