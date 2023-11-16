"""Light wrapper around the Win32 Console API - this module should only be imported on Windows

The API that this module wraps is documented at https://docs.microsoft.com/en-us/windows/console/console-functions
"""
import ctypes
import sys
from typing import Any
windll: Any = None
if sys.platform == 'win32':
    windll = ctypes.LibraryLoader(ctypes.WinDLL)
else:
    raise ImportError(f'{__name__} can only be imported on Windows')
import time
from ctypes import Structure, byref, wintypes
from typing import IO, NamedTuple, Type, cast
from rich.color import ColorSystem
from rich.style import Style
STDOUT = -11
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 4
COORD = wintypes._COORD

class LegacyWindowsError(Exception):
    pass

class WindowsCoordinates(NamedTuple):
    """Coordinates in the Windows Console API are (y, x), not (x, y).
    This class is intended to prevent that confusion.
    Rows and columns are indexed from 0.
    This class can be used in place of wintypes._COORD in arguments and argtypes.
    """
    row: int
    col: int

    @classmethod
    def from_param(cls, value: 'WindowsCoordinates') -> COORD:
        if False:
            i = 10
            return i + 15
        'Converts a WindowsCoordinates into a wintypes _COORD structure.\n        This classmethod is internally called by ctypes to perform the conversion.\n\n        Args:\n            value (WindowsCoordinates): The input coordinates to convert.\n\n        Returns:\n            wintypes._COORD: The converted coordinates struct.\n        '
        return COORD(value.col, value.row)

class CONSOLE_SCREEN_BUFFER_INFO(Structure):
    _fields_ = [('dwSize', COORD), ('dwCursorPosition', COORD), ('wAttributes', wintypes.WORD), ('srWindow', wintypes.SMALL_RECT), ('dwMaximumWindowSize', COORD)]

class CONSOLE_CURSOR_INFO(ctypes.Structure):
    _fields_ = [('dwSize', wintypes.DWORD), ('bVisible', wintypes.BOOL)]
_GetStdHandle = windll.kernel32.GetStdHandle
_GetStdHandle.argtypes = [wintypes.DWORD]
_GetStdHandle.restype = wintypes.HANDLE

def GetStdHandle(handle: int=STDOUT) -> wintypes.HANDLE:
    if False:
        i = 10
        return i + 15
    'Retrieves a handle to the specified standard device (standard input, standard output, or standard error).\n\n    Args:\n        handle (int): Integer identifier for the handle. Defaults to -11 (stdout).\n\n    Returns:\n        wintypes.HANDLE: The handle\n    '
    return cast(wintypes.HANDLE, _GetStdHandle(handle))
_GetConsoleMode = windll.kernel32.GetConsoleMode
_GetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
_GetConsoleMode.restype = wintypes.BOOL

def GetConsoleMode(std_handle: wintypes.HANDLE) -> int:
    if False:
        i = 10
        return i + 15
    "Retrieves the current input mode of a console's input buffer\n    or the current output mode of a console screen buffer.\n\n    Args:\n        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.\n\n    Raises:\n        LegacyWindowsError: If any error occurs while calling the Windows console API.\n\n    Returns:\n        int: Value representing the current console mode as documented at\n            https://docs.microsoft.com/en-us/windows/console/getconsolemode#parameters\n    "
    console_mode = wintypes.DWORD()
    success = bool(_GetConsoleMode(std_handle, console_mode))
    if not success:
        raise LegacyWindowsError('Unable to get legacy Windows Console Mode')
    return console_mode.value
_FillConsoleOutputCharacterW = windll.kernel32.FillConsoleOutputCharacterW
_FillConsoleOutputCharacterW.argtypes = [wintypes.HANDLE, ctypes.c_char, wintypes.DWORD, cast(Type[COORD], WindowsCoordinates), ctypes.POINTER(wintypes.DWORD)]
_FillConsoleOutputCharacterW.restype = wintypes.BOOL

def FillConsoleOutputCharacter(std_handle: wintypes.HANDLE, char: str, length: int, start: WindowsCoordinates) -> int:
    if False:
        while True:
            i = 10
    'Writes a character to the console screen buffer a specified number of times, beginning at the specified coordinates.\n\n    Args:\n        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.\n        char (str): The character to write. Must be a string of length 1.\n        length (int): The number of times to write the character.\n        start (WindowsCoordinates): The coordinates to start writing at.\n\n    Returns:\n        int: The number of characters written.\n    '
    character = ctypes.c_char(char.encode())
    num_characters = wintypes.DWORD(length)
    num_written = wintypes.DWORD(0)
    _FillConsoleOutputCharacterW(std_handle, character, num_characters, start, byref(num_written))
    return num_written.value
_FillConsoleOutputAttribute = windll.kernel32.FillConsoleOutputAttribute
_FillConsoleOutputAttribute.argtypes = [wintypes.HANDLE, wintypes.WORD, wintypes.DWORD, cast(Type[COORD], WindowsCoordinates), ctypes.POINTER(wintypes.DWORD)]
_FillConsoleOutputAttribute.restype = wintypes.BOOL

def FillConsoleOutputAttribute(std_handle: wintypes.HANDLE, attributes: int, length: int, start: WindowsCoordinates) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Sets the character attributes for a specified number of character cells,\n    beginning at the specified coordinates in a screen buffer.\n\n    Args:\n        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.\n        attributes (int): Integer value representing the foreground and background colours of the cells.\n        length (int): The number of cells to set the output attribute of.\n        start (WindowsCoordinates): The coordinates of the first cell whose attributes are to be set.\n\n    Returns:\n        int: The number of cells whose attributes were actually set.\n    '
    num_cells = wintypes.DWORD(length)
    style_attrs = wintypes.WORD(attributes)
    num_written = wintypes.DWORD(0)
    _FillConsoleOutputAttribute(std_handle, style_attrs, num_cells, start, byref(num_written))
    return num_written.value
_SetConsoleTextAttribute = windll.kernel32.SetConsoleTextAttribute
_SetConsoleTextAttribute.argtypes = [wintypes.HANDLE, wintypes.WORD]
_SetConsoleTextAttribute.restype = wintypes.BOOL

def SetConsoleTextAttribute(std_handle: wintypes.HANDLE, attributes: wintypes.WORD) -> bool:
    if False:
        i = 10
        return i + 15
    'Set the colour attributes for all text written after this function is called.\n\n    Args:\n        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.\n        attributes (int): Integer value representing the foreground and background colours.\n\n\n    Returns:\n        bool: True if the attribute was set successfully, otherwise False.\n    '
    return bool(_SetConsoleTextAttribute(std_handle, attributes))
_GetConsoleScreenBufferInfo = windll.kernel32.GetConsoleScreenBufferInfo
_GetConsoleScreenBufferInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(CONSOLE_SCREEN_BUFFER_INFO)]
_GetConsoleScreenBufferInfo.restype = wintypes.BOOL

def GetConsoleScreenBufferInfo(std_handle: wintypes.HANDLE) -> CONSOLE_SCREEN_BUFFER_INFO:
    if False:
        for i in range(10):
            print('nop')
    'Retrieves information about the specified console screen buffer.\n\n    Args:\n        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.\n\n    Returns:\n        CONSOLE_SCREEN_BUFFER_INFO: A CONSOLE_SCREEN_BUFFER_INFO ctype struct contain information about\n            screen size, cursor position, colour attributes, and more.'
    console_screen_buffer_info = CONSOLE_SCREEN_BUFFER_INFO()
    _GetConsoleScreenBufferInfo(std_handle, byref(console_screen_buffer_info))
    return console_screen_buffer_info
_SetConsoleCursorPosition = windll.kernel32.SetConsoleCursorPosition
_SetConsoleCursorPosition.argtypes = [wintypes.HANDLE, cast(Type[COORD], WindowsCoordinates)]
_SetConsoleCursorPosition.restype = wintypes.BOOL

def SetConsoleCursorPosition(std_handle: wintypes.HANDLE, coords: WindowsCoordinates) -> bool:
    if False:
        while True:
            i = 10
    'Set the position of the cursor in the console screen\n\n    Args:\n        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.\n        coords (WindowsCoordinates): The coordinates to move the cursor to.\n\n    Returns:\n        bool: True if the function succeeds, otherwise False.\n    '
    return bool(_SetConsoleCursorPosition(std_handle, coords))
_GetConsoleCursorInfo = windll.kernel32.GetConsoleCursorInfo
_GetConsoleCursorInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(CONSOLE_CURSOR_INFO)]
_GetConsoleCursorInfo.restype = wintypes.BOOL

def GetConsoleCursorInfo(std_handle: wintypes.HANDLE, cursor_info: CONSOLE_CURSOR_INFO) -> bool:
    if False:
        return 10
    "Get the cursor info - used to get cursor visibility and width\n\n    Args:\n        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.\n        cursor_info (CONSOLE_CURSOR_INFO): CONSOLE_CURSOR_INFO ctype struct that receives information\n            about the console's cursor.\n\n    Returns:\n          bool: True if the function succeeds, otherwise False.\n    "
    return bool(_GetConsoleCursorInfo(std_handle, byref(cursor_info)))
_SetConsoleCursorInfo = windll.kernel32.SetConsoleCursorInfo
_SetConsoleCursorInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(CONSOLE_CURSOR_INFO)]
_SetConsoleCursorInfo.restype = wintypes.BOOL

def SetConsoleCursorInfo(std_handle: wintypes.HANDLE, cursor_info: CONSOLE_CURSOR_INFO) -> bool:
    if False:
        i = 10
        return i + 15
    'Set the cursor info - used for adjusting cursor visibility and width\n\n    Args:\n        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.\n        cursor_info (CONSOLE_CURSOR_INFO): CONSOLE_CURSOR_INFO ctype struct containing the new cursor info.\n\n    Returns:\n          bool: True if the function succeeds, otherwise False.\n    '
    return bool(_SetConsoleCursorInfo(std_handle, byref(cursor_info)))
_SetConsoleTitle = windll.kernel32.SetConsoleTitleW
_SetConsoleTitle.argtypes = [wintypes.LPCWSTR]
_SetConsoleTitle.restype = wintypes.BOOL

def SetConsoleTitle(title: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Sets the title of the current console window\n\n    Args:\n        title (str): The new title of the console window.\n\n    Returns:\n        bool: True if the function succeeds, otherwise False.\n    '
    return bool(_SetConsoleTitle(title))

class LegacyWindowsTerm:
    """This class allows interaction with the legacy Windows Console API. It should only be used in the context
    of environments where virtual terminal processing is not available. However, if it is used in a Windows environment,
    the entire API should work.

    Args:
        file (IO[str]): The file which the Windows Console API HANDLE is retrieved from, defaults to sys.stdout.
    """
    BRIGHT_BIT = 8
    ANSI_TO_WINDOWS = [0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15]

    def __init__(self, file: 'IO[str]') -> None:
        if False:
            print('Hello World!')
        handle = GetStdHandle(STDOUT)
        self._handle = handle
        default_text = GetConsoleScreenBufferInfo(handle).wAttributes
        self._default_text = default_text
        self._default_fore = default_text & 7
        self._default_back = default_text >> 4 & 7
        self._default_attrs = self._default_fore | self._default_back << 4
        self._file = file
        self.write = file.write
        self.flush = file.flush

    @property
    def cursor_position(self) -> WindowsCoordinates:
        if False:
            print('Hello World!')
        'Returns the current position of the cursor (0-based)\n\n        Returns:\n            WindowsCoordinates: The current cursor position.\n        '
        coord: COORD = GetConsoleScreenBufferInfo(self._handle).dwCursorPosition
        return WindowsCoordinates(row=cast(int, coord.Y), col=cast(int, coord.X))

    @property
    def screen_size(self) -> WindowsCoordinates:
        if False:
            i = 10
            return i + 15
        'Returns the current size of the console screen buffer, in character columns and rows\n\n        Returns:\n            WindowsCoordinates: The width and height of the screen as WindowsCoordinates.\n        '
        screen_size: COORD = GetConsoleScreenBufferInfo(self._handle).dwSize
        return WindowsCoordinates(row=cast(int, screen_size.Y), col=cast(int, screen_size.X))

    def write_text(self, text: str) -> None:
        if False:
            while True:
                i = 10
        'Write text directly to the terminal without any modification of styles\n\n        Args:\n            text (str): The text to write to the console\n        '
        self.write(text)
        self.flush()

    def write_styled(self, text: str, style: Style) -> None:
        if False:
            print('Hello World!')
        'Write styled text to the terminal.\n\n        Args:\n            text (str): The text to write\n            style (Style): The style of the text\n        '
        color = style.color
        bgcolor = style.bgcolor
        if style.reverse:
            (color, bgcolor) = (bgcolor, color)
        if color:
            fore = color.downgrade(ColorSystem.WINDOWS).number
            fore = fore if fore is not None else 7
            if style.bold:
                fore = fore | self.BRIGHT_BIT
            if style.dim:
                fore = fore & ~self.BRIGHT_BIT
            fore = self.ANSI_TO_WINDOWS[fore]
        else:
            fore = self._default_fore
        if bgcolor:
            back = bgcolor.downgrade(ColorSystem.WINDOWS).number
            back = back if back is not None else 0
            back = self.ANSI_TO_WINDOWS[back]
        else:
            back = self._default_back
        assert fore is not None
        assert back is not None
        SetConsoleTextAttribute(self._handle, attributes=ctypes.c_ushort(fore | back << 4))
        self.write_text(text)
        SetConsoleTextAttribute(self._handle, attributes=self._default_text)

    def move_cursor_to(self, new_position: WindowsCoordinates) -> None:
        if False:
            print('Hello World!')
        'Set the position of the cursor\n\n        Args:\n            new_position (WindowsCoordinates): The WindowsCoordinates representing the new position of the cursor.\n        '
        if new_position.col < 0 or new_position.row < 0:
            return
        SetConsoleCursorPosition(self._handle, coords=new_position)

    def erase_line(self) -> None:
        if False:
            while True:
                i = 10
        'Erase all content on the line the cursor is currently located at'
        screen_size = self.screen_size
        cursor_position = self.cursor_position
        cells_to_erase = screen_size.col
        start_coordinates = WindowsCoordinates(row=cursor_position.row, col=0)
        FillConsoleOutputCharacter(self._handle, ' ', length=cells_to_erase, start=start_coordinates)
        FillConsoleOutputAttribute(self._handle, self._default_attrs, length=cells_to_erase, start=start_coordinates)

    def erase_end_of_line(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Erase all content from the cursor position to the end of that line'
        cursor_position = self.cursor_position
        cells_to_erase = self.screen_size.col - cursor_position.col
        FillConsoleOutputCharacter(self._handle, ' ', length=cells_to_erase, start=cursor_position)
        FillConsoleOutputAttribute(self._handle, self._default_attrs, length=cells_to_erase, start=cursor_position)

    def erase_start_of_line(self) -> None:
        if False:
            print('Hello World!')
        'Erase all content from the cursor position to the start of that line'
        (row, col) = self.cursor_position
        start = WindowsCoordinates(row, 0)
        FillConsoleOutputCharacter(self._handle, ' ', length=col, start=start)
        FillConsoleOutputAttribute(self._handle, self._default_attrs, length=col, start=start)

    def move_cursor_up(self) -> None:
        if False:
            i = 10
            return i + 15
        'Move the cursor up a single cell'
        cursor_position = self.cursor_position
        SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row=cursor_position.row - 1, col=cursor_position.col))

    def move_cursor_down(self) -> None:
        if False:
            print('Hello World!')
        'Move the cursor down a single cell'
        cursor_position = self.cursor_position
        SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row=cursor_position.row + 1, col=cursor_position.col))

    def move_cursor_forward(self) -> None:
        if False:
            i = 10
            return i + 15
        'Move the cursor forward a single cell. Wrap to the next line if required.'
        (row, col) = self.cursor_position
        if col == self.screen_size.col - 1:
            row += 1
            col = 0
        else:
            col += 1
        SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row=row, col=col))

    def move_cursor_to_column(self, column: int) -> None:
        if False:
            i = 10
            return i + 15
        'Move cursor to the column specified by the zero-based column index, staying on the same row\n\n        Args:\n            column (int): The zero-based column index to move the cursor to.\n        '
        (row, _) = self.cursor_position
        SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row, column))

    def move_cursor_backward(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Move the cursor backward a single cell. Wrap to the previous line if required.'
        (row, col) = self.cursor_position
        if col == 0:
            row -= 1
            col = self.screen_size.col - 1
        else:
            col -= 1
        SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row=row, col=col))

    def hide_cursor(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Hide the cursor'
        current_cursor_size = self._get_cursor_size()
        invisible_cursor = CONSOLE_CURSOR_INFO(dwSize=current_cursor_size, bVisible=0)
        SetConsoleCursorInfo(self._handle, cursor_info=invisible_cursor)

    def show_cursor(self) -> None:
        if False:
            return 10
        'Show the cursor'
        current_cursor_size = self._get_cursor_size()
        visible_cursor = CONSOLE_CURSOR_INFO(dwSize=current_cursor_size, bVisible=1)
        SetConsoleCursorInfo(self._handle, cursor_info=visible_cursor)

    def set_title(self, title: str) -> None:
        if False:
            return 10
        'Set the title of the terminal window\n\n        Args:\n            title (str): The new title of the console window\n        '
        assert len(title) < 255, 'Console title must be less than 255 characters'
        SetConsoleTitle(title)

    def _get_cursor_size(self) -> int:
        if False:
            return 10
        'Get the percentage of the character cell that is filled by the cursor'
        cursor_info = CONSOLE_CURSOR_INFO()
        GetConsoleCursorInfo(self._handle, cursor_info=cursor_info)
        return int(cursor_info.dwSize)
if __name__ == '__main__':
    handle = GetStdHandle()
    from rich.console import Console
    console = Console()
    term = LegacyWindowsTerm(sys.stdout)
    term.set_title('Win32 Console Examples')
    style = Style(color='black', bgcolor='red')
    heading = Style.parse('black on green')
    console.rule('Checking colour output')
    console.print('[on red]on red!')
    console.print('[blue]blue!')
    console.print('[yellow]yellow!')
    console.print('[bold yellow]bold yellow!')
    console.print('[bright_yellow]bright_yellow!')
    console.print('[dim bright_yellow]dim bright_yellow!')
    console.print('[italic cyan]italic cyan!')
    console.print('[bold white on blue]bold white on blue!')
    console.print('[reverse bold white on blue]reverse bold white on blue!')
    console.print('[bold black on cyan]bold black on cyan!')
    console.print('[black on green]black on green!')
    console.print('[blue on green]blue on green!')
    console.print('[white on black]white on black!')
    console.print('[black on white]black on white!')
    console.print('[#1BB152 on #DA812D]#1BB152 on #DA812D!')
    console.rule('Checking cursor movement')
    console.print()
    term.move_cursor_backward()
    term.move_cursor_backward()
    term.write_text('went back and wrapped to prev line')
    time.sleep(1)
    term.move_cursor_up()
    term.write_text('we go up')
    time.sleep(1)
    term.move_cursor_down()
    term.write_text('and down')
    time.sleep(1)
    term.move_cursor_up()
    term.move_cursor_backward()
    term.move_cursor_backward()
    term.write_text('we went up and back 2')
    time.sleep(1)
    term.move_cursor_down()
    term.move_cursor_backward()
    term.move_cursor_backward()
    term.write_text('we went down and back 2')
    time.sleep(1)
    term.hide_cursor()
    console.print()
    console.rule('Checking line erasing')
    console.print('\n...Deleting to the start of the line...')
    term.write_text('The red arrow shows the cursor location, and direction of erase')
    time.sleep(1)
    term.move_cursor_to_column(16)
    term.write_styled('<', Style.parse('black on red'))
    term.move_cursor_backward()
    time.sleep(1)
    term.erase_start_of_line()
    time.sleep(1)
    console.print('\n\n...And to the end of the line...')
    term.write_text('The red arrow shows the cursor location, and direction of erase')
    time.sleep(1)
    term.move_cursor_to_column(16)
    term.write_styled('>', Style.parse('black on red'))
    time.sleep(1)
    term.erase_end_of_line()
    time.sleep(1)
    console.print('\n\n...Now the whole line will be erased...')
    term.write_styled("I'm going to disappear!", style=Style.parse('black on cyan'))
    time.sleep(1)
    term.erase_line()
    term.show_cursor()
    print('\n')