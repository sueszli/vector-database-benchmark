"""
Functions for working with the codepage on Windows systems
"""
import logging
from contextlib import contextmanager
from salt.exceptions import CodePageError
log = logging.getLogger(__name__)
try:
    import pywintypes
    import win32console
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if Win32 Libraries are installed\n    '
    if not HAS_WIN32:
        return (False, 'This utility requires pywin32')
    return 'win_chcp'

@contextmanager
def chcp(page_id, raise_error=False):
    if False:
        i = 10
        return i + 15
    '\n    Gets or sets the codepage of the shell.\n\n    Args:\n\n        page_id (str, int):\n            A number representing the codepage.\n\n        raise_error (bool):\n            ``True`` will raise an error if the codepage fails to change.\n            ``False`` will suppress the error\n\n    Returns:\n        int: A number representing the codepage\n\n    Raises:\n        CodePageError: On unsuccessful codepage change\n    '
    if not isinstance(page_id, int):
        try:
            page_id = int(page_id)
        except ValueError:
            error = 'The `page_id` needs to be an integer, not {}'.format(type(page_id))
            if raise_error:
                raise CodePageError(error)
            log.error(error)
            return -1
    previous_page_id = get_codepage_id(raise_error=raise_error)
    if page_id and previous_page_id and (page_id != previous_page_id):
        set_code_page = True
    else:
        set_code_page = False
    try:
        if set_code_page:
            set_codepage_id(page_id, raise_error=raise_error)
        yield
    finally:
        if set_code_page:
            set_codepage_id(previous_page_id, raise_error=raise_error)

def get_codepage_id(raise_error=False):
    if False:
        i = 10
        return i + 15
    '\n    Get the currently set code page on windows\n\n    Args:\n\n        raise_error (bool):\n            ``True`` will raise an error if the codepage fails to change.\n            ``False`` will suppress the error\n\n    Returns:\n        int: A number representing the codepage\n\n    Raises:\n        CodePageError: On unsuccessful codepage change\n    '
    try:
        return win32console.GetConsoleCP()
    except pywintypes.error as exc:
        (_, _, msg) = exc.args
        error = 'Failed to get the windows code page: {}'.format(msg)
        if raise_error:
            raise CodePageError(error)
        else:
            log.error(error)
        return -1

def set_codepage_id(page_id, raise_error=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the code page on windows\n\n    Args:\n\n        page_id (str, int):\n            A number representing the codepage.\n\n        raise_error (bool):\n            ``True`` will raise an error if the codepage fails to change.\n            ``False`` will suppress the error\n\n    Returns:\n        int: A number representing the codepage\n\n    Raises:\n        CodePageError: On unsuccessful codepage change\n    '
    if not isinstance(page_id, int):
        try:
            page_id = int(page_id)
        except ValueError:
            error = 'The `page_id` needs to be an integer, not {}'.format(type(page_id))
            if raise_error:
                raise CodePageError(error)
            log.error(error)
            return -1
    try:
        win32console.SetConsoleCP(page_id)
        return get_codepage_id(raise_error=raise_error)
    except pywintypes.error as exc:
        (_, _, msg) = exc.args
        error = 'Failed to set the windows code page: {}'.format(msg)
        if raise_error:
            raise CodePageError(error)
        else:
            log.error(error)
        return -1