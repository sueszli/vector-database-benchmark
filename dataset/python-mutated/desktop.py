from enum import Enum
from os import PathLike
import os.path as op
import logging

class SpecialFolder(Enum):
    APPDATA = 1
    CACHE = 2

def open_url(url: str) -> None:
    if False:
        return 10
    'Open ``url`` with the default browser.'
    _open_url(url)

def open_path(path: PathLike) -> None:
    if False:
        while True:
            i = 10
    'Open ``path`` with its associated application.'
    _open_path(str(path))

def reveal_path(path: PathLike) -> None:
    if False:
        while True:
            i = 10
    'Open the folder containing ``path`` with the default file browser.'
    _reveal_path(str(path))

def special_folder_path(special_folder: SpecialFolder, portable: bool=False) -> str:
    if False:
        i = 10
        return i + 15
    "Returns the path of ``special_folder``.\n\n    ``special_folder`` is a SpecialFolder.* const. The result is the special folder for the current\n    application. The running process' application info is used to determine relevant information.\n\n    You can override the application name with ``appname``. This argument is ingored under Qt.\n    "
    return _special_folder_path(special_folder, portable=portable)
try:
    from PyQt5.QtCore import QUrl, QStandardPaths
    from PyQt5.QtGui import QDesktopServices
    from qt.util import get_appdata
    from core.util import executable_folder
    from hscommon.plat import ISWINDOWS, ISOSX
    import subprocess

    def _open_url(url: str) -> None:
        if False:
            i = 10
            return i + 15
        QDesktopServices.openUrl(QUrl(url))

    def _open_path(path: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        url = QUrl.fromLocalFile(str(path))
        QDesktopServices.openUrl(url)

    def _reveal_path(path: str) -> None:
        if False:
            i = 10
            return i + 15
        if ISWINDOWS:
            subprocess.run(['explorer', '/select,', op.abspath(path)])
        elif ISOSX:
            subprocess.run(['open', '-R', op.abspath(path)])
        else:
            _open_path(op.dirname(str(path)))

    def _special_folder_path(special_folder: SpecialFolder, portable: bool=False) -> str:
        if False:
            while True:
                i = 10
        if special_folder == SpecialFolder.CACHE:
            if ISWINDOWS and portable:
                folder = op.join(executable_folder(), 'cache')
            else:
                folder = QStandardPaths.standardLocations(QStandardPaths.CacheLocation)[0]
        else:
            folder = get_appdata(portable)
        return folder
except ImportError:
    logging.warning("Can't setup desktop functions!")

    def _open_url(url: str) -> None:
        if False:
            print('Hello World!')
        pass

    def _open_path(path: str) -> None:
        if False:
            return 10
        pass

    def _reveal_path(path: str) -> None:
        if False:
            while True:
                i = 10
        pass

    def _special_folder_path(special_folder: SpecialFolder, portable: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '/tmp'