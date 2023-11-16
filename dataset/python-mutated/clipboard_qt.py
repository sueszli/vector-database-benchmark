"""
QtClipboard Functions
"""
import threading
from PyQt5.QtGui import QClipboard, QImage
from PyQt5.QtWidgets import QApplication
from pathlib import Path

class QtClipboard:
    """
    Read/write access to the X selection and clipboard - QT version
    """

    def __init__(self, app):
        if False:
            return 10
        '\n        Initialize the Qt version of the clipboard\n\n        Usage: Called when QtClipboard is imported.\n\n        @param app: refers to the application instance\n        '
        self.clipBoard = QApplication.clipboard()
        '\n        Refers to the Qt clipboard object\n        '
        self.app = app
        '\n        Refers to the application instance\n        '
        self.text = None
        '\n        Used to temporarily store the value of the selection or clipboard\n        '
        self.sem = None
        '\n        Qt semaphore object used for asynchronous method execution\n        '

    def fill_selection(self, contents):
        if False:
            while True:
                i = 10
        '\n        Copy text into the selection\n\n        Usage: C{clipboard.fill_selection(contents)}\n\n        @param contents: string to be placed in the selection\n        '
        self.__execAsync(self.__fillSelection, contents)

    def __fillSelection(self, string):
        if False:
            for i in range(10):
                print('nop')
        '\n        Backend for the C{fill_selection} method\n\n        Sets the selection text to the C{string} value\n\n        @param string: Value to change the selection to\n        '
        self.clipBoard.setText(string, QClipboard.Selection)
        self.sem.release()

    def get_selection(self):
        if False:
            i = 10
            return i + 15
        '\n        Read text from the selection\n\n        Usage: C{clipboard.get_selection()}\n\n        @return: text contents of the selection\n        @rtype: C{str}\n        '
        self.__execAsync(self.__getSelection)
        return str(self.text)

    def __getSelection(self):
        if False:
            return 10
        self.text = self.clipBoard.text(QClipboard.Selection)
        self.sem.release()

    def fill_clipboard(self, contents):
        if False:
            print('Hello World!')
        '\n        Copy text onto the clipboard\n\n        Usage: C{clipboard.fill_clipboard(contents)}\n\n        @param contents: string to be placed onto the clipboard\n        '
        self.__execAsync(self.__fillClipboard, contents)

    def set_clipboard_image(self, path):
        if False:
            print('Hello World!')
        '\n        Set clipboard to image\n\n        Usage: C{clipboard.set_clipboard_image(path)}\n\n        @param path: Path to image file\n        @raise OSError: If path does not exist\n        '
        self.__execAsync(self.__set_clipboard_image, path)

    def __set_clipboard_image(self, path):
        if False:
            return 10
        image_path = Path(path).expanduser()
        if image_path.exists():
            copied_image = QImage()
            copied_image.load(str(image_path))
            self.clipBoard.setImage(copied_image)
        else:
            raise OSError

    def __fillClipboard(self, string):
        if False:
            i = 10
            return i + 15
        self.clipBoard.setText(string, QClipboard.Clipboard)
        self.sem.release()

    def get_clipboard(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read text from the clipboard\n\n        Usage: C{clipboard.get_clipboard()}\n\n        @return: text contents of the clipboard\n        @rtype: C{str}\n        '
        self.__execAsync(self.__getClipboard)
        return str(self.text)

    def __getClipboard(self):
        if False:
            return 10
        '\n        Backend for the C{get_clipboard} method\n\n        Stores the value of the clipboard into the C{self.text} variable\n        '
        self.text = self.clipBoard.text(QClipboard.Clipboard)
        self.sem.release()

    def __execAsync(self, callback, *args):
        if False:
            print('Hello World!')
        '\n        Backend to execute methods asynchronously in Qt\n        '
        self.sem = threading.Semaphore(0)
        self.app.exec_in_main(callback, *args)
        self.sem.acquire()