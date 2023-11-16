"""
GtkClipboard Functions
"""
from gi.repository import Gtk, Gdk
from pathlib import Path

class GtkClipboard:
    """
    Read/write access to the X selection and clipboard - GTK version
    """

    def __init__(self, app):
        if False:
            return 10
        '\n        Initialize the Gtk version of the Clipboard\n\n        Usage: Called when GtkClipboard is imported\n\n        @param app: refers to the application instance\n        '
        self.clipBoard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        '\n        Refers to the data contained in the Gtk Clipboard (conventional clipboard)\n        '
        self.selection = Gtk.Clipboard.get(Gdk.SELECTION_PRIMARY)
        '\n        Refers to the selection of the clipboard or the highlighted text\n        '
        self.app = app
        '\n        Refers to the application instance\n        '

    def fill_selection(self, contents):
        if False:
            print('Hello World!')
        '\n        Copy text into the selection\n\n        Usage: C{clipboard.fill_selection(contents)}\n\n        @param contents: string to be placed in the selection\n        '
        self.__fillSelection(contents)

    def __fillSelection(self, string):
        if False:
            while True:
                i = 10
        '\n        Backend for the C{fill_selection} method\n        \n        Sets the selection text to the C{string} value\n\n        @param string: Value to change the selection to\n        '
        Gdk.threads_enter()
        self.selection.set_text(string, -1)
        Gdk.threads_leave()

    def get_selection(self):
        if False:
            print('Hello World!')
        '\n        Read text from the selection\n\n        Refers to the currently-highlighted text\n\n        Usage: C{clipboard.get_selection()}\n\n        @return: text contents of the selection\n        @rtype: C{str}\n        @raise Exception: if no text was found in the selection\n        '
        Gdk.threads_enter()
        text = self.selection.wait_for_text()
        Gdk.threads_leave()
        if text is not None:
            return text
        else:
            raise Exception('No text found in X selection')

    def fill_clipboard(self, contents):
        if False:
            print('Hello World!')
        '\n        Copy text into the clipboard\n\n        Usage: C{clipboard.fill_clipboard(contents)}\n\n        @param contents: string to be placed onto the clipboard\n        '
        Gdk.threads_enter()
        if Gtk.get_major_version() >= 3:
            self.clipBoard.set_text(contents, -1)
        else:
            self.clipBoard.set_text(contents)
        Gdk.threads_leave()

    def get_clipboard(self):
        if False:
            while True:
                i = 10
        '\n        Read text from the clipboard\n\n        Usage: C{clipboard.get_clipboard()}\n\n        @return: text contents of the clipboard\n        @rtype: C{str}\n        @raise Exception: if no text was found on the clipboard\n        '
        Gdk.threads_enter()
        text = self.clipBoard.wait_for_text()
        Gdk.threads_leave()
        if text is not None:
            return text
        else:
            raise Exception('No text found on clipboard')

    def set_clipboard_image(self, path):
        if False:
            return 10
        '\n        Set clipboard to image\n\n        Usage: C{clipboard.set_clipboard_image(path)}\n\n        @param path: path to image file\n        @raise OSError: if path does not exist\n\n        '
        image_path = Path(path).expanduser()
        if image_path.exists():
            Gdk.threads_enter()
            copied_image = Gtk.Image.new_from_file(str(image_path))
            self.clipBoard.set_image(copied_image.get_pixbuf())
            Gdk.threads_leave()
        else:
            raise OSError('Image file not found')