import os
import shutil
import subprocess
import sys
from shlex import quote
from . import Image
_viewers = []

def register(viewer, order=1):
    if False:
        return 10
    '\n    The :py:func:`register` function is used to register additional viewers::\n\n        from PIL import ImageShow\n        ImageShow.register(MyViewer())  # MyViewer will be used as a last resort\n        ImageShow.register(MySecondViewer(), 0)  # MySecondViewer will be prioritised\n        ImageShow.register(ImageShow.XVViewer(), 0)  # XVViewer will be prioritised\n\n    :param viewer: The viewer to be registered.\n    :param order:\n        Zero or a negative integer to prepend this viewer to the list,\n        a positive integer to append it.\n    '
    try:
        if issubclass(viewer, Viewer):
            viewer = viewer()
    except TypeError:
        pass
    if order > 0:
        _viewers.append(viewer)
    else:
        _viewers.insert(0, viewer)

def show(image, title=None, **options):
    if False:
        i = 10
        return i + 15
    '\n    Display a given image.\n\n    :param image: An image object.\n    :param title: Optional title. Not all viewers can display the title.\n    :param \\**options: Additional viewer options.\n    :returns: ``True`` if a suitable viewer was found, ``False`` otherwise.\n    '
    for viewer in _viewers:
        if viewer.show(image, title=title, **options):
            return True
    return False

class Viewer:
    """Base class for viewers."""

    def show(self, image, **options):
        if False:
            i = 10
            return i + 15
        '\n        The main function for displaying an image.\n        Converts the given image to the target format and displays it.\n        '
        if not (image.mode in ('1', 'RGBA') or (self.format == 'PNG' and image.mode in ('I;16', 'LA'))):
            base = Image.getmodebase(image.mode)
            if image.mode != base:
                image = image.convert(base)
        return self.show_image(image, **options)
    format = None
    'The format to convert the image into.'
    options = {}
    'Additional options used to convert the image.'

    def get_format(self, image):
        if False:
            i = 10
            return i + 15
        'Return format name, or ``None`` to save as PGM/PPM.'
        return self.format

    def get_command(self, file, **options):
        if False:
            while True:
                i = 10
        '\n        Returns the command used to display the file.\n        Not implemented in the base class.\n        '
        msg = 'unavailable in base viewer'
        raise NotImplementedError(msg)

    def save_image(self, image):
        if False:
            while True:
                i = 10
        'Save to temporary file and return filename.'
        return image._dump(format=self.get_format(image), **self.options)

    def show_image(self, image, **options):
        if False:
            print('Hello World!')
        'Display the given image.'
        return self.show_file(self.save_image(image), **options)

    def show_file(self, path, **options):
        if False:
            while True:
                i = 10
        '\n        Display given file.\n        '
        os.system(self.get_command(path, **options))
        return 1

class WindowsViewer(Viewer):
    """The default viewer on Windows is the default system application for PNG files."""
    format = 'PNG'
    options = {'compress_level': 1, 'save_all': True}

    def get_command(self, file, **options):
        if False:
            while True:
                i = 10
        return f'start "Pillow" /WAIT "{file}" && ping -n 4 127.0.0.1 >NUL && del /f "{file}"'
if sys.platform == 'win32':
    register(WindowsViewer)

class MacViewer(Viewer):
    """The default viewer on macOS using ``Preview.app``."""
    format = 'PNG'
    options = {'compress_level': 1, 'save_all': True}

    def get_command(self, file, **options):
        if False:
            while True:
                i = 10
        command = 'open -a Preview.app'
        command = f'({command} {quote(file)}; sleep 20; rm -f {quote(file)})&'
        return command

    def show_file(self, path, **options):
        if False:
            while True:
                i = 10
        '\n        Display given file.\n        '
        subprocess.call(['open', '-a', 'Preview.app', path])
        executable = sys.executable or shutil.which('python3')
        if executable:
            subprocess.Popen([executable, '-c', 'import os, sys, time; time.sleep(20); os.remove(sys.argv[1])', path])
        return 1
if sys.platform == 'darwin':
    register(MacViewer)

class UnixViewer(Viewer):
    format = 'PNG'
    options = {'compress_level': 1, 'save_all': True}

    def get_command(self, file, **options):
        if False:
            return 10
        command = self.get_command_ex(file, **options)[0]
        return f'({command} {quote(file)}'

class XDGViewer(UnixViewer):
    """
    The freedesktop.org ``xdg-open`` command.
    """

    def get_command_ex(self, file, **options):
        if False:
            i = 10
            return i + 15
        command = executable = 'xdg-open'
        return (command, executable)

    def show_file(self, path, **options):
        if False:
            while True:
                i = 10
        '\n        Display given file.\n        '
        subprocess.Popen(['xdg-open', path])
        return 1

class DisplayViewer(UnixViewer):
    """
    The ImageMagick ``display`` command.
    This viewer supports the ``title`` parameter.
    """

    def get_command_ex(self, file, title=None, **options):
        if False:
            i = 10
            return i + 15
        command = executable = 'display'
        if title:
            command += f' -title {quote(title)}'
        return (command, executable)

    def show_file(self, path, **options):
        if False:
            i = 10
            return i + 15
        '\n        Display given file.\n        '
        args = ['display']
        title = options.get('title')
        if title:
            args += ['-title', title]
        args.append(path)
        subprocess.Popen(args)
        return 1

class GmDisplayViewer(UnixViewer):
    """The GraphicsMagick ``gm display`` command."""

    def get_command_ex(self, file, **options):
        if False:
            i = 10
            return i + 15
        executable = 'gm'
        command = 'gm display'
        return (command, executable)

    def show_file(self, path, **options):
        if False:
            for i in range(10):
                print('nop')
        '\n        Display given file.\n        '
        subprocess.Popen(['gm', 'display', path])
        return 1

class EogViewer(UnixViewer):
    """The GNOME Image Viewer ``eog`` command."""

    def get_command_ex(self, file, **options):
        if False:
            print('Hello World!')
        executable = 'eog'
        command = 'eog -n'
        return (command, executable)

    def show_file(self, path, **options):
        if False:
            print('Hello World!')
        '\n        Display given file.\n        '
        subprocess.Popen(['eog', '-n', path])
        return 1

class XVViewer(UnixViewer):
    """
    The X Viewer ``xv`` command.
    This viewer supports the ``title`` parameter.
    """

    def get_command_ex(self, file, title=None, **options):
        if False:
            i = 10
            return i + 15
        command = executable = 'xv'
        if title:
            command += f' -name {quote(title)}'
        return (command, executable)

    def show_file(self, path, **options):
        if False:
            print('Hello World!')
        '\n        Display given file.\n        '
        args = ['xv']
        title = options.get('title')
        if title:
            args += ['-name', title]
        args.append(path)
        subprocess.Popen(args)
        return 1
if sys.platform not in ('win32', 'darwin'):
    if shutil.which('xdg-open'):
        register(XDGViewer)
    if shutil.which('display'):
        register(DisplayViewer)
    if shutil.which('gm'):
        register(GmDisplayViewer)
    if shutil.which('eog'):
        register(EogViewer)
    if shutil.which('xv'):
        register(XVViewer)

class IPythonViewer(Viewer):
    """The viewer for IPython frontends."""

    def show_image(self, image, **options):
        if False:
            i = 10
            return i + 15
        ipython_display(image)
        return 1
try:
    from IPython.display import display as ipython_display
except ImportError:
    pass
else:
    register(IPythonViewer)
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Syntax: python3 ImageShow.py imagefile [title]')
        sys.exit()
    with Image.open(sys.argv[1]) as im:
        print(show(im, *sys.argv[2:]))