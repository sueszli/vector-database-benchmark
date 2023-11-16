import os
import subprocess
import sys
try:
    import wx
except ImportError:
    wx = None
try:
    from gtk import gdk
except ImportError:
    gdk = None
try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None
from robot.api import logger
from robot.libraries.BuiltIn import BuiltIn
from robot.version import get_version
from robot.utils import abspath, get_error_message, get_link_path

class Screenshot:
    """Library for taking screenshots on the machine where tests are executed.

    Taking the actual screenshot requires a suitable tool or module that may
    need to be installed separately. Taking screenshots also requires tests
    to be run with a physical or virtual display.

    == Table of contents ==

    %TOC%

    = Supported screenshot taking tools and modules =

    How screenshots are taken depends on the operating system. On OSX
    screenshots are taken using the built-in ``screencapture`` utility. On
    other operating systems you need to have one of the following tools or
    Python modules installed. You can specify the tool/module to use when
    `importing` the library. If no tool or module is specified, the first
    one found will be used.

    - wxPython :: http://wxpython.org :: Generic Python GUI toolkit.
    - PyGTK :: http://pygtk.org :: This module is available by default on most
      Linux distributions.
    - Pillow :: http://python-pillow.github.io ::
      Only works on Windows. Also the original PIL package is supported.
    - Scrot :: http://en.wikipedia.org/wiki/Scrot :: Not used on Windows.
      Install with ``apt-get install scrot`` or similar.

    = Where screenshots are saved =

    By default screenshots are saved into the same directory where the Robot
    Framework log file is written. If no log is created, screenshots are saved
    into the directory where the XML output file is written.

    It is possible to specify a custom location for screenshots using
    ``screenshot_directory`` argument when `importing` the library and
    using `Set Screenshot Directory` keyword during execution. It is also
    possible to save screenshots using an absolute path.

    = ScreenCapLibrary =

    [https://github.com/mihaiparvu/ScreenCapLibrary|ScreenCapLibrary] is an
    external Robot Framework library that can be used as an alternative,
    which additionally provides support for multiple formats, adjusting the
    quality, using GIFs and video capturing.
    """
    ROBOT_LIBRARY_SCOPE = 'TEST SUITE'
    ROBOT_LIBRARY_VERSION = get_version()

    def __init__(self, screenshot_directory=None, screenshot_module=None):
        if False:
            print('Hello World!')
        'Configure where screenshots are saved.\n\n        If ``screenshot_directory`` is not given, screenshots are saved into\n        same directory as the log file. The directory can also be set using\n        `Set Screenshot Directory` keyword.\n\n        ``screenshot_module`` specifies the module or tool to use when using\n        this library outside OSX. Possible values are ``wxPython``,\n        ``PyGTK``, ``PIL`` and ``scrot``, case-insensitively. If no value is\n        given, the first module/tool found is used in that order.\n\n        Examples:\n        | =Setting= |  =Value=   |  =Value=   |\n        | Library   | Screenshot |            |\n        | Library   | Screenshot | ${TEMPDIR} |\n        | Library   | Screenshot | screenshot_module=PyGTK |\n        '
        self._given_screenshot_dir = self._norm_path(screenshot_directory)
        self._screenshot_taker = ScreenshotTaker(screenshot_module)

    def _norm_path(self, path):
        if False:
            return 10
        if not path:
            return path
        elif isinstance(path, os.PathLike):
            path = str(path)
        else:
            path = path.replace('/', os.sep)
        return os.path.normpath(path)

    @property
    def _screenshot_dir(self):
        if False:
            while True:
                i = 10
        return self._given_screenshot_dir or self._log_dir

    @property
    def _log_dir(self):
        if False:
            print('Hello World!')
        variables = BuiltIn().get_variables()
        outdir = variables['${OUTPUTDIR}']
        log = variables['${LOGFILE}']
        log = os.path.dirname(log) if log != 'NONE' else '.'
        return self._norm_path(os.path.join(outdir, log))

    def set_screenshot_directory(self, path):
        if False:
            while True:
                i = 10
        'Sets the directory where screenshots are saved.\n\n        It is possible to use ``/`` as a path separator in all operating\n        systems. Path to the old directory is returned.\n\n        The directory can also be set in `importing`.\n        '
        path = self._norm_path(path)
        if not os.path.isdir(path):
            raise RuntimeError("Directory '%s' does not exist." % path)
        old = self._screenshot_dir
        self._given_screenshot_dir = path
        return old

    def take_screenshot(self, name='screenshot', width='800px'):
        if False:
            return 10
        'Takes a screenshot in JPEG format and embeds it into the log file.\n\n        Name of the file where the screenshot is stored is derived from the\n        given ``name``. If the ``name`` ends with extension ``.jpg`` or\n        ``.jpeg``, the screenshot will be stored with that exact name.\n        Otherwise a unique name is created by adding an underscore, a running\n        index and an extension to the ``name``.\n\n        The name will be interpreted to be relative to the directory where\n        the log file is written. It is also possible to use absolute paths.\n        Using ``/`` as a path separator works in all operating systems.\n\n        ``width`` specifies the size of the screenshot in the log file.\n\n        Examples: (LOGDIR is determined automatically by the library)\n        | Take Screenshot |                  |     | # LOGDIR/screenshot_1.jpg (index automatically incremented) |\n        | Take Screenshot | mypic            |     | # LOGDIR/mypic_1.jpg (index automatically incremented) |\n        | Take Screenshot | ${TEMPDIR}/mypic |     | # /tmp/mypic_1.jpg (index automatically incremented) |\n        | Take Screenshot | pic.jpg          |     | # LOGDIR/pic.jpg (always uses this file) |\n        | Take Screenshot | images/login.jpg | 80% | # Specify both name and width. |\n        | Take Screenshot | width=550px      |     | # Specify only width. |\n\n        The path where the screenshot is saved is returned.\n        '
        path = self._save_screenshot(name)
        self._embed_screenshot(path, width)
        return path

    def take_screenshot_without_embedding(self, name='screenshot'):
        if False:
            i = 10
            return i + 15
        'Takes a screenshot and links it from the log file.\n\n        This keyword is otherwise identical to `Take Screenshot` but the saved\n        screenshot is not embedded into the log file. The screenshot is linked\n        so it is nevertheless easily available.\n        '
        path = self._save_screenshot(name)
        self._link_screenshot(path)
        return path

    def _save_screenshot(self, name):
        if False:
            while True:
                i = 10
        name = str(name) if isinstance(name, os.PathLike) else name.replace('/', os.sep)
        path = self._get_screenshot_path(name)
        return self._screenshot_to_file(path)

    def _screenshot_to_file(self, path):
        if False:
            for i in range(10):
                print('nop')
        path = self._validate_screenshot_path(path)
        logger.debug('Using %s module/tool for taking screenshot.' % self._screenshot_taker.module)
        try:
            self._screenshot_taker(path)
        except:
            logger.warn('Taking screenshot failed: %s\nMake sure tests are run with a physical or virtual display.' % get_error_message())
        return path

    def _validate_screenshot_path(self, path):
        if False:
            i = 10
            return i + 15
        path = abspath(self._norm_path(path))
        if not os.path.exists(os.path.dirname(path)):
            raise RuntimeError("Directory '%s' where to save the screenshot does not exist" % os.path.dirname(path))
        return path

    def _get_screenshot_path(self, basename):
        if False:
            i = 10
            return i + 15
        if basename.lower().endswith(('.jpg', '.jpeg')):
            return os.path.join(self._screenshot_dir, basename)
        index = 0
        while True:
            index += 1
            path = os.path.join(self._screenshot_dir, '%s_%d.jpg' % (basename, index))
            if not os.path.exists(path):
                return path

    def _embed_screenshot(self, path, width):
        if False:
            return 10
        link = get_link_path(path, self._log_dir)
        logger.info('<a href="%s"><img src="%s" width="%s"></a>' % (link, link, width), html=True)

    def _link_screenshot(self, path):
        if False:
            for i in range(10):
                print('nop')
        link = get_link_path(path, self._log_dir)
        logger.info('Screenshot saved to \'<a href="%s">%s</a>\'.' % (link, path), html=True)

class ScreenshotTaker:

    def __init__(self, module_name=None):
        if False:
            while True:
                i = 10
        self._screenshot = self._get_screenshot_taker(module_name)
        self.module = self._screenshot.__name__.split('_')[1]
        self._wx_app_reference = None

    def __call__(self, path):
        if False:
            while True:
                i = 10
        self._screenshot(path)

    def __bool__(self):
        if False:
            print('Hello World!')
        return self.module != 'no'

    def test(self, path=None):
        if False:
            return 10
        if not self:
            print('Cannot take screenshots.')
            return False
        print("Using '%s' to take screenshot." % self.module)
        if not path:
            print('Not taking test screenshot.')
            return True
        print("Taking test screenshot to '%s'." % path)
        try:
            self(path)
        except:
            print('Failed: %s' % get_error_message())
            return False
        else:
            print('Success!')
            return True

    def _get_screenshot_taker(self, module_name=None):
        if False:
            print('Hello World!')
        if sys.platform == 'darwin':
            return self._osx_screenshot
        if module_name:
            return self._get_named_screenshot_taker(module_name.lower())
        return self._get_default_screenshot_taker()

    def _get_named_screenshot_taker(self, name):
        if False:
            while True:
                i = 10
        screenshot_takers = {'wxpython': (wx, self._wx_screenshot), 'pygtk': (gdk, self._gtk_screenshot), 'pil': (ImageGrab, self._pil_screenshot), 'scrot': (self._scrot, self._scrot_screenshot)}
        if name not in screenshot_takers:
            raise RuntimeError("Invalid screenshot module or tool '%s'." % name)
        (supported, screenshot_taker) = screenshot_takers[name]
        if not supported:
            raise RuntimeError("Screenshot module or tool '%s' not installed." % name)
        return screenshot_taker

    def _get_default_screenshot_taker(self):
        if False:
            i = 10
            return i + 15
        for (module, screenshot_taker) in [(wx, self._wx_screenshot), (gdk, self._gtk_screenshot), (ImageGrab, self._pil_screenshot), (self._scrot, self._scrot_screenshot), (True, self._no_screenshot)]:
            if module:
                return screenshot_taker

    def _osx_screenshot(self, path):
        if False:
            print('Hello World!')
        if self._call('screencapture', '-t', 'jpg', path) != 0:
            raise RuntimeError("Using 'screencapture' failed.")

    def _call(self, *command):
        if False:
            print('Hello World!')
        try:
            return subprocess.call(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError:
            return -1

    @property
    def _scrot(self):
        if False:
            return 10
        return os.sep == '/' and self._call('scrot', '--version') == 0

    def _scrot_screenshot(self, path):
        if False:
            i = 10
            return i + 15
        if not path.endswith(('.jpg', '.jpeg')):
            raise RuntimeError("Scrot requires extension to be '.jpg' or '.jpeg', got '%s'." % os.path.splitext(path)[1])
        if os.path.exists(path):
            os.remove(path)
        if self._call('scrot', '--silent', path) != 0:
            raise RuntimeError("Using 'scrot' failed.")

    def _wx_screenshot(self, path):
        if False:
            while True:
                i = 10
        if not self._wx_app_reference:
            self._wx_app_reference = wx.App(False)
        context = wx.ScreenDC()
        (width, height) = context.GetSize()
        if wx.__version__ >= '4':
            bitmap = wx.Bitmap(width, height, -1)
        else:
            bitmap = wx.EmptyBitmap(width, height, -1)
        memory = wx.MemoryDC()
        memory.SelectObject(bitmap)
        memory.Blit(0, 0, width, height, context, -1, -1)
        memory.SelectObject(wx.NullBitmap)
        bitmap.SaveFile(path, wx.BITMAP_TYPE_JPEG)

    def _gtk_screenshot(self, path):
        if False:
            for i in range(10):
                print('nop')
        window = gdk.get_default_root_window()
        if not window:
            raise RuntimeError('Taking screenshot failed.')
        (width, height) = window.get_size()
        pb = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, width, height)
        pb = pb.get_from_drawable(window, window.get_colormap(), 0, 0, 0, 0, width, height)
        if not pb:
            raise RuntimeError('Taking screenshot failed.')
        pb.save(path, 'jpeg')

    def _pil_screenshot(self, path):
        if False:
            return 10
        ImageGrab.grab().save(path, 'JPEG')

    def _no_screenshot(self, path):
        if False:
            i = 10
            return i + 15
        raise RuntimeError('Taking screenshots is not supported on this platform by default. See library documentation for details.')
if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        sys.exit('Usage: %s <path>|test [wxpython|pygtk|pil|scrot]' % os.path.basename(sys.argv[0]))
    path = sys.argv[1] if sys.argv[1] != 'test' else None
    module = sys.argv[2] if len(sys.argv) > 2 else None
    ScreenshotTaker(module).test(path)