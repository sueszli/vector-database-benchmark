import os
import shutil
import subprocess
import sys
import pytest
from PIL import Image, ImageGrab
from .helper import assert_image_equal_tofile, skip_unless_feature

class TestImageGrab:

    @pytest.mark.skipif(sys.platform not in ('win32', 'darwin'), reason='requires Windows or macOS')
    def test_grab(self):
        if False:
            return 10
        ImageGrab.grab()
        ImageGrab.grab(include_layered_windows=True)
        ImageGrab.grab(all_screens=True)
        im = ImageGrab.grab(bbox=(10, 20, 50, 80))
        assert im.size == (40, 60)

    @skip_unless_feature('xcb')
    def test_grab_x11(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            if sys.platform not in ('win32', 'darwin'):
                ImageGrab.grab()
            ImageGrab.grab(xdisplay='')
        except OSError as e:
            pytest.skip(str(e))

    @pytest.mark.skipif(Image.core.HAVE_XCB, reason='tests missing XCB')
    def test_grab_no_xcb(self):
        if False:
            i = 10
            return i + 15
        if sys.platform not in ('win32', 'darwin') and (not shutil.which('gnome-screenshot')):
            with pytest.raises(OSError) as e:
                ImageGrab.grab()
            assert str(e.value).startswith('Pillow was built without XCB support')
        with pytest.raises(OSError) as e:
            ImageGrab.grab(xdisplay='')
        assert str(e.value).startswith('Pillow was built without XCB support')

    @skip_unless_feature('xcb')
    def test_grab_invalid_xdisplay(self):
        if False:
            while True:
                i = 10
        with pytest.raises(OSError) as e:
            ImageGrab.grab(xdisplay='error.test:0.0')
        assert str(e.value).startswith('X connection failed')

    def test_grabclipboard(self):
        if False:
            for i in range(10):
                print('nop')
        if sys.platform == 'darwin':
            subprocess.call(['screencapture', '-cx'])
        elif sys.platform == 'win32':
            p = subprocess.Popen(['powershell', '-command', '-'], stdin=subprocess.PIPE)
            p.stdin.write(b'[Reflection.Assembly]::LoadWithPartialName("System.Drawing")\n[Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms")\n$bmp = New-Object Drawing.Bitmap 200, 200\n[Windows.Forms.Clipboard]::SetImage($bmp)')
            p.communicate()
        else:
            if not shutil.which('wl-paste') and (not shutil.which('xclip')):
                with pytest.raises(NotImplementedError, match='wl-paste or xclip is required for ImageGrab.grabclipboard\\(\\) on Linux'):
                    ImageGrab.grabclipboard()
            return
        ImageGrab.grabclipboard()

    @pytest.mark.skipif(sys.platform != 'win32', reason='Windows only')
    def test_grabclipboard_file(self):
        if False:
            print('Hello World!')
        p = subprocess.Popen(['powershell', '-command', '-'], stdin=subprocess.PIPE)
        p.stdin.write(b'Set-Clipboard -Path "Tests\\images\\hopper.gif"')
        p.communicate()
        im = ImageGrab.grabclipboard()
        assert len(im) == 1
        assert os.path.samefile(im[0], 'Tests/images/hopper.gif')

    @pytest.mark.skipif(sys.platform != 'win32', reason='Windows only')
    def test_grabclipboard_png(self):
        if False:
            while True:
                i = 10
        p = subprocess.Popen(['powershell', '-command', '-'], stdin=subprocess.PIPE)
        p.stdin.write(b'$bytes = [System.IO.File]::ReadAllBytes("Tests\\images\\hopper.png")\n$ms = new-object System.IO.MemoryStream(, $bytes)\n[Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms")\n[Windows.Forms.Clipboard]::SetData("PNG", $ms)')
        p.communicate()
        im = ImageGrab.grabclipboard()
        assert_image_equal_tofile(im, 'Tests/images/hopper.png')

    @pytest.mark.skipif(sys.platform != 'linux' or not all((shutil.which(cmd) for cmd in ('wl-paste', 'wl-copy'))), reason='Linux with wl-clipboard only')
    @pytest.mark.parametrize('ext', ('gif', 'png', 'ico'))
    def test_grabclipboard_wl_clipboard(self, ext):
        if False:
            i = 10
            return i + 15
        image_path = 'Tests/images/hopper.' + ext
        with open(image_path, 'rb') as fp:
            subprocess.call(['wl-copy'], stdin=fp)
        im = ImageGrab.grabclipboard()
        assert_image_equal_tofile(im, image_path)