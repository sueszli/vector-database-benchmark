"""
Clipboard xclip: an implementation of the Clipboard using xclip
command line tool.
"""
__all__ = ('ClipboardXclip',)
from kivy.utils import platform
from kivy.core.clipboard._clipboard_ext import ClipboardExternalBase
if platform != 'linux':
    raise SystemError('unsupported platform for xclip clipboard')
try:
    import subprocess
    p = subprocess.Popen(['xclip', '-version'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p.communicate()
except:
    raise

class ClipboardXclip(ClipboardExternalBase):

    @staticmethod
    def _clip(inout, selection):
        if False:
            for i in range(10):
                print('nop')
        pipe = {'std' + inout: subprocess.PIPE}
        return subprocess.Popen(['xclip', '-' + inout, '-selection', selection], **pipe)