"""
Explore encoding settings on a platform.
"""
from __future__ import print_function
import sys
import platform
import locale
from behave.textutil import select_best_encoding

def explore_platform_encoding():
    if False:
        i = 10
        return i + 15
    python_version = platform.python_version()
    print('python %s (platform: %s, %s, %s)' % (python_version, sys.platform, platform.python_implementation(), platform.platform()))
    print('sys.getfilesystemencoding():   %s' % sys.getfilesystemencoding())
    print('locale.getpreferredencoding(): %s' % locale.getpreferredencoding())
    print('behave.textutil.select_best_encoding(): %s' % select_best_encoding())
    return 0
if __name__ == '__main__':
    sys.exit(explore_platform_encoding())