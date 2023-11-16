"""SCons.Platform.cygwin

Platform-specific initialization for Cygwin systems.

There normally shouldn't be any need to import this module directly.  It
will usually be imported through the generic SCons.Platform.Platform()
selection method.
"""
__revision__ = 'src/engine/SCons/Platform/cygwin.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import sys
from . import posix
from SCons.Platform import TempFileMunge
CYGWIN_DEFAULT_PATHS = []
if sys.platform == 'win32':
    CYGWIN_DEFAULT_PATHS = ['C:\\cygwin64\\bin', 'C:\\cygwin\\bin']

def generate(env):
    if False:
        i = 10
        return i + 15
    posix.generate(env)
    env['PROGPREFIX'] = ''
    env['PROGSUFFIX'] = '.exe'
    env['SHLIBPREFIX'] = ''
    env['SHLIBSUFFIX'] = '.dll'
    env['LIBPREFIXES'] = ['$LIBPREFIX', '$SHLIBPREFIX', '$IMPLIBPREFIX']
    env['LIBSUFFIXES'] = ['$LIBSUFFIX', '$SHLIBSUFFIX', '$IMPLIBSUFFIX']
    env['TEMPFILE'] = TempFileMunge
    env['TEMPFILEPREFIX'] = '@'
    env['MAXLINELENGTH'] = 2048