"""SCons.Platform.os2

Platform-specific initialization for OS/2 systems.

There normally shouldn't be any need to import this module directly.  It
will usually be imported through the generic SCons.Platform.Platform()
selection method.
"""
__revision__ = 'src/engine/SCons/Platform/os2.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
from . import win32

def generate(env):
    if False:
        return 10
    if 'ENV' not in env:
        env['ENV'] = {}
    env['OBJPREFIX'] = ''
    env['OBJSUFFIX'] = '.obj'
    env['SHOBJPREFIX'] = '$OBJPREFIX'
    env['SHOBJSUFFIX'] = '$OBJSUFFIX'
    env['PROGPREFIX'] = ''
    env['PROGSUFFIX'] = '.exe'
    env['LIBPREFIX'] = ''
    env['LIBSUFFIX'] = '.lib'
    env['SHLIBPREFIX'] = ''
    env['SHLIBSUFFIX'] = '.dll'
    env['LIBPREFIXES'] = '$LIBPREFIX'
    env['LIBSUFFIXES'] = ['$LIBSUFFIX', '$SHLIBSUFFIX']
    env['HOST_OS'] = 'os2'
    env['HOST_ARCH'] = win32.get_architecture().arch