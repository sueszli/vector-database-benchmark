"""Tool-specific initialization for the Intel Fortran compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
import SCons.Defaults
from SCons.Scanner.Fortran import FortranScan
from .FortranCommon import add_all_to_env

def generate(env):
    if False:
        return 10
    'Add Builders and construction variables for ifl to an Environment.'
    fscan = FortranScan('FORTRANPATH')
    SCons.Tool.SourceFileScanner.add_scanner('.i', fscan)
    SCons.Tool.SourceFileScanner.add_scanner('.i90', fscan)
    if 'FORTRANFILESUFFIXES' not in env:
        env['FORTRANFILESUFFIXES'] = ['.i']
    else:
        env['FORTRANFILESUFFIXES'].append('.i')
    if 'F90FILESUFFIXES' not in env:
        env['F90FILESUFFIXES'] = ['.i90']
    else:
        env['F90FILESUFFIXES'].append('.i90')
    add_all_to_env(env)
    env['FORTRAN'] = 'ifl'
    env['SHFORTRAN'] = '$FORTRAN'
    env['FORTRANCOM'] = '$FORTRAN $FORTRANFLAGS $_FORTRANINCFLAGS /c $SOURCES /Fo$TARGET'
    env['FORTRANPPCOM'] = '$FORTRAN $FORTRANFLAGS $CPPFLAGS $_CPPDEFFLAGS $_FORTRANINCFLAGS /c $SOURCES /Fo$TARGET'
    env['SHFORTRANCOM'] = '$SHFORTRAN $SHFORTRANFLAGS $_FORTRANINCFLAGS /c $SOURCES /Fo$TARGET'
    env['SHFORTRANPPCOM'] = '$SHFORTRAN $SHFORTRANFLAGS $CPPFLAGS $_CPPDEFFLAGS $_FORTRANINCFLAGS /c $SOURCES /Fo$TARGET'

def exists(env):
    if False:
        i = 10
        return i + 15
    return env.Detect('ifl')