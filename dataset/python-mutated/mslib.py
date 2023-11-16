"""SCons.Tool.mslib

Tool-specific initialization for lib (MicroSoft library archiver).

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/mslib.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os
import SCons.Defaults
import SCons.Tool
import SCons.Tool.msvs
import SCons.Tool.msvc
import SCons.Util
from .MSCommon import msvc_exists, msvc_setup_env_once

def generate(env):
    if False:
        return 10
    'Add Builders and construction variables for lib to an Environment.'
    SCons.Tool.createStaticLibBuilder(env)
    msvc_setup_env_once(env)
    env['AR'] = 'lib'
    env['ARFLAGS'] = SCons.Util.CLVar('/nologo')
    env['ARCOM'] = "${TEMPFILE('$AR $ARFLAGS /OUT:$TARGET $SOURCES','$ARCOMSTR')}"
    env['LIBPREFIX'] = ''
    env['LIBSUFFIX'] = '.lib'
    env['TEMPFILEARGJOIN'] = os.linesep

def exists(env):
    if False:
        while True:
            i = 10
    return msvc_exists(env)