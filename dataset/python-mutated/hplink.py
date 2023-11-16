"""SCons.Tool.hplink

Tool-specific initialization for the HP linker.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
__revision__ = 'src/engine/SCons/Tool/hplink.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os
import os.path
import SCons.Util
from . import link
ccLinker = None
try:
    dirs = os.listdir('/opt')
except (IOError, OSError):
    dirs = []
for dir in dirs:
    linker = '/opt/' + dir + '/bin/aCC'
    if os.path.exists(linker):
        ccLinker = linker
        break

def generate(env):
    if False:
        i = 10
        return i + 15
    '\n    Add Builders and construction variables for Visual Age linker to\n    an Environment.\n    '
    link.generate(env)
    env['LINKFLAGS'] = SCons.Util.CLVar('-Wl,+s -Wl,+vnocompatwarnings')
    env['SHLINKFLAGS'] = SCons.Util.CLVar('$LINKFLAGS -b')
    env['SHLIBSUFFIX'] = '.sl'

def exists(env):
    if False:
        i = 10
        return i + 15
    return ccLinker