"""SCons.Tool.sunlink

Tool-specific initialization for the Sun Solaris (Forte) linker.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
__revision__ = 'src/engine/SCons/Tool/sunlink.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os
import os.path
import SCons.Util
from . import link
ccLinker = None
try:
    dirs = os.listdir('/opt')
except (IOError, OSError):
    dirs = []
for d in dirs:
    linker = '/opt/' + d + '/bin/CC'
    if os.path.exists(linker):
        ccLinker = linker
        break

def generate(env):
    if False:
        while True:
            i = 10
    'Add Builders and construction variables for Forte to an Environment.'
    link.generate(env)
    env['SHLINKFLAGS'] = SCons.Util.CLVar('$LINKFLAGS -G')
    env['RPATHPREFIX'] = '-R'
    env['RPATHSUFFIX'] = ''
    env['_RPATH'] = '${_concat(RPATHPREFIX, RPATH, RPATHSUFFIX, __env__)}'
    link._setup_versioned_lib_variables(env, tool='sunlink', use_soname=True)
    env['LINKCALLBACKS'] = link._versioned_lib_callbacks()

def exists(env):
    if False:
        i = 10
        return i + 15
    return ccLinker