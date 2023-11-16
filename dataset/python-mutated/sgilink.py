"""SCons.Tool.sgilink

Tool-specific initialization for the SGI MIPSPro linker on SGI.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/sgilink.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Util
from . import link
linkers = ['CC', 'cc']

def generate(env):
    if False:
        return 10
    'Add Builders and construction variables for MIPSPro to an Environment.'
    link.generate(env)
    env['LINK'] = env.Detect(linkers) or 'cc'
    env['SHLINKFLAGS'] = SCons.Util.CLVar('$LINKFLAGS -shared')
    env['RPATHPREFIX'] = '-rpath '
    env['RPATHSUFFIX'] = ''
    env['_RPATH'] = '${_concat(RPATHPREFIX, RPATH, RPATHSUFFIX, __env__)}'

def exists(env):
    if False:
        for i in range(10):
            print('nop')
    return env.Detect(linkers)