"""SCons.Tool.hpcc

Tool-specific initialization for HP aCC and cc.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
__revision__ = 'src/engine/SCons/Tool/hpcc.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Util
from . import cc

def generate(env):
    if False:
        return 10
    'Add Builders and construction variables for aCC & cc to an Environment.'
    cc.generate(env)
    env['CXX'] = 'aCC'
    env['SHCCFLAGS'] = SCons.Util.CLVar('$CCFLAGS +Z')

def exists(env):
    if False:
        i = 10
        return i + 15
    return env.Detect('aCC')