"""SCons.Tool.sgicc

Tool-specific initialization for MIPSPro cc on SGI.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/sgicc.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
from . import cc

def generate(env):
    if False:
        print('Hello World!')
    'Add Builders and construction variables for gcc to an Environment.'
    cc.generate(env)
    env['CXX'] = 'CC'
    env['SHOBJSUFFIX'] = '.o'
    env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1

def exists(env):
    if False:
        while True:
            i = 10
    return env.Detect('cc')