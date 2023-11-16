"""SCons.Tool.sgic++

Tool-specific initialization for MIPSpro C++ on SGI.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/sgicxx.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Util
import SCons.Tool.cxx
cplusplus = SCons.Tool.cxx

def generate(env):
    if False:
        i = 10
        return i + 15
    'Add Builders and construction variables for SGI MIPS C++ to an Environment.'
    cplusplus.generate(env)
    env['CXX'] = 'CC'
    env['CXXFLAGS'] = SCons.Util.CLVar('-LANG:std')
    env['SHCXX'] = '$CXX'
    env['SHOBJSUFFIX'] = '.o'
    env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1

def exists(env):
    if False:
        print('Hello World!')
    return env.Detect('CC')