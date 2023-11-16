"""SCons.Tool.aixc++

Tool-specific initialization for IBM xlC / Visual Age C++ compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/aixcxx.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os.path
import SCons.Platform.aix
import SCons.Tool.cxx
cplusplus = SCons.Tool.cxx
packages = ['vacpp.cmp.core', 'vacpp.cmp.batch', 'vacpp.cmp.C', 'ibmcxx.cmp']

def get_xlc(env):
    if False:
        return 10
    xlc = env.get('CXX', 'xlC')
    return SCons.Platform.aix.get_xlc(env, xlc, packages)

def generate(env):
    if False:
        while True:
            i = 10
    'Add Builders and construction variables for xlC / Visual Age\n    suite to an Environment.'
    (path, _cxx, version) = get_xlc(env)
    if path and _cxx:
        _cxx = os.path.join(path, _cxx)
    if 'CXX' not in env:
        env['CXX'] = _cxx
    cplusplus.generate(env)
    if version:
        env['CXXVERSION'] = version

def exists(env):
    if False:
        return 10
    (path, _cxx, version) = get_xlc(env)
    if path and _cxx:
        xlc = os.path.join(path, _cxx)
        if os.path.exists(xlc):
            return xlc
    return None