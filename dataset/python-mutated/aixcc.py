"""SCons.Tool.aixcc

Tool-specific initialization for IBM xlc / Visual Age C compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
__revision__ = 'src/engine/SCons/Tool/aixcc.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os.path
import SCons.Platform.aix
from . import cc
packages = ['vac.C', 'ibmcxx.cmp']

def get_xlc(env):
    if False:
        print('Hello World!')
    xlc = env.get('CC', 'xlc')
    return SCons.Platform.aix.get_xlc(env, xlc, packages)

def generate(env):
    if False:
        i = 10
        return i + 15
    'Add Builders and construction variables for xlc / Visual Age\n    suite to an Environment.'
    (path, _cc, version) = get_xlc(env)
    if path and _cc:
        _cc = os.path.join(path, _cc)
    if 'CC' not in env:
        env['CC'] = _cc
    cc.generate(env)
    if version:
        env['CCVERSION'] = version

def exists(env):
    if False:
        while True:
            i = 10
    (path, _cc, version) = get_xlc(env)
    if path and _cc:
        xlc = os.path.join(path, _cc)
        if os.path.exists(xlc):
            return xlc
    return None