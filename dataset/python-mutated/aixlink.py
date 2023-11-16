"""SCons.Tool.aixlink

Tool-specific initialization for the IBM Visual Age linker.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
__revision__ = 'src/engine/SCons/Tool/aixlink.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os
import os.path
import SCons.Util
from . import aixcc
from . import link
import SCons.Tool.cxx
cplusplus = SCons.Tool.cxx

def smart_linkflags(source, target, env, for_signature):
    if False:
        return 10
    if cplusplus.iscplusplus(source):
        build_dir = env.subst('$BUILDDIR', target=target, source=source)
        if build_dir:
            return '-qtempinc=' + os.path.join(build_dir, 'tempinc')
    return ''

def generate(env):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add Builders and construction variables for Visual Age linker to\n    an Environment.\n    '
    link.generate(env)
    env['SMARTLINKFLAGS'] = smart_linkflags
    env['LINKFLAGS'] = SCons.Util.CLVar('$SMARTLINKFLAGS')
    env['SHLINKFLAGS'] = SCons.Util.CLVar('$LINKFLAGS -qmkshrobj -qsuppress=1501-218')
    env['SHLIBSUFFIX'] = '.a'

def exists(env):
    if False:
        while True:
            i = 10
    linkers = {'CXX': ['aixc++'], 'CC': ['aixcc']}
    alltools = []
    for (langvar, linktools) in linkers.items():
        if langvar in env:
            return SCons.Tool.FindTool(linktools, env)
        alltools.extend(linktools)
    return SCons.Tool.FindTool(alltools, env)