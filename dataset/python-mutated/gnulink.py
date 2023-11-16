"""SCons.Tool.gnulink

Tool-specific initialization for the gnu linker.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/gnulink.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Util
import SCons.Tool
import os
import sys
import re
from . import link

def generate(env):
    if False:
        for i in range(10):
            print('nop')
    'Add Builders and construction variables for gnulink to an Environment.'
    link.generate(env)
    if env['PLATFORM'] == 'hpux':
        env['SHLINKFLAGS'] = SCons.Util.CLVar('$LINKFLAGS -shared -fPIC')
    env['RPATHPREFIX'] = '-Wl,-rpath='
    env['RPATHSUFFIX'] = ''
    env['_RPATH'] = '${_concat(RPATHPREFIX, RPATH, RPATHSUFFIX, __env__)}'
    use_soname = not sys.platform.startswith('openbsd')
    link._setup_versioned_lib_variables(env, tool='gnulink', use_soname=use_soname)
    env['LINKCALLBACKS'] = link._versioned_lib_callbacks()
    env['SHLIBVERSIONFLAGS'] = SCons.Util.CLVar('-Wl,-Bsymbolic')

def exists(env):
    if False:
        print('Hello World!')
    linkers = {'CXX': ['g++'], 'CC': ['gcc']}
    alltools = []
    for (langvar, linktools) in linkers.items():
        if langvar in env:
            return SCons.Tool.FindTool(linktools, env)
        alltools.extend(linktools)
    return SCons.Tool.FindTool(alltools, env)