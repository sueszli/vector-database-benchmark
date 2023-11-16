"""SCons.Tool.hpc++

Tool-specific initialization for c++ on HP/UX.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/hpcxx.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os.path
import SCons.Util
import SCons.Tool.cxx
cplusplus = SCons.Tool.cxx
acc = None
try:
    dirs = os.listdir('/opt')
except (IOError, OSError):
    dirs = []
for dir in dirs:
    cc = '/opt/' + dir + '/bin/aCC'
    if os.path.exists(cc):
        acc = cc
        break

def generate(env):
    if False:
        while True:
            i = 10
    'Add Builders and construction variables for g++ to an Environment.'
    cplusplus.generate(env)
    if acc:
        env['CXX'] = acc or 'aCC'
        env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS +Z')
        with os.popen(acc + ' -V 2>&1') as p:
            line = p.readline().rstrip()
        if line.find('aCC: HP ANSI C++') == 0:
            env['CXXVERSION'] = line.split()[-1]
        if env['PLATFORM'] == 'cygwin':
            env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS')
        else:
            env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS +Z')

def exists(env):
    if False:
        while True:
            i = 10
    return acc