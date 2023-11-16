"""SCons.Tool.Perforce.py

Tool-specific initialization for Perforce Source Code Management system.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/Perforce.py  2014/07/05 09:42:21 garyo'
import os
import SCons.Action
import SCons.Builder
import SCons.Node.FS
import SCons.Util
from SCons.Tool.PharLapCommon import addPathIfNotExists
_import_env = ['P4PORT', 'P4CLIENT', 'P4USER', 'USER', 'USERNAME', 'P4PASSWD', 'P4CHARSET', 'P4LANGUAGE', 'SystemRoot']
PerforceAction = SCons.Action.Action('$P4COM', '$P4COMSTR')

def generate(env):
    if False:
        while True:
            i = 10
    'Add a Builder factory function and construction variables for\n    Perforce to an Environment.'

    def PerforceFactory(env=env):
        if False:
            i = 10
            return i + 15
        ' '
        import SCons.Warnings as W
        W.warn(W.DeprecatedSourceCodeWarning, 'The Perforce() factory is deprecated and there is no replacement.')
        return SCons.Builder.Builder(action=PerforceAction, env=env)
    env.Perforce = PerforceFactory
    env['P4'] = 'p4'
    env['P4FLAGS'] = SCons.Util.CLVar('')
    env['P4COM'] = '$P4 $P4FLAGS sync $TARGET'
    try:
        environ = env['ENV']
    except KeyError:
        environ = {}
        env['ENV'] = environ
    environ['PWD'] = env.Dir('#').get_abspath()
    for var in _import_env:
        v = os.environ.get(var)
        if v:
            environ[var] = v
    if SCons.Util.can_read_reg:
        try:
            k = SCons.Util.RegOpenKeyEx(SCons.Util.hkey_mod.HKEY_LOCAL_MACHINE, 'Software\\Perforce\\environment')
            (val, tok) = SCons.Util.RegQueryValueEx(k, 'P4INSTROOT')
            addPathIfNotExists(environ, 'PATH', val)
        except SCons.Util.RegError:
            pass

def exists(env):
    if False:
        i = 10
        return i + 15
    return env.Detect('p4')