"""SCons.Tool.linkloc

Tool specification for the LinkLoc linker for the Phar Lap ETS embedded
operating system.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/linkloc.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os.path
import re
import SCons.Action
import SCons.Defaults
import SCons.Errors
import SCons.Tool
import SCons.Util
from SCons.Tool.MSCommon import msvs_exists, merge_default_version
from SCons.Tool.PharLapCommon import addPharLapPaths
_re_linker_command = re.compile('(\\s)@\\s*([^\\s]+)')

def repl_linker_command(m):
    if False:
        i = 10
        return i + 15
    try:
        with open(m.group(2), 'r') as f:
            return m.group(1) + f.read()
    except IOError:
        return m.group(1) + '#' + m.group(2)

class LinklocGenerator(object):

    def __init__(self, cmdline):
        if False:
            print('Hello World!')
        self.cmdline = cmdline

    def __call__(self, env, target, source, for_signature):
        if False:
            while True:
                i = 10
        if for_signature:
            subs = 1
            strsub = env.subst(self.cmdline, target=target, source=source)
            while subs:
                (strsub, subs) = _re_linker_command.subn(repl_linker_command, strsub)
            return strsub
        else:
            return "${TEMPFILE('" + self.cmdline + "')}"

def generate(env):
    if False:
        while True:
            i = 10
    'Add Builders and construction variables for ar to an Environment.'
    SCons.Tool.createSharedLibBuilder(env)
    SCons.Tool.createProgBuilder(env)
    env['SUBST_CMD_FILE'] = LinklocGenerator
    env['SHLINK'] = '$LINK'
    env['SHLINKFLAGS'] = SCons.Util.CLVar('$LINKFLAGS')
    env['SHLINKCOM'] = '${SUBST_CMD_FILE("$SHLINK $SHLINKFLAGS $_LIBDIRFLAGS $_LIBFLAGS -dll $TARGET $SOURCES")}'
    env['SHLIBEMITTER'] = None
    env['LDMODULEEMITTER'] = None
    env['LINK'] = 'linkloc'
    env['LINKFLAGS'] = SCons.Util.CLVar('')
    env['LINKCOM'] = '${SUBST_CMD_FILE("$LINK $LINKFLAGS $_LIBDIRFLAGS $_LIBFLAGS -exe $TARGET $SOURCES")}'
    env['LIBDIRPREFIX'] = '-libpath '
    env['LIBDIRSUFFIX'] = ''
    env['LIBLINKPREFIX'] = '-lib '
    env['LIBLINKSUFFIX'] = '$LIBSUFFIX'
    merge_default_version(env)
    addPharLapPaths(env)

def exists(env):
    if False:
        i = 10
        return i + 15
    if msvs_exists():
        return env.Detect('linkloc')
    else:
        return 0