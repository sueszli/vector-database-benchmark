""" xgettext tool

Tool specific initialization of `xgettext` tool.
"""
__revision__ = 'src/engine/SCons/Tool/xgettext.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os
import re
import subprocess
import sys
import SCons.Action
import SCons.Node.FS
import SCons.Tool
import SCons.Util
from SCons.Builder import BuilderBase
from SCons.Environment import _null
from SCons.Platform.cygwin import CYGWIN_DEFAULT_PATHS
from SCons.Platform.mingw import MINGW_DEFAULT_PATHS
from SCons.Tool.GettextCommon import _POTargetFactory
from SCons.Tool.GettextCommon import RPaths, _detect_xgettext
from SCons.Tool.GettextCommon import _xgettext_exists

class _CmdRunner(object):
    """ Callable object, which runs shell command storing its stdout and stderr to
    variables. It also provides `strfunction()` method, which shall be used by
    scons Action objects to print command string. """

    def __init__(self, command, commandstr=None):
        if False:
            while True:
                i = 10
        self.out = None
        self.err = None
        self.status = None
        self.command = command
        self.commandstr = commandstr

    def __call__(self, target, source, env):
        if False:
            while True:
                i = 10
        kw = {'stdin': 'devnull', 'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE, 'universal_newlines': True, 'shell': True}
        command = env.subst(self.command, target=target, source=source)
        proc = SCons.Action._subproc(env, command, **kw)
        (self.out, self.err) = proc.communicate()
        self.status = proc.wait()
        if self.err:
            sys.stderr.write(SCons.Util.UnicodeType(self.err))
        return self.status

    def strfunction(self, target, source, env):
        if False:
            i = 10
            return i + 15
        comstr = self.commandstr
        if env.subst(comstr, target=target, source=source) == '':
            comstr = self.command
        s = env.subst(comstr, target=target, source=source)
        return s

def _update_pot_file(target, source, env):
    if False:
        print('Hello World!')
    ' Action function for `POTUpdate` builder '
    nop = lambda target, source, env: 0
    save_cwd = env.fs.getcwd()
    save_os_cwd = os.getcwd()
    chdir = target[0].dir
    chdir_str = repr(chdir.get_abspath())
    env.Execute(SCons.Action.Action(nop, 'Entering ' + chdir_str))
    env.fs.chdir(chdir, 1)
    try:
        cmd = _CmdRunner('$XGETTEXTCOM', '$XGETTEXTCOMSTR')
        action = SCons.Action.Action(cmd, strfunction=cmd.strfunction)
        status = action([target[0]], source, env)
    except:
        env.Execute(SCons.Action.Action(nop, 'Leaving ' + chdir_str))
        env.fs.chdir(save_cwd, 0)
        os.chdir(save_os_cwd)
        raise
    env.Execute(SCons.Action.Action(nop, 'Leaving ' + chdir_str))
    env.fs.chdir(save_cwd, 0)
    os.chdir(save_os_cwd)
    if status:
        return status
    new_content = cmd.out
    if not new_content:
        needs_update = False
        explain = 'no internationalized messages encountered'
    elif target[0].exists():
        old_content = target[0].get_text_contents()
        re_cdate = re.compile('^"POT-Creation-Date: .*"$[\\r\\n]?', re.M)
        old_content_nocdate = re.sub(re_cdate, '', old_content)
        new_content_nocdate = re.sub(re_cdate, '', new_content)
        if old_content_nocdate == new_content_nocdate:
            needs_update = False
            explain = 'messages in file found to be up-to-date'
        else:
            needs_update = True
            explain = 'messages in file were outdated'
    else:
        needs_update = True
        explain = 'new file'
    if needs_update:
        msg = 'Writing ' + repr(str(target[0])) + ' (' + explain + ')'
        env.Execute(SCons.Action.Action(nop, msg))
        f = open(str(target[0]), 'w')
        f.write(new_content)
        f.close()
        return 0
    else:
        msg = 'Not writing ' + repr(str(target[0])) + ' (' + explain + ')'
        env.Execute(SCons.Action.Action(nop, msg))
        return 0

class _POTBuilder(BuilderBase):

    def _execute(self, env, target, source, *args):
        if False:
            print('Hello World!')
        if not target:
            if 'POTDOMAIN' in env and env['POTDOMAIN']:
                domain = env['POTDOMAIN']
            else:
                domain = 'messages'
            target = [domain]
        return BuilderBase._execute(self, env, target, source, *args)

def _scan_xgettext_from_files(target, source, env, files=None, path=None):
    if False:
        return 10
    ' Parses `POTFILES.in`-like file and returns list of extracted file names.\n    '
    if files is None:
        return 0
    if not SCons.Util.is_List(files):
        files = [files]
    if path is None:
        if 'XGETTEXTPATH' in env:
            path = env['XGETTEXTPATH']
        else:
            path = []
    if not SCons.Util.is_List(path):
        path = [path]
    path = SCons.Util.flatten(path)
    dirs = ()
    for p in path:
        if not isinstance(p, SCons.Node.FS.Base):
            if SCons.Util.is_String(p):
                p = env.subst(p, source=source, target=target)
            p = env.arg2nodes(p, env.fs.Dir)
        dirs += tuple(p)
    if not dirs:
        dirs = (env.fs.getcwd(),)
    re_comment = re.compile('^#[^\\n\\r]*$\\r?\\n?', re.M)
    re_emptyln = re.compile('^[ \\t\\r]*$\\r?\\n?', re.M)
    re_trailws = re.compile('[ \\t\\r]+$')
    for f in files:
        if isinstance(f, SCons.Node.FS.Base) and f.rexists():
            contents = f.get_text_contents()
            contents = re_comment.sub('', contents)
            contents = re_emptyln.sub('', contents)
            contents = re_trailws.sub('', contents)
            depnames = contents.splitlines()
            for depname in depnames:
                depfile = SCons.Node.FS.find_file(depname, dirs)
                if not depfile:
                    depfile = env.arg2nodes(depname, dirs[0].File)
                env.Depends(target, depfile)
    return 0

def _pot_update_emitter(target, source, env):
    if False:
        while True:
            i = 10
    ' Emitter function for `POTUpdate` builder '
    if 'XGETTEXTFROM' in env:
        xfrom = env['XGETTEXTFROM']
    else:
        return (target, source)
    if not SCons.Util.is_List(xfrom):
        xfrom = [xfrom]
    xfrom = SCons.Util.flatten(xfrom)
    files = []
    for xf in xfrom:
        if not isinstance(xf, SCons.Node.FS.Base):
            if SCons.Util.is_String(xf):
                xf = env.subst(xf, source=source, target=target)
            xf = env.arg2nodes(xf)
        files.extend(xf)
    if files:
        env.Depends(target, files)
        _scan_xgettext_from_files(target, source, env, files)
    return (target, source)

def _POTUpdateBuilderWrapper(env, target=None, source=_null, **kw):
    if False:
        for i in range(10):
            print('nop')
    return env._POTUpdateBuilder(target, source, **kw)

def _POTUpdateBuilder(env, **kw):
    if False:
        while True:
            i = 10
    ' Creates `POTUpdate` builder object '
    kw['action'] = SCons.Action.Action(_update_pot_file, None)
    kw['suffix'] = '$POTSUFFIX'
    kw['target_factory'] = _POTargetFactory(env, alias='$POTUPDATE_ALIAS').File
    kw['emitter'] = _pot_update_emitter
    return _POTBuilder(**kw)

def generate(env, **kw):
    if False:
        while True:
            i = 10
    ' Generate `xgettext` tool '
    if sys.platform == 'win32':
        xgettext = SCons.Tool.find_program_path(env, 'xgettext', default_paths=MINGW_DEFAULT_PATHS + CYGWIN_DEFAULT_PATHS)
        if xgettext:
            xgettext_bin_dir = os.path.dirname(xgettext)
            env.AppendENVPath('PATH', xgettext_bin_dir)
        else:
            SCons.Warnings.Warning('xgettext tool requested, but binary not found in ENV PATH')
    try:
        env['XGETTEXT'] = _detect_xgettext(env)
    except:
        env['XGETTEXT'] = 'xgettext'
    sources = '$( ${_concat( "", SOURCES, "", __env__, XgettextRPaths, TARGET' + ', SOURCES)} $)'
    xgettextcom = '$XGETTEXT $XGETTEXTFLAGS $_XGETTEXTPATHFLAGS' + ' $_XGETTEXTFROMFLAGS -o - ' + sources
    xgettextpathflags = '$( ${_concat( XGETTEXTPATHPREFIX, XGETTEXTPATH' + ', XGETTEXTPATHSUFFIX, __env__, RDirs, TARGET, SOURCES)} $)'
    xgettextfromflags = '$( ${_concat( XGETTEXTFROMPREFIX, XGETTEXTFROM' + ', XGETTEXTFROMSUFFIX, __env__, target=TARGET, source=SOURCES)} $)'
    env.SetDefault(_XGETTEXTDOMAIN='${TARGET.filebase}', XGETTEXTFLAGS=[], XGETTEXTCOM=xgettextcom, XGETTEXTCOMSTR='', XGETTEXTPATH=[], XGETTEXTPATHPREFIX='-D', XGETTEXTPATHSUFFIX='', XGETTEXTFROM=None, XGETTEXTFROMPREFIX='-f', XGETTEXTFROMSUFFIX='', _XGETTEXTPATHFLAGS=xgettextpathflags, _XGETTEXTFROMFLAGS=xgettextfromflags, POTSUFFIX=['.pot'], POTUPDATE_ALIAS='pot-update', XgettextRPaths=RPaths(env))
    env.Append(BUILDERS={'_POTUpdateBuilder': _POTUpdateBuilder(env)})
    env.AddMethod(_POTUpdateBuilderWrapper, 'POTUpdate')
    env.AlwaysBuild(env.Alias('$POTUPDATE_ALIAS'))

def exists(env):
    if False:
        while True:
            i = 10
    ' Check, whether the tool exists '
    try:
        return _xgettext_exists(env)
    except:
        return False