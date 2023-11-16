""" msgmerget tool 

Tool specific initialization for `msgmerge` tool.
"""
__revision__ = 'src/engine/SCons/Tool/msgmerge.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'

def _update_or_init_po_files(target, source, env):
    if False:
        for i in range(10):
            print('nop')
    ' Action function for `POUpdate` builder '
    import SCons.Action
    from SCons.Tool.GettextCommon import _init_po_files
    for tgt in target:
        if tgt.rexists():
            action = SCons.Action.Action('$MSGMERGECOM', '$MSGMERGECOMSTR')
        else:
            action = _init_po_files
        status = action([tgt], source, env)
        if status:
            return status
    return 0

def _POUpdateBuilder(env, **kw):
    if False:
        while True:
            i = 10
    ' Create an object of `POUpdate` builder '
    import SCons.Action
    from SCons.Tool.GettextCommon import _POFileBuilder
    action = SCons.Action.Action(_update_or_init_po_files, None)
    return _POFileBuilder(env, action=action, target_alias='$POUPDATE_ALIAS')
from SCons.Environment import _null

def _POUpdateBuilderWrapper(env, target=None, source=_null, **kw):
    if False:
        for i in range(10):
            print('nop')
    " Wrapper for `POUpdate` builder - make user's life easier "
    if source is _null:
        if 'POTDOMAIN' in kw:
            domain = kw['POTDOMAIN']
        elif 'POTDOMAIN' in env and env['POTDOMAIN']:
            domain = env['POTDOMAIN']
        else:
            domain = 'messages'
        source = [domain]
    return env._POUpdateBuilder(target, source, **kw)

def generate(env, **kw):
    if False:
        for i in range(10):
            print('nop')
    ' Generate the `msgmerge` tool '
    import sys
    import os
    import SCons.Tool
    from SCons.Tool.GettextCommon import _detect_msgmerge
    from SCons.Platform.mingw import MINGW_DEFAULT_PATHS
    from SCons.Platform.cygwin import CYGWIN_DEFAULT_PATHS
    if sys.platform == 'win32':
        msgmerge = SCons.Tool.find_program_path(env, 'msgmerge', default_paths=MINGW_DEFAULT_PATHS + CYGWIN_DEFAULT_PATHS)
        if msgmerge:
            msgmerge_bin_dir = os.path.dirname(msgmerge)
            env.AppendENVPath('PATH', msgmerge_bin_dir)
        else:
            SCons.Warnings.Warning('msgmerge tool requested, but binary not found in ENV PATH')
    try:
        env['MSGMERGE'] = _detect_msgmerge(env)
    except:
        env['MSGMERGE'] = 'msgmerge'
    env.SetDefault(POTSUFFIX=['.pot'], POSUFFIX=['.po'], MSGMERGECOM='$MSGMERGE  $MSGMERGEFLAGS --update $TARGET $SOURCE', MSGMERGECOMSTR='', MSGMERGEFLAGS=[], POUPDATE_ALIAS='po-update')
    env.Append(BUILDERS={'_POUpdateBuilder': _POUpdateBuilder(env)})
    env.AddMethod(_POUpdateBuilderWrapper, 'POUpdate')
    env.AlwaysBuild(env.Alias('$POUPDATE_ALIAS'))

def exists(env):
    if False:
        print('Hello World!')
    ' Check if the tool exists '
    from SCons.Tool.GettextCommon import _msgmerge_exists
    try:
        return _msgmerge_exists(env)
    except:
        return False