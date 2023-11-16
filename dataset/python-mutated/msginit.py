""" msginit tool 

Tool specific initialization of msginit tool.
"""
__revision__ = 'src/engine/SCons/Tool/msginit.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Warnings
import SCons.Builder
import re

def _optional_no_translator_flag(env):
    if False:
        while True:
            i = 10
    " Return '--no-translator' flag if we run *msginit(1)*  in non-interactive\n      mode."
    import SCons.Util
    if 'POAUTOINIT' in env:
        autoinit = env['POAUTOINIT']
    else:
        autoinit = False
    if autoinit:
        return [SCons.Util.CLVar('--no-translator')]
    else:
        return [SCons.Util.CLVar('')]

def _POInitBuilder(env, **kw):
    if False:
        for i in range(10):
            print('nop')
    ' Create builder object for `POInit` builder. '
    import SCons.Action
    from SCons.Tool.GettextCommon import _init_po_files, _POFileBuilder
    action = SCons.Action.Action(_init_po_files, None)
    return _POFileBuilder(env, action=action, target_alias='$POCREATE_ALIAS')
from SCons.Environment import _null

def _POInitBuilderWrapper(env, target=None, source=_null, **kw):
    if False:
        while True:
            i = 10
    " Wrapper for _POFileBuilder. We use it to make user's life easier.\n  \n  This wrapper checks for `$POTDOMAIN` construction variable (or override in\n  `**kw`) and treats it appropriatelly. \n  "
    if source is _null:
        if 'POTDOMAIN' in kw:
            domain = kw['POTDOMAIN']
        elif 'POTDOMAIN' in env:
            domain = env['POTDOMAIN']
        else:
            domain = 'messages'
        source = [domain]
    return env._POInitBuilder(target, source, **kw)

def generate(env, **kw):
    if False:
        for i in range(10):
            print('nop')
    ' Generate the `msginit` tool '
    import sys
    import os
    import SCons.Util
    import SCons.Tool
    from SCons.Tool.GettextCommon import _detect_msginit
    from SCons.Platform.mingw import MINGW_DEFAULT_PATHS
    from SCons.Platform.cygwin import CYGWIN_DEFAULT_PATHS
    if sys.platform == 'win32':
        msginit = SCons.Tool.find_program_path(env, 'msginit', default_paths=MINGW_DEFAULT_PATHS + CYGWIN_DEFAULT_PATHS)
        if msginit:
            msginit_bin_dir = os.path.dirname(msginit)
            env.AppendENVPath('PATH', msginit_bin_dir)
        else:
            SCons.Warnings.Warning('msginit tool requested, but binary not found in ENV PATH')
    try:
        env['MSGINIT'] = _detect_msginit(env)
    except:
        env['MSGINIT'] = 'msginit'
    msginitcom = '$MSGINIT ${_MSGNoTranslator(__env__)} -l ${_MSGINITLOCALE}' + ' $MSGINITFLAGS -i $SOURCE -o $TARGET'
    env.SetDefault(POSUFFIX=['.po'], POTSUFFIX=['.pot'], _MSGINITLOCALE='${TARGET.filebase}', _MSGNoTranslator=_optional_no_translator_flag, MSGINITCOM=msginitcom, MSGINITCOMSTR='', MSGINITFLAGS=[], POAUTOINIT=False, POCREATE_ALIAS='po-create')
    env.Append(BUILDERS={'_POInitBuilder': _POInitBuilder(env)})
    env.AddMethod(_POInitBuilderWrapper, 'POInit')
    env.AlwaysBuild(env.Alias('$POCREATE_ALIAS'))

def exists(env):
    if False:
        while True:
            i = 10
    ' Check if the tool exists '
    from SCons.Tool.GettextCommon import _msginit_exists
    try:
        return _msginit_exists(env)
    except:
        return False