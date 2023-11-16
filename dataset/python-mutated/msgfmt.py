""" msgfmt tool """
__revision__ = 'src/engine/SCons/Tool/msgfmt.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
from SCons.Builder import BuilderBase

class _MOFileBuilder(BuilderBase):
    """ The builder class for `MO` files.
  
  The reason for this builder to exists and its purpose is quite simillar 
  as for `_POFileBuilder`. This time, we extend list of sources, not targets,
  and call `BuilderBase._execute()` only once (as we assume single-target
  here).
  """

    def _execute(self, env, target, source, *args, **kw):
        if False:
            while True:
                i = 10
        import SCons.Util
        from SCons.Tool.GettextCommon import _read_linguas_from_files
        linguas_files = None
        if 'LINGUAS_FILE' in env and env['LINGUAS_FILE'] is not None:
            linguas_files = env['LINGUAS_FILE']
            env['LINGUAS_FILE'] = None
            linguas = _read_linguas_from_files(env, linguas_files)
            if SCons.Util.is_List(source):
                source.extend(linguas)
            elif source is not None:
                source = [source] + linguas
            else:
                source = linguas
        result = BuilderBase._execute(self, env, target, source, *args, **kw)
        if linguas_files is not None:
            env['LINGUAS_FILE'] = linguas_files
        return result

def _create_mo_file_builder(env, **kw):
    if False:
        print('Hello World!')
    ' Create builder object for `MOFiles` builder '
    import SCons.Action
    kw['action'] = SCons.Action.Action('$MSGFMTCOM', '$MSGFMTCOMSTR')
    kw['suffix'] = '$MOSUFFIX'
    kw['src_suffix'] = '$POSUFFIX'
    kw['src_builder'] = '_POUpdateBuilder'
    kw['single_source'] = True
    return _MOFileBuilder(**kw)

def generate(env, **kw):
    if False:
        i = 10
        return i + 15
    ' Generate `msgfmt` tool '
    import sys
    import os
    import SCons.Util
    import SCons.Tool
    from SCons.Tool.GettextCommon import _detect_msgfmt
    from SCons.Platform.mingw import MINGW_DEFAULT_PATHS
    from SCons.Platform.cygwin import CYGWIN_DEFAULT_PATHS
    if sys.platform == 'win32':
        msgfmt = SCons.Tool.find_program_path(env, 'msgfmt', default_paths=MINGW_DEFAULT_PATHS + CYGWIN_DEFAULT_PATHS)
        if msgfmt:
            msgfmt_bin_dir = os.path.dirname(msgfmt)
            env.AppendENVPath('PATH', msgfmt_bin_dir)
        else:
            SCons.Warnings.Warning('msgfmt tool requested, but binary not found in ENV PATH')
    try:
        env['MSGFMT'] = _detect_msgfmt(env)
    except:
        env['MSGFMT'] = 'msgfmt'
    env.SetDefault(MSGFMTFLAGS=[SCons.Util.CLVar('-c')], MSGFMTCOM='$MSGFMT $MSGFMTFLAGS -o $TARGET $SOURCE', MSGFMTCOMSTR='', MOSUFFIX=['.mo'], POSUFFIX=['.po'])
    env.Append(BUILDERS={'MOFiles': _create_mo_file_builder(env)})

def exists(env):
    if False:
        i = 10
        return i + 15
    ' Check if the tool exists '
    from SCons.Tool.GettextCommon import _msgfmt_exists
    try:
        return _msgfmt_exists(env)
    except:
        return False