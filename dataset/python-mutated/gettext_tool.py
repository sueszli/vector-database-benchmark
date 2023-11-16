"""gettext tool
"""
__revision__ = 'src/engine/SCons/Tool/gettext_tool.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'

def generate(env, **kw):
    if False:
        print('Hello World!')
    import sys
    import os
    import SCons.Tool
    from SCons.Platform.mingw import MINGW_DEFAULT_PATHS
    from SCons.Platform.cygwin import CYGWIN_DEFAULT_PATHS
    from SCons.Tool.GettextCommon import _translate, tool_list
    for t in tool_list(env['PLATFORM'], env):
        if sys.platform == 'win32':
            tool = SCons.Tool.find_program_path(env, t, default_paths=MINGW_DEFAULT_PATHS + CYGWIN_DEFAULT_PATHS)
            if tool:
                tool_bin_dir = os.path.dirname(tool)
                env.AppendENVPath('PATH', tool_bin_dir)
            else:
                SCons.Warnings.Warning(t + ' tool requested, but binary not found in ENV PATH')
        env.Tool(t)
    env.AddMethod(_translate, 'Translate')

def exists(env):
    if False:
        i = 10
        return i + 15
    from SCons.Tool.GettextCommon import _xgettext_exists, _msginit_exists, _msgmerge_exists, _msgfmt_exists
    try:
        return _xgettext_exists(env) and _msginit_exists(env) and _msgmerge_exists(env) and _msgfmt_exists(env)
    except:
        return False