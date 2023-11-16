"""SCons.Tool.pyext

Tool-specific initialization for python extensions builder.

AUTHORS:
 - David Cournapeau
 - Dag Sverre Seljebotn

"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import sys
import SCons
from SCons.Tool import SourceFileScanner, ProgramScanner

def createPythonObjectBuilder(env):
    if False:
        for i in range(10):
            print('nop')
    'This is a utility function that creates the PythonObject Builder in an\n    Environment if it is not there already.\n\n    If it is already there, we return the existing one.\n    '
    try:
        pyobj = env['BUILDERS']['PythonObject']
    except KeyError:
        pyobj = SCons.Builder.Builder(action={}, emitter={}, prefix='$PYEXTOBJPREFIX', suffix='$PYEXTOBJSUFFIX', src_builder=['CFile', 'CXXFile'], source_scanner=SourceFileScanner, single_source=1)
        env['BUILDERS']['PythonObject'] = pyobj
    return pyobj

def createPythonExtensionBuilder(env):
    if False:
        while True:
            i = 10
    'This is a utility function that creates the PythonExtension Builder in\n    an Environment if it is not there already.\n\n    If it is already there, we return the existing one.\n    '
    try:
        pyext = env['BUILDERS']['PythonExtension']
    except KeyError:
        import SCons.Action
        import SCons.Defaults
        action = SCons.Action.Action('$PYEXTLINKCOM', '$PYEXTLINKCOMSTR')
        action_list = [SCons.Defaults.SharedCheck, action]
        pyext = SCons.Builder.Builder(action=action_list, emitter='$SHLIBEMITTER', prefix='$PYEXTPREFIX', suffix='$PYEXTSUFFIX', target_scanner=ProgramScanner, src_suffix='$PYEXTOBJSUFFIX', src_builder='PythonObject')
        env['BUILDERS']['PythonExtension'] = pyext
    return pyext

def pyext_coms(platform):
    if False:
        for i in range(10):
            print('nop')
    'Return PYEXTCCCOM, PYEXTCXXCOM and PYEXTLINKCOM for the given\n    platform.'
    if platform == 'win32':
        pyext_cccom = '$PYEXTCC /Fo$TARGET /c $PYEXTCCSHARED $PYEXTCFLAGS $PYEXTCCFLAGS $_CCCOMCOM $_PYEXTCPPINCFLAGS $SOURCES'
        pyext_cxxcom = '$PYEXTCXX /Fo$TARGET /c $PYEXTCSHARED $PYEXTCXXFLAGS $PYEXTCCFLAGS $_CCCOMCOM $_PYEXTCPPINCFLAGS $SOURCES'
        pyext_linkcom = '${TEMPFILE("$PYEXTLINK $PYEXTLINKFLAGS /OUT:$TARGET.windows $( $_LIBDIRFLAGS $) $_LIBFLAGS $_PYEXTRUNTIME $SOURCES.windows")}'
    else:
        pyext_cccom = '$PYEXTCC -o $TARGET -c $PYEXTCCSHARED $PYEXTCFLAGS $PYEXTCCFLAGS $_CCCOMCOM $_PYEXTCPPINCFLAGS $SOURCES'
        pyext_cxxcom = '$PYEXTCXX -o $TARGET -c $PYEXTCSHARED $PYEXTCXXFLAGS $PYEXTCCFLAGS $_CCCOMCOM $_PYEXTCPPINCFLAGS $SOURCES'
        pyext_linkcom = '$PYEXTLINK -o $TARGET $PYEXTLINKFLAGS $SOURCES $_LIBDIRFLAGS $_LIBFLAGS $_PYEXTRUNTIME'
    if platform == 'darwin':
        pyext_linkcom += ' $_FRAMEWORKPATH $_FRAMEWORKS $FRAMEWORKSFLAGS'
    return (pyext_cccom, pyext_cxxcom, pyext_linkcom)

def set_basic_vars(env):
    if False:
        i = 10
        return i + 15
    env['PYEXTCPPPATH'] = SCons.Util.CLVar('$PYEXTINCPATH')
    env['_PYEXTCPPINCFLAGS'] = '$( ${_concat(INCPREFIX, PYEXTCPPPATH, INCSUFFIX, __env__, RDirs, TARGET, SOURCE)} $)'
    env['PYEXTOBJSUFFIX'] = '$SHOBJSUFFIX'
    env['PYEXTOBJPREFIX'] = '$SHOBJPREFIX'
    env['PYEXTRUNTIME'] = SCons.Util.CLVar('')
    env['_PYEXTRUNTIME'] = '$( ${_concat(LIBLINKPREFIX, PYEXTRUNTIME, LIBLINKSUFFIX, __env__)} $)'
    (pycc, pycxx, pylink) = pyext_coms(sys.platform)
    env['PYEXTLINKFLAGSEND'] = SCons.Util.CLVar('$LINKFLAGSEND')
    env['PYEXTCCCOM'] = pycc
    env['PYEXTCXXCOM'] = pycxx
    env['PYEXTLINKCOM'] = pylink

def _set_configuration_nodistutils(env):
    if False:
        return 10
    def_cfg = {'PYEXTCC': '$SHCC', 'PYEXTCFLAGS': '$SHCFLAGS', 'PYEXTCCFLAGS': '$SHCCFLAGS', 'PYEXTCXX': '$SHCXX', 'PYEXTCXXFLAGS': '$SHCXXFLAGS', 'PYEXTLINK': '$LDMODULE', 'PYEXTSUFFIX': '$LDMODULESUFFIX', 'PYEXTPREFIX': ''}
    if sys.platform == 'darwin':
        def_cfg['PYEXTSUFFIX'] = '.so'
    for (k, v) in def_cfg.items():
        ifnotset(env, k, v)
    ifnotset(env, 'PYEXT_ALLOW_UNDEFINED', SCons.Util.CLVar('$ALLOW_UNDEFINED'))
    ifnotset(env, 'PYEXTLINKFLAGS', SCons.Util.CLVar('$LDMODULEFLAGS'))
    env.AppendUnique(PYEXTLINKFLAGS=env['PYEXT_ALLOW_UNDEFINED'])

def ifnotset(env, name, value):
    if False:
        i = 10
        return i + 15
    if name not in env:
        env[name] = value

def set_configuration(env, use_distutils):
    if False:
        while True:
            i = 10
    "Set construction variables which are platform dependants.\n\n    If use_distutils == True, use distutils configuration. Otherwise, use\n    'sensible' default.\n\n    Any variable already defined is untouched."
    dist_cfg = {'PYEXTCC': ("sysconfig.get_config_var('CC')", False), 'PYEXTCFLAGS': ("sysconfig.get_config_var('CFLAGS')", True), 'PYEXTCCSHARED': ("sysconfig.get_config_var('CCSHARED')", False), 'PYEXTLINKFLAGS': ("sysconfig.get_config_var('LDFLAGS')", True), 'PYEXTLINK': ("sysconfig.get_config_var('LDSHARED')", False), 'PYEXTINCPATH': ('sysconfig.get_python_inc()', False), 'PYEXTSUFFIX': ("sysconfig.get_config_var('SO')", False)}
    from distutils import sysconfig
    ifnotset(env, 'PYEXTINCPATH', sysconfig.get_python_inc())
    if use_distutils:
        for (k, (v, should_split)) in dist_cfg.items():
            val = eval(v)
            if should_split:
                val = val.split()
            ifnotset(env, k, val)
    else:
        _set_configuration_nodistutils(env)

def generate(env):
    if False:
        i = 10
        return i + 15
    'Add Builders and construction variables for python extensions to an\n    Environment.'
    if 'PYEXT_USE_DISTUTILS' not in env:
        env['PYEXT_USE_DISTUTILS'] = False
    set_basic_vars(env)
    set_configuration(env, env['PYEXT_USE_DISTUTILS'])
    pyobj = createPythonObjectBuilder(env)
    action = SCons.Action.Action('$PYEXTCCCOM', '$PYEXTCCCOMSTR')
    pyobj.add_emitter('.c', SCons.Defaults.SharedObjectEmitter)
    pyobj.add_action('.c', action)
    action = SCons.Action.Action('$PYEXTCXXCOM', '$PYEXTCXXCOMSTR')
    pyobj.add_emitter('$CXXFILESUFFIX', SCons.Defaults.SharedObjectEmitter)
    pyobj.add_action('$CXXFILESUFFIX', action)
    createPythonExtensionBuilder(env)

def exists(env):
    if False:
        for i in range(10):
            print('nop')
    try:
        from distutils import sysconfig
        return True
    except ImportError:
        return False