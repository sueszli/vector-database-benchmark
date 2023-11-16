from SCons.Tool import createLoadableModuleBuilder
from .SharedLibrary import shlib_symlink_emitter
from . import lib_emitter

def ldmod_symlink_emitter(target, source, env, **kw):
    if False:
        print('Hello World!')
    return shlib_symlink_emitter(target, source, env, variable_prefix='LDMODULE')

def _get_ldmodule_stem(target, source, env, for_signature):
    if False:
        print('Hello World!')
    '\n    Get the basename for a library (so for libxyz.so, return xyz)\n    :param target:\n    :param source:\n    :param env:\n    :param for_signature:\n    :return:\n    '
    target_name = str(target)
    ldmodule_prefix = env.subst('$LDMODULEPREFIX')
    ldmodule_suffix = env.subst('$_LDMODULESUFFIX')
    if target_name.startswith(ldmodule_prefix):
        target_name = target_name[len(ldmodule_prefix):]
    if target_name.endswith(ldmodule_suffix):
        target_name = target_name[:-len(ldmodule_suffix)]
    return target_name

def _ldmodule_soversion(target, source, env, for_signature):
    if False:
        return 10
    'Function to determine what to use for SOVERSION'
    if 'SOVERSION' in env:
        return '.$SOVERSION'
    elif 'LDMODULEVERSION' in env:
        ldmod_version = env.subst('$LDMODULEVERSION')
        return '.' + ldmod_version.split('.')[0]
    else:
        return ''

def _ldmodule_soname(target, source, env, for_signature):
    if False:
        i = 10
        return i + 15
    if 'SONAME' in env:
        return '$SONAME'
    else:
        return '$LDMODULEPREFIX$_get_ldmodule_stem${LDMODULESUFFIX}$_LDMODULESOVERSION'

def _LDMODULEVERSION(target, source, env, for_signature):
    if False:
        while True:
            i = 10
    '\n    Return "." + version if it\'s set, otherwise just a blank\n    '
    value = env.subst('$LDMODULEVERSION', target=target, source=source)
    if value:
        return '.' + value
    else:
        return ''

def setup_loadable_module_logic(env):
    if False:
        i = 10
        return i + 15
    '\n    Just the logic for loadable modules\n\n    For most platforms, a loadable module is the same as a shared\n    library.  Platforms which are different can override these, but\n    setting them the same means that LoadableModule works everywhere.\n\n    :param env:\n    :return:\n    '
    createLoadableModuleBuilder(env)
    env['_get_ldmodule_stem'] = _get_ldmodule_stem
    env['_LDMODULESOVERSION'] = _ldmodule_soversion
    env['_LDMODULESONAME'] = _ldmodule_soname
    env['LDMODULENAME'] = '${LDMODULEPREFIX}$_get_ldmodule_stem${_LDMODULESUFFIX}'
    env['LDMODULE_NOVERSION_SYMLINK'] = '$_get_shlib_dir${LDMODULEPREFIX}$_get_ldmodule_stem${LDMODULESUFFIX}'
    env['LDMODULE_SONAME_SYMLINK'] = '$_get_shlib_dir$_LDMODULESONAME'
    env['_LDMODULEVERSION'] = _LDMODULEVERSION
    env['_LDMODULEVERSIONFLAGS'] = '$LDMODULEVERSIONFLAGS -Wl,-soname=$_LDMODULESONAME'
    env['LDMODULEEMITTER'] = [lib_emitter, ldmod_symlink_emitter]
    env['LDMODULEPREFIX'] = '$SHLIBPREFIX'
    env['_LDMODULESUFFIX'] = '${LDMODULESUFFIX}${_LDMODULEVERSION}'
    env['LDMODULESUFFIX'] = '$SHLIBSUFFIX'
    env['LDMODULE'] = '$SHLINK'
    env['LDMODULEFLAGS'] = '$SHLINKFLAGS'
    env['LDMODULECOM'] = '$LDMODULE -o $TARGET $LDMODULEFLAGS $__LDMODULEVERSIONFLAGS $__RPATH $SOURCES $_LIBDIRFLAGS $_LIBFLAGS '
    env['LDMODULEVERSION'] = '$SHLIBVERSION'
    env['LDMODULENOVERSIONSYMLINKS'] = '$SHLIBNOVERSIONSYMLINKS'