from SCons.Errors import UserError
from SCons.Tool import createSharedLibBuilder
from SCons.Util import CLVar
from . import lib_emitter, EmitLibSymlinks, StringizeLibSymlinks

def shlib_symlink_emitter(target, source, env, **kw):
    if False:
        for i in range(10):
            print('nop')
    verbose = False
    if 'variable_prefix' in kw:
        var_prefix = kw['variable_prefix']
    else:
        var_prefix = 'SHLIB'
    do_symlinks = env.subst('$%sNOVERSIONSYMLINKS' % var_prefix)
    if do_symlinks in ['1', 'True', 'true', True]:
        return (target, source)
    shlibversion = env.subst('$%sVERSION' % var_prefix)
    if shlibversion:
        if verbose:
            print('shlib_symlink_emitter: %sVERSION=%s' % (var_prefix, shlibversion))
        libnode = target[0]
        shlib_soname_symlink = env.subst('$%s_SONAME_SYMLINK' % var_prefix, target=target, source=source)
        shlib_noversion_symlink = env.subst('$%s_NOVERSION_SYMLINK' % var_prefix, target=target, source=source)
        if verbose:
            print('shlib_soname_symlink    :%s' % shlib_soname_symlink)
            print('shlib_noversion_symlink :%s' % shlib_noversion_symlink)
            print('libnode                 :%s' % libnode)
        shlib_soname_symlink = env.File(shlib_soname_symlink)
        shlib_noversion_symlink = env.File(shlib_noversion_symlink)
        symlinks = []
        if shlib_soname_symlink != libnode:
            symlinks.append((env.File(shlib_soname_symlink), libnode))
        symlinks.append((env.File(shlib_noversion_symlink), libnode))
        if verbose:
            print('_lib_emitter: symlinks={!r}'.format(', '.join(['%r->%r' % (k, v) for (k, v) in StringizeLibSymlinks(symlinks)])))
        if symlinks:
            EmitLibSymlinks(env, symlinks, target[0])
            target[0].attributes.shliblinks = symlinks
    return (target, source)

def _soversion(target, source, env, for_signature):
    if False:
        return 10
    'Function to determine what to use for SOVERSION'
    if 'SOVERSION' in env:
        return '.$SOVERSION'
    elif 'SHLIBVERSION' in env:
        shlibversion = env.subst('$SHLIBVERSION')
        return '.' + shlibversion.split('.')[0]
    else:
        return ''

def _soname(target, source, env, for_signature):
    if False:
        for i in range(10):
            print('nop')
    if 'SONAME' in env:
        if 'SOVERSION' in env:
            raise UserError('Ambiguous library .so naming, both SONAME: %s and SOVERSION: %s are defined. Only one can be defined for a target library.' % (env['SONAME'], env['SOVERSION']))
        return '$SONAME'
    else:
        return '$SHLIBPREFIX$_get_shlib_stem${SHLIBSUFFIX}$_SHLIBSOVERSION'

def _get_shlib_stem(target, source, env, for_signature: bool) -> str:
    if False:
        i = 10
        return i + 15
    'Get the base name of a shared library.\n\n    Args:\n        target: target node containing the lib name\n        source: source node, not used\n        env: environment context for running subst\n        for_signature: whether this is being done for signature generation\n\n    Returns:\n        the library name without prefix/suffix\n    '
    verbose = False
    target_name = str(target.name)
    shlibprefix = env.subst('$SHLIBPREFIX')
    shlibsuffix = env.subst('$_SHLIBSUFFIX')
    if verbose and (not for_signature):
        print('_get_shlib_stem: target_name:%s shlibprefix:%s shlibsuffix:%s' % (target_name, shlibprefix, shlibsuffix))
    if shlibsuffix and target_name.endswith(shlibsuffix):
        target_name = target_name[:-len(shlibsuffix)]
    if shlibprefix and target_name.startswith(shlibprefix):
        if target_name != shlibprefix:
            target_name = target_name[len(shlibprefix):]
    if verbose and (not for_signature):
        print('_get_shlib_stem: target_name:%s AFTER' % (target_name,))
    return target_name

def _get_shlib_dir(target, source, env, for_signature: bool) -> str:
    if False:
        for i in range(10):
            print('nop')
    "Get the directory the shared library is in.\n\n    Args:\n        target: target node\n        source: source node, not used\n        env: environment context, not used\n        for_signature: whether this is being done for signature generation\n\n    Returns:\n        the directory the library will be in (empty string if '.')\n    "
    verbose = False
    if target.dir and str(target.dir) != '.':
        if verbose:
            print('_get_shlib_dir: target.dir:%s' % target.dir)
        return '%s/' % str(target.dir)
    else:
        return ''

def setup_shared_lib_logic(env):
    if False:
        while True:
            i = 10
    'Initialize an environment for shared library building.\n\n    Args:\n        env: environment to set up\n    '
    createSharedLibBuilder(env)
    env['_get_shlib_stem'] = _get_shlib_stem
    env['_get_shlib_dir'] = _get_shlib_dir
    env['_SHLIBSOVERSION'] = _soversion
    env['_SHLIBSONAME'] = _soname
    env['SHLIBNAME'] = '${_get_shlib_dir}${SHLIBPREFIX}$_get_shlib_stem${_SHLIBSUFFIX}'
    env['SHLIB_NOVERSION_SYMLINK'] = '${_get_shlib_dir}${SHLIBPREFIX}$_get_shlib_stem${SHLIBSUFFIX}'
    env['SHLIB_SONAME_SYMLINK'] = '${_get_shlib_dir}$_SHLIBSONAME'
    env['SHLIBSONAMEFLAGS'] = '-Wl,-soname=$_SHLIBSONAME'
    env['_SHLIBVERSION'] = "${SHLIBVERSION and '.'+SHLIBVERSION or ''}"
    env['_SHLIBVERSIONFLAGS'] = '$SHLIBVERSIONFLAGS -Wl,-soname=$_SHLIBSONAME'
    env['SHLIBEMITTER'] = [lib_emitter, shlib_symlink_emitter]
    env['SHLIBPREFIX'] = env.get('SHLIBPREFIX', 'lib')
    env['_SHLIBSUFFIX'] = '${SHLIBSUFFIX}${_SHLIBVERSION}'
    env['SHLINKFLAGS'] = CLVar('$LINKFLAGS -shared')
    env['SHLINKCOM'] = '$SHLINK -o $TARGET $SHLINKFLAGS $__SHLIBVERSIONFLAGS $__RPATH $SOURCES $_LIBDIRFLAGS $_LIBFLAGS'
    env['SHLINK'] = '$LINK'