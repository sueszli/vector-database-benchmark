"""SCons.Tool.mwld

Tool-specific initialization for the Metrowerks CodeWarrior linker.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
__revision__ = 'src/engine/SCons/Tool/mwld.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Tool

def generate(env):
    if False:
        print('Hello World!')
    'Add Builders and construction variables for lib to an Environment.'
    SCons.Tool.createStaticLibBuilder(env)
    SCons.Tool.createSharedLibBuilder(env)
    SCons.Tool.createProgBuilder(env)
    env['AR'] = 'mwld'
    env['ARCOM'] = '$AR $ARFLAGS -library -o $TARGET $SOURCES'
    env['LIBDIRPREFIX'] = '-L'
    env['LIBDIRSUFFIX'] = ''
    env['LIBLINKPREFIX'] = '-l'
    env['LIBLINKSUFFIX'] = '.lib'
    env['LINK'] = 'mwld'
    env['LINKCOM'] = '$LINK $LINKFLAGS -o $TARGET $SOURCES $_LIBDIRFLAGS $_LIBFLAGS'
    env['SHLINK'] = '$LINK'
    env['SHLINKFLAGS'] = '$LINKFLAGS'
    env['SHLINKCOM'] = shlib_action
    env['SHLIBEMITTER'] = shlib_emitter
    env['LDMODULEEMITTER'] = shlib_emitter

def exists(env):
    if False:
        return 10
    import SCons.Tool.mwcc
    return SCons.Tool.mwcc.set_vars(env)

def shlib_generator(target, source, env, for_signature):
    if False:
        return 10
    cmd = ['$SHLINK', '$SHLINKFLAGS', '-shared']
    no_import_lib = env.get('no_import_lib', 0)
    if no_import_lib:
        cmd.extend('-noimplib')
    dll = env.FindIxes(target, 'SHLIBPREFIX', 'SHLIBSUFFIX')
    if dll:
        cmd.extend(['-o', dll])
    implib = env.FindIxes(target, 'LIBPREFIX', 'LIBSUFFIX')
    if implib:
        cmd.extend(['-implib', implib.get_string(for_signature)])
    cmd.extend(['$SOURCES', '$_LIBDIRFLAGS', '$_LIBFLAGS'])
    return [cmd]

def shlib_emitter(target, source, env):
    if False:
        while True:
            i = 10
    dll = env.FindIxes(target, 'SHLIBPREFIX', 'SHLIBSUFFIX')
    no_import_lib = env.get('no_import_lib', 0)
    if not dll:
        raise SCons.Errors.UserError('A shared library should have exactly one target with the suffix: %s' % env.subst('$SHLIBSUFFIX'))
    if not no_import_lib and (not env.FindIxes(target, 'LIBPREFIX', 'LIBSUFFIX')):
        target.append(env.ReplaceIxes(dll, 'SHLIBPREFIX', 'SHLIBSUFFIX', 'LIBPREFIX', 'LIBSUFFIX'))
    return (target, source)
shlib_action = SCons.Action.Action(shlib_generator, generator=1)