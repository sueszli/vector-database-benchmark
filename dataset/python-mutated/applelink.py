"""SCons.Tool.applelink

Tool-specific initialization for Apple's gnu-like linker.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/applelink.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Util
from . import link

class AppleLinkInvalidCurrentVersionException(Exception):
    pass

class AppleLinkInvalidCompatibilityVersionException(Exception):
    pass

def _applelib_versioned_lib_suffix(env, suffix, version):
    if False:
        print('Hello World!')
    "For suffix='.dylib' and version='0.1.2' it returns '.0.1.2.dylib'"
    Verbose = False
    if Verbose:
        print('_applelib_versioned_lib_suffix: suffix={!r}'.format(suffix))
        print('_applelib_versioned_lib_suffix: version={!r}'.format(version))
    if version not in suffix:
        suffix = '.' + version + suffix
    if Verbose:
        print('_applelib_versioned_lib_suffix: return suffix={!r}'.format(suffix))
    return suffix

def _applelib_versioned_lib_soname(env, libnode, version, prefix, suffix, name_func):
    if False:
        print('Hello World!')
    "For libnode='/optional/dir/libfoo.X.Y.Z.dylib' it returns 'libfoo.X.dylib'"
    Verbose = False
    if Verbose:
        print('_applelib_versioned_lib_soname: version={!r}'.format(version))
    name = name_func(env, libnode, version, prefix, suffix)
    if Verbose:
        print('_applelib_versioned_lib_soname: name={!r}'.format(name))
    major = version.split('.')[0]
    (libname, _suffix) = name.split('.')
    soname = '.'.join([libname, major, _suffix])
    if Verbose:
        print('_applelib_versioned_lib_soname: soname={!r}'.format(soname))
    return soname

def _applelib_versioned_shlib_soname(env, libnode, version, prefix, suffix):
    if False:
        while True:
            i = 10
    return _applelib_versioned_lib_soname(env, libnode, version, prefix, suffix, link._versioned_shlib_name)
_applelib_max_version_values = (65535, 255, 255)

def _applelib_check_valid_version(version_string):
    if False:
        i = 10
        return i + 15
    '\n    Check that the version # is valid.\n    X[.Y[.Z]]\n    where X 0-65535\n    where Y either not specified or 0-255\n    where Z either not specified or 0-255\n    :param version_string:\n    :return:\n    '
    parts = version_string.split('.')
    if len(parts) > 3:
        return (False, 'Version string has too many periods [%s]' % version_string)
    if len(parts) <= 0:
        return (False, 'Version string unspecified [%s]' % version_string)
    for (i, p) in enumerate(parts):
        try:
            p_i = int(p)
        except ValueError:
            return (False, 'Version component %s (from %s) is not a number' % (p, version_string))
        if p_i < 0 or p_i > _applelib_max_version_values[i]:
            return (False, 'Version component %s (from %s) is not valid value should be between 0 and %d' % (p, version_string, _applelib_max_version_values[i]))
    return (True, '')

def _applelib_currentVersionFromSoVersion(source, target, env, for_signature):
    if False:
        return 10
    "\n    A generator function to create the -Wl,-current_version flag if needed.\n    If env['APPLELINK_NO_CURRENT_VERSION'] contains a true value no flag will be generated\n    Otherwise if APPLELINK_CURRENT_VERSION is not specified, env['SHLIBVERSION']\n    will be used.\n\n    :param source:\n    :param target:\n    :param env:\n    :param for_signature:\n    :return: A string providing the flag to specify the current_version of the shared library\n    "
    if env.get('APPLELINK_NO_CURRENT_VERSION', False):
        return ''
    elif env.get('APPLELINK_CURRENT_VERSION', False):
        version_string = env['APPLELINK_CURRENT_VERSION']
    elif env.get('SHLIBVERSION', False):
        version_string = env['SHLIBVERSION']
    else:
        return ''
    version_string = '.'.join(version_string.split('.')[:3])
    (valid, reason) = _applelib_check_valid_version(version_string)
    if not valid:
        raise AppleLinkInvalidCurrentVersionException(reason)
    return '-Wl,-current_version,%s' % version_string

def _applelib_compatVersionFromSoVersion(source, target, env, for_signature):
    if False:
        for i in range(10):
            print('nop')
    "\n    A generator function to create the -Wl,-compatibility_version flag if needed.\n    If env['APPLELINK_NO_COMPATIBILITY_VERSION'] contains a true value no flag will be generated\n    Otherwise if APPLELINK_COMPATIBILITY_VERSION is not specified\n    the first two parts of env['SHLIBVERSION'] will be used with a .0 appended.\n\n    :param source:\n    :param target:\n    :param env:\n    :param for_signature:\n    :return: A string providing the flag to specify the compatibility_version of the shared library\n    "
    if env.get('APPLELINK_NO_COMPATIBILITY_VERSION', False):
        return ''
    elif env.get('APPLELINK_COMPATIBILITY_VERSION', False):
        version_string = env['APPLELINK_COMPATIBILITY_VERSION']
    elif env.get('SHLIBVERSION', False):
        version_string = '.'.join(env['SHLIBVERSION'].split('.')[:2] + ['0'])
    else:
        return ''
    if version_string is None:
        return ''
    (valid, reason) = _applelib_check_valid_version(version_string)
    if not valid:
        raise AppleLinkInvalidCompatibilityVersionException(reason)
    return '-Wl,-compatibility_version,%s' % version_string

def generate(env):
    if False:
        for i in range(10):
            print('nop')
    'Add Builders and construction variables for applelink to an\n    Environment.'
    link.generate(env)
    env['FRAMEWORKPATHPREFIX'] = '-F'
    env['_FRAMEWORKPATH'] = '${_concat(FRAMEWORKPATHPREFIX, FRAMEWORKPATH, "", __env__, RDirs)}'
    env['_FRAMEWORKS'] = '${_concat("-framework ", FRAMEWORKS, "", __env__)}'
    env['LINKCOM'] = env['LINKCOM'] + ' $_FRAMEWORKPATH $_FRAMEWORKS $FRAMEWORKSFLAGS'
    env['SHLINKFLAGS'] = SCons.Util.CLVar('$LINKFLAGS -dynamiclib')
    env['SHLINKCOM'] = env['SHLINKCOM'] + ' $_FRAMEWORKPATH $_FRAMEWORKS $FRAMEWORKSFLAGS'
    link._setup_versioned_lib_variables(env, tool='applelink')
    env['LINKCALLBACKS'] = link._versioned_lib_callbacks()
    env['LINKCALLBACKS']['VersionedShLibSuffix'] = _applelib_versioned_lib_suffix
    env['LINKCALLBACKS']['VersionedShLibSoname'] = _applelib_versioned_shlib_soname
    env['_APPLELINK_CURRENT_VERSION'] = _applelib_currentVersionFromSoVersion
    env['_APPLELINK_COMPATIBILITY_VERSION'] = _applelib_compatVersionFromSoVersion
    env['_SHLIBVERSIONFLAGS'] = '$_APPLELINK_CURRENT_VERSION $_APPLELINK_COMPATIBILITY_VERSION '
    env['_LDMODULEVERSIONFLAGS'] = '$_APPLELINK_CURRENT_VERSION $_APPLELINK_COMPATIBILITY_VERSION '
    env['LDMODULEPREFIX'] = ''
    env['LDMODULESUFFIX'] = ''
    env['LDMODULEFLAGS'] = SCons.Util.CLVar('$LINKFLAGS -bundle')
    env['LDMODULECOM'] = '$LDMODULE -o ${TARGET} $LDMODULEFLAGS $SOURCES $_LIBDIRFLAGS $_LIBFLAGS $_FRAMEWORKPATH $_FRAMEWORKS $FRAMEWORKSFLAGS'
    env['__SHLIBVERSIONFLAGS'] = '${__libversionflags(__env__,"SHLIBVERSION","_SHLIBVERSIONFLAGS")}'

def exists(env):
    if False:
        for i in range(10):
            print('nop')
    return env['PLATFORM'] == 'darwin'