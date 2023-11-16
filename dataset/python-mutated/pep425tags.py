"""Generate and work with PEP 425 Compatibility Tags."""
from __future__ import absolute_import
import distutils.util
import logging
import platform
import re
import sys
import sysconfig
import warnings
from collections import OrderedDict
import pip._internal.utils.glibc
from pip._internal.utils.compat import get_extension_suffixes
from pip._internal.utils.typing import MYPY_CHECK_RUNNING
if MYPY_CHECK_RUNNING:
    from typing import Tuple, Callable, List, Optional, Union, Dict
    Pep425Tag = Tuple[str, str, str]
logger = logging.getLogger(__name__)
_osx_arch_pat = re.compile('(.+)_(\\d+)_(\\d+)_(.+)')

def get_config_var(var):
    if False:
        print('Hello World!')
    try:
        return sysconfig.get_config_var(var)
    except IOError as e:
        warnings.warn('{}'.format(e), RuntimeWarning)
        return None

def get_abbr_impl():
    if False:
        return 10
    'Return abbreviated implementation name.'
    if hasattr(sys, 'pypy_version_info'):
        pyimpl = 'pp'
    elif sys.platform.startswith('java'):
        pyimpl = 'jy'
    elif sys.platform == 'cli':
        pyimpl = 'ip'
    else:
        pyimpl = 'cp'
    return pyimpl

def get_impl_ver():
    if False:
        while True:
            i = 10
    'Return implementation version.'
    impl_ver = get_config_var('py_version_nodot')
    if not impl_ver or get_abbr_impl() == 'pp':
        impl_ver = ''.join(map(str, get_impl_version_info()))
    return impl_ver

def get_impl_version_info():
    if False:
        for i in range(10):
            print('nop')
    'Return sys.version_info-like tuple for use in decrementing the minor\n    version.'
    if get_abbr_impl() == 'pp':
        return (sys.version_info[0], sys.pypy_version_info.major, sys.pypy_version_info.minor)
    else:
        return (sys.version_info[0], sys.version_info[1])

def get_impl_tag():
    if False:
        i = 10
        return i + 15
    '\n    Returns the Tag for this specific implementation.\n    '
    return '{}{}'.format(get_abbr_impl(), get_impl_ver())

def get_flag(var, fallback, expected=True, warn=True):
    if False:
        return 10
    'Use a fallback method for determining SOABI flags if the needed config\n    var is unset or unavailable.'
    val = get_config_var(var)
    if val is None:
        if warn:
            logger.debug("Config variable '%s' is unset, Python ABI tag may be incorrect", var)
        return fallback()
    return val == expected

def get_abi_tag():
    if False:
        while True:
            i = 10
    'Return the ABI tag based on SOABI (if available) or emulate SOABI\n    (CPython 2, PyPy).'
    soabi = get_config_var('SOABI')
    impl = get_abbr_impl()
    if not soabi and impl in {'cp', 'pp'} and hasattr(sys, 'maxunicode'):
        d = ''
        m = ''
        u = ''
        if get_flag('Py_DEBUG', lambda : hasattr(sys, 'gettotalrefcount'), warn=impl == 'cp'):
            d = 'd'
        if get_flag('WITH_PYMALLOC', lambda : impl == 'cp', warn=impl == 'cp'):
            m = 'm'
        if get_flag('Py_UNICODE_SIZE', lambda : sys.maxunicode == 1114111, expected=4, warn=impl == 'cp' and sys.version_info < (3, 3)) and sys.version_info < (3, 3):
            u = 'u'
        abi = '%s%s%s%s%s' % (impl, get_impl_ver(), d, m, u)
    elif soabi and soabi.startswith('cpython-'):
        abi = 'cp' + soabi.split('-')[1]
    elif soabi:
        abi = soabi.replace('.', '_').replace('-', '_')
    else:
        abi = None
    return abi

def _is_running_32bit():
    if False:
        print('Hello World!')
    return sys.maxsize == 2147483647

def get_platform():
    if False:
        i = 10
        return i + 15
    "Return our platform name 'win32', 'linux_x86_64'"
    if sys.platform == 'darwin':
        (release, _, machine) = platform.mac_ver()
        split_ver = release.split('.')
        if machine == 'x86_64' and _is_running_32bit():
            machine = 'i386'
        elif machine == 'ppc64' and _is_running_32bit():
            machine = 'ppc'
        return 'macosx_{}_{}_{}'.format(split_ver[0], split_ver[1], machine)
    result = distutils.util.get_platform().replace('.', '_').replace('-', '_')
    if result == 'linux_x86_64' and _is_running_32bit():
        result = 'linux_i686'
    return result

def is_manylinux1_compatible():
    if False:
        for i in range(10):
            print('nop')
    if get_platform() not in {'linux_x86_64', 'linux_i686'}:
        return False
    try:
        import _manylinux
        return bool(_manylinux.manylinux1_compatible)
    except (ImportError, AttributeError):
        pass
    return pip._internal.utils.glibc.have_compatible_glibc(2, 5)

def is_manylinux2010_compatible():
    if False:
        i = 10
        return i + 15
    if get_platform() not in {'linux_x86_64', 'linux_i686'}:
        return False
    try:
        import _manylinux
        return bool(_manylinux.manylinux2010_compatible)
    except (ImportError, AttributeError):
        pass
    return pip._internal.utils.glibc.have_compatible_glibc(2, 12)

def get_darwin_arches(major, minor, machine):
    if False:
        for i in range(10):
            print('nop')
    'Return a list of supported arches (including group arches) for\n    the given major, minor and machine architecture of an macOS machine.\n    '
    arches = []

    def _supports_arch(major, minor, arch):
        if False:
            for i in range(10):
                print('nop')
        if arch == 'ppc':
            return (major, minor) <= (10, 5)
        if arch == 'ppc64':
            return (major, minor) == (10, 5)
        if arch == 'i386':
            return (major, minor) >= (10, 4)
        if arch == 'x86_64':
            return (major, minor) >= (10, 5)
        if arch in groups:
            for garch in groups[arch]:
                if _supports_arch(major, minor, garch):
                    return True
        return False
    groups = OrderedDict([('fat', ('i386', 'ppc')), ('intel', ('x86_64', 'i386')), ('fat64', ('x86_64', 'ppc64')), ('fat32', ('x86_64', 'i386', 'ppc'))])
    if _supports_arch(major, minor, machine):
        arches.append(machine)
    for garch in groups:
        if machine in groups[garch] and _supports_arch(major, minor, garch):
            arches.append(garch)
    arches.append('universal')
    return arches

def get_all_minor_versions_as_strings(version_info):
    if False:
        print('Hello World!')
    versions = []
    major = version_info[:-1]
    for minor in range(version_info[-1], -1, -1):
        versions.append(''.join(map(str, major + (minor,))))
    return versions

def get_supported(versions=None, noarch=False, platform=None, impl=None, abi=None):
    if False:
        while True:
            i = 10
    'Return a list of supported tags for each version specified in\n    `versions`.\n\n    :param versions: a list of string versions, of the form ["33", "32"],\n        or None. The first version will be assumed to support our ABI.\n    :param platform: specify the exact platform you want valid\n        tags for, or None. If None, use the local system platform.\n    :param impl: specify the exact implementation you want valid\n        tags for, or None. If None, use the local interpreter impl.\n    :param abi: specify the exact abi you want valid\n        tags for, or None. If None, use the local interpreter abi.\n    '
    supported = []
    if versions is None:
        version_info = get_impl_version_info()
        versions = get_all_minor_versions_as_strings(version_info)
    impl = impl or get_abbr_impl()
    abis = []
    abi = abi or get_abi_tag()
    if abi:
        abis[0:0] = [abi]
    abi3s = set()
    for suffix in get_extension_suffixes():
        if suffix.startswith('.abi'):
            abi3s.add(suffix.split('.', 2)[1])
    abis.extend(sorted(list(abi3s)))
    abis.append('none')
    if not noarch:
        arch = platform or get_platform()
        (arch_prefix, arch_sep, arch_suffix) = arch.partition('_')
        if arch.startswith('macosx'):
            match = _osx_arch_pat.match(arch)
            if match:
                (name, major, minor, actual_arch) = match.groups()
                tpl = '{}_{}_%i_%s'.format(name, major)
                arches = []
                for m in reversed(range(int(minor) + 1)):
                    for a in get_darwin_arches(int(major), m, actual_arch):
                        arches.append(tpl % (m, a))
            else:
                arches = [arch]
        elif arch_prefix == 'manylinux2010':
            arches = [arch, 'manylinux1' + arch_sep + arch_suffix]
        elif platform is None:
            arches = []
            if is_manylinux2010_compatible():
                arches.append('manylinux2010' + arch_sep + arch_suffix)
            if is_manylinux1_compatible():
                arches.append('manylinux1' + arch_sep + arch_suffix)
            arches.append(arch)
        else:
            arches = [arch]
        for abi in abis:
            for arch in arches:
                supported.append(('%s%s' % (impl, versions[0]), abi, arch))
        for version in versions[1:]:
            if version in {'31', '30'}:
                break
            for abi in abi3s:
                for arch in arches:
                    supported.append(('%s%s' % (impl, version), abi, arch))
        for arch in arches:
            supported.append(('py%s' % versions[0][0], 'none', arch))
    supported.append(('%s%s' % (impl, versions[0]), 'none', 'any'))
    supported.append(('%s%s' % (impl, versions[0][0]), 'none', 'any'))
    for (i, version) in enumerate(versions):
        supported.append(('py%s' % (version,), 'none', 'any'))
        if i == 0:
            supported.append(('py%s' % version[0], 'none', 'any'))
    return supported
implementation_tag = get_impl_tag()