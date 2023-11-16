"""
Common functions for working with RPM packages
"""
import collections
import datetime
import logging
import platform
import subprocess
import salt.utils.path
import salt.utils.stringutils
log = logging.getLogger(__name__)
ARCHES_64 = ('x86_64', 'athlon', 'amd64', 'ia32e', 'ia64', 'geode')
ARCHES_32 = ('i386', 'i486', 'i586', 'i686')
ARCHES_PPC = ('ppc', 'ppc64', 'ppc64le', 'ppc64iseries', 'ppc64pseries')
ARCHES_S390 = ('s390', 's390x')
ARCHES_SPARC = ('sparc', 'sparcv8', 'sparcv9', 'sparcv9v', 'sparc64', 'sparc64v')
ARCHES_ALPHA = ('alpha', 'alphaev4', 'alphaev45', 'alphaev5', 'alphaev56', 'alphapca56', 'alphaev6', 'alphaev67', 'alphaev68', 'alphaev7')
ARCHES_ARM_32 = ('armv5tel', 'armv5tejl', 'armv6l', 'armv6hl', 'armv7l', 'armv7hl', 'armv7hnl')
ARCHES_ARM_64 = ('aarch64',)
ARCHES_SH = ('sh3', 'sh4', 'sh4a')
ARCHES = ARCHES_64 + ARCHES_32 + ARCHES_PPC + ARCHES_S390 + ARCHES_ALPHA + ARCHES_ARM_32 + ARCHES_ARM_64 + ARCHES_SH
QUERYFORMAT = '%{NAME}_|-%{EPOCH}_|-%{VERSION}_|-%{RELEASE}_|-%{ARCH}_|-%{REPOID}_|-%{INSTALLTIME}'

def get_osarch():
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the os architecture using rpm --eval\n    '
    if salt.utils.path.which('rpm'):
        ret = subprocess.Popen(['rpm', '--eval', '%{_host_cpu}'], close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
    else:
        ret = ''.join([x for x in platform.uname()[-2:] if x][-1:])
    return salt.utils.stringutils.to_str(ret).strip() or 'unknown'

def check_32(arch, osarch=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns True if both the OS arch and the passed arch are x86 or ARM 32-bit\n    '
    if osarch is None:
        osarch = get_osarch()
    return all((x in ARCHES_32 for x in (osarch, arch))) or all((x in ARCHES_ARM_32 for x in (osarch, arch)))

def pkginfo(name, version, arch, repoid, install_date=None, install_date_time_t=None):
    if False:
        while True:
            i = 10
    '\n    Build and return a pkginfo namedtuple\n    '
    pkginfo_tuple = collections.namedtuple('PkgInfo', ('name', 'version', 'arch', 'repoid', 'install_date', 'install_date_time_t'))
    return pkginfo_tuple(name, version, arch, repoid, install_date, install_date_time_t)

def resolve_name(name, arch, osarch=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Resolve the package name and arch into a unique name referred to by salt.\n    For example, on a 64-bit OS, a 32-bit package will be pkgname.i386.\n    '
    if osarch is None:
        osarch = get_osarch()
    if not check_32(arch, osarch) and arch not in (osarch, 'noarch'):
        name += '.{}'.format(arch)
    return name

def parse_pkginfo(line, osarch=None):
    if False:
        i = 10
        return i + 15
    "\n    A small helper to parse an rpm/repoquery command's output. Returns a\n    pkginfo namedtuple.\n    "
    try:
        (name, epoch, version, release, arch, repoid, install_time) = line.split('_|-')
    except ValueError:
        return None
    name = resolve_name(name, arch, osarch)
    if release:
        version += '-{}'.format(release)
    if epoch not in ('(none)', '0'):
        version = ':'.join((epoch, version))
    if install_time not in ('(none)', '0'):
        install_date = datetime.datetime.utcfromtimestamp(int(install_time)).isoformat() + 'Z'
        install_date_time_t = int(install_time)
    else:
        install_date = None
        install_date_time_t = None
    return pkginfo(name, version, arch, repoid, install_date, install_date_time_t)

def combine_comments(comments):
    if False:
        print('Hello World!')
    "\n    Given a list of comments, strings, a single comment or a single string,\n    return a single string of text containing all of the comments, prepending\n    the '#' and joining with newlines as necessary.\n    "
    if not isinstance(comments, list):
        comments = [comments]
    ret = []
    for comment in comments:
        if not isinstance(comment, str):
            comment = str(comment)
        ret.append('# {}\n'.format(comment.lstrip('#').lstrip()))
    return ''.join(ret)

def version_to_evr(verstring):
    if False:
        return 10
    '\n    Split the package version string into epoch, version and release.\n    Return this as tuple.\n\n    The epoch is always not empty. The version and the release can be an empty\n    string if such a component could not be found in the version string.\n\n    "2:1.0-1.2" => (\'2\', \'1.0\', \'1.2)\n    "1.0" => (\'0\', \'1.0\', \'\')\n    "" => (\'0\', \'\', \'\')\n    '
    if verstring in [None, '']:
        return ('0', '', '')
    idx_e = verstring.find(':')
    if idx_e != -1:
        try:
            epoch = str(int(verstring[:idx_e]))
        except ValueError:
            epoch = '0'
    else:
        epoch = '0'
    idx_r = verstring.find('-')
    if idx_r != -1:
        version = verstring[idx_e + 1:idx_r]
        release = verstring[idx_r + 1:]
    else:
        version = verstring[idx_e + 1:]
        release = ''
    return (epoch, version, release)