import platform as py_platform
import re
from subprocess import check_output
from spack.version import Version
from ._operating_system import OperatingSystem

def kernel_version():
    if False:
        while True:
            i = 10
    "Return the kernel version as a Version object.\n    Note that the kernel version is distinct from OS and/or\n    distribution versions. For instance:\n    >>> distro.id()\n    'centos'\n    >>> distro.version()\n    '7'\n    >>> platform.release()\n    '5.10.84+'\n    "
    clean_version = re.sub('\\+', '', py_platform.release())
    return Version(clean_version)

class LinuxDistro(OperatingSystem):
    """This class will represent the autodetected operating system
    for a Linux System. Since there are many different flavors of
    Linux, this class will attempt to encompass them all through
    autodetection using the python module platform and the method
    platform.dist()
    """

    def __init__(self):
        if False:
            print('Hello World!')
        try:
            import distro
            (distname, version) = (distro.id(), distro.version())
        except ImportError:
            (distname, version) = ('unknown', '')
        version = re.split('[^\\w-]', version)
        if 'ubuntu' in distname:
            version = '.'.join(version[0:2])
        elif 'opensuse-tumbleweed' in distname or 'opensusetumbleweed' in distname:
            distname = 'opensuse'
            output = check_output(['ldd', '--version']).decode()
            libcvers = re.findall('ldd \\(GNU libc\\) (.*)', output)
            if len(libcvers) == 1:
                version = 'tumbleweed' + libcvers[0]
            else:
                version = 'tumbleweed' + version[0]
        else:
            version = version[0]
        super().__init__(distname, version)