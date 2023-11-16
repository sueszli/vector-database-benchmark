import os
import re
import llnl.util.tty as tty
import spack.error
import spack.version
from spack.util.module_cmd import module
from .linux_distro import LinuxDistro
_cle_release_file = '/etc/opt/cray/release/cle-release'
_clerelease_file = '/etc/opt/cray/release/clerelease'

def read_cle_release_file():
    if False:
        return 10
    'Read the CLE release file and return a dict with its attributes.\n\n    This file is present on newer versions of Cray.\n\n    The release file looks something like this::\n\n        RELEASE=6.0.UP07\n        BUILD=6.0.7424\n        ...\n\n    The dictionary we produce looks like this::\n\n        {\n          "RELEASE": "6.0.UP07",\n          "BUILD": "6.0.7424",\n          ...\n        }\n\n    Returns:\n        dict: dictionary of release attributes\n    '
    with open(_cle_release_file) as release_file:
        result = {}
        for line in release_file:
            (key, _, value) = line.partition('=')
            result[key] = value.strip()
        return result

def read_clerelease_file():
    if False:
        print('Hello World!')
    'Read the CLE release file and return the Cray OS version.\n\n    This file is present on older versions of Cray.\n\n    The release file looks something like this::\n\n        5.2.UP04\n\n    Returns:\n        str: the Cray OS version\n    '
    with open(_clerelease_file) as release_file:
        for line in release_file:
            return line.strip()

class CrayBackend(LinuxDistro):
    """Compute Node Linux (CNL) is the operating system used for the Cray XC
    series super computers. It is a very stripped down version of GNU/Linux.
    Any compilers found through this operating system will be used with
    modules. If updated, user must make sure that version and name are
    updated to indicate that OS has been upgraded (or downgraded)
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        name = 'cnl'
        version = self._detect_crayos_version()
        if version:
            super(LinuxDistro, self).__init__(name, version)
        else:
            super().__init__()
        self.modulecmd = module

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name + str(self.version)

    @classmethod
    def _detect_crayos_version(cls):
        if False:
            print('Hello World!')
        if os.path.isfile(_cle_release_file):
            release_attrs = read_cle_release_file()
            if 'RELEASE' not in release_attrs:
                return None
            v = spack.version.Version(release_attrs['RELEASE'])
            return v[0]
        elif os.path.isfile(_clerelease_file):
            v = read_clerelease_file()
            return spack.version.Version(v)[0]
        else:
            return None

    def arguments_to_detect_version_fn(self, paths):
        if False:
            i = 10
            return i + 15
        import spack.compilers
        command_arguments = []
        for compiler_name in spack.compilers.supported_compilers():
            cmp_cls = spack.compilers.class_for_compiler_name(compiler_name)
            if cmp_cls.PrgEnv is None:
                continue
            if cmp_cls.PrgEnv_compiler is None:
                tty.die('Must supply PrgEnv_compiler with PrgEnv')
            compiler_id = spack.compilers.CompilerID(self, compiler_name, None)
            detect_version_args = spack.compilers.DetectVersionArgs(id=compiler_id, variation=(None, None), language='cc', path='cc')
            command_arguments.append(detect_version_args)
        return command_arguments

    def detect_version(self, detect_version_args):
        if False:
            i = 10
            return i + 15
        import spack.compilers
        modulecmd = self.modulecmd
        compiler_name = detect_version_args.id.compiler_name
        compiler_cls = spack.compilers.class_for_compiler_name(compiler_name)
        output = modulecmd('avail', compiler_cls.PrgEnv_compiler)
        version_regex = '({0})/([\\d\\.]+[\\d]-?[\\w]*)'.format(compiler_cls.PrgEnv_compiler)
        matches = re.findall(version_regex, output)
        version = tuple((version for (_, version) in matches if 'classic' not in version))
        compiler_id = detect_version_args.id
        value = detect_version_args._replace(id=compiler_id._replace(version=version))
        return (value, None)

    def make_compilers(self, compiler_id, paths):
        if False:
            print('Hello World!')
        import spack.spec
        name = compiler_id.compiler_name
        cmp_cls = spack.compilers.class_for_compiler_name(name)
        compilers = []
        for v in compiler_id.version:
            comp = cmp_cls(spack.spec.CompilerSpec(name + '@=' + v), self, 'any', ['cc', 'CC', 'ftn'], [cmp_cls.PrgEnv, name + '/' + v])
            compilers.append(comp)
        return compilers