import llnl.util.filesystem as fs
import spack.builder
import spack.package_base
from spack.directives import build_system, depends_on
from spack.multimethod import when
from spack.util.executable import which
from ._checks import BaseBuilder

class MavenPackage(spack.package_base.PackageBase):
    """Specialized class for packages that are built using the
    Maven build system. See https://maven.apache.org/index.html
    for more information.
    """
    build_system_class = 'MavenPackage'
    legacy_buildsystem = 'maven'
    build_system('maven')
    with when('build_system=maven'):
        depends_on('java', type=('build', 'run'))
        depends_on('maven', type='build')

@spack.builder.builder('maven')
class MavenBuilder(BaseBuilder):
    """The Maven builder encodes the default way to build software with Maven.
    It has two phases that can be overridden, if need be:

        1. :py:meth:`~.MavenBuilder.build`
        2. :py:meth:`~.MavenBuilder.install`
    """
    phases = ('build', 'install')
    legacy_methods = ('build_args',)
    legacy_attributes = ('build_directory',)

    @property
    def build_directory(self):
        if False:
            return 10
        'The directory containing the ``pom.xml`` file.'
        return self.pkg.stage.source_path

    def build_args(self):
        if False:
            i = 10
            return i + 15
        'List of args to pass to build phase.'
        return []

    def build(self, pkg, spec, prefix):
        if False:
            for i in range(10):
                print('nop')
        'Compile code and package into a JAR file.'
        with fs.working_dir(self.build_directory):
            mvn = which('mvn')
            if self.pkg.run_tests:
                mvn('verify', *self.build_args())
            else:
                mvn('package', '-DskipTests', *self.build_args())

    def install(self, pkg, spec, prefix):
        if False:
            for i in range(10):
                print('nop')
        'Copy to installation prefix.'
        with fs.working_dir(self.build_directory):
            fs.install_tree('.', prefix)