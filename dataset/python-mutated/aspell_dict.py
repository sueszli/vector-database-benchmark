import llnl.util.filesystem as fs
import spack.directives
import spack.package_base
import spack.util.executable
from .autotools import AutotoolsBuilder, AutotoolsPackage

class AspellBuilder(AutotoolsBuilder):
    """The Aspell builder is close enough to an autotools builder to allow
    specializing the builder class, so to use variables that are specific
    to the Aspell extensions.
    """

    def configure(self, pkg, spec, prefix):
        if False:
            i = 10
            return i + 15
        aspell = spec['aspell'].prefix.bin.aspell
        prezip = spec['aspell'].prefix.bin.prezip
        destdir = prefix
        sh = spack.util.executable.which('sh')
        sh('./configure', '--vars', 'ASPELL={0}'.format(aspell), 'PREZIP={0}'.format(prezip), 'DESTDIR={0}'.format(destdir))

class AspellDictPackage(AutotoolsPackage):
    """Specialized class for building aspell dictionairies."""
    spack.directives.extends('aspell', when='build_system=autotools')
    AutotoolsBuilder = AspellBuilder

    def view_destination(self, view):
        if False:
            print('Hello World!')
        aspell_spec = self.spec['aspell']
        if view.get_projection_for_spec(aspell_spec) != aspell_spec.prefix:
            raise spack.package_base.ExtensionError('aspell does not support non-global extensions')
        aspell = aspell_spec.command
        return aspell('dump', 'config', 'dict-dir', output=str).strip()

    def view_source(self):
        if False:
            while True:
                i = 10
        return self.prefix.lib

    def patch(self):
        if False:
            for i in range(10):
                print('nop')
        fs.filter_file('^dictdir=.*$', 'dictdir=/lib', 'configure')
        fs.filter_file('^datadir=.*$', 'datadir=/lib', 'configure')