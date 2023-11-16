"""Definitions that control how Spack creates Spec hashes."""
import spack.deptypes as dt
import spack.repo
hashes = []

class SpecHashDescriptor:
    """This class defines how hashes are generated on Spec objects.

    Spec hashes in Spack are generated from a serialized (e.g., with
    YAML) representation of the Spec graph.  The representation may only
    include certain dependency types, and it may optionally include a
    canonicalized hash of the package.py for each node in the graph.

    We currently use different hashes for different use cases."""

    def __init__(self, depflag: dt.DepFlag, package_hash, name, override=None):
        if False:
            print('Hello World!')
        self.depflag = depflag
        self.package_hash = package_hash
        self.name = name
        hashes.append(self)
        self.override = override

    @property
    def attr(self):
        if False:
            i = 10
            return i + 15
        'Private attribute stored on spec'
        return '_' + self.name

    def __call__(self, spec):
        if False:
            i = 10
            return i + 15
        'Run this hash on the provided spec.'
        return spec.spec_hash(self)
dag_hash = SpecHashDescriptor(depflag=dt.BUILD | dt.LINK | dt.RUN, package_hash=True, name='hash')
process_hash = SpecHashDescriptor(depflag=dt.BUILD | dt.LINK | dt.RUN | dt.TEST, package_hash=True, name='process_hash')

def _content_hash_override(spec):
    if False:
        for i in range(10):
            print('nop')
    pkg_cls = spack.repo.PATH.get_pkg_class(spec.name)
    pkg = pkg_cls(spec)
    return pkg.content_hash()
package_hash = SpecHashDescriptor(depflag=0, package_hash=True, name='package_hash', override=_content_hash_override)
full_hash = SpecHashDescriptor(depflag=dt.BUILD | dt.LINK | dt.RUN, package_hash=True, name='full_hash')
build_hash = SpecHashDescriptor(depflag=dt.BUILD | dt.LINK | dt.RUN, package_hash=False, name='build_hash')