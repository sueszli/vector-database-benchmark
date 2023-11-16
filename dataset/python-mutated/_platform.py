from typing import Optional
import llnl.util.lang
import spack.error

class NoPlatformError(spack.error.SpackError):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        msg = 'Could not determine a platform for this machine'
        super().__init__(msg)

@llnl.util.lang.lazy_lexicographic_ordering
class Platform:
    """Platform is an abstract class extended by subclasses.

    To add a new type of platform (such as cray_xe), create a subclass and set all the
    class attributes such as priority, front_target, back_target, front_os, back_os.

    Platform also contain a priority class attribute. A lower number signifies higher
    priority. These numbers are arbitrarily set and can be changed though often there
    isn't much need unless a new platform is added and the user wants that to be
    detected first.

    Targets are created inside the platform subclasses. Most architecture (like linux,
    and darwin) will have only one target family (x86_64) but in the case of Cray
    machines, there is both a frontend and backend processor. The user can specify
    which targets are present on front-end and back-end architecture.

    Depending on the platform, operating systems are either autodetected or are
    set. The user can set the frontend and backend operating setting by the class
    attributes front_os and back_os. The operating system will be responsible for
    compiler detection.
    """
    priority: Optional[int] = None
    binary_formats = ['elf']
    front_end: Optional[str] = None
    back_end: Optional[str] = None
    default: Optional[str] = None
    front_os: Optional[str] = None
    back_os: Optional[str] = None
    default_os: Optional[str] = None
    reserved_targets = ['default_target', 'frontend', 'fe', 'backend', 'be']
    reserved_oss = ['default_os', 'frontend', 'fe', 'backend', 'be']

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.targets = {}
        self.operating_sys = {}
        self.name = name

    def add_target(self, name, target):
        if False:
            return 10
        'Used by the platform specific subclass to list available targets.\n        Raises an error if the platform specifies a name\n        that is reserved by spack as an alias.\n        '
        if name in Platform.reserved_targets:
            msg = '{0} is a spack reserved alias and cannot be the name of a target'
            raise ValueError(msg.format(name))
        self.targets[name] = target

    def target(self, name):
        if False:
            while True:
                i = 10
        'This is a getter method for the target dictionary\n        that handles defaulting based on the values provided by default,\n        front-end, and back-end. This can be overwritten\n        by a subclass for which we want to provide further aliasing options.\n        '
        name = str(name)
        if name == 'default_target':
            name = self.default
        elif name == 'frontend' or name == 'fe':
            name = self.front_end
        elif name == 'backend' or name == 'be':
            name = self.back_end
        return self.targets.get(name, None)

    def add_operating_system(self, name, os_class):
        if False:
            print('Hello World!')
        'Add the operating_system class object into the\n        platform.operating_sys dictionary.\n        '
        if name in Platform.reserved_oss:
            msg = '{0} is a spack reserved alias and cannot be the name of an OS'
            raise ValueError(msg.format(name))
        self.operating_sys[name] = os_class

    def operating_system(self, name):
        if False:
            while True:
                i = 10
        if name == 'default_os':
            name = self.default_os
        if name == 'frontend' or name == 'fe':
            name = self.front_os
        if name == 'backend' or name == 'be':
            name = self.back_os
        return self.operating_sys.get(name, None)

    def setup_platform_environment(self, pkg, env):
        if False:
            print('Hello World!')
        'Subclass can override this method if it requires any\n        platform-specific build environment modifications.\n        '
        pass

    @classmethod
    def detect(cls):
        if False:
            for i in range(10):
                print('nop')
        'Return True if the the host platform is detected to be the current\n        Platform class, False otherwise.\n\n        Derived classes are responsible for implementing this method.\n        '
        raise NotImplementedError()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__str__()

    def __str__(self):
        if False:
            return 10
        return self.name

    def _cmp_iter(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.name
        yield self.default
        yield self.front_end
        yield self.back_end
        yield self.default_os
        yield self.front_os
        yield self.back_os

        def targets():
            if False:
                i = 10
                return i + 15
            for t in sorted(self.targets.values()):
                yield t._cmp_iter
        yield targets

        def oses():
            if False:
                while True:
                    i = 10
            for o in sorted(self.operating_sys.values()):
                yield o._cmp_iter
        yield oses