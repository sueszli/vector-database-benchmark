"""This module contains utilities for using multi-methods in
spack. You can think of multi-methods like overloaded methods --
they're methods with the same name, and we need to select a version
of the method based on some criteria.  e.g., for overloaded
methods, you would select a version of the method to call based on
the types of its arguments.

In spack, multi-methods are used to ease the life of package
authors.  They allow methods like install() (or other methods
called by install()) to declare multiple versions to be called when
the package is instantiated with different specs.  e.g., if the
package is built with OpenMPI on x86_64,, you might want to call a
different install method than if it was built for mpich2 on
BlueGene/Q.  Likewise, you might want to do a different type of
install for different versions of the package.

Multi-methods provide a simple decorator-based syntax for this that
avoids overly complicated rat nests of if statements.  Obviously,
depending on the scenario, regular old conditionals might be clearer,
so package authors should use their judgement.
"""
import functools
import inspect
from contextlib import contextmanager
from llnl.util.lang import caller_locals
import spack.directives
import spack.error
from spack.spec import Spec

class MultiMethodMeta(type):
    """This allows us to track the class's dict during instantiation."""
    _locals = None

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        if False:
            while True:
                i = 10
        'Save the dictionary that will be used for the class namespace.'
        MultiMethodMeta._locals = dict()
        return MultiMethodMeta._locals

    def __init__(cls, name, bases, attr_dict):
        if False:
            return 10
        'Clear out the cached locals dict once the class is built.'
        MultiMethodMeta._locals = None
        super(MultiMethodMeta, cls).__init__(name, bases, attr_dict)

class SpecMultiMethod:
    """This implements a multi-method for Spack specs.  Packages are
    instantiated with a particular spec, and you may want to
    execute different versions of methods based on what the spec
    looks like.  For example, you might want to call a different
    version of install() for one platform than you call on another.

    The SpecMultiMethod class implements a callable object that
    handles method dispatch.  When it is called, it looks through
    registered methods and their associated specs, and it tries
    to find one that matches the package's spec.  If it finds one
    (and only one), it will call that method.

    This is intended for use with decorators (see below).  The
    decorator (see docs below) creates SpecMultiMethods and
    registers method versions with them.

    To register a method, you can do something like this:
        mm = SpecMultiMethod()
        mm.register("^chaos_5_x86_64_ib", some_method)

    The object registered needs to be a Spec or some string that
    will parse to be a valid spec.

    When the mm is actually called, it selects a version of the
    method to call based on the sys_type of the object it is
    called on.

    See the docs for decorators below for more details.
    """

    def __init__(self, default=None):
        if False:
            i = 10
            return i + 15
        self.method_list = []
        self.default = default
        if default:
            functools.update_wrapper(self, default)

    def register(self, spec, method):
        if False:
            print('Hello World!')
        'Register a version of a method for a particular spec.'
        self.method_list.append((spec, method))
        if not hasattr(self, '__name__'):
            functools.update_wrapper(self, method)
        else:
            assert self.__name__ == method.__name__

    def __get__(self, obj, objtype):
        if False:
            print('Hello World!')
        'This makes __call__ support instance methods.'
        wrapped_method = self.method_list[0][1]
        func = functools.wraps(wrapped_method)(functools.partial(self.__call__, obj))
        return func

    def _get_method_by_spec(self, spec):
        if False:
            for i in range(10):
                print('nop')
        'Find the method of this SpecMultiMethod object that satisfies the\n        given spec, if one exists\n        '
        for (condition, method) in self.method_list:
            if spec.satisfies(condition):
                return method
        return self.default or None

    def __call__(self, package_or_builder_self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "Find the first method with a spec that matches the\n        package's spec.  If none is found, call the default\n        or if there is none, then raise a NoSuchMethodError.\n        "
        spec_method = self._get_method_by_spec(package_or_builder_self.spec)
        if spec_method:
            return spec_method(package_or_builder_self, *args, **kwargs)
        for cls in inspect.getmro(package_or_builder_self.__class__)[1:]:
            superself = cls.__dict__.get(self.__name__, None)
            if isinstance(superself, SpecMultiMethod):
                superself_method = superself._get_method_by_spec(package_or_builder_self.spec)
                if superself_method:
                    return superself_method(package_or_builder_self, *args, **kwargs)
            elif superself:
                return superself(package_or_builder_self, *args, **kwargs)
        raise NoSuchMethodError(type(package_or_builder_self), self.__name__, package_or_builder_self.spec, [m[0] for m in self.method_list])

class when:

    def __init__(self, condition):
        if False:
            while True:
                i = 10
        'Can be used both as a decorator, for multimethods, or as a context\n        manager to group ``when=`` arguments together.\n\n        Examples are given in the docstrings below.\n\n        Args:\n            condition (str): condition to be met\n        '
        if isinstance(condition, bool):
            self.spec = Spec() if condition else None
        else:
            self.spec = Spec(condition)

    def __call__(self, method):
        if False:
            i = 10
            return i + 15
        "This annotation lets packages declare multiple versions of\n        methods like install() that depend on the package's spec.\n\n        For example:\n\n           .. code-block:: python\n\n              class SomePackage(Package):\n                  ...\n\n                  def install(self, prefix):\n                      # Do default install\n\n                  @when('target=x86_64:')\n                  def install(self, prefix):\n                      # This will be executed instead of the default install if\n                      # the package's target is in the x86_64 family.\n\n                  @when('target=ppc64:')\n                  def install(self, prefix):\n                      # This will be executed if the package's target is in\n                      # the ppc64 family\n\n           This allows each package to have a default version of install() AND\n           specialized versions for particular platforms.  The version that is\n           called depends on the architecutre of the instantiated package.\n\n           Note that this works for methods other than install, as well.  So,\n           if you only have part of the install that is platform specific, you\n           could do this:\n\n           .. code-block:: python\n\n              class SomePackage(Package):\n                  ...\n                  # virtual dependence on MPI.\n                  # could resolve to mpich, mpich2, OpenMPI\n                  depends_on('mpi')\n\n                  def setup(self):\n                      # do nothing in the default case\n                      pass\n\n                  @when('^openmpi')\n                  def setup(self):\n                      # do something special when this is built with OpenMPI for\n                      # its MPI implementations.\n\n\n                  def install(self, prefix):\n                      # Do common install stuff\n                      self.setup()\n                      # Do more common install stuff\n\n           Note that the default version of decorated methods must\n           *always* come first.  Otherwise it will override all of the\n           platform-specific versions.  There's not much we can do to get\n           around this because of the way decorators work.\n        "
        if MultiMethodMeta._locals is None:
            MultiMethodMeta._locals = caller_locals()
        original_method = MultiMethodMeta._locals.get(method.__name__)
        if not isinstance(original_method, SpecMultiMethod):
            original_method = SpecMultiMethod(original_method)
        if self.spec is not None:
            original_method.register(self.spec, method)
        return original_method

    def __enter__(self):
        if False:
            while True:
                i = 10
        "Inject the constraint spec into the `when=` argument of directives\n        in the context.\n\n        This context manager allows you to write:\n\n            with when('+nvptx'):\n                conflicts('@:6', msg='NVPTX only supported from gcc 7')\n                conflicts('languages=ada')\n                conflicts('languages=brig')\n\n        instead of writing:\n\n             conflicts('@:6', when='+nvptx', msg='NVPTX only supported from gcc 7')\n             conflicts('languages=ada', when='+nvptx')\n             conflicts('languages=brig', when='+nvptx')\n\n        Context managers can be nested (but this is not recommended for readability)\n        and add their constraint to whatever may be already present in the directive\n        `when=` argument.\n        "
        spack.directives.DirectiveMeta.push_to_context(str(self.spec))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        spack.directives.DirectiveMeta.pop_from_context()

@contextmanager
def default_args(**kwargs):
    if False:
        print('Hello World!')
    spack.directives.DirectiveMeta.push_default_args(kwargs)
    yield
    spack.directives.DirectiveMeta.pop_default_args()

class MultiMethodError(spack.error.SpackError):
    """Superclass for multimethod dispatch errors"""

    def __init__(self, message):
        if False:
            return 10
        super().__init__(message)

class NoSuchMethodError(spack.error.SpackError):
    """Raised when we can't find a version of a multi-method."""

    def __init__(self, cls, method_name, spec, possible_specs):
        if False:
            return 10
        super().__init__('Package %s does not support %s called with %s.  Options are: %s' % (cls.__name__, method_name, spec, ', '.join((str(s) for s in possible_specs))))