from collections import OrderedDict
from weakref import WeakKeyDictionary
from .compiler import Compiler

class ShaderObject(object):
    """Base class for all objects that may be included in a GLSL program
    (Functions, Variables, Expressions).

    Shader objects have a *definition* that defines the object in GLSL, an
    *expression* that is used to reference the object, and a set of
    *dependencies* that must be declared before the object is used.

    Dependencies are tracked hierarchically such that changes to any object
    will be propagated up the dependency hierarchy to trigger a recompile.
    """

    @classmethod
    def create(self, obj, ref=None):
        if False:
            return 10
        'Convert *obj* to a new ShaderObject. If the output is a Variable\n        with no name, then set its name using *ref*.\n        '
        if isinstance(ref, Variable):
            ref = ref.name
        elif isinstance(ref, str) and ref.startswith('gl_'):
            ref = ref[3:].lower()
        if hasattr(obj, '_shader_object'):
            obj = obj._shader_object()
        if isinstance(obj, ShaderObject):
            if isinstance(obj, Variable) and obj.name is None:
                obj.name = ref
        elif isinstance(obj, str):
            obj = TextExpression(obj)
        else:
            obj = Variable(ref, obj)
            if obj.vtype and obj.vtype[0] in 'auv':
                obj.name = obj.vtype[0] + '_' + obj.name
        return obj

    def __init__(self):
        if False:
            print('Hello World!')
        self._deps = OrderedDict()
        self._dependents = WeakKeyDictionary()

    @property
    def name(self):
        if False:
            return 10
        'The name of this shader object.'
        return None

    @property
    def version_pragma(self):
        if False:
            while True:
                i = 10
        'Return version number and extra qualifiers from pragma if present.'
        return None

    def definition(self, obj_names, version, shader):
        if False:
            print('Hello World!')
        'Return the GLSL definition for this object. Use *obj_names* to\n        determine the names of dependencies, and *version* (number, qualifier)\n        to adjust code output.\n        '
        return None

    def expression(self, obj_names):
        if False:
            print('Hello World!')
        'Return the GLSL expression used to reference this object inline.'
        return obj_names[self]

    def dependencies(self, sort=False):
        if False:
            for i in range(10):
                print('nop')
        'Return all dependencies required to use this object. The last item\n        in the list is *self*.\n        '
        alldeps = []
        if sort:

            def key(obj):
                if False:
                    print('Hello World!')
                if not isinstance(obj, Variable):
                    return (0, 0)
                else:
                    return (1, obj.vtype)
            deps = sorted(self._deps, key=key)
        else:
            deps = self._deps
        for dep in deps:
            alldeps.extend(dep.dependencies(sort=sort))
        alldeps.append(self)
        return alldeps

    def static_names(self):
        if False:
            while True:
                i = 10
        "Return a list of names that are declared in this object's\n        definition (not including the name of the object itself).\n\n        These names will be reserved by the compiler when automatically\n        determining object names.\n        "
        return []

    def _add_dep(self, dep):
        if False:
            print('Hello World!')
        'Increment the reference count for *dep*. If this is a new\n        dependency, then connect to its *changed* event.\n        '
        if dep in self._deps:
            self._deps[dep] += 1
        else:
            self._deps[dep] = 1
            dep._dependents[self] = None

    def _remove_dep(self, dep):
        if False:
            while True:
                i = 10
        'Decrement the reference count for *dep*. If the reference count\n        reaches 0, then the dependency is removed and its *changed* event is\n        disconnected.\n        '
        refcount = self._deps[dep]
        if refcount == 1:
            self._deps.pop(dep)
            dep._dependents.pop(self)
        else:
            self._deps[dep] -= 1

    def _dep_changed(self, dep, code_changed=False, value_changed=False):
        if False:
            i = 10
            return i + 15
        "Called when a dependency's expression has changed."
        self.changed(code_changed, value_changed)

    def changed(self, code_changed=False, value_changed=False):
        if False:
            return 10
        'Inform dependents that this shaderobject has changed.'
        for d in self._dependents:
            d._dep_changed(self, code_changed=code_changed, value_changed=value_changed)

    def compile(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a compilation of this object and its dependencies.\n\n        Note: this is mainly for debugging purposes; the names in this code\n        are not guaranteed to match names in any other compilations. Use\n        Compiler directly to ensure consistent naming across multiple objects.\n        '
        compiler = Compiler(obj=self)
        return compiler.compile()['obj']

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.name is not None:
            return '<%s "%s" at 0x%x>' % (self.__class__.__name__, self.name, id(self))
        else:
            return '<%s at 0x%x>' % (self.__class__.__name__, id(self))
from .variable import Variable
from .expression import TextExpression