"""Implementation of special members of Python's abc library."""
from pytype.abstract import abstract
from pytype.overlays import overlay
from pytype.overlays import special_builtins

def _set_abstract(args, argname):
    if False:
        while True:
            i = 10
    if args.posargs:
        func_var = args.posargs[0]
    else:
        func_var = args.namedargs[argname]
    for func in func_var.data:
        if isinstance(func, abstract.FUNCTION_TYPES):
            func.is_abstract = True
    return func_var

class ABCOverlay(overlay.Overlay):
    """A custom overlay for the 'abc' module."""

    def __init__(self, ctx):
        if False:
            while True:
                i = 10
        member_map = {'abstractclassmethod': AbstractClassMethod.make, 'abstractmethod': AbstractMethod.make, 'abstractproperty': AbstractProperty.make, 'abstractstaticmethod': AbstractStaticMethod.make, 'ABCMeta': overlay.add_name('ABCMeta', special_builtins.Type.make_alias)}
        ast = ctx.loader.import_name('abc')
        super().__init__(ctx, 'abc', member_map, ast)

class AbstractClassMethod(special_builtins.ClassMethod):
    """Implements abc.abstractclassmethod."""

    @classmethod
    def make(cls, ctx, module):
        if False:
            for i in range(10):
                print('nop')
        return super().make_alias('abstractclassmethod', ctx, module)

    def call(self, node, func, args, alias_map=None):
        if False:
            return 10
        _ = _set_abstract(args, 'callable')
        return super().call(node, func, args, alias_map)

class AbstractMethod(abstract.PyTDFunction):
    """Implements the @abc.abstractmethod decorator."""

    @classmethod
    def make(cls, ctx, module):
        if False:
            print('Hello World!')
        return super().make('abstractmethod', ctx, module)

    def call(self, node, func, args, alias_map=None):
        if False:
            return 10
        'Marks that the given function is abstract.'
        del func, alias_map
        self.match_args(node, args)
        return (node, _set_abstract(args, 'funcobj'))

class AbstractProperty(special_builtins.Property):
    """Implements the @abc.abstractproperty decorator."""

    @classmethod
    def make(cls, ctx, module):
        if False:
            for i in range(10):
                print('nop')
        return super().make_alias('abstractproperty', ctx, module)

    def call(self, node, func, args, alias_map=None):
        if False:
            return 10
        property_args = self._get_args(args)
        for v in property_args.values():
            for b in v.bindings:
                f = b.data
                if isinstance(f, abstract.Function):
                    f.is_abstract = True
        return (node, special_builtins.PropertyInstance(self.ctx, self.name, self, **property_args).to_variable(node))

class AbstractStaticMethod(special_builtins.StaticMethod):
    """Implements abc.abstractstaticmethod."""

    @classmethod
    def make(cls, ctx, module):
        if False:
            while True:
                i = 10
        return super().make_alias('abstractstaticmethod', ctx, module)

    def call(self, node, func, args, alias_map=None):
        if False:
            print('Hello World!')
        _ = _set_abstract(args, 'callable')
        return super().call(node, func, args, alias_map)