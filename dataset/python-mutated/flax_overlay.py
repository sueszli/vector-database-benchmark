"""Support for flax.struct dataclasses."""
from pytype.abstract import abstract
from pytype.abstract import abstract_utils
from pytype.abstract import function
from pytype.overlays import classgen
from pytype.overlays import dataclass_overlay
from pytype.overlays import overlay
from pytype.pytd import pytd

class DataclassOverlay(overlay.Overlay):
    """A custom overlay for the 'flax.struct' module."""

    def __init__(self, ctx):
        if False:
            while True:
                i = 10
        member_map = {'dataclass': Dataclass.make}
        ast = ctx.loader.import_name('flax.struct')
        super().__init__(ctx, 'flax.struct', member_map, ast)

class Dataclass(dataclass_overlay.Dataclass):
    """Implements the @dataclass decorator."""

    def decorate(self, node, cls):
        if False:
            return 10
        super().decorate(node, cls)
        if not isinstance(cls, abstract.InterpreterClass):
            return
        cls.members['replace'] = classgen.make_replace_method(self.ctx, node, cls)

class LinenOverlay(overlay.Overlay):
    """A custom overlay for the 'flax.linen' module."""

    def __init__(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        member_map = {'Module': Module}
        ast = ctx.loader.import_name('flax.linen')
        super().__init__(ctx, 'flax.linen', member_map, ast)

class LinenModuleOverlay(overlay.Overlay):
    """A custom overlay for the 'flax.linen.module' module."""

    def __init__(self, ctx):
        if False:
            return 10
        member_map = {'Module': Module}
        ast = ctx.loader.import_name('flax.linen.module')
        super().__init__(ctx, 'flax.linen.module', member_map, ast)

class ModuleDataclass(dataclass_overlay.Dataclass):
    """Dataclass with automatic 'name' and 'parent' members."""

    def _add_implicit_field(self, node, cls_locals, key, typ):
        if False:
            i = 10
            return i + 15
        if key in cls_locals:
            self.ctx.errorlog.invalid_annotation(self.ctx.vm.frames, None, name=key, details=f"flax.linen.Module defines field '{key}' implicitly")
        default = typ.to_variable(node)
        cls_locals[key] = abstract_utils.Local(node, None, typ, default, self.ctx)

    def get_class_locals(self, node, cls):
        if False:
            while True:
                i = 10
        cls_locals = super().get_class_locals(node, cls)
        initvar = self.ctx.convert.lookup_value('dataclasses', 'InitVar')

        def make_initvar(t):
            if False:
                i = 10
                return i + 15
            return abstract.ParameterizedClass(initvar, {abstract_utils.T: t}, self.ctx)
        name_type = make_initvar(self.ctx.convert.str_type)
        parent_type = make_initvar(self.ctx.convert.unsolvable)
        self._add_implicit_field(node, cls_locals, 'name', name_type)
        self._add_implicit_field(node, cls_locals, 'parent', parent_type)
        return cls_locals

    def decorate(self, node, cls):
        if False:
            i = 10
            return i + 15
        super().decorate(node, cls)
        if not isinstance(cls, abstract.InterpreterClass):
            return
        cls.members['replace'] = classgen.make_replace_method(self.ctx, node, cls)

class Module(abstract.PyTDClass):
    """Construct a dataclass for any class inheriting from Module."""
    IMPLICIT_FIELDS = ('name', 'parent')
    _MODULE = 'flax.linen.module'

    def __init__(self, ctx, module):
        if False:
            i = 10
            return i + 15
        del module
        pytd_cls = ctx.loader.lookup_pytd(self._MODULE, 'Module')
        if isinstance(pytd_cls, pytd.Constant):
            pytd_cls = ctx.convert.constant_to_value(pytd_cls).pytd_cls
        super().__init__('Module', pytd_cls, ctx)

    def init_subclass(self, node, cls):
        if False:
            i = 10
            return i + 15
        cls.additional_init_methods.append('setup')
        dc = ModuleDataclass.make(self.ctx)
        cls_var = cls.to_variable(node)
        args = function.Args(posargs=(cls_var,), namedargs={})
        (node, _) = dc.call(node, None, args)
        return node

    def get_instance_type(self, node=None, instance=None, seen=None, view=None):
        if False:
            return 10
        'Get the type an instance of us would have.'
        return pytd.NamedType(self.full_name)

    @property
    def full_name(self):
        if False:
            return 10
        return f'{self._MODULE}.{self.name}'

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'Overlay({self.full_name})'