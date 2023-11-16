"""Intermediate representation of classes."""
from __future__ import annotations
from typing import List, NamedTuple
from mypyc.common import PROPSET_PREFIX, JsonDict
from mypyc.ir.func_ir import FuncDecl, FuncIR, FuncSignature
from mypyc.ir.ops import DeserMaps, Value
from mypyc.ir.rtypes import RInstance, RType, deserialize_type
from mypyc.namegen import NameGenerator, exported_name

class VTableMethod(NamedTuple):
    cls: 'ClassIR'
    name: str
    method: FuncIR
    shadow_method: FuncIR | None
VTableEntries = List[VTableMethod]

class ClassIR:
    """Intermediate representation of a class.

    This also describes the runtime structure of native instances.
    """

    def __init__(self, name: str, module_name: str, is_trait: bool=False, is_generated: bool=False, is_abstract: bool=False, is_ext_class: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.module_name = module_name
        self.is_trait = is_trait
        self.is_generated = is_generated
        self.is_abstract = is_abstract
        self.is_ext_class = is_ext_class
        self.is_augmented = False
        self.inherits_python = False
        self.has_dict = False
        self.allow_interpreted_subclasses = False
        self.needs_getseters = False
        self._serializable = False
        self.builtin_base: str | None = None
        self.ctor = FuncDecl(name, None, module_name, FuncSignature([], RInstance(self)))
        self.attributes: dict[str, RType] = {}
        self.deletable: list[str] = []
        self.method_decls: dict[str, FuncDecl] = {}
        self.methods: dict[str, FuncIR] = {}
        self.glue_methods: dict[tuple[ClassIR, str], FuncIR] = {}
        self.properties: dict[str, tuple[FuncIR, FuncIR | None]] = {}
        self.property_types: dict[str, RType] = {}
        self.vtable: dict[str, int] | None = None
        self.vtable_entries: VTableEntries = []
        self.trait_vtables: dict[ClassIR, VTableEntries] = {}
        self.base: ClassIR | None = None
        self.traits: list[ClassIR] = []
        self.mro: list[ClassIR] = [self]
        self.base_mro: list[ClassIR] = [self]
        self.children: list[ClassIR] | None = []
        self.attrs_with_defaults: set[str] = set()
        self._always_initialized_attrs: set[str] = set()
        self._sometimes_initialized_attrs: set[str] = set()
        self.init_self_leak = False
        self.bitmap_attrs: list[str] = []

    def __repr__(self) -> str:
        if False:
            return 10
        return 'ClassIR(name={self.name}, module_name={self.module_name}, is_trait={self.is_trait}, is_generated={self.is_generated}, is_abstract={self.is_abstract}, is_ext_class={self.is_ext_class})'.format(self=self)

    @property
    def fullname(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{self.module_name}.{self.name}'

    def real_base(self) -> ClassIR | None:
        if False:
            print('Hello World!')
        'Return the actual concrete base class, if there is one.'
        if len(self.mro) > 1 and (not self.mro[1].is_trait):
            return self.mro[1]
        return None

    def vtable_entry(self, name: str) -> int:
        if False:
            print('Hello World!')
        assert self.vtable is not None, 'vtable not computed yet'
        assert name in self.vtable, f'{self.name!r} has no attribute {name!r}'
        return self.vtable[name]

    def attr_details(self, name: str) -> tuple[RType, ClassIR]:
        if False:
            i = 10
            return i + 15
        for ir in self.mro:
            if name in ir.attributes:
                return (ir.attributes[name], ir)
            if name in ir.property_types:
                return (ir.property_types[name], ir)
        raise KeyError(f'{self.name!r} has no attribute {name!r}')

    def attr_type(self, name: str) -> RType:
        if False:
            for i in range(10):
                print('nop')
        return self.attr_details(name)[0]

    def method_decl(self, name: str) -> FuncDecl:
        if False:
            for i in range(10):
                print('nop')
        for ir in self.mro:
            if name in ir.method_decls:
                return ir.method_decls[name]
        raise KeyError(f'{self.name!r} has no attribute {name!r}')

    def method_sig(self, name: str) -> FuncSignature:
        if False:
            print('Hello World!')
        return self.method_decl(name).sig

    def has_method(self, name: str) -> bool:
        if False:
            return 10
        try:
            self.method_decl(name)
        except KeyError:
            return False
        return True

    def is_method_final(self, name: str) -> bool:
        if False:
            print('Hello World!')
        subs = self.subclasses()
        if subs is None:
            return False
        if self.has_method(name):
            method_decl = self.method_decl(name)
            for subc in subs:
                if subc.method_decl(name) != method_decl:
                    return False
            return True
        else:
            return not any((subc.has_method(name) for subc in subs))

    def has_attr(self, name: str) -> bool:
        if False:
            return 10
        try:
            self.attr_type(name)
        except KeyError:
            return False
        return True

    def is_deletable(self, name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return any((name in ir.deletable for ir in self.mro))

    def is_always_defined(self, name: str) -> bool:
        if False:
            while True:
                i = 10
        if self.is_deletable(name):
            return False
        return name in self._always_initialized_attrs

    def name_prefix(self, names: NameGenerator) -> str:
        if False:
            return 10
        return names.private_name(self.module_name, self.name)

    def struct_name(self, names: NameGenerator) -> str:
        if False:
            i = 10
            return i + 15
        return f'{exported_name(self.fullname)}Object'

    def get_method_and_class(self, name: str, *, prefer_method: bool=False) -> tuple[FuncIR, ClassIR] | None:
        if False:
            i = 10
            return i + 15
        for ir in self.mro:
            if name in ir.methods:
                func_ir = ir.methods[name]
                if not prefer_method and func_ir.decl.implicit:
                    return None
                return (func_ir, ir)
        return None

    def get_method(self, name: str, *, prefer_method: bool=False) -> FuncIR | None:
        if False:
            return 10
        res = self.get_method_and_class(name, prefer_method=prefer_method)
        return res[0] if res else None

    def has_method_decl(self, name: str) -> bool:
        if False:
            while True:
                i = 10
        return any((name in ir.method_decls for ir in self.mro))

    def has_no_subclasses(self) -> bool:
        if False:
            print('Hello World!')
        return self.children == [] and (not self.allow_interpreted_subclasses)

    def subclasses(self) -> set[ClassIR] | None:
        if False:
            for i in range(10):
                print('nop')
        'Return all subclasses of this class, both direct and indirect.\n\n        Return None if it is impossible to identify all subclasses, for example\n        because we are performing separate compilation.\n        '
        if self.children is None or self.allow_interpreted_subclasses:
            return None
        result = set(self.children)
        for child in self.children:
            if child.children:
                child_subs = child.subclasses()
                if child_subs is None:
                    return None
                result.update(child_subs)
        return result

    def concrete_subclasses(self) -> list[ClassIR] | None:
        if False:
            for i in range(10):
                print('nop')
        'Return all concrete (i.e. non-trait and non-abstract) subclasses.\n\n        Include both direct and indirect subclasses. Place classes with no children first.\n        '
        subs = self.subclasses()
        if subs is None:
            return None
        concrete = {c for c in subs if not (c.is_trait or c.is_abstract)}
        return sorted(concrete, key=lambda c: (len(c.children or []), c.name))

    def is_serializable(self) -> bool:
        if False:
            print('Hello World!')
        return any((ci._serializable for ci in self.mro))

    def serialize(self) -> JsonDict:
        if False:
            while True:
                i = 10
        return {'name': self.name, 'module_name': self.module_name, 'is_trait': self.is_trait, 'is_ext_class': self.is_ext_class, 'is_abstract': self.is_abstract, 'is_generated': self.is_generated, 'is_augmented': self.is_augmented, 'inherits_python': self.inherits_python, 'has_dict': self.has_dict, 'allow_interpreted_subclasses': self.allow_interpreted_subclasses, 'needs_getseters': self.needs_getseters, '_serializable': self._serializable, 'builtin_base': self.builtin_base, 'ctor': self.ctor.serialize(), 'attributes': [(k, t.serialize()) for (k, t) in self.attributes.items()], 'method_decls': [(k, d.id if k in self.methods else d.serialize()) for (k, d) in self.method_decls.items()], 'methods': [(k, m.id) for (k, m) in self.methods.items()], 'glue_methods': [((cir.fullname, k), m.id) for ((cir, k), m) in self.glue_methods.items()], 'property_types': [(k, t.serialize()) for (k, t) in self.property_types.items()], 'properties': list(self.properties), 'vtable': self.vtable, 'vtable_entries': serialize_vtable(self.vtable_entries), 'trait_vtables': [(cir.fullname, serialize_vtable(v)) for (cir, v) in self.trait_vtables.items()], 'base': self.base.fullname if self.base else None, 'traits': [cir.fullname for cir in self.traits], 'mro': [cir.fullname for cir in self.mro], 'base_mro': [cir.fullname for cir in self.base_mro], 'children': [cir.fullname for cir in self.children] if self.children is not None else None, 'deletable': self.deletable, 'attrs_with_defaults': sorted(self.attrs_with_defaults), '_always_initialized_attrs': sorted(self._always_initialized_attrs), '_sometimes_initialized_attrs': sorted(self._sometimes_initialized_attrs), 'init_self_leak': self.init_self_leak}

    @classmethod
    def deserialize(cls, data: JsonDict, ctx: DeserMaps) -> ClassIR:
        if False:
            return 10
        fullname = data['module_name'] + '.' + data['name']
        assert fullname in ctx.classes, 'Class %s not in deser class map' % fullname
        ir = ctx.classes[fullname]
        ir.is_trait = data['is_trait']
        ir.is_generated = data['is_generated']
        ir.is_abstract = data['is_abstract']
        ir.is_ext_class = data['is_ext_class']
        ir.is_augmented = data['is_augmented']
        ir.inherits_python = data['inherits_python']
        ir.has_dict = data['has_dict']
        ir.allow_interpreted_subclasses = data['allow_interpreted_subclasses']
        ir.needs_getseters = data['needs_getseters']
        ir._serializable = data['_serializable']
        ir.builtin_base = data['builtin_base']
        ir.ctor = FuncDecl.deserialize(data['ctor'], ctx)
        ir.attributes = {k: deserialize_type(t, ctx) for (k, t) in data['attributes']}
        ir.method_decls = {k: ctx.functions[v].decl if isinstance(v, str) else FuncDecl.deserialize(v, ctx) for (k, v) in data['method_decls']}
        ir.methods = {k: ctx.functions[v] for (k, v) in data['methods']}
        ir.glue_methods = {(ctx.classes[c], k): ctx.functions[v] for ((c, k), v) in data['glue_methods']}
        ir.property_types = {k: deserialize_type(t, ctx) for (k, t) in data['property_types']}
        ir.properties = {k: (ir.methods[k], ir.methods.get(PROPSET_PREFIX + k)) for k in data['properties']}
        ir.vtable = data['vtable']
        ir.vtable_entries = deserialize_vtable(data['vtable_entries'], ctx)
        ir.trait_vtables = {ctx.classes[k]: deserialize_vtable(v, ctx) for (k, v) in data['trait_vtables']}
        base = data['base']
        ir.base = ctx.classes[base] if base else None
        ir.traits = [ctx.classes[s] for s in data['traits']]
        ir.mro = [ctx.classes[s] for s in data['mro']]
        ir.base_mro = [ctx.classes[s] for s in data['base_mro']]
        ir.children = data['children'] and [ctx.classes[s] for s in data['children']]
        ir.deletable = data['deletable']
        ir.attrs_with_defaults = set(data['attrs_with_defaults'])
        ir._always_initialized_attrs = set(data['_always_initialized_attrs'])
        ir._sometimes_initialized_attrs = set(data['_sometimes_initialized_attrs'])
        ir.init_self_leak = data['init_self_leak']
        return ir

class NonExtClassInfo:
    """Information needed to construct a non-extension class (Python class).

    Includes the class dictionary, a tuple of base classes,
    the class annotations dictionary, and the metaclass.
    """

    def __init__(self, dict: Value, bases: Value, anns: Value, metaclass: Value) -> None:
        if False:
            while True:
                i = 10
        self.dict = dict
        self.bases = bases
        self.anns = anns
        self.metaclass = metaclass

def serialize_vtable_entry(entry: VTableMethod) -> JsonDict:
    if False:
        return 10
    return {'.class': 'VTableMethod', 'cls': entry.cls.fullname, 'name': entry.name, 'method': entry.method.decl.id, 'shadow_method': entry.shadow_method.decl.id if entry.shadow_method else None}

def serialize_vtable(vtable: VTableEntries) -> list[JsonDict]:
    if False:
        while True:
            i = 10
    return [serialize_vtable_entry(v) for v in vtable]

def deserialize_vtable_entry(data: JsonDict, ctx: DeserMaps) -> VTableMethod:
    if False:
        print('Hello World!')
    if data['.class'] == 'VTableMethod':
        return VTableMethod(ctx.classes[data['cls']], data['name'], ctx.functions[data['method']], ctx.functions[data['shadow_method']] if data['shadow_method'] else None)
    assert False, 'Bogus vtable .class: %s' % data['.class']

def deserialize_vtable(data: list[JsonDict], ctx: DeserMaps) -> VTableEntries:
    if False:
        return 10
    return [deserialize_vtable_entry(x, ctx) for x in data]

def all_concrete_classes(class_ir: ClassIR) -> list[ClassIR] | None:
    if False:
        return 10
    'Return all concrete classes among the class itself and its subclasses.'
    concrete = class_ir.concrete_subclasses()
    if concrete is None:
        return None
    if not (class_ir.is_abstract or class_ir.is_trait):
        concrete.append(class_ir)
    return concrete