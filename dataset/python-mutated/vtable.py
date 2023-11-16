"""Compute vtables of native (extension) classes."""
from __future__ import annotations
import itertools
from mypyc.ir.class_ir import ClassIR, VTableEntries, VTableMethod
from mypyc.sametype import is_same_method_signature

def compute_vtable(cls: ClassIR) -> None:
    if False:
        while True:
            i = 10
    'Compute the vtable structure for a class.'
    if cls.vtable is not None:
        return
    if not cls.is_generated:
        cls.has_dict = any((x.inherits_python for x in cls.mro))
    for t in cls.mro[1:]:
        compute_vtable(t)
        if not t.is_trait:
            continue
        for (name, typ) in t.attributes.items():
            if not cls.is_trait and (not any((name in b.attributes for b in cls.base_mro))):
                cls.attributes[name] = typ
    cls.vtable = {}
    if cls.base:
        assert cls.base.vtable is not None
        cls.vtable.update(cls.base.vtable)
        cls.vtable_entries = specialize_parent_vtable(cls, cls.base)
    entries = cls.vtable_entries
    all_traits = [t for t in cls.mro if t.is_trait]
    for t in [cls] + cls.traits:
        for fn in itertools.chain(t.methods.values()):
            if fn == cls.get_method(fn.name, prefer_method=True):
                cls.vtable[fn.name] = len(entries)
                shadow = cls.glue_methods.get((cls, fn.name))
                entries.append(VTableMethod(t, fn.name, fn, shadow))
    if not cls.is_trait:
        for trait in all_traits:
            compute_vtable(trait)
            cls.trait_vtables[trait] = specialize_parent_vtable(cls, trait)

def specialize_parent_vtable(cls: ClassIR, parent: ClassIR) -> VTableEntries:
    if False:
        i = 10
        return i + 15
    'Generate the part of a vtable corresponding to a parent class or trait'
    updated = []
    for entry in parent.vtable_entries:
        orig_parent_method = entry.cls.get_method(entry.name, prefer_method=True)
        assert orig_parent_method
        method_cls = cls.get_method_and_class(entry.name, prefer_method=True)
        if method_cls:
            (child_method, defining_cls) = method_cls
            if is_same_method_signature(orig_parent_method.sig, child_method.sig) or orig_parent_method.name == '__init__':
                entry = VTableMethod(entry.cls, entry.name, child_method, entry.shadow_method)
            else:
                entry = VTableMethod(entry.cls, entry.name, defining_cls.glue_methods[entry.cls, entry.name], entry.shadow_method)
        updated.append(entry)
    return updated