"""Same type check for RTypes."""
from __future__ import annotations
from mypyc.ir.func_ir import FuncSignature
from mypyc.ir.rtypes import RArray, RInstance, RPrimitive, RStruct, RTuple, RType, RTypeVisitor, RUnion, RVoid

def is_same_type(a: RType, b: RType) -> bool:
    if False:
        while True:
            i = 10
    return a.accept(SameTypeVisitor(b))

def is_same_signature(a: FuncSignature, b: FuncSignature) -> bool:
    if False:
        while True:
            i = 10
    return len(a.args) == len(b.args) and is_same_type(a.ret_type, b.ret_type) and all((is_same_type(t1.type, t2.type) and t1.name == t2.name for (t1, t2) in zip(a.args, b.args)))

def is_same_method_signature(a: FuncSignature, b: FuncSignature) -> bool:
    if False:
        i = 10
        return i + 15
    return len(a.args) == len(b.args) and is_same_type(a.ret_type, b.ret_type) and all((is_same_type(t1.type, t2.type) and (t1.pos_only and t2.pos_only or t1.name == t2.name) and (t1.optional == t2.optional) for (t1, t2) in zip(a.args[1:], b.args[1:])))

class SameTypeVisitor(RTypeVisitor[bool]):

    def __init__(self, right: RType) -> None:
        if False:
            print('Hello World!')
        self.right = right

    def visit_rinstance(self, left: RInstance) -> bool:
        if False:
            while True:
                i = 10
        return isinstance(self.right, RInstance) and left.name == self.right.name

    def visit_runion(self, left: RUnion) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.right, RUnion):
            items = list(self.right.items)
            for left_item in left.items:
                for (j, right_item) in enumerate(items):
                    if is_same_type(left_item, right_item):
                        del items[j]
                        break
                else:
                    return False
            return not items
        return False

    def visit_rprimitive(self, left: RPrimitive) -> bool:
        if False:
            print('Hello World!')
        return left is self.right

    def visit_rtuple(self, left: RTuple) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(self.right, RTuple) and len(self.right.types) == len(left.types) and all((is_same_type(t1, t2) for (t1, t2) in zip(left.types, self.right.types)))

    def visit_rstruct(self, left: RStruct) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return isinstance(self.right, RStruct) and self.right.name == left.name

    def visit_rarray(self, left: RArray) -> bool:
        if False:
            i = 10
            return i + 15
        return left == self.right

    def visit_rvoid(self, left: RVoid) -> bool:
        if False:
            while True:
                i = 10
        return isinstance(self.right, RVoid)