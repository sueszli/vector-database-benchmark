import dis
from compiler.static.types import _TMP_VAR_PREFIX, TypedSyntaxError
from unittest import skip
from .common import StaticTestBase

class RefineFieldsTests(StaticTestBase):

    def test_can_refine_loaded_field(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n\n                def f(self) -> None:\n                   if self.x is not None:\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'int')

    def test_cannot_refine_property(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                @property\n                def x(self) -> int | None:\n                    return 42\n\n                def f(self) -> None:\n                   if self.x is not None:\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[int]')

    def test_refinements_are_invalidated_with_calls(self) -> None:
        if False:
            return 10
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n\n                def f(self) -> None:\n                   if self.x is not None:\n                       open("a.py")\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[int]')

    def test_refinements_are_invalidated_with_stores(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n\n                def f(self) -> None:\n                   if self.x is not None:\n                       self.x = None\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Exact[None]')

    def test_refinements_restored_after_write_with_interfering_calls(self) -> None:
        if False:
            return 10
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n\n                def f(self) -> None:\n                   if self.x is not None:\n                       self.x = None\n                       open("a.py")\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[int]')

    def test_refinements_are_not_invalidated_with_known_safe_attr_stores(self) -> None:
        if False:
            return 10
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n                    self.y: None = None\n\n                def f(self) -> None:\n                   if self.x is not None:\n                       self.y = None\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'int')

    def test_refinements_are_invalidated_with_unknown_attr_stores(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n\n                def f(self, other) -> None:\n                   if self.x is not None:\n                       other.y = None\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[int]')

    def test_refinements_are_preserved_with_simple_assignments(self) -> None:
        if False:
            return 10
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n                    self.y: None = None\n\n                def f(self) -> None:\n                   if self.x is not None:\n                       a = self.x\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'int')

    def test_isinstance_refinement(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n                    self.y: None = None\n\n                def f(self) -> None:\n                   if isinstance(self.x, int):\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'int')

    def test_refinements_cleared_when_merging_branches(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n\n                def f(self) -> None:\n                   if self.x is not None:\n                      pass\n                   reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[int]')

    def test_type_not_refined_outside_while_loop(self) -> None:
        if False:
            return 10
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n\n                def f(self) -> None:\n                   while self.x is None:\n                       pass\n                   reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[int]')

    def test_type_not_refined_when_visiting_name(self) -> None:
        if False:
            return 10
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n\n                def f(self) -> None:\n                   if self.x:\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[int]')

    def test_type_not_refined_for_attribute_test_with_custom_bool(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            class D:\n                def __bool__(self) -> bool:\n                    return True\n\n            class C:\n                def __init__(self) -> None:\n                    self.x: D | None = None\n\n                def f(self) -> None:\n                   if self.x:\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[<module>.D]')

    def test_type_not_refined_for_attribute_test_without_custom_bool(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n            class D:\n                pass\n\n            class C:\n                def __init__(self) -> None:\n                    self.x: D | None = None\n\n                def f(self) -> None:\n                   if self.x:\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[<module>.D]')

    def test_type_refined_after_if_branch(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n\n                def f(self) -> None:\n                   if self.x is None:\n                      self.x = 4\n                   reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'int')

    def test_refined_field_codegen(self) -> None:
        if False:
            return 10
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n\n                def f(self) -> int | None:\n                   if self.x is not None:\n                       a = self.x\n                       return a * 2\n        '
        with self.in_module(codestr) as mod:
            self.assertInBytecode(mod.C.f, 'STORE_FAST')
            self.assertInBytecode(mod.C.f, 'LOAD_FAST')
            self.assertEqual(mod.C(21).f(), 42)

    def test_refinements_cleared_in_if_with_implicit_bool(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n\n                def f(self, y) -> None:\n                   if self.x is not None:\n                       if y:\n                           reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[int]')

    def test_refinements_cleared_in_assert_with_implicit_bool(self) -> None:
        if False:
            return 10
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    self.x: int | None = None\n\n                def f(self, y) -> None:\n                   if self.x is not None:\n                       assert(y)\n                       reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[int]')

    def test_refined_field_assert_unoptimized(self) -> None:
        if False:
            return 10
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n\n                def f(self) -> int:\n                   assert self.x is not None\n                   return self.x\n        '
        with self.in_module(codestr) as mod:
            self.assertInBytecode(mod.C.f, 'STORE_FAST')
            self.assertInBytecode(mod.C.f, 'LOAD_FAST')
            self.assertEqual(mod.C(21).f(), 21)

    def test_refined_field_assert_optimized(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n\n                def f(self) -> int:\n                   assert self.x is not None\n                   return self.x\n        '
        with self.in_module(codestr, optimize=2) as mod:
            self.assertInBytecode(mod.C.f, 'STORE_FAST')
            self.assertInBytecode(mod.C.f, 'LOAD_FAST')
            self.assertInBytecode(mod.C.f, 'CAST')
            self.assertEqual(mod.C(21).f(), 21)

    def test_field_not_refined_if_one_branch_is_unrefined(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n\n                def f(self, b) -> int:\n                   if b:\n                       assert self.x is not None\n                   reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[int]')

    def test_refined_field_if_merge_branch_to_default(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n\n                def f(self, b: bool) -> int:\n                   assert self.x is not None\n                   if b:\n                       assert self.x is not None\n                   reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'int')

    def test_fields_not_refined_if_dunder_bool_called_in_if(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n\n                def f(self, b) -> int:\n                   assert self.x is not None\n                   if b:\n                       assert self.x is not None\n                   reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'Optional[int]')

    def test_refined_field_if_merge_branch_to_orelse(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n\n                def f(self, b) -> int:\n                   if b:\n                       assert self.x is not None\n                   else:\n                       assert self.x is not None\n                   reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'int')

    def test_refined_field_if_merge_branch_to_default_codegen(self) -> None:
        if False:
            return 10
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n\n                def f(self, b: bool) -> int:\n                   if self.x is None:\n                       open("a.py") # Add a call to clear refinements.\n                       assert self.x is not None\n                   return self.x\n        '
        with self.in_module(codestr) as mod:
            refined_write_count = 0
            tmp_name = f'{_TMP_VAR_PREFIX}.__refined_field__.0'
            for instr in dis.get_instructions(mod.C.f):
                if instr.opname == 'STORE_FAST' and instr.argval == tmp_name:
                    refined_write_count += 1
            self.assertEqual(refined_write_count, 2)
            self.assertInBytecode(mod.C.f, 'LOAD_FAST', tmp_name)

    def test_refined_field_if_merge_branch_to_orelse_codegen(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n\n                def f(self, b) -> int:\n                   if b:\n                       assert self.x is not None\n                   else:\n                       assert self.x is not None\n                   return self.x\n        '
        with self.in_module(codestr) as mod:
            refined_write_count = 0
            tmp_name = f'{_TMP_VAR_PREFIX}.__refined_field__.0'
            for instr in dis.get_instructions(mod.C.f):
                if instr.opname == 'STORE_FAST' and instr.argval == tmp_name:
                    refined_write_count += 1
            self.assertEqual(refined_write_count, 2)
            self.assertInBytecode(mod.C.f, 'LOAD_FAST', tmp_name)

    def test_refined_field_if_merge_branch_to_orelse_no_refinement(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            from typing import Optional\n            class C:\n\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n\n                def f(self, b) -> Optional[int]:\n                   if b:\n                       assert self.x is not None\n                   else:\n                       assert self.x is not None\n                   open("a.py")\n                   return self.x\n        '
        with self.in_module(codestr) as mod:
            refined_write_count = 0
            tmp_name = f'{_TMP_VAR_PREFIX}.__refined_field__.1'
            for instr in dis.get_instructions(mod.C.f):
                if instr.opname == 'STORE_FAST' and instr.argval == tmp_name:
                    refined_write_count += 1
            self.assertEqual(refined_write_count, 0)
            self.assertNotInBytecode(mod.C.f, 'LOAD_FAST', tmp_name)

    def test_refined_field_while_merge_branch(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n\n                def f(self, b) -> int:\n                   assert self.x is not None\n                   while b is not None:\n                       b = not b\n                       assert self.x is not None\n                   reveal_type(self.x)\n        '
        self.revealed_type(codestr, 'int')

    def test_refined_field_when_storing(self) -> None:
        if False:
            return 10
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n\n                def f(self, x: int) -> int:\n                   self.x = x\n                   return self.x\n        '
        with self.in_module(codestr) as mod:
            c = mod.C(21)
            self.assertEqual(c.x, 21)
            self.assertEqual(c.f(42), 42)
            self.assertEqual(c.x, 42)

    def test_refined_field_at_source_codegen(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n                    self.y: int | None = None\n\n                def f(self) -> int:\n                   if self.x is None or self.y is None:\n                      return 2\n                   return 3\n        '
        with self.in_module(codestr) as mod:
            c = mod.C(None)
            self.assertEqual(c.f(), 2)
            self.assertNotInBytecode(mod.C.f, 'STORE_FAST')

    def test_refined_field_at_source_used_codegen(self) -> None:
        if False:
            return 10
        codestr = '\n            class C:\n                def __init__(self, x: int | None) -> None:\n                    self.x: int | None = x\n                    self.y: int | None = None\n\n                def f(self) -> int:\n                   if self.x is not None and self.y is None:\n                      return self.x\n                   return 3\n        '
        with self.in_module(codestr) as mod:
            c = mod.C(42)
            self.assertEqual(c.f(), 42)
            self.assertInBytecode(mod.C.f, 'STORE_FAST')
            self.assertInBytecode(mod.C.f, 'LOAD_FAST')