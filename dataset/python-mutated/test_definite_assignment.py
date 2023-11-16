from __future__ import annotations
from textwrap import dedent
from typing import final, Optional, Sequence
from cinderx.strictmodule import StrictAnalysisResult, StrictModuleLoader
from .common import StrictTestBase

@final
class DefiniteAssignmentTests(StrictTestBase):

    def analyze(self, code: str, mod_name: str='mod', import_path: Optional[Sequence[str]]=None, allow_list_prefix: Optional[Sequence[str]]=None, stub_root: str='') -> StrictAnalysisResult:
        if False:
            while True:
                i = 10
        code = dedent(code)
        compiler = StrictModuleLoader(import_path or [], stub_root, allow_list_prefix or [], [], True)
        module = compiler.check_source(code, f'{mod_name}.py', mod_name, [])
        return module

    def assertNoError(self, code: str, mod_name: str='mod', import_path: Optional[Sequence[str]]=None, allow_list_prefix: Optional[Sequence[str]]=None, stub_root: str=''):
        if False:
            while True:
                i = 10
        m = self.analyze(code, mod_name, import_path, allow_list_prefix, stub_root)
        self.assertEqual(m.is_valid, True)
        self.assertEqual(m.errors, [])

    def assertError(self, code: str, err: str, mod_name: str='mod', import_path: Optional[Sequence[str]]=None, allow_list_prefix: Optional[Sequence[str]]=None, stub_root: str=''):
        if False:
            while True:
                i = 10
        m = self.analyze(code, mod_name, import_path, allow_list_prefix, stub_root)
        self.assertEqual(m.is_valid, True)
        self.assertTrue(len(m.errors) > 0)
        self.assertTrue(err in m.errors[0][0])

    def test_simple_not_assigned(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\nabc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_simple_del_not_assigned(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\ndel abc\n'
        self.assertNoError(test_exec)

    def test_simple_assign_del_ok(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\nabc = 1\ndel abc\n'
        self.assertNoError(test_exec)

    def test_simple_assign_double_del(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nabc = 1\ndel abc\ndel abc\n'
        self.assertNoError(test_exec)

    def test_simple_if(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\nif False:\n    abc = 1\nabc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_simple_if_del(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\nabc = 1\nif True:\n    del abc\nabc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_simple_if_else(self) -> None:
        if False:
            i = 10
            return i + 15
        test_exec = '\nimport __strict__\nif str:\n    foo = 1\nelse:\n    abc = 2\nabc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_simple_if_else_del(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\nabc = 1\nif str:\n    pass\nelse:\n    del abc\nabc + 1\n'
        self.assertNoError(test_exec)

    def test_simple_if_ok(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nif str:\n    abc = 1\nelse:\n    abc = 2\nabc\n'
        self.assertNoError(test_exec)

    def test_func_dec(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\n@abc\ndef f(x): pass\n'
        self.assertError(test_exec, 'NameError')

    def test_func_self_default(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\ndef f(x = f()): pass\n'
        self.assertError(test_exec, 'NameError')

    def test_async_func_dec(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\n@abc\nasync def f(x): pass\n'
        self.assertError(test_exec, 'NameError')

    def test_async_func_self_default(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\nasync def f(x = f()): pass\n'
        self.assertError(test_exec, 'NameError')

    def test_while(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nwhile False:\n    abc = 1\nabc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_while_else(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\nwhile False:\n    abc = 1\nelse:\n    abc = 1\nabc\n'
        self.assertNoError(test_exec)

    def test_while_del(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nabc = 1\nwhile str:\n    del abc\n    break\nabc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_while_else_del(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nabc = 1\nwhile False:\n    pass\nelse:\n    del abc\nx = abc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_while_del_else(self) -> None:
        if False:
            i = 10
            return i + 15
        test_exec = '\nimport __strict__\nabc = 1\nx = 1\nwhile x > 0:\n    del abc\n    x = x - 1\nelse:\n    abc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_class_defined(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\nclass C:\n    pass\n\nC\n'
        self.assertNoError(test_exec)

    def test_class_defined_with_func(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nclass C:\n    def __init__(self):\n        pass\n\nC\n'
        self.assertNoError(test_exec)

    def test_class_scoping(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nclass C:\n    abc = 42\n\nx = abc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_class_uninit_global_read(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\nclass C:\n    x = abc + 1\n\n'
        self.assertError(test_exec, 'NameError')

    def test_class_uninit_class_read(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\nclass C:\n    if str:\n        abc = 42\n    abc + 1\n'
        self.assertNoError(test_exec)

    def test_nested_class_uninit_read(self) -> None:
        if False:
            i = 10
            return i + 15
        test_exec = '\nimport __strict__\nclass C:\n    abc = 42\n    class D:\n        x = abc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_class_undef_dec(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\n@abc\nclass C:\n    pass\n'
        self.assertError(test_exec, 'NameError')

    def test_uninit_aug_assign(self) -> None:
        if False:
            i = 10
            return i + 15
        test_exec = '\nimport __strict__\nabc += 1\n'
        self.assertError(test_exec, 'NameError')

    def test_aug_assign(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nabc = 0\nabc += 1\n    '
        self.assertNoError(test_exec)

    def test_with_no_assign(self) -> None:
        if False:
            i = 10
            return i + 15
        test_exec = '\nimport __strict__\nclass A:\n    def __enter__(self):\n        pass\n    def __exit__(self, exc_tp, exc, tb):\n        pass\nwith A():\n    abc = 1\nabc + 1\n'
        self.assertNoError(test_exec)

    def test_with_var(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\nclass A:\n    def __enter__(self):\n        pass\n    def __exit__(self, exc_tp, exc, tb):\n        pass\nwith A() as abc:\n    pass\nabc\n'
        self.assertNoError(test_exec)

    def test_with_var_destructured(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\nclass A:\n    def __enter__(self):\n        return 1, 3\n    def __exit__(self, exc_tp, exc, tb):\n        pass\nwith A() as (abc, foo):\n    pass\nabc\nfoo\n'
        self.assertNoError(test_exec)

    def test_import(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\nimport abc\nabc\n'
        self.assertNoError(test_exec)

    def test_import_as(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\nimport foo as abc\nabc\n'
        self.assertNoError(test_exec)

    def test_import_from(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\nfrom foo import abc\nabc\n'
        self.assertNoError(test_exec)

    def test_import_from_as(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nfrom foo import bar as abc\nabc\n'
        self.assertNoError(test_exec)

    def test_del_in_finally(self) -> None:
        if False:
            i = 10
            return i + 15
        test_exec = '\nimport __strict__\ntry:\n    abc = 1\nfinally:\n    del abc\n'
        self.assertNoError(test_exec)

    def test_del_in_finally_2(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nabc = 1\ntry:\n    pass\nfinally:\n    del abc\nabc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_finally_no_del(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\ntry:\n    abc = 1\nfinally:\n    pass\nabc\n    '
        self.assertNoError(test_exec)

    def test_finally_not_defined(self) -> None:
        if False:
            i = 10
            return i + 15
        test_exec = '\nimport __strict__\ntry:\n    abc = 1\nfinally:\n    abc + 1\n'
        self.assertNoError(test_exec)

    def test_try_finally_deletes_apply(self) -> None:
        if False:
            i = 10
            return i + 15
        test_exec = '\nimport __strict__\nabc = 1\ntry:\n    del abc\nfinally:\n    pass\nabc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_try_except_var_defined(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\ntry:\n    pass\nexcept Exception as abc:\n    abc\n'
        self.assertNoError(test_exec)

    def test_try_except_var_not_defined_after(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\ntry:\n    pass\nexcept Exception as abc:\n    pass\nabc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_try_except_no_try_define(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\ntry:\n    abc = 1\nexcept Exception:\n    pass\nabc + 1\n'
        self.assertNoError(test_exec)

    def test_try_except_no_except_define(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\ntry:\n    pass\nexcept Exception:\n    abc = 1\nabc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_try_except_dels_assumed(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\nabc = 1\ntry:\n    del abc\nexcept Exception:\n    pass\nabc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_try_except_dels_assumed_in_except(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nabc = 1\ntry:\n    del abc\nexcept Exception:\n    abc + 1\n'
        self.assertNoError(test_exec)

    def test_try_except_except_dels_assumed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\nabc = 1\ntry:\n    pass\nexcept Exception:\n    del abc\nabc + 1\n'
        self.assertNoError(test_exec)

    def test_try_except_finally(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\ntry:\n    pass\nexcept Exception:\n    pass\nfinally:\n    abc = 1\nabc\n'
        self.assertNoError(test_exec)

    def test_try_except_finally_try_not_assumed(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\ntry:\n    abc = 1\nexcept Exception:\n    pass\nfinally:\n    abc + 1\n'
        self.assertNoError(test_exec)

    def test_try_except_finally_except_not_assumed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\ntry:\n    pass\nexcept Exception:\n    abc = 1\nfinally:\n    abc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_try_except_else_try_assumed(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\ntry:\n    abc = 1\nexcept Exception:\n    pass\nelse:\n    abc\n'
        self.assertNoError(test_exec)

    def test_try_except_else_try_assumed_del(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\ntry:\n    abc = 1\nexcept Exception:\n    pass\nelse:\n    del abc\n'
        self.assertNoError(test_exec)

    def test_try_except_else_except_not_assumed(self) -> None:
        if False:
            i = 10
            return i + 15
        test_exec = '\nimport __strict__\ntry:\n    pass\nexcept Exception:\n    abc = 1\nelse:\n    x = abc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_try_except_else_except_del_not_assumed(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\nabc = 1\ntry:\n    pass\nexcept Exception:\n    del abc\nelse:\n    x = abc + 1\n'
        self.assertNoError(test_exec)

    def test_try_except_else_assign_not_assumed_for_finally(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\ntry:\n    pass\nexcept Exception:\n    pass\nelse:\n    abc = 1\nfinally:\n    x = abc + 1\n'
        self.assertNoError(test_exec)

    def test_try_except_finally_del_assumed(self) -> None:
        if False:
            i = 10
            return i + 15
        test_exec = '\nimport __strict__\nabc = 1\ntry:\n    pass\nexcept Exception:\n    del abc\nfinally:\n    x = abc + 1\n'
        self.assertNoError(test_exec)

    def test_lambda_not_assigned(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\nx = (lambda x=abc + 1: 42)\n'
        self.assertError(test_exec, 'NameError')

    def test_lambda_ok(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nx = lambda x: abc\n'
        self.assertNoError(test_exec)

    def test_list_comp(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\nfoo = [1, 2, 3]\nbar = [x for x in foo]\n'
        self.assertNoError(test_exec)

    def test_list_comp_undef(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\nbar = [x for x in abc]\n'
        self.assertError(test_exec, 'NameError')

    def test_list_comp_if(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_exec = '\nimport __strict__\nfoo = [1, 2, 3]\nbar = [x for x in foo if x]\n'
        self.assertNoError(test_exec)

    def test_set_comp(self) -> None:
        if False:
            i = 10
            return i + 15
        test_exec = '\nimport __strict__\nfoo = [1, 2, 3]\nbar = {x for x in foo}\n'
        self.assertNoError(test_exec)

    def test_set_comp_undef(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nbar = {x for x in abc}\n'
        self.assertError(test_exec, 'NameError')

    def test_set_comp_undef_value(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\nfoo = [1, 2, 3]\nbar = {(x, abc) for x in foo}\n'
        self.assertError(test_exec, 'NameError')

    def test_set_comp_if(self) -> None:
        if False:
            i = 10
            return i + 15
        test_exec = '\nimport __strict__\nfoo = [1, 2, 3]\nbar = {x for x in foo if x}\n'
        self.assertNoError(test_exec)

    def test_gen_comp(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\nfoo = [1, 2, 3]\nbar = (x for x in foo)\n'
        self.assertNoError(test_exec)

    def test_gen_comp_undef(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nbar = (x for x in abc)\n'
        self.assertError(test_exec, 'NameError')

    def test_gen_comp_undef_value(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nfoo = [1, 2, 3]\nbar = ((x, abc) for x in foo)\n'
        self.assertError(test_exec, 'NameError')

    def test_gen_comp_if(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nfoo = [1, 2, 3]\nbar = (x for x in foo if x)\n'
        self.assertNoError(test_exec)

    def test_dict_comp(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\nfoo = [1, 2, 3]\nbar = {x:x for x in foo}\n'
        self.assertNoError(test_exec)

    def test_dict_comp_undef(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\nbar = {x:x for x in abc}\n'
        self.assertError(test_exec, 'NameError')

    def test_dict_comp_if(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nfoo = [1, 2, 3]\nbar = {x:x for x in foo if x}\n'
        self.assertNoError(test_exec)

    def test_self_assign(self) -> None:
        if False:
            print('Hello World!')
        test_exec = '\nimport __strict__\nabc = abc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_ann_assign_not_defined(self) -> None:
        if False:
            return 10
        test_exec = '\nimport __strict__\nabc: int\nabc + 1\n'
        self.assertError(test_exec, 'NameError')

    def test_expected_globals_name(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\nx = __name__\n'
        self.assertNoError(test_exec)

    def test_raise_unreachable(self) -> None:
        if False:
            while True:
                i = 10
        test_exec = '\nimport __strict__\nx = 0\nif x:\n    raise Exception\n    abc = 2\nelse:\n    abc = 1\n\nabc + 1\n'
        self.assertNoError(test_exec)