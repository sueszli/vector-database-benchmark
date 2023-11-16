import os
import sys
import inspect
import unittest
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from textwrap import dedent
from collections import OrderedDict
from torch import Tensor
import torch
import torch.nn as nn
import types
from torch.testing import FileCheck
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, make_global
from torch.testing._internal.common_utils import skipIfTorchDynamo
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestList(JitTestCase):

    def test_list_bool_conversion(self):
        if False:
            return 10

        def if_predicate(l: List[int]):
            if False:
                i = 10
                return i + 15
            if l:
                s = 0
                for n in l:
                    s += n
                return s
            else:
                return -1
        self.checkScript(if_predicate, ([1, 2, 3],))
        self.checkScript(if_predicate, ([],))

        def while_predicate(l: List[int]):
            if False:
                return 10
            s = 0
            while l:
                s += l.pop()
        self.checkScript(while_predicate, ([1, 2, 3],))
        self.checkScript(while_predicate, ([],))

        def ternary_predicate(l: List[int]):
            if False:
                while True:
                    i = 10
            return 'non-empty' if l else 'empty'
        self.checkScript(ternary_predicate, ([1, 2, 3],))
        self.checkScript(ternary_predicate, ([],))

    def test_in_check(self):
        if False:
            for i in range(10):
                print('nop')

        def int_in(x: List[int]) -> bool:
            if False:
                print('Hello World!')
            return 2 in x
        self.checkScript(int_in, ([1, 2, 3],))
        self.checkScript(int_in, ([1, 3, 3],))

        def float_in(x: List[float]) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return 2.0 in x
        self.checkScript(float_in, ([1.0, 2.0, 3.0],))
        self.checkScript(float_in, ([1.0, 3.0, 3.0],))

        def str_in(x: List[str]) -> bool:
            if False:
                return 10
            return 'hi' in x
        self.checkScript(str_in, (['not', 'here'],))
        self.checkScript(str_in, (['hi', 'bye'],))
        self.checkScript(str_in, ([],))

    def test_list_literal(self):
        if False:
            print('Hello World!')

        def reassign():
            if False:
                for i in range(10):
                    print('nop')
            x = [1]
            if 1 == 1:
                x = [2, 3]
            return
        self.checkScript(reassign, (), optimize=False)

        def reassign_arity_change():
            if False:
                for i in range(10):
                    print('nop')
            x = [1]
            if 1 == 1:
                x = [1, 2, 3]
            return
        self.checkScript(reassign_arity_change, (), optimize=False)

        def reassign_from_empty_literal():
            if False:
                i = 10
                return i + 15
            x = []
            if 1 == 1:
                x = [1, 2, 3]
            return
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'previously had type List\\[Tensor\\]', 'x'):
            self.checkScript(reassign_from_empty_literal, (), optimize=False)

        def reassign_from_empty_builtin():
            if False:
                while True:
                    i = 10
            x = torch.jit.annotate(List[int], [])
            if 1 == 1:
                x = [1, 2, 3]
            y = torch.jit.annotate(List[float], [])
            if 1 == 1:
                y = [1.0, 2.0, 3.0]
            z = []
            if 1 == 1:
                z = [torch.randn([1])]
            return
        self.checkScript(reassign_from_empty_builtin, (), optimize=False)

        def reassign_bad_type():
            if False:
                i = 10
                return i + 15
            x = [1]
            if 1 == 1:
                x = [1.0]
            return
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'previously had type', 'x'):
            self.checkScript(reassign_bad_type, (), optimize=False)

        def reassign_nested():
            if False:
                print('Hello World!')
            x = torch.jit.annotate(List[int], [])
            if 1 == 1:
                x = [1, 2, 3]
                if 1 == 1:
                    x = [1.0]
            return
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'previously had type', 'x'):
            self.checkScript(reassign_nested, (), optimize=False)

    def test_list_variance(self):
        if False:
            return 10
        '\n        `List[T1]` is not a subtype of `List[T2]`, even if `T1` is a\n        subtype of `T2`. However, if we have a temporary list object\n        (that is, a list comprehension or a list literal) on the rhs of\n        an assignment statement, we want to ignore the inferred type of\n        the rhs if we can prove that: 1) both the lhs and the rhs are\n        lists, and 2) the inner type of the lhs list is a subtype of the\n        inner type of the rhs list.\n\n        # This should pass\n        x: List[Optional[int]] = [None, None, None]\n\n        # This should fail\n        y: List[None] = [None, None, None]\n        x: List[Optional[int]] = y\n        '

        def test_listliteral_is_typed_from_annotation():
            if False:
                print('Hello World!')
            x: List[Optional[int]] = [None, None, None]
            return x
        self.checkScript(test_listliteral_is_typed_from_annotation, ())

        def test_listcomprehension_is_typed_from_annotation():
            if False:
                i = 10
                return i + 15
            x: List[Optional[int]] = [None for _ in range(3)]
            return x
        self.checkScript(test_listcomprehension_is_typed_from_annotation, ())

        def test_lists_with_different_internal_types_are_invariant(self):
            if False:
                for i in range(10):
                    print('nop')
            x: List[int] = [1, 2, 3]
            y: List[Optional[int]] = x
            return x
        with self.assertRaisesRegex(RuntimeError, "Variable 'y' is annotated with type List\\[Optional\\[int\\]\\] but is being assigned to a value of type List\\[int\\]"):
            torch.jit.script(test_lists_with_different_internal_types_are_invariant)

        def test_lists_with_different_internal_types_are_invariant_recursive(self):
            if False:
                i = 10
                return i + 15
            x: List[List[int]] = [[1, 2], [3]]
            y: List[List[Optional[int]]] = x
            return x
        with self.assertRaisesRegex(RuntimeError, "Variable 'y' is annotated with type List\\[List\\[Optional\\[int\\]\\]\\] but is being assigned to a value of type List\\[List\\[int\\]\\]"):
            torch.jit.script(test_lists_with_different_internal_types_are_invariant_recursive)

    def test_del(self):
        if False:
            while True:
                i = 10

        def inputs():
            if False:
                print('Hello World!')
            return [1, 2, 3, 4]

        def fn(x: List[int]) -> List[int]:
            if False:
                print('Hello World!')
            del x[1]
            return x
        python_out = fn(inputs())
        cu = torch.jit.CompilationUnit()
        cu.define(dedent(inspect.getsource(fn)))
        self.assertEqual(cu.fn(inputs()), python_out)
        self.assertEqual(torch.jit.script(fn)(inputs()), python_out)

        @torch.jit.script
        def fn2(x: List[int]) -> List[int]:
            if False:
                return 10
            del x[100]
            return x
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'out of range', 'x[100]'):
            fn2([])
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'deletion at a single index', 'x[1:3]'):

            @torch.jit.script
            def fn(x: List[int]) -> List[int]:
                if False:
                    return 10
                del x[1:3]
                return x

    def test_list_keyword(self):
        if False:
            return 10

        def foo():
            if False:
                i = 10
                return i + 15
            return (list([1, 2, 3]), list(('a', 'b')), list(range(5)), list('abcdefg'))
        self.checkScript(foo, ())

        def foo2():
            if False:
                while True:
                    i = 10
            x: List[int] = list()
            x.append(1)
            return (x,)
        self.checkScript(foo2, ())

        def foo3():
            if False:
                print('Hello World!')
            return list(list('abc'))
        self.checkScript(foo3, ())
        FileCheck().check_count('aten::list', 2, exactly=True).run(torch.jit.script(foo3).graph)

    def test_dict_keyword_with_kwargs(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                return 10
            return dict(foo=1, bar=2, baz=3)
        self.checkScript(fn, ())

    def test_dict_keyword_with_kwargs_using_container_values(self):
        if False:
            return 10

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            return dict(foo=[1, 2, 3], bar=[4, 5, 6], baz=[7, 8, 9])
        self.checkScript(fn, ())

    def test_dict_keyword_with_iterable(self):
        if False:
            return 10

        def fn():
            if False:
                i = 10
                return i + 15
            return dict([('foo', 1), ('bar', 2), ('baz', 3)])
        self.checkScript(fn, ())

    def test_dict_keyword_with_empty_iterable(self):
        if False:
            return 10

        def fn():
            if False:
                i = 10
                return i + 15
            return dict([])
        self.checkScript(fn, ())

    def test_dict_keyword_with_internal_aggregate_function(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                print('Hello World!')
            return dict(zip(['foo', 'baz', 'bar'], [1, 2, 3]))
        self.checkScript(fn, ())

    def test_dict_keyword_with_mapping(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                print('Hello World!')
            return {'foo': 1, 'bar': 2, 'baz': 3}
        self.checkScript(fn, ())

    def test_dict_keyword_with_mapping_and_kwargs(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                while True:
                    i = 10
            return dict({'foo': 1, 'bar': 2}, baz=3)
        self.checkScript(fn, ())

    def test_dict_keyword_with_dict_comprehension(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                return 10
            return {i: chr(i + 65) for i in range(4)}
        self.checkScript(fn, ())

    def test_dict_keyword_with_dict_comprehension_and_kwargs(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            return dict({chr(65 + i): i for i in range(4)}, foo=2)
        self.checkScript(fn, ())

    def test_dict_keyword_with_empty_dict_comprehension(self):
        if False:
            while True:
                i = 10

        def fn():
            if False:
                print('Hello World!')
            return {}
        self.checkScript(fn, ())

    def test_dict_keyword_is_correctly_typed(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                print('Hello World!')
            x: Dict[str, int] = dict()
            x['foo'] = 1
            return x
        self.checkScript(fn, ())

    def test_dict_keyword_with_mismatched_annotations(self):
        if False:
            while True:
                i = 10
        err_msg = 'Dict type annotation `Dict\\[int, str\\]` did not match the type of an actual key type `str`'
        with self.assertRaisesRegex(RuntimeError, err_msg):

            @torch.jit.script
            def fn():
                if False:
                    return 10
                x: Dict[int, str] = dict([('foo', 1), ('bar', 2), ('baz', 3)])
                return x

    def test_dict_keyword_with_nested_call(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                print('Hello World!')
            return dict(dict(foo=1, bar=2, baz=3))
        self.checkScript(fn, ())

    def test_dict_keyword_with_previously_declared_variable(self):
        if False:
            return 10

        def fn():
            if False:
                while True:
                    i = 10
            d = {'foo': 1, 'bar': 2}
            return dict(d)
        self.checkScript(fn, ())

    def test_dict_keyword_with_previously_declared_variable_and_kwargs(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                print('Hello World!')
            d = {'foo': 1, 'bar': 2}
            return dict(d, baz=3)
        self.checkScript(fn, ())

    def test_min_bool_list(self):
        if False:
            while True:
                i = 10

        def jit_min_list(a: List[bool], b: List[bool]) -> List[bool]:
            if False:
                for i in range(10):
                    print('nop')
            return min(a, b)
        self.checkScript(jit_min_list, ([True, False], [False, True]))

    def test_min_max_list(self):
        if False:
            i = 10
            return i + 15

        def jit_min_list(a: List[int], b: List[int]) -> List[int]:
            if False:
                while True:
                    i = 10
            return min(a, b)

        def jit_min_list_float(a: List[float], b: List[float]) -> List[float]:
            if False:
                while True:
                    i = 10
            return min(a, b)

        def jit_min_list_bool(a: List[bool], b: List[bool]) -> List[bool]:
            if False:
                for i in range(10):
                    print('nop')
            return min(a, b)

        def run_tests(func, a, b):
            if False:
                for i in range(10):
                    print('nop')
            for t in zip(a, b):
                self.checkScript(func, t)
        args_left_int = [[1, 8, 8], [2, 1, 1], [], [2], [1], [1, 2, 3]]
        args_right_int = [[2, 1, 1], [1, 8, 8], [], [1], [], [1, 2]]
        run_tests(jit_min_list, args_left_int, args_right_int)
        args_left_float = [[1.0, 8.0, 8.0], [2.0, 1.0, 1.0], [], [2.0], [1.0], [1.0, 2.0, 3.0]]
        args_right_float = [[2.0, 1.0, 1.0], [1.0, 8.0, 8.0], [], [1.0], [], [1.0, 2.0]]
        run_tests(jit_min_list_float, args_left_float, args_right_float)
        args_left_bool = [[], [], [], [False], [True], [False, True], [True, True], [False, False, False], [False, False, True]]
        args_right_bool = [[], [False], [True], [True], [False], [True, True], [False, True], [False, False, True], [False, False, False]]
        run_tests(jit_min_list_bool, args_left_bool, args_right_bool)

        def jit_max_list(a: List[int], b: List[int]) -> List[int]:
            if False:
                for i in range(10):
                    print('nop')
            return max(a, b)

        def jit_max_list_float(a: List[float], b: List[float]) -> List[float]:
            if False:
                for i in range(10):
                    print('nop')
            return max(a, b)

        def jit_max_list_bool(a: List[bool], b: List[bool]) -> List[bool]:
            if False:
                for i in range(10):
                    print('nop')
            return max(a, b)
        args_left_int = [[1, 8, 8], [8, 1, 1], [], [1], [], [1, 2]]
        args_right_int = [[8, 1, 1], [1, 8, 8], [], [2], [1], [1, 2, 3]]
        run_tests(jit_max_list, args_left_int, args_right_int)
        args_left_float = [[1.0, 8.0, 8.0], [8.0, 1.0, 1.0], [], [1.0], [], [1.0, 2.0]]
        args_right_float = [[8.0, 1.0, 1.0], [1.0, 8.0, 8.0], [], [2.0], [1.0], [1.0, 2.0, 3.0]]
        run_tests(jit_max_list_float, args_left_float, args_right_float)
        run_tests(jit_max_list_bool, args_left_bool, args_right_bool)

    def test_list_gather(self):
        if False:
            print('Hello World!')

        def index():
            if False:
                while True:
                    i = 10
            a = [1, 2, 3]
            return a[1]
        self.checkScript(index, ())

        def negative_index():
            if False:
                while True:
                    i = 10
            a = [1, 2, 3]
            return a[-1]
        self.checkScript(negative_index, ())

        def bad_index():
            if False:
                for i in range(10):
                    print('nop')
            a = [1, 2, 3]
            return a[4]
        self.checkScriptRaisesRegex(bad_index, (), Exception, 'list index out of range')

        def bad_negative_index():
            if False:
                return 10
            a = [1, 2, 3]
            return a[-5]
        self.checkScriptRaisesRegex(bad_negative_index, (), Exception, 'list index out of range')

    def test_list_len(self):
        if False:
            print('Hello World!')

        def func():
            if False:
                return 10
            a = [1, 2, 3]
            return len(a) == 3
        self.checkScript(func, ())

        def func2():
            if False:
                return 10
            a = []
            return len(a) == 0
        self.checkScript(func2, ())

    @skipIfTorchDynamo('TorchDynamo fails to raise on this checkScriptRaisesRegex, because we trace it properly now')
    def test_list_ops(self):
        if False:
            for i in range(10):
                print('nop')

        def test_equality():
            if False:
                print('Hello World!')
            a = [1, 2, 3]
            b = [1, 2, 3]
            return a == b
        self.checkScript(test_equality, (), optimize=True)

        def test_equality_str():
            if False:
                for i in range(10):
                    print('nop')
            a = ['foo', 'bar']
            b = ['foo', 'bar']
            return a == b
        self.checkScript(test_equality_str, (), optimize=True)

        def test_inequality():
            if False:
                return 10
            a = [1, 2, 3]
            b = [1, 2, 3]
            return a != b
        self.checkScript(test_inequality, (), optimize=True)

        def test_inequality_str():
            if False:
                print('Hello World!')
            a = ['foo', 'bar']
            b = ['foo', 'bar', 'food']
            return a != b
        self.checkScript(test_inequality_str, (), optimize=True)

        def test_non_equality():
            if False:
                for i in range(10):
                    print('nop')
            a = [1, 2, 3]
            b = [3]
            return a == b
        self.checkScript(test_non_equality, (), optimize=True)

        def test_non_inequality():
            if False:
                while True:
                    i = 10
            a = [1, 2, 3]
            b = [3]
            return a != b
        self.checkScript(test_non_equality, (), optimize=True)

        def test_list_equality_as_cond():
            if False:
                return 10
            a = [1, 2, 3]
            b = [3]
            if a == b:
                c = 1
            else:
                c = 2
            return c
        self.checkScript(test_list_equality_as_cond, (), optimize=True)

        def test_list_add():
            if False:
                print('Hello World!')
            a = [1, 2, 3]
            b = [2]
            c = a + b
            return c == [1, 2, 3, 2]
        self.checkScript(test_list_add, (), optimize=True)

        def test_list_add_empty():
            if False:
                return 10
            a = [1, 2, 3]
            b = torch.jit.annotate(List[int], [])
            c = a + b
            return c == [1, 2, 3]
        self.checkScript(test_list_add_empty, (), optimize=True)

        def test_tensor_list_equality():
            if False:
                return 10
            t1 = torch.ones([1, 1])
            t2 = torch.ones([1, 1])
            x = [t1, t2]
            y = [t2, t1]
            return x == y
        self.checkScript(test_tensor_list_equality, (), optimize=True)

        def test_invalid_list_equality():
            if False:
                print('Hello World!')
            t1 = torch.ones([2, 2])
            t2 = torch.ones([2, 2])
            x = [t1, t2]
            y = [t2, t1]
            return x == y
        self.checkScriptRaisesRegex(test_invalid_list_equality, (), RuntimeError, 'Boolean value of Tensor')

    def test_list_sort(self):
        if False:
            while True:
                i = 10
        template = dedent('\n        def func():\n            li_1 = {list_create}\n            li_2 = {list_create}\n            li_3 = {list_create}\n            li_1.sort()\n            li_2.sort(reverse=True)\n            li_4 = sorted(li_3)\n            return li_1, li_2, li_3, li_4\n        ')
        lists = ['[]', '[1, 3, 2]', '[True, False, True]', '[1.2, .2, 3.2]', '[torch.tensor(1.0), torch.tensor(0.2), torch.tensor(0.5)]', '[torch.tensor(5), torch.tensor(-2), torch.tensor(4)]']
        for li in lists:
            code = template.format(list_create=li)
            scope = {}
            exec(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            t1 = cu.func()
            t2 = scope['func']()
            self.assertEqual(t1, t2)

        def test_fail(x: List[Tensor]) -> List[Tensor]:
            if False:
                print('Hello World!')
            x.sort()
            return x
        self.checkScriptRaisesRegex(test_fail, ([torch.zeros([2]), torch.zeros([2])],), Exception, 'Boolean value of Tensor with more than one value')

        @torch.jit.script
        def test_mutation():
            if False:
                i = 10
                return i + 15
            a = [1, 2, 3]
            a.sort()
            return a
        test_mutation()
        FileCheck().check('aten::sort').run(test_mutation.graph_for())

        def test_sorted_copy():
            if False:
                while True:
                    i = 10
            a = [torch.tensor(2), torch.tensor(0), torch.tensor(1)]
            b = sorted(a)
            a[0] = torch.tensor(10)
            return (a, b)
        self.checkScript(test_sorted_copy, ())

    def test_list_slice(self):
        if False:
            i = 10
            return i + 15

        def test_regular_slice():
            if False:
                i = 10
                return i + 15
            a = [0, 1, 2, 3, 4]
            return a[2:3] == [2]
        self.checkScript(test_regular_slice, ())

        def test_open_ended_slice():
            if False:
                return 10
            a = [0, 1, 2, 3, 4]
            return a[2:] == [2, 3, 4]
        self.checkScript(test_open_ended_slice, ())

        def test_open_ended_slice2():
            if False:
                for i in range(10):
                    print('nop')
            a = [0, 1, 2, 3, 4]
            return a[:2] == [0, 1]
        self.checkScript(test_open_ended_slice2, ())

        def test_negative_slice():
            if False:
                while True:
                    i = 10
            a = [0, 1, 2, 3, 4]
            return a[:-1] == [0, 1, 2, 3]
        self.checkScript(test_negative_slice, ())

        def test_negative_slice2():
            if False:
                for i in range(10):
                    print('nop')
            a = [0, 1, 2, 3, 4]
            return a[-3:-1] == [2, 3]
        self.checkScript(test_negative_slice2, ())

        def test_backward_slice():
            if False:
                while True:
                    i = 10
            a = [0, 1, 2, 3, 4]
            return a[3:2] == torch.jit.annotate(List[int], [])
        self.checkScript(test_backward_slice, ())

        def test_over_slice():
            if False:
                for i in range(10):
                    print('nop')
            a = [0, 1, 2, 3, 4]
            return a[3:10] == [3, 4]
        self.checkScript(test_backward_slice, ())

    def test_slice_index(self):
        if False:
            while True:
                i = 10
        a = torch.tensor([[[1, 11], [2, 22]], [[3, 33], [4, 44]], [[5, 55], [6, 66]]])

        def test_index_slice1(x):
            if False:
                print('Hello World!')
            x = x[:, :, [0, 1]]
            return x
        self.checkScript(test_index_slice1, (a,))

        def test_index_slice2(x):
            if False:
                return 10
            x = x[[2, 1, 0], :, :]
            return x
        self.checkScript(test_index_slice2, (a,))

        def test_index_slice3(x):
            if False:
                return 10
            x = x[[0, 1], :, [1]]
            return x
        self.checkScript(test_index_slice3, (a,))

        def test_index_slice_empty_list(x):
            if False:
                while True:
                    i = 10
            empty_list: List[int] = []
            x = x[empty_list, :, :]
            return x
        self.checkScript(test_index_slice_empty_list, (a,))

        def test_index_slice_out_of_bounds_index(x):
            if False:
                for i in range(10):
                    print('nop')
            x = x[[4], :, :]
            return x
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'index 4 is out of bounds for dimension 0 with size 3', 'x[[4], :, :]'):
            self.checkScript(test_index_slice_out_of_bounds_index, (a,))

    def test_mutable_list_append(self):
        if False:
            i = 10
            return i + 15

        def test_append():
            if False:
                for i in range(10):
                    print('nop')
            a = [0, 1]
            a.append(2)
            a.append(3)
            return a == [0, 1, 2, 3]
        self.checkScript(test_append, ())

    def test_comprehensions_basic(self):
        if False:
            print('Hello World!')

        def comp(l: List[int]) -> List[int]:
            if False:
                for i in range(10):
                    print('nop')
            n = [x * 3 for x in l]
            return n
        comp([1, 2, 3])
        self.checkScript(comp, ([1, 2, 3],))

    def test_comprehensions_basic_float(self):
        if False:
            while True:
                i = 10

        def comp(l: List[float]) -> List[float]:
            if False:
                for i in range(10):
                    print('nop')
            n = [x * 3 for x in l]
            return n
        self.checkScript(comp, ([1.0, 2.0, 3.0],))

    def test_comprehensions_two_comps(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def comp(l1: List[int], l2: List[int]) -> List[int]:
            if False:
                print('Hello World!')
            n = [x * 3 for x in l1]
            n2 = [x + 2 for x in l2]
            return n + n2
        self.assertEqual(comp([1, 2, 3], [4, 5]), [3, 6, 9, 6, 7])

    def test_comprehension_out_type_not_in_type(self):
        if False:
            for i in range(10):
                print('nop')

        def list_cast() -> int:
            if False:
                i = 10
                return i + 15
            li = [int(i) for i in [torch.tensor(0), torch.tensor(1), torch.tensor(2)]]
            return li[0] + li[1] + li[2]
        self.checkScript(list_cast, ())

    def test_comprehension_iterable(self):
        if False:
            i = 10
            return i + 15

        def test_func(fn, inputs):
            if False:
                print('Hello World!')
            self.assertEqual(fn(*inputs), torch.jit.script(fn)(*inputs))

        def foo(names: List[int], results: List[int]) -> List[Tuple[int, int]]:
            if False:
                while True:
                    i = 10
            return [(k + 5, v - 2) for (k, v) in zip(names, results)]
        test_func(foo, ([1, 2, 4], [4, 7, 9]))
        test_func(foo, ([5], [4, 7, 9]))

        def fn(x: int) -> List[int]:
            if False:
                i = 10
                return i + 15
            return [i for i in range(x)]
        test_func(fn, (9,))
        test_func(fn, (0,))
        test_func(fn, (-1,))

        def changes_type():
            if False:
                for i in range(10):
                    print('nop')
            a = [float(i) for i in range(5)]
            b = [float(i) for i in [1, 2, 3, 4]]
            c = [(float(i), j) for (i, j) in enumerate([1, 2, 3, 8])]
            return (a, b, c)
        test_func(changes_type, ())

        def test_zero_iter():
            if False:
                i = 10
                return i + 15
            return [str(i) for (i, j) in zip('', '')]
        test_func(test_zero_iter, ())

    def test_mutable_list_append_2(self):
        if False:
            return 10

        def test_append_2():
            if False:
                i = 10
                return i + 15
            a = [0, 1]
            a.append(2)
            a = [1]
            a.append(4)
            return a == [1, 4]
        self.checkScript(test_append_2, ())

    def test_mutable_list_append_if(self):
        if False:
            for i in range(10):
                print('nop')

        def test_append_if():
            if False:
                while True:
                    i = 10
            a = [1]
            if 1 == 1:
                a.append(4)
            return a == [1, 4]
        self.checkScript(test_append_if, ())

    def test_mutable_list_append_if_else(self):
        if False:
            return 10

        def test_append_if_else():
            if False:
                print('Hello World!')
            a = [1]
            if 1 == 2:
                a.append(4)
            else:
                a.append(10)
            return a == [1, 10]
        self.checkScript(test_append_if_else, ())

    def test_mutable_list_append_loop(self):
        if False:
            i = 10
            return i + 15

        def test_append_loop():
            if False:
                while True:
                    i = 10
            a = torch.jit.annotate(List[int], [])
            for i in range(5):
                a.append(i)
            return a == [0, 1, 2, 3, 4]
        self.checkScript(test_append_loop, ())

    def test_mutable_list_append_loop_if(self):
        if False:
            return 10

        def test_append_loop_if():
            if False:
                while True:
                    i = 10
            a = torch.jit.annotate(List[int], [])
            for i in range(5):
                if i > 3:
                    a.append(i)
                else:
                    a.append(0)
            return a == [0, 0, 0, 0, 4]
        self.checkScript(test_append_loop_if, ())

    def test_mutable_list_nested_loop(self):
        if False:
            while True:
                i = 10

        def test_nested_loop():
            if False:
                return 10
            a = torch.jit.annotate(List[int], [])
            for i in range(2):
                for j in range(2):
                    a.append(i + j)
            return a == [0, 1, 1, 2]
        self.checkScript(test_nested_loop, ())

    def test_mutable_list_function_inline(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def bar(y: List[int]) -> None:
            if False:
                while True:
                    i = 10
            y.append(4)

        @torch.jit.script
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            x = [1, 2, 3]
            bar(x)
            return x
        self.assertEqual(foo(), [1, 2, 3, 4])

    def test_mutable_list_reverse_empty(self):
        if False:
            print('Hello World!')

        def test_reverse_empty():
            if False:
                return 10
            a = []
            a.reverse()
            return a == []
        self.checkScript(test_reverse_empty, ())

    def test_mutable_list_reverse(self):
        if False:
            i = 10
            return i + 15

        def test_reverse():
            if False:
                for i in range(10):
                    print('nop')
            a = [1, 2, 3, 4]
            a.reverse()
            return a == [4, 3, 2, 1]
        self.checkScript(test_reverse, ())

    def test_mutable_tensor_list_reverse(self):
        if False:
            for i in range(10):
                print('nop')

        def test_tensor_reverse():
            if False:
                i = 10
                return i + 15
            a = [torch.tensor(1), torch.tensor(2)]
            a.reverse()
            return a == [torch.tensor(2), torch.tensor(1)]
        self.checkScript(test_tensor_reverse, ())

    def test_mutable_list_pop_empty(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def test_pop_empty():
            if False:
                print('Hello World!')
            a = torch.jit.annotate(List[int], [])
            return a.pop()
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'pop from empty list', 'a.pop'):
            test_pop_empty()

    def test_mutable_list_pop(self):
        if False:
            return 10

        def test_pop():
            if False:
                i = 10
                return i + 15
            a = [1, 2, 3, 4]
            b = a.pop()
            return b == 4
        self.checkScript(test_pop, ())

    def test_mutable_list_pop2(self):
        if False:
            while True:
                i = 10

        def test_pop2():
            if False:
                print('Hello World!')
            a = [1, 2, 3, 4]
            b = a.pop()
            return len(a) == 3
        self.checkScript(test_pop2, ())

    def test_mutable_list_pop_at(self):
        if False:
            return 10

        def test_pop_at():
            if False:
                while True:
                    i = 10
            a = [1, 2, 3, 4]
            b = a.pop(1)
            return b == 2
        self.checkScript(test_pop_at, ())

    def test_mutable_list_pop_at2(self):
        if False:
            return 10

        def test_pop_at2():
            if False:
                while True:
                    i = 10
            a = [1, 2, 3, 4]
            b = a.pop(1)
            return len(a) == 3
        self.checkScript(test_pop_at2, ())

    def test_mutable_list_pop_at_negative(self):
        if False:
            print('Hello World!')

        def test_pop_at_negative():
            if False:
                i = 10
                return i + 15
            a = [1, 2, 3, 4]
            b = a.pop(-2)
            return b == 3
        self.checkScript(test_pop_at_negative, ())

    def test_mutable_list_pop_at_negative2(self):
        if False:
            print('Hello World!')

        def test_pop_at_negative2():
            if False:
                while True:
                    i = 10
            a = [1, 2, 3, 4]
            b = a.pop(-2)
            return len(a) == 3
        self.checkScript(test_pop_at_negative2, ())

    def test_mutable_list_pop_slice(self):
        if False:
            for i in range(10):
                print('nop')

        def test_pop_slice():
            if False:
                while True:
                    i = 10
            a = [1, 2, 3, 4]
            b = [1, 2, 3, 4]
            a.pop()
            b = b[:-1]
            return a == b
        self.checkScript(test_pop_slice, ())

    def test_mutable_list_clear_empty(self):
        if False:
            i = 10
            return i + 15

        def test_clear_empty():
            if False:
                print('Hello World!')
            a = torch.jit.annotate(List[int], [])
            a.clear()
            return len(a) == 0
        self.checkScript(test_clear_empty, ())

    def test_mutable_list_clear(self):
        if False:
            while True:
                i = 10

        def test_clear():
            if False:
                for i in range(10):
                    print('nop')
            a = [1, 2, 3, 4]
            a.clear()
            return len(a) == 0
        self.checkScript(test_clear, ())

    def test_mutable_list_insert(self):
        if False:
            return 10

        def test_list_insert():
            if False:
                i = 10
                return i + 15
            a = [1, 2, 3, 4]
            a.insert(2, 5)
            return a == [1, 2, 5, 3, 4]
        self.checkScript(test_list_insert, ())

    def test_mutable_list_insert_negative(self):
        if False:
            for i in range(10):
                print('nop')

        def test_list_insert_negative():
            if False:
                return 10
            a = [1, 2, 3, 4]
            a.insert(-1, 5)
            return a == [1, 2, 3, 5, 4]
        self.checkScript(test_list_insert_negative, ())

    def test_mutable_list_insert_neg_out_of_bounds(self):
        if False:
            for i in range(10):
                print('nop')

        def test_list_insert_neg_out_of_bounds():
            if False:
                return 10
            a = [1, 2, 3, 4]
            a.insert(-10, 5)
            return a == [5, 1, 2, 3, 4]
        self.checkScript(test_list_insert_neg_out_of_bounds, ())

    def test_mutable_list_insert_out_of_bounds(self):
        if False:
            return 10

        def test_list_insert_out_of_bounds():
            if False:
                while True:
                    i = 10
            a = [1, 2, 3, 4]
            a.insert(10, 5)
            return a == [1, 2, 3, 4, 5]
        self.checkScript(test_list_insert_out_of_bounds, ())

    def test_mutable_list_remove_not_existing(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def test_list_remove_not_existing():
            if False:
                i = 10
                return i + 15
            a = [1, 2, 3, 4]
            a.remove(5)
            return a
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'x not in list', 'a.remove'):
            test_list_remove_not_existing()

    def test_mutable_list_remove(self):
        if False:
            return 10

        def test_list_remove():
            if False:
                return 10
            a = [1, 2, 3, 4]
            a.remove(3)
            return a == [1, 2, 4]
        self.checkScript(test_list_remove, ())

        def test_str_list_remove():
            if False:
                return 10
            a = ['foo', 'bar']
            a.remove('foo')
            return a == ['bar']
        self.checkScript(test_str_list_remove, ())

    def test_list_index_not_existing(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def list_index_not_existing():
            if False:
                for i in range(10):
                    print('nop')
            a = [4, 1, 3, 2]
            i = a.index(5)
            return i
        with self.assertRaisesRegexWithHighlight(RuntimeError, "'5' is not in list", 'a.index'):
            list_index_not_existing()

    def test_list_index(self):
        if False:
            print('Hello World!')

        def list_index():
            if False:
                return 10
            a = [4, 1, 3, 2]
            i = a.index(3)
            return i == 2
        self.checkScript(list_index, ())

        def list_str_index():
            if False:
                print('Hello World!')
            a = ['foo', 'bar']
            i = a.index('bar')
            return i == 1
        self.checkScript(list_str_index, ())

    def test_tensor_list_index(self):
        if False:
            while True:
                i = 10

        def tensor_list_index():
            if False:
                i = 10
                return i + 15
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(3), torch.tensor(2)]
            i = a.index(torch.tensor(3))
            return i == 2
        self.checkScript(tensor_list_index, ())

    def test_tensor_list_index_not_existing(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def tensor_list_index_not_existing():
            if False:
                i = 10
                return i + 15
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(3), torch.tensor(2)]
            i = a.index(torch.tensor(5))
            return i
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'is not in list', 'a.index'):
            tensor_list_index_not_existing()

    def test_list_count(self):
        if False:
            return 10

        def list_count():
            if False:
                for i in range(10):
                    print('nop')
            a = [4, 1, 4, 2, 4]
            i = a.count(4)
            return i == 3
        self.checkScript(list_count, ())

        def list_str_count():
            if False:
                for i in range(10):
                    print('nop')
            a = ['foo', 'bar', 'foo']
            i = a.count('foo')
            return i == 2
        self.checkScript(list_str_count, ())

    def test_list_count_not_existing(self):
        if False:
            return 10

        def list_count_not_existing():
            if False:
                i = 10
                return i + 15
            a = [4, 1, 4, 2, 4]
            i = a.count(5)
            return i == 0
        self.checkScript(list_count_not_existing, ())

    def test_tensor_list_count(self):
        if False:
            for i in range(10):
                print('nop')

        def tensor_list_count():
            if False:
                for i in range(10):
                    print('nop')
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(4), torch.tensor(4)]
            i = a.count(torch.tensor(4))
            return i == 3
        self.checkScript(tensor_list_count, ())

    def test_tensor_list_count_not_existing(self):
        if False:
            return 10

        def tensor_list_count_not_existing():
            if False:
                return 10
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(4), torch.tensor(4)]
            i = a.count(torch.tensor(5))
            return i == 0
        self.checkScript(tensor_list_count_not_existing, ())

    def test_mutable_list_remove_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        def test_list_remove_tensor():
            if False:
                for i in range(10):
                    print('nop')
            a = [torch.ones(1), torch.zeros(1), torch.ones(2)]
            a.remove(torch.zeros(1))
            return len(a) == 2
        self.checkScript(test_list_remove_tensor, ())

    def test_mutable_list_remove2(self):
        if False:
            while True:
                i = 10

        def test_list_remove2():
            if False:
                while True:
                    i = 10
            a = [1]
            a.remove(1)
            return len(a) == 0
        self.checkScript(test_list_remove2, ())

    def test_extend_list_mutable(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def extend_list(a: List[Tensor], b: List[Tensor]) -> List[Tensor]:
            if False:
                i = 10
                return i + 15
            a.extend(b)
            return a
        for l in [[], [torch.rand(2)], [torch.rand(2), torch.rand(2), torch.rand(2)]]:
            for r in [[], [torch.rand(2)], [torch.rand(2), torch.rand(2), torch.rand(2)]]:
                self.assertEqual(extend_list(l, r), l + r)

    def test_extend_list_immutable(self):
        if False:
            return 10

        @torch.jit.script
        def extend_list(a: List[int], b: List[int]) -> List[int]:
            if False:
                print('Hello World!')
            a.extend(b)
            return a
        for l in [[], [1], [1, 2, 3]]:
            for r in [[], [1], [1, 2, 3]]:
                self.assertEqual(extend_list(l, r), l + r)

    def test_copy_list_mutable(self):
        if False:
            return 10

        @torch.jit.script
        def copy_list(a: List[Tensor]) -> List[Tensor]:
            if False:
                print('Hello World!')
            return a.copy()
        for l in [[], [torch.rand(2)], [torch.rand(2), torch.rand(2), torch.rand(2)]]:
            self.assertEqual(copy_list(l), l)

    def test_copy_list_immutable(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def copy_list(a: List[int]) -> List[int]:
            if False:
                for i in range(10):
                    print('nop')
            return a.copy()
        for l in [[], [1], [1, 2, 3]]:
            self.assertEqual(copy_list(l), l)

    def test_min_max_single_list(self):
        if False:
            print('Hello World!')

        def min_intlist(li: List[int]) -> int:
            if False:
                return 10
            return min(li)

        def max_intlist(li: List[int]) -> int:
            if False:
                print('Hello World!')
            return max(li)

        def min_boollist(li: List[bool]) -> bool:
            if False:
                i = 10
                return i + 15
            return min(li)

        def max_boollist(li: List[bool]) -> bool:
            if False:
                print('Hello World!')
            return max(li)

        def min_floatlist(li: List[float]) -> float:
            if False:
                i = 10
                return i + 15
            return min(li)

        def max_floatlist(li: List[float]) -> float:
            if False:
                print('Hello World!')
            return max(li)
        int_lists = ([1], [2, 1, 2], [-3, 4, 2], [-2, -7, 1, 4], [2, 1, 0, 4], [])

        def check_list(fn, li):
            if False:
                for i in range(10):
                    print('nop')
            if len(li) == 0:
                self.checkScriptRaisesRegex(fn, (li,), Exception, 'arg is an empty sequence')
            else:
                self.checkScript(fn, (li,))
        for int_list in int_lists:
            check_list(min_intlist, int_list)
            check_list(max_intlist, int_list)
            bool_li = [bool(x) for x in int_list]
            check_list(min_boollist, bool_li)
            check_list(max_boollist, bool_li)
            float_li = [float(x) for x in int_list]
            check_list(min_floatlist, float_li)
            check_list(max_floatlist, float_li)

    def test_to_list(self):
        if False:
            return 10
        'Unit tests for Tensor.tolist() function.'
        '\n        Boolean dtype unit tests.\n        '

        def to_list_bool_0D(x: torch.Tensor) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            li = torch.jit.annotate(bool, x.tolist())
            return li

        def to_list_bool_1D(x: torch.Tensor) -> List[bool]:
            if False:
                print('Hello World!')
            li = torch.jit.annotate(List[bool], x.tolist())
            return li

        def to_list_bool_2D(x: torch.Tensor) -> List[List[bool]]:
            if False:
                for i in range(10):
                    print('nop')
            li = torch.jit.annotate(List[List[bool]], x.tolist())
            return li

        def to_list_bool_3D(x: torch.Tensor) -> List[List[List[bool]]]:
            if False:
                print('Hello World!')
            li = torch.jit.annotate(List[List[List[bool]]], x.tolist())
            return li
        self.checkScript(to_list_bool_0D, (torch.tensor(False, dtype=torch.bool),))
        bool_input_1D = torch.tensor([True, False, True, False], dtype=torch.bool)
        self.checkScript(to_list_bool_1D, (bool_input_1D,))
        bool_input_2D = torch.tensor([[True, True, False], [False, True, False]], dtype=torch.bool)
        self.checkScript(to_list_bool_2D, (bool_input_2D,))
        bool_input_3D = torch.tensor([[[True, False], [False, True]], [[True, False], [False, False]]], dtype=torch.bool)
        self.checkScript(to_list_bool_3D, (bool_input_3D,))
        bool_input_noncontiguous = torch.tensor([[[True, False], [False, True]], [[True, False], [False, False]]], dtype=torch.bool).transpose(0, 1)
        self.checkScript(to_list_bool_3D, (bool_input_noncontiguous,))
        '\n        Int dtype unit tests.\n        '

        def to_list_int_0D(x: torch.Tensor) -> int:
            if False:
                print('Hello World!')
            li = torch.jit.annotate(int, x.tolist())
            return li

        def to_list_int_1D(x: torch.Tensor) -> List[int]:
            if False:
                return 10
            li = torch.jit.annotate(List[int], x.tolist())
            return li

        def to_list_int_2D(x: torch.Tensor) -> List[List[int]]:
            if False:
                i = 10
                return i + 15
            li = torch.jit.annotate(List[List[int]], x.tolist())
            return li

        def to_list_int_3D(x: torch.Tensor) -> List[List[List[int]]]:
            if False:
                while True:
                    i = 10
            li = torch.jit.annotate(List[List[List[int]]], x.tolist())
            return li
        self.checkScript(to_list_int_0D, (torch.tensor(1, dtype=torch.long),))
        int_input_1D = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        self.checkScript(to_list_int_1D, (int_input_1D,))
        int_input_2D = torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=torch.long)
        self.checkScript(to_list_int_2D, (int_input_2D,))
        int_input_3D = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.long)
        self.checkScript(to_list_int_3D, (int_input_3D,))
        int_input_noncontiguous = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.long).transpose(0, 1)
        self.checkScript(to_list_int_3D, (int_input_noncontiguous,))
        '\n        Float dtype unit tests.\n        '

        def to_list_float_0D(x: torch.Tensor) -> float:
            if False:
                i = 10
                return i + 15
            li = torch.jit.annotate(float, x.tolist())
            return li

        def to_list_float_1D(x: torch.Tensor) -> List[float]:
            if False:
                print('Hello World!')
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        def to_list_float_2D(x: torch.Tensor) -> List[List[float]]:
            if False:
                for i in range(10):
                    print('nop')
            li = torch.jit.annotate(List[List[float]], x.tolist())
            return li

        def to_list_float_3D(x: torch.Tensor) -> List[List[List[float]]]:
            if False:
                for i in range(10):
                    print('nop')
            li = torch.jit.annotate(List[List[List[float]]], x.tolist())
            return li
        self.checkScript(to_list_float_0D, (torch.randn(5, dtype=torch.float)[0],))
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.float),))
        self.checkScript(to_list_float_2D, (torch.randn(5, 6, dtype=torch.float),))
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.float),))
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.float).transpose(0, 1),))
        self.checkScript(to_list_float_0D, (torch.randn(5, dtype=torch.double)[0],))
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.double),))
        self.checkScript(to_list_float_2D, (torch.randn(5, 6, dtype=torch.double),))
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.double),))
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.double).transpose(0, 1),))
        '\n        Complex dtype unit tests.\n        '

        def to_list_complex_0D(x: torch.Tensor) -> complex:
            if False:
                for i in range(10):
                    print('nop')
            li = torch.jit.annotate(complex, x.tolist())
            return li

        def to_list_complex_1D(x: torch.Tensor) -> List[complex]:
            if False:
                return 10
            li = torch.jit.annotate(List[complex], x.tolist())
            return li

        def to_list_complex_2D(x: torch.Tensor) -> List[List[complex]]:
            if False:
                print('Hello World!')
            li = torch.jit.annotate(List[List[complex]], x.tolist())
            return li

        def to_list_complex_3D(x: torch.Tensor) -> List[List[List[complex]]]:
            if False:
                while True:
                    i = 10
            li = torch.jit.annotate(List[List[List[complex]]], x.tolist())
            return li
        self.checkScript(to_list_complex_0D, (torch.randn(5, dtype=torch.cfloat)[0],))
        self.checkScript(to_list_complex_1D, (torch.randn(5, dtype=torch.cfloat),))
        self.checkScript(to_list_complex_2D, (torch.randn(5, 6, dtype=torch.cfloat),))
        self.checkScript(to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cfloat),))
        self.checkScript(to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cfloat).transpose(0, 1),))
        self.checkScript(to_list_complex_0D, (torch.randn(5, dtype=torch.cdouble)[0],))
        self.checkScript(to_list_complex_1D, (torch.randn(5, dtype=torch.cdouble),))
        self.checkScript(to_list_complex_2D, (torch.randn(5, 6, dtype=torch.cdouble),))
        self.checkScript(to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cdouble),))
        self.checkScript(to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cdouble).transpose(0, 1),))
        "\n        Non-happy path tests:\n            - missing type annotation\n            - mismatch between type annotation and input\n            - type annotation with unsupported type\n            - type annotation with the wrong dimension\n            - type annotation with scalar type that doesn't match the input scalar type\n        "

        def to_list_missing_type_annotation(x: torch.Tensor) -> List[float]:
            if False:
                print('Hello World!')
            li = x.tolist()
            return li

        def to_list_incorrect_type_annotation(x: torch.Tensor) -> List[float]:
            if False:
                print('Hello World!')
            li = torch.jit.annotate(float, x.tolist())
            return li

        def to_list_unsupported_type_annotation(x: torch.Tensor) -> List[float]:
            if False:
                while True:
                    i = 10
            li = torch.jit.annotate(List[str], x.tolist())
            return li

        def to_list_type_annotation_wrong_dim(x: torch.Tensor) -> List[List[float]]:
            if False:
                for i in range(10):
                    print('nop')
            li = torch.jit.annotate(List[List[float]], x.tolist())
            return li

        def to_list_type_annotation_incorrect_scalar_type(x: torch.Tensor) -> List[float]:
            if False:
                i = 10
                return i + 15
            li = torch.jit.annotate(List[float], x.tolist())
            return li
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Expected type hint for result of tolist()', 'x.tolist('):
            self.checkScript(to_list_missing_type_annotation, (torch.randn(5),))
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Return value was annotated as having type List\\[float\\] but is actually of type float', 'return li'):
            self.checkScript(to_list_incorrect_type_annotation, (torch.randn(5),))
        with self.assertRaisesRegex(RuntimeError, 'str is not one of the supported element types for tolist'):
            self.checkScript(to_list_unsupported_type_annotation, (torch.randn(5),))
        with self.assertRaisesRegex(RuntimeError, 'Output annotation list dimension and runtime tensor dimension must match'):
            self.checkScript(to_list_type_annotation_wrong_dim, (torch.randn(5, dtype=torch.double),))
        with self.assertRaisesRegex(RuntimeError, 'Output annotation element type and runtime tensor element type must match'):
            self.checkScript(to_list_type_annotation_incorrect_scalar_type, (torch.ones(5, dtype=torch.long),))

    def test_to_list_gpu(self):
        if False:
            i = 10
            return i + 15
        'GPU tests for Tensor.tolist() function.'
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            self.skipTest('CUDA is not available')

        def to_list_bool_1D(x: torch.Tensor) -> List[bool]:
            if False:
                i = 10
                return i + 15
            li = torch.jit.annotate(List[bool], x.tolist())
            return li

        def to_list_int_1D(x: torch.Tensor) -> List[int]:
            if False:
                i = 10
                return i + 15
            li = torch.jit.annotate(List[int], x.tolist())
            return li

        def to_list_float_1D(x: torch.Tensor) -> List[float]:
            if False:
                return 10
            li = torch.jit.annotate(List[float], x.tolist())
            return li
        self.checkScript(to_list_bool_1D, (torch.tensor([True, False, True, False], dtype=torch.bool).cuda(),))
        self.checkScript(to_list_int_1D, (torch.tensor([1, 2, 3, 4], dtype=torch.long).cuda(),))
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.double).cuda(),))

    def test_no_element_type_annotation(self):
        if False:
            return 10

        def fn_with_comment(x: torch.Tensor) -> List:
            if False:
                return 10
            a: List = x.tolist()
            return a

        def annotated_fn(x: torch.Tensor) -> List:
            if False:
                return 10
            a: List = x.tolist()
            return a
        with self.assertRaisesRegex(RuntimeError, 'Attempted to use List without a contained type'):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))
        with self.assertRaisesRegex(RuntimeError, 'Attempted to use List without a contained type'):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))
        with self.assertRaisesRegex(RuntimeError, 'Attempted to use List without a contained type'):
            torch.jit.script(fn_with_comment)
        with self.assertRaisesRegex(RuntimeError, 'Attempted to use List without a contained type'):
            torch.jit.script(annotated_fn)

    def test_list_none(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'Can not create ListType with None type'):
            x = torch._C.ListType(None)

    def test_list_unification_hint(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'Expected an annotation of type List'):

            @torch.jit.script
            def x():
                if False:
                    print('Hello World!')
                b: int = [2, 3]
                return b

class TestDict(JitTestCase):

    def dict(self):
        if False:
            print('Hello World!')
        return {u'a': torch.ones(1), u'b': torch.ones(1) + 1, u'c': torch.ones(1) + 2}

    def dict2(self):
        if False:
            return 10
        return {'x': torch.ones(1) + 100, 'y': torch.ones(1) + 101, 'z': torch.ones(1) + 102}

    def dict_bool(self):
        if False:
            print('Hello World!')
        return {True: 1}

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_dict_bool_conversion(self):
        if False:
            i = 10
            return i + 15

        def if_predicate(d: Dict[int, int]):
            if False:
                return 10
            if d:
                (s, t) = (0, 0)
                for (k, v) in d.items():
                    s += k
                    t += v
                return (s, t)
            else:
                return (-1, -1)
        self.checkScript(if_predicate, ({1: 2, 3: 5},))
        self.checkScript(if_predicate, ({},))

        def while_predicate(d: Dict[int, int]):
            if False:
                while True:
                    i = 10
            while d:
                d.clear()
        self.checkScript(while_predicate, ({1: 2, 3: 5},))
        self.checkScript(while_predicate, ({},))

        def ternary_predicate(d: Dict[int, int]):
            if False:
                print('Hello World!')
            return 'non-empty' if d else 'empty'
        self.checkScript(ternary_predicate, ({1: 2, 3: 5},))
        self.checkScript(ternary_predicate, ({},))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_del(self):
        if False:
            for i in range(10):
                print('nop')

        def inputs():
            if False:
                i = 10
                return i + 15
            return {'hi': 2, 'bye': 3}

        def fn(x: Dict[str, int]) -> Dict[str, int]:
            if False:
                return 10
            del x['hi']
            return x
        python_out = fn(inputs())
        cu = torch.jit.CompilationUnit()
        cu.define(dedent(inspect.getsource(fn)))
        self.assertEqual(cu.fn(inputs()), python_out)
        self.assertEqual(torch.jit.script(fn)(inputs()), python_out)
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'KeyError', "x['hi']"):
            self.checkScript(fn, [{}])

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_dict_variance(self):
        if False:
            return 10
        "\n        `Dict[T1, _]` is not a subtype of `Dict[T2, _]`, even if `T1` is\n        a subtype of `T2`; similarly `Dict[_, T1]` would not be a\n        subtype of `Dict[_, T2]`.\n\n        However, if we have a temporary dict object (that is, a dict\n        comprehension or a dict literal) on the rhs of an assignment\n        statement, we want to ignore the inferred type of the rhs if we\n        can prove that: 1) both the lhs and the rhs are dicts with the\n        same key types (TorchScript has a restricted set of allowed key\n        types, so we don't need to worry about subtyping relationships\n        here), and 2) the value type of the dict is a subtype of the\n        value type of the rhs dict.\n        "

        def test_dictliteral_is_typed_from_annotation():
            if False:
                for i in range(10):
                    print('nop')
            x: Dict[str, Optional[int]] = {'foo': None, 'bar': None, 'baz': None}
            return x
        self.checkScript(test_dictliteral_is_typed_from_annotation, ())

        def test_dictcomprehension_is_typed_from_annotation():
            if False:
                for i in range(10):
                    print('nop')
            metasyntactics = ['foo', 'bar', 'baz']
            x: Dict[str, Optional[int]] = {word: None for word in metasyntactics}
            return x
        self.checkScript(test_dictcomprehension_is_typed_from_annotation, ())

        def test_dicts_with_different_value_types_are_invariant(self):
            if False:
                return 10
            x: Dict[str, int] = {'foo': 1, 'bar': 2, 'baz': 3}
            y: Dict[str, Optional[int]] = x
            return x
        with self.assertRaisesRegex(RuntimeError, "Variable 'y' is annotated with type Dict\\[str, Optional\\[int\\]\\] but is being assigned to a value of type Dict\\[str, int\\]"):
            torch.jit.script(test_dicts_with_different_value_types_are_invariant)

        def test_dicts_with_different_value_types_are_invariant_recursive(self):
            if False:
                i = 10
                return i + 15
            x: Dict[str, int] = {'foo': 1, 'bar': 2, 'baz': 3}
            y: Dict[str, Dict[str, int]] = {'foo': x, 'bar': x, 'baz': x}
            z: Dict[str, Dict[str, Optional[int]]] = y
            return x
        with self.assertRaisesRegex(RuntimeError, "Variable 'z' is annotated with type Dict\\[str, Dict\\[str, Optional\\[int\\]\\]\\] but is being assigned to a value of type Dict\\[str, Dict\\[str, int\\]\\]"):
            torch.jit.script(test_dicts_with_different_value_types_are_invariant_recursive)

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_keys(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def keys(x: Dict[str, Tensor]) -> List[str]:
            if False:
                i = 10
                return i + 15
            return list(x.keys())
        self.assertEqual(set(keys(self.dict())), set(self.dict().keys()))

        @torch.jit.script
        def specialized_list():
            if False:
                return 10
            li = {1: 1, 2: 2}.keys()
            li.append(3)
            return li
        self.assertTrue(set(specialized_list()) == {1, 2, 3})

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_values(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def values(x: Dict[str, Tensor]) -> List[Tensor]:
            if False:
                i = 10
                return i + 15
            return list(x.values())
        the_dict = self.dict()
        self.assertEqual(set(values(the_dict)), set(the_dict.values()))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_len(self):
        if False:
            print('Hello World!')

        def length(x: Dict[str, Tensor]) -> int:
            if False:
                while True:
                    i = 10
            return len(x)
        self.checkScript(length, (self.dict(),))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_copy(self):
        if False:
            return 10

        def func(x: Dict[str, Tensor]) -> Dict[str, Tensor]:
            if False:
                print('Hello World!')
            return x.copy()
        self.checkScript(func, (self.dict(),))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_items(self):
        if False:
            return 10

        def func(x: Dict[str, Tensor]) -> List[Tuple[str, Tensor]]:
            if False:
                return 10
            return x.items()
        scripted_func = torch.jit.script(func)
        eager_out = func(self.dict())
        script_out = scripted_func(self.dict())
        self.assertEqual(len(eager_out), len(script_out))
        for item in eager_out:
            self.assertTrue(item in script_out)

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_pop(self):
        if False:
            print('Hello World!')

        def pop(x: Dict[str, Tensor], key: str) -> Tuple[Tensor, Dict[str, Tensor]]:
            if False:
                while True:
                    i = 10
            return (x.pop(key), x)

        def tester(fn, *args):
            if False:
                while True:
                    i = 10
            eager_out = fn(self.dict(), *args)
            script_out = torch.jit.script(fn)(self.dict(), *args)
            self.assertEqual(eager_out, script_out)
        tester(pop, 'a')
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'KeyError', 'x.pop'):
            torch.jit.script(pop)(self.dict(), 'x')

        def default_pop(x: Dict[str, Tensor], key: str, default: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
            if False:
                i = 10
                return i + 15
            return (x.pop(key, default), x)
        tester(default_pop, 'a', torch.randn(2, 2))
        tester(default_pop, 'x', torch.randn(2, 2))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_setdefault(self):
        if False:
            i = 10
            return i + 15

        def setdefault(x: Dict[str, Tensor], key: str, default: Tensor) -> Dict[str, Tensor]:
            if False:
                for i in range(10):
                    print('nop')
            x.setdefault(key, default)
            return x
        self.checkScript(setdefault, (self.dict(), 'a', torch.randn(2, 2)))
        self.checkScript(setdefault, (self.dict(), 'nonexistant', torch.randn(2, 2)))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_update(self):
        if False:
            return 10

        def update(a: Dict[str, Tensor], b: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
            if False:
                i = 10
                return i + 15
            a.update(b)
            return (a, b)
        self.checkScript(update, (self.dict(), self.dict()))
        self.checkScript(update, (self.dict(), self.dict2()))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_update_existing_key(self):
        if False:
            i = 10
            return i + 15

        def foo() -> Dict[str, int]:
            if False:
                print('Hello World!')
            a: Dict[str, int] = {}
            for i in range(3):
                a.update({'a': i})
            return a
        self.checkScript(foo, ())

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_aug_assign(self):
        if False:
            i = 10
            return i + 15

        def aug_assign_dict_tensor(a: Dict[str, Tensor]) -> Dict[str, Tensor]:
            if False:
                while True:
                    i = 10
            a['a'] += 1
            a['b'] -= 12
            a['c'] *= 122
            a['c'] /= 2
            a['c'] %= 2
            return a

        def aug_assign_dict_prim(a: Dict[str, float]) -> Dict[str, float]:
            if False:
                for i in range(10):
                    print('nop')
            a['a'] += 3.4
            a['b'] -= 2.4
            a['c'] *= 3.0
            a['c'] /= 2.0
            a['c'] %= 2.0
            return a
        self.checkScript(aug_assign_dict_tensor, (self.dict(),))
        self.checkScript(aug_assign_dict_prim, ({'a': 3.0, 'b': 2.0, 'c': 4.0},))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_popitem(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def popitem(x: Dict[str, Tensor]) -> Tuple[Tuple[str, Tensor], Dict[str, Tensor]]:
            if False:
                for i in range(10):
                    print('nop')
            item = x.popitem()
            return (item, x)
        eager_in = self.dict()
        eager_out = (eager_in.popitem(), eager_in)
        script_out = popitem(self.dict())
        self.assertEqual(len(eager_out[1]), len(script_out[1]))
        self.assertTrue(isinstance(script_out[0][0], str))
        self.assertTrue(isinstance(script_out[0][1], torch.Tensor))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_clear(self):
        if False:
            while True:
                i = 10

        def clear(x: Dict[str, Tensor]) -> Dict[str, Tensor]:
            if False:
                while True:
                    i = 10
            x.clear()
            return x
        self.checkScript(clear, (self.dict(),))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_get(self):
        if False:
            while True:
                i = 10

        def get(x: Dict[str, Tensor], key: str) -> Optional[Tensor]:
            if False:
                for i in range(10):
                    print('nop')
            return x.get(key)
        self.checkScript(get, (self.dict(), 'a'))
        self.checkScript(get, (self.dict(), "doesn't exist"))

        def get_default(x: Dict[str, Tensor], key: str) -> Optional[Tensor]:
            if False:
                for i in range(10):
                    print('nop')
            return x.get(key, torch.randn(2, 2))
        self.checkScript(get, (self.dict(), 'a'))
        self.checkScript(get, (self.dict(), "doesn't exist"))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_get_boolkey(self):
        if False:
            while True:
                i = 10

        def get(x: Dict[bool, int], key: bool) -> Optional[int]:
            if False:
                i = 10
                return i + 15
            return x.get(key)
        self.checkScript(get, (self.dict_bool(), True))
        self.checkScript(get, (self.dict_bool(), False))

        def get_default(x: Dict[bool, int], key: bool) -> int:
            if False:
                return 10
            return x.get(key, 42)
        self.checkScript(get_default, (self.dict_bool(), True))
        self.checkScript(get_default, (self.dict_bool(), False))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_basic(self):
        if False:
            i = 10
            return i + 15

        def simple(x: Dict[str, int]) -> Dict[str, int]:
            if False:
                return 10
            return x
        self.checkScript(simple, ({'item': 20, 'other_item': 120},))

        def index(x: Dict[str, int]) -> int:
            if False:
                while True:
                    i = 10
            return x['item']
        self.checkScript(index, ({'item': 20, 'other_item': 120},))

        def type_default() -> Dict[str, Tensor]:
            if False:
                print('Hello World!')
            return {}
        self.checkScript(type_default, ())

        @torch.jit.script
        def missing_index(x: Dict[str, int]) -> int:
            if False:
                return 10
            return x['dne']
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'KeyError', "x['dne'"):
            missing_index({'item': 20, 'other_item': 120})
        code = dedent('\n            def literal1():\n                return torch.jit.annotate(Dict[int, float], {})\n            def literal2():\n                return torch.jit.annotate(Dict[int, float], {10: 1.2})\n        ')
        cu = torch.jit.CompilationUnit(code)
        self.assertEqual({}, cu.literal1())
        self.assertEqual({10: 1.2}, cu.literal2())
        cu = torch.jit.CompilationUnit(dedent('\n            def literal3():\n                return torch.jit.annotate(Dict[int, float], {10: 1.2, 11: 1.3})\n        '))
        self.assertEqual({10: 1.2, 11: 1.3}, cu.literal3())

        def list_of_dicts() -> List[Dict[str, Tensor]]:
            if False:
                for i in range(10):
                    print('nop')
            return [{'word': torch.ones(2) + 3}, {'other word': torch.ones(1) + 2}]
        self.checkScript(list_of_dicts, ())

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_mutability(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fn() -> Dict[str, int]:
            if False:
                return 10
            a = torch.jit.annotate(Dict[str, int], {})
            a['ok'] = 10
            return a
        self.assertEqual(fn(), {'ok': 10})

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_key_type(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'but instead found type', 'a[None]'):

            @torch.jit.script
            def fn(a: Dict[str, int]) -> int:
                if False:
                    i = 10
                    return i + 15
                return a[None]

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_loop(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fn(x: int) -> Dict[str, int]:
            if False:
                return 10
            a = torch.jit.annotate(Dict[str, int], {})
            for i in range(x):
                a['ok'] = i
            return a
        self.assertEqual(fn(10), {'ok': 9})

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_view(self):
        if False:
            i = 10
            return i + 15

        def fn(x, y):
            if False:
                print('Hello World!')
            l = {'a': x}
            x_view = l['a']
            a = x + x
            x_view.add_(y)
            b = x + x
            return a == b
        self.checkScript(fn, (torch.rand(2, 3), torch.rand(2, 3)))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_membership(self):
        if False:
            i = 10
            return i + 15

        def fn(x: Dict[int, int], y: int) -> int:
            if False:
                return 10
            return x.get(y, 3)
        d = {1: 2, 3: 4}
        self.checkScript(fn, (d, 3))
        self.checkScript(fn, (d, 2))

        def optional(x: Dict[int, int], y: int) -> bool:
            if False:
                return 10
            res = x.get(y)
            return res is None
        self.checkScript(fn, (d, 3))
        self.checkScript(fn, (d, 2))
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'is actually of type Optional', 'return x.get(y'):

            @torch.jit.script
            def bad_types(x: Dict[int, int], y: int) -> int:
                if False:
                    return 10
                return x.get(y)

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_dict_to_python(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.ignore
        def python_lookup(my_dict: Dict[str, int], keys: List[str]) -> List[int]:
            if False:
                while True:
                    i = 10
            return [my_dict[k] for k in keys]

        def fn(my_dict: Dict[str, int], keys: List[str]) -> List[int]:
            if False:
                for i in range(10):
                    print('nop')
            return python_lookup(my_dict, keys)
        a_dict = {'a': torch.ones(1), 'b': torch.ones(1) + 1, 'c': torch.ones(1) + 2}
        self.checkScript(fn, (a_dict, ('a', 'c')))

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_ordered_dict(self):
        if False:
            return 10

        def test_func(fn, inputs):
            if False:
                print('Hello World!')
            self.assertEqual(fn(*inputs), torch.jit.script(fn)(*inputs))

        def repeated_key():
            if False:
                while True:
                    i = 10
            return OrderedDict([(1, 2), (2, 3), (1, 4)])
        test_func(repeated_key, ())

        def no_args():
            if False:
                for i in range(10):
                    print('nop')
            a = OrderedDict()
            a['one'] = torch.tensor(1)
            a['two'] = torch.tensor(2)
        test_func(no_args, ())

        def test_dict_constructor():
            if False:
                while True:
                    i = 10
            a = dict()
            a['one'] = torch.tensor(1)
            return (a, dict([(1, 2), (2, 3), (1, 4)]))
        test_func(test_dict_constructor, ())

        def test_dict_initializer_list():
            if False:
                i = 10
                return i + 15
            a = {'1': torch.tensor(1), '2': torch.tensor(2)}
            output_order = []
            for key in a:
                output_order.append(a[key])
            return output_order
        test_func(test_dict_initializer_list, ())

        def test_dict_error():
            if False:
                while True:
                    i = 10
            a = dict()
            a[1] = 2
            return a
        with self.assertRaisesRegexWithHighlight(Exception, 'Arguments for call are not', 'a[1] = 2'):
            torch.jit.script(test_dict_error)

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_type_annotation_missing_contained_type(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that the use of a Dict type annotation without contained\n        key and value types produces an error.\n        '

        def fn_with_comment(input: Dict) -> Any:
            if False:
                i = 10
                return i + 15
            return input

        def annotated_fn(input: Dict) -> Any:
            if False:
                for i in range(10):
                    print('nop')
            return input
        with self.assertRaisesRegex(RuntimeError, 'Attempted to use Dict without contained types'):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))
        with self.assertRaisesRegex(RuntimeError, 'Attempted to use Dict without contained types'):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))
        with self.assertRaisesRegex(RuntimeError, 'Attempted to use Dict without contained types'):
            m = torch.jit.script(fn_with_comment)
        with self.assertRaisesRegex(RuntimeError, 'Attempted to use Dict without contained types'):
            m = torch.jit.script(annotated_fn)

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_dict_preserves_order(self):
        if False:
            while True:
                i = 10

        def dict_ordering():
            if False:
                i = 10
                return i + 15
            a: Dict[int, int] = {}
            for i in range(1000):
                a[i] = i + 1
            return a
        self.checkScript(dict_ordering, ())
        di = torch.jit.script(dict_ordering)()
        res = list(di.items())
        for i in range(1000):
            (key, value) = res[i]
            self.assertTrue(key == i and value == i + 1)

    @skipIfTorchDynamo('TorchDynamo fails for this test for unknown reason')
    def test_optional_dict_construct(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def use(self, buffer: Dict[str, Optional[torch.Tensor]]):
                if False:
                    for i in range(10):
                        print('nop')
                return buffer['prev_key']

            def forward(self, x):
                if False:
                    print('Hello World!')
                prev_key = torch.rand(2, 3)
                next_key = torch.rand(2, 3)
                saved_state: Dict[str, Optional[torch.Tensor]] = {'prev_key': prev_key, 'next_key': next_key}
                return self.use(saved_state)
        self.checkModule(M(), (torch.rand(2, 2),))

class TestNamedTuple(JitTestCase):

    def test_namedtuple(self):
        if False:
            while True:
                i = 10

        class FeatureVector(NamedTuple):
            float_features: float
            sequence_features: List[float]
            time_since_first: float

        @torch.jit.script
        def foo(x) -> float:
            if False:
                while True:
                    i = 10
            fv = FeatureVector(3.0, [3.0], 3.0)
            rv = fv.float_features
            for val in fv.sequence_features:
                rv += val
            rv *= fv.time_since_first
            return rv
        self.assertEqual(foo(torch.rand(3, 4)), 18.0)

    def test_namedtuple_constant(self):
        if False:
            while True:
                i = 10

        class Tup(NamedTuple):
            a: int
            b: int

        @torch.jit.script
        def foo():
            if False:
                while True:
                    i = 10
            return Tup(1, 2)
        self.assertEqual(foo(), Tup(1, 2))

    def test_return_named_tuple(self):
        if False:
            for i in range(10):
                print('nop')

        class FeatureVector(NamedTuple):
            float_features: float
            sequence_features: List[float]
            time_since_first: float

        @torch.jit.script
        def foo(x):
            if False:
                print('Hello World!')
            fv = FeatureVector(3.0, [3.0], 3.0)
            return fv
        out = foo(torch.rand(3, 4))
        out = foo(torch.rand(3, 4))
        self.assertEqual(out.float_features, 3.0)
        self.assertEqual(out.sequence_features, [3.0])
        self.assertEqual(out.time_since_first, 3.0)

    def test_namedtuple_as_attr(self):
        if False:
            i = 10
            return i + 15

        class Config(NamedTuple):
            size: int

        class MyMod(nn.Module):
            configs: Dict[int, Config]

            def __init__(self, configs):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.configs = configs

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                for config in self.configs.values():
                    x += config.size
                return x
        s = torch.jit.script(MyMod({0: Config(size=16)}))

    def test_namedtuple_resolution(self):
        if False:
            return 10

        class TheType(NamedTuple):
            t: int

        class MyModule(types.ModuleType):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__('MyModule')

            def __getattr__(self, attr):
                if False:
                    print('Hello World!')
                return TheType
        some_module = MyModule()

        def fn() -> some_module.Type:
            if False:
                for i in range(10):
                    print('nop')
            return some_module.Type(1)
        self.checkScript(fn, [])

    def test_namedtuple_slice_unpack(self):
        if False:
            for i in range(10):
                print('nop')

        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo(a: int, b: float, c: List[int]):
            if False:
                print('Hello World!')
            tup = MyCoolNamedTuple(a, b, c)
            (my_a, my_b, my_c) = tup
            return (tup[:1], my_a, my_c)
        self.assertEqual(foo(3, 3.5, [6]), ((3,), 3, [6]))

    def test_namedtuple_lower(self):
        if False:
            for i in range(10):
                print('nop')

        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo(a: int):
            if False:
                for i in range(10):
                    print('nop')
            tup = MyCoolNamedTuple(a, 3.14, [9])
            return tup
        FileCheck().check('TupleConstruct').run(foo.graph)
        torch._C._jit_pass_lower_all_tuples(foo.graph)
        FileCheck().check_not('TupleConstruct').run(foo.graph)

    def test_namedtuple_type_annotation(self):
        if False:
            while True:
                i = 10
        global MyCoolNamedTuple

        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo(x: MyCoolNamedTuple) -> MyCoolNamedTuple:
            if False:
                return 10
            return x
        mnt = MyCoolNamedTuple(42, 420.0, [666])
        self.assertEqual(foo(mnt), mnt)

    def test_namedtuple_wrong_types(self):
        if False:
            for i in range(10):
                print('nop')

        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]
        with self.assertRaisesRegex(RuntimeError, "Expected a value of type 'int' for argument 'a' but instead found type 'str'"):

            @torch.jit.script
            def foo():
                if False:
                    i = 10
                    return i + 15
                tup = MyCoolNamedTuple('foo', 'bar', 'baz')
                return tup

    def test_namedtuple_kwarg_construct(self):
        if False:
            i = 10
            return i + 15

        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo():
            if False:
                i = 10
                return i + 15
            tup = MyCoolNamedTuple(c=[1, 2, 3], b=3.5, a=9)
            return tup
        tup = foo()
        self.assertEqual(tup.a, 9)
        self.assertEqual(tup.b, 3.5)
        self.assertEqual(tup.c, [1, 2, 3])

    @unittest.skipIf(True, 'broken while these tests were not in CI')
    def test_namedtuple_serialization(self):
        if False:
            return 10

        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        class MyMod(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self):
                if False:
                    i = 10
                    return i + 15
                return MyCoolNamedTuple(3, 3.5, [3, 4, 5])
        mm = MyMod()
        mm.save('foo.zip')
        torch.testing._internal.jit_utils.clear_class_registry()
        loaded = torch.jit.load('foo.zip')
        out = mm()
        out_loaded = loaded()
        for name in ['a', 'b', 'c']:
            self.assertEqual(getattr(out_loaded, name), getattr(out, name))

    def test_namedtuple_inside_forwardref(self):
        if False:
            while True:
                i = 10

        class FeatureVector(NamedTuple):
            float_features: 'float'
            sequence_features: 'List[float]'
            time_since_first: 'float'

        @torch.jit.script
        def foo(x) -> float:
            if False:
                return 10
            fv = FeatureVector(3.0, [3.0], 3.0)
            rv = fv.float_features
            for val in fv.sequence_features:
                rv += val
            rv *= fv.time_since_first
            return rv
        self.assertEqual(foo(torch.rand(3, 4)), 18.0)

    def test_namedtuple_input_forwardref(self):
        if False:
            while True:
                i = 10

        class MyNamedTuple(NamedTuple):
            a: 'int'
            b: 'float'
            c: 'torch.Tensor'
        make_global(MyNamedTuple)
        nt = MyNamedTuple(4, 2.5, torch.rand((2, 2)))

        def fn(obj: MyNamedTuple):
            if False:
                i = 10
                return i + 15
            return ((obj.c + obj.b) ** obj.a).sin()
        expected = fn(nt)
        fn_s = torch.jit.script(fn)
        actual = fn_s(nt)
        self.assertEqual(expected, actual)

    @unittest.expectedFailure
    def test_namedtuple_resolution_forwardref(self):
        if False:
            while True:
                i = 10

        class TheType(NamedTuple):
            t: 'int'

        class MyModule(types.ModuleType):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__('MyModule')

            def __getattr__(self, attr):
                if False:
                    return 10
                return TheType
        some_module = MyModule()

        def fn() -> some_module.Type:
            if False:
                print('Hello World!')
            return some_module.Type(1)
        self.checkScript(fn, [])

class TestScriptDict(JitTestCase):
    """
    This class contains a suite of tests for torch.jit.script, a
    function that returns a dictionary-like object that has reference
    semantics across the Python/TorchScript boundary. That is,
    it can be passed to a TorchScript function that mutates it
    and those modifications are visible in the scope of the Python
    caller of said TorchScript function.

    The vast majority of tests are for making sure that objects returned
    by torch.jit.script behave like dictionaries do so that they are fungible
    in almost all cirumstances with regular dictionaries.
    """

    def _script_dict_add(self, d: torch._C.ScriptDict, k: int, v: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        This is a helper function that inserts the pair (k, v) into the\n        dictionary d in TorchScript. It is used for testing reference\n        semantics.\n        '

        @torch.jit.script
        def dict_add(d: Dict[int, int], k: int, v: int):
            if False:
                for i in range(10):
                    print('nop')
            d[k] = v
        dict_add(d, k, v)

    def _compare_eager_and_script(self, fn, input_dict, script_input_dict=None):
        if False:
            return 10
        '\n        This is a helper function that facilitates comparing behaviour between\n        Python dictionaries and "scripted" dictionaries.\n\n        Args:\n            fn: The function to test and compare the behaviour of.\n            input_dict: The input dictionary to use for the test (passed to fn).\n            script_input_dict: The scripted input dictionary to use for the tests.\n                                If None, input_dict is scripted with torch.jit.script\n                                and used instead.\n        '
        script_input_dict = script_input_dict or torch.jit.script(input_dict)
        (eager_raised, script_raised) = (False, False)
        try:
            eager_out = fn(input_dict)
        except Exception as e:
            eager_exception = e
            eager_raised = True
        try:
            script_out = fn(script_input_dict)
        except Exception as e:
            script_exception = e
            script_raised = True
        self.assertEqual(eager_raised, script_raised)
        if eager_raised:
            self.assertEqual(type(eager_exception), type(script_exception))
        else:
            self.assertEqual(eager_out, script_out)
            self.assertEqual(input_dict, script_input_dict)

    def test_repr(self):
        if False:
            print('Hello World!')
        '\n        Test the __repr__ method.\n        '
        self._compare_eager_and_script(lambda d: repr(d), {1: 2})

    def test_bool(self):
        if False:
            print('Hello World!')
        '\n        Test the __bool__ method. This should return True\n        if the dictionary is non-empty and False otherwise.\n        '
        self._compare_eager_and_script(lambda d: bool(d), {1: 2})
        self._compare_eager_and_script(lambda d: bool(d), {})

    def test_iter(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test iteration over a dictionary's keys.\n        "

        def sum_keys(input_dict):
            if False:
                print('Hello World!')
            s = 0
            for k in input_dict:
                s += k
            return s
        self._compare_eager_and_script(sum_keys, {1: 2, 3: 4})

    def test_items(self):
        if False:
            return 10
        '\n        Test .items().\n        '

        def sum_pair_product(input_dict):
            if False:
                for i in range(10):
                    print('nop')
            s = 0
            for (k, v) in input_dict.items():
                s += k * v
            return s
        self._compare_eager_and_script(sum_pair_product, {1: 2, 3: 4})

    def test_getitem(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test accessing dictionary values using the [] operator.\n        '
        data = {1: 2, 3: 4}
        self._compare_eager_and_script(lambda d: d[1], data)
        self._compare_eager_and_script(lambda d: d[4], data)
        self._compare_eager_and_script(lambda d: d[2], data)
        self._compare_eager_and_script(lambda d: d['key'], data)

    def test_setitem(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test setting dictionary values using the [] operator.\n        '
        data = {1: 2, 3: 4}

        def fn(input_dict):
            if False:
                i = 10
                return i + 15
            input_dict[1] = 10
            input_dict[3] = 11
        self._compare_eager_and_script(fn, data)
        script_data = torch.jit.script(data)
        with self.assertRaises(TypeError):
            script_data['str'] = 3
        with self.assertRaises(TypeError):
            script_data[3] = 'str'

    def test_contains(self):
        if False:
            while True:
                i = 10
        '\n        Test membership checks (x in y, x not in y).\n        '
        data = {1: 2, 3: 4}

        def fn(input_dict):
            if False:
                i = 10
                return i + 15
            return (1 in input_dict, 2 not in input_dict, 3 in input_dict, 4 not in input_dict)
        self._compare_eager_and_script(fn, data)
        script_data = torch.jit.script(data)
        with self.assertRaises(KeyError):
            a = 'str' in script_data

    def test_delitem(self):
        if False:
            while True:
                i = 10
        '\n        Test deletion.\n        '
        data = {1: 2, 3: 4}

        def del_fn(input_dict):
            if False:
                print('Hello World!')
            del input_dict[1]

        def del_fn_raises(input_dict):
            if False:
                return 10
            del input_dict[10]
        self._compare_eager_and_script(del_fn, data)
        self._compare_eager_and_script(del_fn_raises, data)
        script_data = torch.jit.script(data)
        with self.assertRaises(TypeError):
            del script_data['str']

    def test_len(self):
        if False:
            i = 10
            return i + 15
        '\n        Test len() builtin function.\n        '
        self._compare_eager_and_script(lambda d: len(d), {1: 2})
        self._compare_eager_and_script(lambda d: len(d), {})

    @unittest.skip('Cannot pass until all dicts returned from TorchScript are ScriptDicts')
    def test_nested(self):
        if False:
            print('Hello World!')
        '\n        Test that reference semantics are honoured when the ScriptDict that is\n        mutated using TorchScript is inside another.\n        '
        nested = torch.jit.script({1: {1: 2}, 2: {3: 4}}, type_hint=Dict[int, Dict[int, int]])
        one = nested[1]
        two = nested[2]
        self._script_dict_add(one, 9, 10)
        self._script_dict_add(two, 11, 12)
        self.assertEqual(len(one), 2)
        self.assertEqual(len(two), 2)
        self.assertEqual(len(nested[1]), 2)
        self.assertEqual(len(nested[2]), 2)

    def test_reference_semantics(self):
        if False:
            while True:
                i = 10
        '\n        Test that reference semantics are honoured; that modifications made\n        to a ScriptDict in TorchScript are visible in Python.\n        '
        data = torch.jit.script({1: 2})
        self._script_dict_add(data, 3, 4)
        self.assertEqual(len(data), 2)
        self.assertTrue(3 in data)
        self.assertEqual(data[3], 4)

class TestScriptList(JitTestCase):
    """
    This class contains a suite of tests for torch._C.ScriptList, a
    function that returns a list-like object that has reference
    semantics across the Python/TorchScript boundary. That is,
    it can be passed to a TorchScript function that mutates it
    and those modifications are visible in the scope of the Python
    caller of said TorchScript function.

    The vast majority of tests are for making sure that instances of
    torch._C.ScriptList behave like lists do so that they are fungible
    in almost all cirumstances with regular list.
    """

    def _script_list_add(self, l: torch._C.ScriptList, e: int):
        if False:
            i = 10
            return i + 15
        '\n        This is a helper function that inserts the element e into the\n        list l in TorchScript. It is used for testing reference\n        semantics.\n        '

        @torch.jit.script
        def list_add(l: List[int], e: int):
            if False:
                i = 10
                return i + 15
            l.append(e)
        list_add(l, e)

    def _compare_eager_and_script(self, fn, input_list, script_input_list=None):
        if False:
            while True:
                i = 10
        '\n        This is a helper function that facilitates comparing behaviour between\n        Python lists and "scripted" lists.\n        Args:\n            fn: The function to test and compare the behaviour of.\n            input_list: The input list to use for the test (passed to fn).\n            script_input_list: The scripted input list to use for the tests.\n                                If None, input_list is scripted with torch.jit.script\n                                and used instead.\n        '
        script_input_list = script_input_list or torch.jit.script(input_list)
        (eager_raised, script_raised) = (False, False)
        try:
            eager_out = fn(input_list)
        except Exception as e:
            eager_exception = e
            eager_raised = True
        try:
            script_out = fn(script_input_list)
        except Exception as e:
            script_exception = e
            script_raised = True
        self.assertEqual(eager_raised, script_raised)
        if eager_raised:
            self.assertEqual(type(eager_exception), type(script_exception))
        else:
            self.assertEqual(eager_out, script_out)
            self.assertEqual(input_list, script_input_list)

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the __repr__ method.\n        '
        self._compare_eager_and_script(lambda l: repr(l), [1])

    def test_bool(self):
        if False:
            return 10
        '\n        Test the __bool__ method. This should return True\n        if the list is non-empty and False otherwise.\n        '
        self._compare_eager_and_script(lambda l: bool(l), [1])
        self._compare_eager_and_script(lambda l: bool(l), [])

    def test_iter(self):
        if False:
            print('Hello World!')
        "\n        Test iteration over a list's elements.\n        "

        def sum_elements(input_list):
            if False:
                print('Hello World!')
            s = 0
            for k in input_list:
                s += k
            return s
        self._compare_eager_and_script(sum_elements, [1, 2, 3, 4])

    def test_getitem(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test accessing list elements using the [] operator.\n        '
        data = [1, 2, 3, 4]
        self._compare_eager_and_script(lambda l: l[1], data)
        self._compare_eager_and_script(lambda l: l[3], data)
        self._compare_eager_and_script(lambda l: l[-1], data)
        self._compare_eager_and_script(lambda l: l[1:3], data)
        self._compare_eager_and_script(lambda l: l[:], data)
        self._compare_eager_and_script(lambda l: l[1:], data)
        self._compare_eager_and_script(lambda l: l[:2], data)
        self._compare_eager_and_script(lambda l: l[-1], data)
        self._compare_eager_and_script(lambda l: l[-1::-1], data)
        self._compare_eager_and_script(lambda l: l[5], data)
        self._compare_eager_and_script(lambda l: l[-7], data)
        self._compare_eager_and_script(lambda l: l['key'], data)

    def test_setitem(self):
        if False:
            print('Hello World!')
        '\n        Test setting list elements using the [] operator.\n        '
        data = [1, 2, 3, 4]

        def setitem(input_list):
            if False:
                i = 10
                return i + 15
            input_list[1] = 10
            input_list[3] = 11
            input_list[-1] = 12
        self._compare_eager_and_script(setitem, data.copy())

        def setitem_slice(input_list):
            if False:
                while True:
                    i = 10
            input_list[:4:2] = [10, 11]
            input_list[-2:] = [15, 16]
        self._compare_eager_and_script(setitem_slice, data)

        def out_of_range(input_list):
            if False:
                for i in range(10):
                    print('nop')
            input_list[11] = 3

        def out_of_range_negative(input_list):
            if False:
                for i in range(10):
                    print('nop')
            input_list[-11] = 3

        def wrong_index_type(input_list):
            if False:
                return 10
            input_list['str'] = 3
        self._compare_eager_and_script(out_of_range, data)
        self._compare_eager_and_script(out_of_range_negative, data)
        self._compare_eager_and_script(wrong_index_type, data)
        script_data = torch.jit.script(data)
        with self.assertRaises(TypeError):
            script_data[0] = 'str'

    def test_contains(self):
        if False:
            i = 10
            return i + 15
        '\n        Test membership checks (x in y, x not in y).\n        '
        data = [1, 2, 3, 4]

        def fn(input_list):
            if False:
                for i in range(10):
                    print('nop')
            return (1 in input_list, 2 not in input_list, 3 in input_list, 4 not in input_list)
        self._compare_eager_and_script(fn, data)
        script_data = torch.jit.script(data)
        with self.assertRaises(TypeError):
            a = 'str' in script_data

    def test_delitem(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test deletion.\n        '
        data = [1, 2, 3, 4]

        def del_fn(input_list):
            if False:
                return 10
            del input_list[1]

        def del_fn_out_of_range(input_list):
            if False:
                while True:
                    i = 10
            del input_list[10]

        def del_fn_wrong_type(input_list):
            if False:
                return 10
            del input_list['str']
        self._compare_eager_and_script(del_fn, data.copy())
        self._compare_eager_and_script(del_fn_out_of_range, data)
        self._compare_eager_and_script(del_fn_wrong_type, data)

    def test_len(self):
        if False:
            return 10
        '\n        Test len() builtin function.\n        '
        self._compare_eager_and_script(lambda l: len(l), [1, 2, 3, 4])
        self._compare_eager_and_script(lambda l: len(l), [])

    def test_count(self):
        if False:
            return 10
        '\n        Test count method.\n        '
        self._compare_eager_and_script(lambda l: l.count(3), [1, 2, 3, 3])
        script_data = torch.jit.script([1])
        with self.assertRaises(TypeError):
            script_data.count('str')

    def test_remove(self):
        if False:
            return 10
        '\n        Test remove method.\n        '
        self._compare_eager_and_script(lambda l: l.remove(1), [1, 2, 3])
        self._compare_eager_and_script(lambda l: l.remove(10), [1, 2, 3])
        script_data = torch.jit.script([1])
        with self.assertRaises(TypeError):
            script_data.remove('str')

    def test_append(self):
        if False:
            while True:
                i = 10
        '\n        Test append method.\n        '
        self._compare_eager_and_script(lambda l: l.append(1), [4, 3, 2])
        script_data = torch.jit.script([1])
        with self.assertRaises(TypeError):
            script_data.append('str')

    @skipIfTorchDynamo('https://github.com/pytorch/torchdynamo/issues/1991')
    def test_clear(self):
        if False:
            while True:
                i = 10
        '\n        Test clear.\n        '
        self._compare_eager_and_script(lambda l: l.clear(), [4, 3, 2])

    def test_extend(self):
        if False:
            while True:
                i = 10
        '\n        Test extend.\n        '

        class Iterable:

            def __init__(self, limit: int):
                if False:
                    while True:
                        i = 10
                self.limit = limit
                self.value = 0

            def __iter__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self

            def __next__(self):
                if False:
                    for i in range(10):
                        print('nop')
                if self.value == limit:
                    raise StopIteration()
                ret = self.value
                self.value += 1
                return ret
        data = [1, 2, 3]

        def extend_list(input_list):
            if False:
                i = 10
                return i + 15
            input_list.extend([4, 5, 6])

        def extend_dict(input_list):
            if False:
                return 10
            input_list.extend({4: 10, 5: 11, 6: 12})

        def extend_iterable(input_list):
            if False:
                while True:
                    i = 10
            input_list.extend(Iterable(3))
        self._compare_eager_and_script(extend_list, data.copy())
        self._compare_eager_and_script(extend_dict, data.copy())
        self._compare_eager_and_script(extend_iterable, data)
        script_data = torch.jit.script([1])
        with self.assertRaises(TypeError):
            script_data.extend(['a'])
        with self.assertRaises(TypeError):
            script_data.extend({'a': 1})

    def test_insert(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test insert.\n        '
        data = [1, 2, 4]
        self._compare_eager_and_script(lambda l: l.insert(3, 3), data.copy())
        self._compare_eager_and_script(lambda l: l.insert(0, 3), data.copy())
        self._compare_eager_and_script(lambda l: l.insert(-2, 3), data)
        script_data = torch.jit.script([1])
        with self.assertRaises(TypeError):
            script_data.insert((0, 'str'))

    def test_pop(self):
        if False:
            while True:
                i = 10
        '\n        Test pop.\n        '
        data = [1, 2, 3, 4, 5]
        self._compare_eager_and_script(lambda l: l.pop(), data.copy())
        self._compare_eager_and_script(lambda l: l.pop(2), data.copy())
        self._compare_eager_and_script(lambda l: l.pop(-3), data.copy())
        self._compare_eager_and_script(lambda l: l.pop(10), data)

    @unittest.skip('Cannot pass until all list returned from TorchScript are ScriptLists')
    def test_nested(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that reference semantics are honoured when the ScriptList that is\n        mutated using TorchScript is inside another.\n        '
        nested = torch.jit.script([[1], [2]], List[List[int]])
        one = nested[0]
        two = nested[1]
        self._script_list_add(one, 3)
        self._script_list_add(two, 4)
        self.assertEqual(len(one), 2)
        self.assertEqual(len(two), 2)
        self.assertEqual(one[len(one) - 1], 3)
        self.assertEqual(two[len(one) - 1], 4)
        self.assertEqual(len(nested[0]), 2)
        self.assertEqual(len(nested[1]), 2)

    def test_reference_semantics(self):
        if False:
            return 10
        '\n        Test that reference semantics are honoured; that modifications made\n        to a ScriptList in TorchScript are visible in Python.\n        '
        l = torch.jit.script([1, 2])
        self._script_list_add(l, 3)
        self.assertEqual(len(l), 3)
        self.assertTrue(3 in l)
        self.assertEqual(l[2], 3)