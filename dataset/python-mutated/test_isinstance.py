import os
import sys
import torch
import warnings
from typing import List, Any, Dict, Tuple, Optional
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestIsinstance(JitTestCase):

    def test_int(self):
        if False:
            for i in range(10):
                print('nop')

        def int_test(x: Any):
            if False:
                return 10
            assert torch.jit.isinstance(x, int)
            assert not torch.jit.isinstance(x, float)
        x = 1
        self.checkScript(int_test, (x,))

    def test_float(self):
        if False:
            i = 10
            return i + 15

        def float_test(x: Any):
            if False:
                for i in range(10):
                    print('nop')
            assert torch.jit.isinstance(x, float)
            assert not torch.jit.isinstance(x, int)
        x = 1.0
        self.checkScript(float_test, (x,))

    def test_bool(self):
        if False:
            for i in range(10):
                print('nop')

        def bool_test(x: Any):
            if False:
                while True:
                    i = 10
            assert torch.jit.isinstance(x, bool)
            assert not torch.jit.isinstance(x, float)
        x = False
        self.checkScript(bool_test, (x,))

    def test_list(self):
        if False:
            while True:
                i = 10

        def list_str_test(x: Any):
            if False:
                i = 10
                return i + 15
            assert torch.jit.isinstance(x, List[str])
            assert not torch.jit.isinstance(x, List[int])
            assert not torch.jit.isinstance(x, Tuple[int])
        x = ['1', '2', '3']
        self.checkScript(list_str_test, (x,))

    def test_list_tensor(self):
        if False:
            print('Hello World!')

        def list_tensor_test(x: Any):
            if False:
                print('Hello World!')
            assert torch.jit.isinstance(x, List[torch.Tensor])
            assert not torch.jit.isinstance(x, Tuple[int])
        x = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]
        self.checkScript(list_tensor_test, (x,))

    def test_dict(self):
        if False:
            i = 10
            return i + 15

        def dict_str_int_test(x: Any):
            if False:
                for i in range(10):
                    print('nop')
            assert torch.jit.isinstance(x, Dict[str, int])
            assert not torch.jit.isinstance(x, Dict[int, str])
            assert not torch.jit.isinstance(x, Dict[str, str])
        x = {'a': 1, 'b': 2}
        self.checkScript(dict_str_int_test, (x,))

    def test_dict_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        def dict_int_tensor_test(x: Any):
            if False:
                while True:
                    i = 10
            assert torch.jit.isinstance(x, Dict[int, torch.Tensor])
        x = {2: torch.tensor([2])}
        self.checkScript(dict_int_tensor_test, (x,))

    def test_tuple(self):
        if False:
            print('Hello World!')

        def tuple_test(x: Any):
            if False:
                print('Hello World!')
            assert torch.jit.isinstance(x, Tuple[str, int, str])
            assert not torch.jit.isinstance(x, Tuple[int, str, str])
            assert not torch.jit.isinstance(x, Tuple[str])
        x = ('a', 1, 'b')
        self.checkScript(tuple_test, (x,))

    def test_tuple_tensor(self):
        if False:
            print('Hello World!')

        def tuple_tensor_test(x: Any):
            if False:
                i = 10
                return i + 15
            assert torch.jit.isinstance(x, Tuple[torch.Tensor, torch.Tensor])
        x = (torch.tensor([1]), torch.tensor([[2], [3]]))
        self.checkScript(tuple_tensor_test, (x,))

    def test_optional(self):
        if False:
            return 10

        def optional_test(x: Any):
            if False:
                print('Hello World!')
            assert torch.jit.isinstance(x, Optional[torch.Tensor])
            assert not torch.jit.isinstance(x, Optional[str])
        x = torch.ones(3, 3)
        self.checkScript(optional_test, (x,))

    def test_optional_none(self):
        if False:
            print('Hello World!')

        def optional_test_none(x: Any):
            if False:
                i = 10
                return i + 15
            assert torch.jit.isinstance(x, Optional[torch.Tensor])
        x = None
        self.checkScript(optional_test_none, (x,))

    def test_list_nested(self):
        if False:
            for i in range(10):
                print('nop')

        def list_nested(x: Any):
            if False:
                print('Hello World!')
            assert torch.jit.isinstance(x, List[Dict[str, int]])
            assert not torch.jit.isinstance(x, List[List[str]])
        x = [{'a': 1, 'b': 2}, {'aa': 11, 'bb': 22}]
        self.checkScript(list_nested, (x,))

    def test_dict_nested(self):
        if False:
            while True:
                i = 10

        def dict_nested(x: Any):
            if False:
                for i in range(10):
                    print('nop')
            assert torch.jit.isinstance(x, Dict[str, Tuple[str, str, str]])
            assert not torch.jit.isinstance(x, Dict[str, Tuple[int, int, int]])
        x = {'a': ('aa', 'aa', 'aa'), 'b': ('bb', 'bb', 'bb')}
        self.checkScript(dict_nested, (x,))

    def test_tuple_nested(self):
        if False:
            for i in range(10):
                print('nop')

        def tuple_nested(x: Any):
            if False:
                return 10
            assert torch.jit.isinstance(x, Tuple[Dict[str, Tuple[str, str, str]], List[bool], Optional[str]])
            assert not torch.jit.isinstance(x, Dict[str, Tuple[int, int, int]])
            assert not torch.jit.isinstance(x, Tuple[str])
            assert not torch.jit.isinstance(x, Tuple[List[bool], List[str], List[int]])
        x = ({'a': ('aa', 'aa', 'aa'), 'b': ('bb', 'bb', 'bb')}, [True, False, True], None)
        self.checkScript(tuple_nested, (x,))

    def test_optional_nested(self):
        if False:
            while True:
                i = 10

        def optional_nested(x: Any):
            if False:
                while True:
                    i = 10
            assert torch.jit.isinstance(x, Optional[List[str]])
        x = ['a', 'b', 'c']
        self.checkScript(optional_nested, (x,))

    def test_list_tensor_type_true(self):
        if False:
            for i in range(10):
                print('nop')

        def list_tensor_type_true(x: Any):
            if False:
                i = 10
                return i + 15
            assert torch.jit.isinstance(x, List[torch.Tensor])
        x = [torch.rand(3, 3), torch.rand(4, 3)]
        self.checkScript(list_tensor_type_true, (x,))

    def test_tensor_type_false(self):
        if False:
            print('Hello World!')

        def list_tensor_type_false(x: Any):
            if False:
                print('Hello World!')
            assert not torch.jit.isinstance(x, List[torch.Tensor])
        x = [1, 2, 3]
        self.checkScript(list_tensor_type_false, (x,))

    def test_in_if(self):
        if False:
            print('Hello World!')

        def list_in_if(x: Any):
            if False:
                print('Hello World!')
            if torch.jit.isinstance(x, List[int]):
                assert True
            if torch.jit.isinstance(x, List[str]):
                assert not True
        x = [1, 2, 3]
        self.checkScript(list_in_if, (x,))

    def test_if_else(self):
        if False:
            while True:
                i = 10

        def list_in_if_else(x: Any):
            if False:
                print('Hello World!')
            if torch.jit.isinstance(x, Tuple[str, str, str]):
                assert True
            else:
                assert not True
        x = ('a', 'b', 'c')
        self.checkScript(list_in_if_else, (x,))

    def test_in_while_loop(self):
        if False:
            for i in range(10):
                print('nop')

        def list_in_while_loop(x: Any):
            if False:
                while True:
                    i = 10
            count = 0
            while torch.jit.isinstance(x, List[Dict[str, int]]) and count <= 0:
                count = count + 1
            assert count == 1
        x = [{'a': 1, 'b': 2}, {'aa': 11, 'bb': 22}]
        self.checkScript(list_in_while_loop, (x,))

    def test_type_refinement(self):
        if False:
            for i in range(10):
                print('nop')

        def type_refinement(obj: Any):
            if False:
                print('Hello World!')
            hit = False
            if torch.jit.isinstance(obj, List[torch.Tensor]):
                hit = not hit
                for el in obj:
                    y = el.clamp(0, 0.5)
            if torch.jit.isinstance(obj, Dict[str, str]):
                hit = not hit
                str_cat = ''
                for val in obj.values():
                    str_cat = str_cat + val
                assert '111222' == str_cat
            assert hit
        x = [torch.rand(3, 3), torch.rand(4, 3)]
        self.checkScript(type_refinement, (x,))
        x = {'1': '111', '2': '222'}
        self.checkScript(type_refinement, (x,))

    def test_list_no_contained_type(self):
        if False:
            for i in range(10):
                print('nop')

        def list_no_contained_type(x: Any):
            if False:
                for i in range(10):
                    print('nop')
            assert torch.jit.isinstance(x, List)
        x = ['1', '2', '3']
        err_msg = 'Attempted to use List without a contained type. Please add a contained type, e.g. List\\[int\\]'
        with self.assertRaisesRegex(RuntimeError, err_msg):
            torch.jit.script(list_no_contained_type)
        with self.assertRaisesRegex(RuntimeError, err_msg):
            list_no_contained_type(x)

    def test_tuple_no_contained_type(self):
        if False:
            while True:
                i = 10

        def tuple_no_contained_type(x: Any):
            if False:
                print('Hello World!')
            assert torch.jit.isinstance(x, Tuple)
        x = ('1', '2', '3')
        err_msg = 'Attempted to use Tuple without a contained type. Please add a contained type, e.g. Tuple\\[int\\]'
        with self.assertRaisesRegex(RuntimeError, err_msg):
            torch.jit.script(tuple_no_contained_type)
        with self.assertRaisesRegex(RuntimeError, err_msg):
            tuple_no_contained_type(x)

    def test_optional_no_contained_type(self):
        if False:
            while True:
                i = 10

        def optional_no_contained_type(x: Any):
            if False:
                return 10
            assert torch.jit.isinstance(x, Optional)
        x = ('1', '2', '3')
        err_msg = 'Attempted to use Optional without a contained type. Please add a contained type, e.g. Optional\\[int\\]'
        with self.assertRaisesRegex(RuntimeError, err_msg):
            torch.jit.script(optional_no_contained_type)
        with self.assertRaisesRegex(RuntimeError, err_msg):
            optional_no_contained_type(x)

    def test_dict_no_contained_type(self):
        if False:
            while True:
                i = 10

        def dict_no_contained_type(x: Any):
            if False:
                return 10
            assert torch.jit.isinstance(x, Dict)
        x = {'a': 'aa'}
        err_msg = 'Attempted to use Dict without contained types. Please add contained type, e.g. Dict\\[int, int\\]'
        with self.assertRaisesRegex(RuntimeError, err_msg):
            torch.jit.script(dict_no_contained_type)
        with self.assertRaisesRegex(RuntimeError, err_msg):
            dict_no_contained_type(x)

    def test_tuple_rhs(self):
        if False:
            i = 10
            return i + 15

        def fn(x: Any):
            if False:
                print('Hello World!')
            assert torch.jit.isinstance(x, (int, List[str]))
            assert not torch.jit.isinstance(x, (List[float], Tuple[int, str]))
            assert not torch.jit.isinstance(x, (List[float], str))
        self.checkScript(fn, (2,))
        self.checkScript(fn, (['foo', 'bar', 'baz'],))

    def test_nontuple_container_rhs_throws_in_eager(self):
        if False:
            print('Hello World!')

        def fn1(x: Any):
            if False:
                while True:
                    i = 10
            assert torch.jit.isinstance(x, [int, List[str]])

        def fn2(x: Any):
            if False:
                i = 10
                return i + 15
            assert not torch.jit.isinstance(x, {List[str], Tuple[int, str]})
        err_highlight = 'must be a type or a tuple of types'
        with self.assertRaisesRegex(RuntimeError, err_highlight):
            fn1(2)
        with self.assertRaisesRegex(RuntimeError, err_highlight):
            fn2(2)

    def test_empty_container_throws_warning_in_eager(self):
        if False:
            while True:
                i = 10

        def fn(x: Any):
            if False:
                return 10
            torch.jit.isinstance(x, List[int])
        with warnings.catch_warnings(record=True) as w:
            x: List[int] = []
            fn(x)
            self.assertEqual(len(w), 1)
        with warnings.catch_warnings(record=True) as w:
            x: int = 2
            fn(x)
            self.assertEqual(len(w), 0)

    def test_empty_container_special_cases(self):
        if False:
            while True:
                i = 10
        torch._jit_internal.check_empty_containers(torch.Tensor([]))
        torch._jit_internal.check_empty_containers(torch.rand(2, 3))