import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import CompileCounter

class TestInputAttrTracking(torch._dynamo.test_case.TestCase):

    def test_tensor_property_on_tensor(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                print('Hello World!')
            return x * x.y
        x_ = torch.randn([2, 2])
        y_ = torch.randn([2, 2])
        x_.y = y_
        eager_result = fn(x_)
        graph = None

        def grab_graph_backend(gm, inps):
            if False:
                return 10
            nonlocal graph
            graph = gm
            return gm
        fn = torch._dynamo.optimize(grab_graph_backend, nopython=True)(fn)
        compile_result = fn(x_)
        self.assertEqual(eager_result, compile_result)
        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == 'placeholder':
                placeholder_cnt += 1
        self.assertEqual(placeholder_cnt, 2)

    def test_tensor_property_assigned_on_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            x.y = y
            return x * x.y
        x_ = torch.randn([2, 2])
        y_ = torch.randn([2, 2])
        eager_result = fn(x_, y_)
        graph = None

        def grab_graph_backend(gm, inps):
            if False:
                while True:
                    i = 10
            nonlocal graph
            graph = gm
            return gm
        fn = torch._dynamo.optimize(grab_graph_backend, nopython=True)(fn)
        compile_result = fn(x_, y_)
        self.assertEqual(eager_result, compile_result)
        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == 'placeholder':
                placeholder_cnt += 1
        self.assertEqual(placeholder_cnt, 2)

    def test_const_property_on_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x):
            if False:
                while True:
                    i = 10
            return x * x.y
        x_ = torch.randn([2, 2])
        y_ = 4
        x_.y = y_
        eager_result = fn(x_)
        graph = None

        def grab_graph_backend(gm, inps):
            if False:
                print('Hello World!')
            nonlocal graph
            graph = gm
            return gm
        fn = torch._dynamo.optimize(grab_graph_backend, nopython=True)(fn)
        compile_result = fn(x_)
        self.assertEqual(eager_result, compile_result)
        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == 'placeholder':
                placeholder_cnt += 1
        self.assertEqual(placeholder_cnt, 1)

    def test_const_property_assigned_on_tensor(self):
        if False:
            while True:
                i = 10

        def fn(x, y):
            if False:
                print('Hello World!')
            x.y = y
            return x * x.y
        x_ = torch.randn([2, 2])
        y_ = 4
        eager_result = fn(x_, y_)
        fn = torch._dynamo.optimize('eager', nopython=True)(fn)
        compile_result = fn(x_, y_)
        self.assertEqual(eager_result, compile_result)

    def test_guards_correctly_property_assigned_on_tensor_type_change(self):
        if False:
            return 10

        def fn(x, y):
            if False:
                return 10
            x.y = y
            return x * x.y
        x_ = torch.randn([2, 2])
        fn = torch._dynamo.optimize('eager', nopython=True)(fn)
        compile_result_const = fn(x_, 4)
        self.assertEqual(compile_result_const, x_ * 4)
        y = torch.randn([2, 2])
        compile_result_tensor = fn(x_, y)
        self.assertEqual(compile_result_tensor, x_ * y)

    def test_guards_correctly_property_assigned_on_tensor_type_change_inductor(self):
        if False:
            return 10

        def fn(x, y):
            if False:
                print('Hello World!')
            x.y = y
            return x * x.y
        x_ = torch.randn([2, 2])
        fn = torch._dynamo.optimize('inductor', nopython=True)(fn)
        compile_result_const = fn(x_, 4)
        self.assertEqual(compile_result_const, x_ * 4)
        y = torch.randn([2, 2])
        compile_result_tensor = fn(x_, y)
        self.assertEqual(compile_result_tensor, x_ * y)

    def test_complex_attr_access_without_graph_breaks(self):
        if False:
            return 10

        def fn(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            for t in x:
                t.y = y
                t.z = y * z
            new_y = 1
            new_z = 1
            for t in x:
                new_y = t.y * new_y
                new_z = t.z * new_z
            return (new_y, new_z)
        x_0 = torch.randn([2, 2])
        x_1 = torch.randn([2, 2])
        x_2 = torch.randn([2, 2])
        x = [x_0, x_1, x_2]
        y = torch.randn([2, 2])
        z = 5
        eager_result = fn(x, y, z)
        counter = CompileCounter()
        fn = torch._dynamo.optimize(counter, nopython=True)(fn)
        compile_result = fn(x, y, z)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 9)

    def test_complex_attr_access_with_graph_breaks(self):
        if False:
            i = 10
            return i + 15

        def fn(x, y, z):
            if False:
                print('Hello World!')
            for t in x:
                t.y = y
                t.z = y * z
            print('Break!')
            new_y = 1
            new_z = 1
            for t in x:
                new_y = t.y * new_y
                new_z = t.z * new_z
            return (new_y, new_z)
        x_0 = torch.randn([2, 2])
        x_1 = torch.randn([2, 2])
        x_2 = torch.randn([2, 2])
        x = [x_0, x_1, x_2]
        y = torch.randn([2, 2])
        z = 5
        eager_result = fn(x, y, z)
        counter = CompileCounter()
        fn = torch._dynamo.optimize(counter, nopython=False)(fn)
        compile_result = fn(x, y, z)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(counter.op_count, 9)

    def test_complex_attr_access_with_inline_reconstruct(self):
        if False:
            while True:
                i = 10

        def inline_test_fn(x, y, z):
            if False:
                print('Hello World!')
            print('f')
            return x.a + y.a + z.a

        def fn(x, y, z):
            if False:
                while True:
                    i = 10
            x.a = 1
            y.a = 2
            z.a = 3
            mult = inline_test_fn(x, y, z)
            y = y * mult
            x = x * mult
            return (x, y)
        x = torch.randn([2, 2])
        y = torch.randn([2, 2])
        z = torch.randn([2, 2])
        eager_result = fn(x, y, z)
        counter = CompileCounter()
        fn = torch._dynamo.optimize(counter, nopython=False)(fn)
        compile_result = fn(x, y, z)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)

    def test_set_data_on_input_tensor(self):
        if False:
            i = 10
            return i + 15

        def fn(x, y):
            if False:
                while True:
                    i = 10
            x.data = y.data
            if x.size() == y.size():
                return x * y
            else:
                return y * y
        x = torch.randn([5, 5])
        y = torch.randn([2, 2])
        eager_result = fn(x, y)
        counter = CompileCounter()
        fn = torch._dynamo.optimize(counter, nopython=True)(fn)
        compile_result = fn(x, y)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)

    def test_set_data_on_scoped_tensor(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            z = torch.zeros([4, 4])
            z.data = x.data
            if x.size() == z.size():
                return z * x
            else:
                return x
        x = torch.randn([5, 5])
        eager_result = fn(x)
        counter = CompileCounter()
        fn = torch._dynamo.optimize(counter, nopython=False)(fn)
        compile_result = fn(x)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(counter.op_count, 3)

    def test_set_data_on_user_defined_class_input_tensor(self):
        if False:
            while True:
                i = 10

        class MyUserDefinedClass:

            def __init__(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = x
                self.y = y

            def do_some_setattr_stuff(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.z = x * y
                self.a = x + x
                return self.z * self.a
        x = torch.randn([5, 5])
        y = torch.randn([5, 5])
        mudc_1 = MyUserDefinedClass(x, y)
        eager_result = mudc_1.do_some_setattr_stuff()
        counter = CompileCounter()
        mudc_2 = MyUserDefinedClass(x, y)
        do_some_setattr_stuff = torch._dynamo.optimize(counter, nopython=True)(mudc_2.do_some_setattr_stuff)
        compile_result = do_some_setattr_stuff()
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()