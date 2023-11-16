import inspect
import unittest
from test_imperative_base import new_program_scope
import paddle
from paddle import base
from paddle.base import framework

class TestTracerMode(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.init_mode = True

    def get_tracer_mode(self):
        if False:
            i = 10
            return i + 15
        assert framework.in_dygraph_mode(), 'Dygraph mode must be enabled'

    @base.dygraph.no_grad
    def no_grad_func(self, a):
        if False:
            print('Hello World!')
        self.assertEqual(self.tracer._has_grad, False)
        return a

    @framework.dygraph_not_support
    def not_support_func(self):
        if False:
            print('Hello World!')
        return True

    def check_not_support_rlt(self, ans):
        if False:
            return 10
        try:
            rlt = self.not_support_func()
        except AssertionError:
            rlt = False
        finally:
            self.assertEqual(rlt, ans)

    def test_main(self):
        if False:
            while True:
                i = 10
        with base.dygraph.guard():
            self.tracer = framework._dygraph_tracer()
            self.tracer._train_mode = self.init_mode
            self.assertEqual(self.no_grad_func(1), 1)
            self.assertEqual(self.no_grad_func.__name__, 'no_grad_func')

            def need_no_grad_func(a, b=1):
                if False:
                    for i in range(10):
                        print('nop')
                return a + b
            decorated_func = base.dygraph.no_grad(need_no_grad_func)
            self.assertTrue(str(inspect.getfullargspec(decorated_func)) == str(inspect.getfullargspec(need_no_grad_func)))
            self.assertEqual(self.tracer._train_mode, self.init_mode)
        with base.dygraph.guard():
            self.check_not_support_rlt(False)
        paddle.enable_static()
        with new_program_scope():
            self.check_not_support_rlt(True)

class TestTracerMode2(TestTracerMode):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_mode = False

class TestNoGradClass(unittest.TestCase):

    @paddle.no_grad()
    def no_grad_func(self, a):
        if False:
            return 10
        self.assertEqual(self.tracer._train_mode, True)
        self.assertEqual(self.tracer._has_grad, False)
        return a

    def test_main(self):
        if False:
            return 10
        paddle.disable_static()
        self.tracer = framework._dygraph_tracer()
        self.tracer._train_mode = True
        self.tracer._has_grad = True
        self.assertEqual(self.no_grad_func(1), 1)
        self.assertEqual(self.no_grad_func.__name__, 'no_grad_func')

        def need_no_grad_func(a, b=1):
            if False:
                for i in range(10):
                    print('nop')
            return a + b
        decorated_func = paddle.no_grad()(need_no_grad_func)
        self.assertEqual(str(inspect.getfullargspec(decorated_func)), str(inspect.getfullargspec(need_no_grad_func)))

        def test_gen():
            if False:
                return 10
            yield from range(3)
        a = 0
        for i in test_gen():
            a += i

        @paddle.no_grad()
        def test_wrapped_gen():
            if False:
                return 10
            yield from range(3)
        b = 0
        for i in test_wrapped_gen():
            b += i
        self.assertEqual(a, b)

class TestEnableGradClass(unittest.TestCase):

    @paddle.enable_grad()
    def enable_grad_func(self, a):
        if False:
            return 10
        self.assertEqual(self.tracer._train_mode, True)
        self.assertEqual(self.tracer._has_grad, True)
        return a

    def test_main(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        self.tracer = framework._dygraph_tracer()
        self.tracer._train_mode = True
        self.tracer._has_grad = False
        self.assertEqual(self.enable_grad_func(1), 1)
        self.assertEqual(self.enable_grad_func.__name__, 'enable_grad_func')

        def need_enable_grad_func(a, b=1):
            if False:
                while True:
                    i = 10
            return a + b
        decorated_func = paddle.enable_grad()(need_enable_grad_func)
        self.assertEqual(str(inspect.getfullargspec(decorated_func)), str(inspect.getfullargspec(need_enable_grad_func)))

        def test_gen():
            if False:
                while True:
                    i = 10
            yield from range(3)
        a = 0
        for i in test_gen():
            a += i

        @paddle.enable_grad()
        def test_wrapped_gen():
            if False:
                i = 10
                return i + 15
            yield from range(3)
        b = 0
        for i in test_wrapped_gen():
            b += i
        self.assertEqual(a, b)

    def test_stop_gradient(self):
        if False:
            return 10
        x = paddle.to_tensor([1.0], stop_gradient=False)
        with paddle.no_grad():
            with paddle.enable_grad():
                y = x * 2
        self.assertTrue(y.stop_gradient is False)
        y.backward()
        self.assertTrue(x.grad is not None)

        @paddle.enable_grad()
        def double(x):
            if False:
                print('Hello World!')
            return x * 2
        with paddle.no_grad():
            z = double(x)
        self.assertTrue(z.stop_gradient is False)

class TestSetGradEnabledClass(unittest.TestCase):

    @paddle.set_grad_enabled(True)
    def enable_grad_func(self, a):
        if False:
            print('Hello World!')
        self.assertEqual(self.tracer._train_mode, True)
        self.assertEqual(self.tracer._has_grad, True)
        return a

    def test_main(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        self.tracer = framework._dygraph_tracer()
        self.tracer._train_mode = True
        self.assertEqual(self.enable_grad_func(1), 1)
        self.assertEqual(self.enable_grad_func.__name__, 'enable_grad_func')

        def need_enable_grad_func(a, b=1):
            if False:
                for i in range(10):
                    print('nop')
            return a + b
        decorated_func = paddle.set_grad_enabled(True)(need_enable_grad_func)
        self.assertEqual(str(inspect.getfullargspec(decorated_func)), str(inspect.getfullargspec(need_enable_grad_func)))

        def test_gen():
            if False:
                i = 10
                return i + 15
            yield from range(3)
        a = 0
        for i in test_gen():
            a += i

        @paddle.set_grad_enabled(True)
        def test_wrapped_gen():
            if False:
                for i in range(10):
                    print('nop')
            yield from range(3)
        b = 0
        for i in test_wrapped_gen():
            b += i
        self.assertEqual(a, b)

    def test_stop_gradient(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.to_tensor([1.0], stop_gradient=False)
        is_train = False
        with paddle.set_grad_enabled(is_train):
            y = x * 2
        self.assertTrue(y.stop_gradient is True)
        paddle.set_grad_enabled(True)
        y = x * 2
        self.assertTrue(y.stop_gradient is False)
        paddle.set_grad_enabled(False)
        y = x * 2
        self.assertTrue(y.stop_gradient is True)

class TestIsGradEnabledClass(unittest.TestCase):

    def test_main(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        self.tracer = framework._dygraph_tracer()
        self.tracer._train_mode = True
        self.tracer._has_grad = True
        flag = paddle.is_grad_enabled()
        self.assertTrue(flag is True)
        with paddle.set_grad_enabled(False):
            flag = paddle.is_grad_enabled()
            self.assertTrue(flag is False)
        flag = paddle.is_grad_enabled()
        self.assertTrue(flag is True)
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()