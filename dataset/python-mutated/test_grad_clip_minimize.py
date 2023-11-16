import unittest
import numpy as np
from paddle import base
from paddle.base.dygraph.base import to_variable
from paddle.nn import ClipGradByGlobalNorm, ClipGradByNorm, ClipGradByValue

class TestGradClipByGlobalNorm(unittest.TestCase):

    def init_value(self):
        if False:
            for i in range(10):
                print('nop')
        self.max_global_norm = 5.0
        self.init_scale = 1.0
        self.shape = (20, 20)

    def generate_p_g(self):
        if False:
            i = 10
            return i + 15
        self.para_and_grad = []
        for i in range(10):
            self.para_and_grad.append((np.random.uniform(-self.init_scale, self.init_scale, self.shape).astype('float32'), np.random.uniform(-self.init_scale, self.init_scale, self.shape).astype('float32')))

    def get_numpy_global_norm_result(self):
        if False:
            while True:
                i = 10
        gloabl_norm = 0.0
        for (p, g) in self.para_and_grad:
            gloabl_norm += np.sum(np.square(g))
        gloabl_norm_np = np.sqrt(gloabl_norm)
        new_np_p_g = []
        scale = 1.0
        if gloabl_norm_np > self.max_global_norm:
            scale = self.max_global_norm / gloabl_norm_np
        for (p, g) in self.para_and_grad:
            new_np_p_g.append((p, g * scale))
        return new_np_p_g

    def get_dygrap_global_norm_result(self):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard():
            gloabl_norm_clip = ClipGradByGlobalNorm(self.max_global_norm)
            p_g_var = []
            for (p, g) in self.para_and_grad:
                new_p = to_variable(p)
                new_g = to_variable(g)
                p_g_var.append((new_p, new_g))
            new_p_g_var = gloabl_norm_clip(p_g_var)
            p_g_dy_out = []
            for (p, g) in new_p_g_var:
                p_g_dy_out.append((p.numpy(), g.numpy()))
            return p_g_dy_out

    def test_clip_by_global_norm(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_value()
        self.generate_p_g()
        np_p_g = self.get_numpy_global_norm_result()
        dy_out_p_g = self.get_dygrap_global_norm_result()
        for ((p_np, g_np), (p_dy, g_dy)) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)

    def test_clip_by_global_norm_2(self):
        if False:
            i = 10
            return i + 15
        self.init_value()
        self.init_scale = 0.2
        self.max_global_norm = 10
        self.generate_p_g()
        np_p_g = self.get_numpy_global_norm_result()
        dy_out_p_g = self.get_dygrap_global_norm_result()
        for ((p_np, g_np), (p_dy, g_dy)) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)

class TestGradClipByNorm(unittest.TestCase):

    def init_value(self):
        if False:
            i = 10
            return i + 15
        self.max_norm = 5.0
        self.init_scale = 1.0
        self.shape = (10, 10)

    def generate_p_g(self):
        if False:
            i = 10
            return i + 15
        self.para_and_grad = []
        for i in range(10):
            self.para_and_grad.append((np.random.uniform(-self.init_scale, self.init_scale, self.shape).astype('float32'), np.random.uniform(-self.init_scale, self.init_scale, self.shape).astype('float32')))

    def get_numpy_norm_result(self):
        if False:
            i = 10
            return i + 15
        new_p_g = []
        for (p, g) in self.para_and_grad:
            norm = np.sqrt(np.sum(np.square(g)))
            if norm > self.max_norm:
                new_p_g.append((p, g * self.max_norm / norm))
            else:
                new_p_g.append((p, g))
        return new_p_g

    def get_dygrap_norm_result(self):
        if False:
            return 10
        with base.dygraph.guard():
            norm_clip = ClipGradByNorm(self.max_norm)
            p_g_var = []
            for (p, g) in self.para_and_grad:
                new_p = to_variable(p)
                new_g = to_variable(g)
                p_g_var.append((new_p, new_g))
            new_p_g_var = norm_clip(p_g_var)
            p_g_dy_out = []
            for (p, g) in new_p_g_var:
                p_g_dy_out.append((p.numpy(), g.numpy()))
            return p_g_dy_out

    def test_clip_by_norm(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_value()
        self.generate_p_g()
        np_p_g = self.get_numpy_norm_result()
        dy_out_p_g = self.get_dygrap_norm_result()
        for ((p_np, g_np), (p_dy, g_dy)) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)

    def test_clip_by_norm_2(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_value()
        self.init_scale = 0.2
        self.max_norm = 10.0
        self.generate_p_g()
        np_p_g = self.get_numpy_norm_result()
        dy_out_p_g = self.get_dygrap_norm_result()
        for ((p_np, g_np), (p_dy, g_dy)) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)

class TestGradClipByValue(unittest.TestCase):

    def init_value(self):
        if False:
            while True:
                i = 10
        self.max_value = 0.8
        self.min_value = -0.1
        self.init_scale = 1.0
        self.shape = (10, 10)

    def generate_p_g(self):
        if False:
            return 10
        self.para_and_grad = []
        for i in range(10):
            self.para_and_grad.append((np.random.uniform(-self.init_scale, self.init_scale, self.shape).astype('float32'), np.random.uniform(-self.init_scale, self.init_scale, self.shape).astype('float32')))

    def get_numpy_clip_result(self):
        if False:
            return 10
        new_p_g = []
        for (p, g) in self.para_and_grad:
            new_p_g.append((p, np.clip(g, self.min_value, self.max_value)))
        return new_p_g

    def get_dygrap_clip_result(self):
        if False:
            return 10
        with base.dygraph.guard():
            value_clip = ClipGradByValue(max=self.max_value, min=self.min_value)
            p_g_var = []
            for (p, g) in self.para_and_grad:
                new_p = to_variable(p)
                new_g = to_variable(g)
                p_g_var.append((new_p, new_g))
            new_p_g_var = value_clip(p_g_var)
            p_g_dy_out = []
            for (p, g) in new_p_g_var:
                p_g_dy_out.append((p.numpy(), g.numpy()))
            return p_g_dy_out

    def test_clip_by_value(self):
        if False:
            while True:
                i = 10
        self.init_value()
        self.generate_p_g()
        np_p_g = self.get_numpy_clip_result()
        dy_out_p_g = self.get_dygrap_clip_result()
        for ((p_np, g_np), (p_dy, g_dy)) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)

    def test_clip_by_value_2(self):
        if False:
            print('Hello World!')
        self.init_value()
        self.init_scale = 0.2
        self.generate_p_g()
        np_p_g = self.get_numpy_clip_result()
        dy_out_p_g = self.get_dygrap_clip_result()
        for ((p_np, g_np), (p_dy, g_dy)) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)

    def test_clip_by_value_3(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_value()
        self.init_scale = 0.5
        self.max_value = 0.6
        self.min_value = None
        self.generate_p_g()
        np_p_g = self.get_numpy_clip_result()
        dy_out_p_g = self.get_dygrap_clip_result()
        for ((p_np, g_np), (p_dy, g_dy)) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)
if __name__ == '__main__':
    unittest.main()