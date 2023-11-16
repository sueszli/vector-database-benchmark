import unittest
import numpy as np
import paddle
from paddle.vision.transforms import transforms
SEED = 2022

class TestTransformUnitTestBase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.img = (np.random.rand(*self.get_shape()) * 255.0).astype(np.float32)
        self.set_trans_api()

    def get_shape(self):
        if False:
            print('Hello World!')
        return (3, 64, 64)

    def set_trans_api(self):
        if False:
            print('Hello World!')
        self.api = transforms.Resize(size=16)

    def dynamic_transform(self):
        if False:
            i = 10
            return i + 15
        paddle.seed(SEED)
        img_t = paddle.to_tensor(self.img)
        return self.api(img_t)

    def static_transform(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        paddle.seed(SEED)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(shape=self.get_shape(), dtype=paddle.float32, name='img')
            out = self.api(x)
        exe = paddle.static.Executor()
        res = exe.run(main_program, fetch_list=[out], feed={'img': self.img})
        paddle.disable_static()
        return res[0]

    def test_transform(self):
        if False:
            i = 10
            return i + 15
        dy_res = self.dynamic_transform()
        if isinstance(dy_res, paddle.Tensor):
            dy_res = dy_res.numpy()
        st_res = self.static_transform()
        np.testing.assert_almost_equal(dy_res, st_res)

class TestResize(TestTransformUnitTestBase):

    def set_trans_api(self):
        if False:
            print('Hello World!')
        self.api = transforms.Resize(size=(16, 16))

class TestResizeError(TestTransformUnitTestBase):

    def test_transform(self):
        if False:
            while True:
                i = 10
        pass

    def test_error(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        with self.assertRaises(NotImplementedError):
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                x = paddle.static.data(shape=[-1, -1, -1], dtype=paddle.float32, name='img')
                self.api(x)
        paddle.disable_static()

class TestRandomVerticalFlip0(TestTransformUnitTestBase):

    def set_trans_api(self):
        if False:
            while True:
                i = 10
        self.api = transforms.RandomVerticalFlip(prob=0)

class TestRandomVerticalFlip1(TestTransformUnitTestBase):

    def set_trans_api(self):
        if False:
            return 10
        self.api = transforms.RandomVerticalFlip(prob=1)

class TestRandomHorizontalFlip0(TestTransformUnitTestBase):

    def set_trans_api(self):
        if False:
            i = 10
            return i + 15
        self.api = transforms.RandomHorizontalFlip(0)

class TestRandomHorizontalFlip1(TestTransformUnitTestBase):

    def set_trans_api(self):
        if False:
            for i in range(10):
                print('nop')
        self.api = transforms.RandomHorizontalFlip(1)

class TestRandomCrop_random(TestTransformUnitTestBase):

    def get_shape(self):
        if False:
            print('Hello World!')
        return (3, 240, 240)

    def set_trans_api(self):
        if False:
            i = 10
            return i + 15
        self.crop_size = (224, 224)
        self.api = transforms.RandomCrop(self.crop_size)

    def assert_test_random_equal(self, res, eps=0.0001):
        if False:
            while True:
                i = 10
        (_, h, w) = self.get_shape()
        (c_h, c_w) = self.crop_size
        res_assert = True
        for y in range(h - c_h):
            for x in range(w - c_w):
                diff_abs_sum = np.abs(self.img[:, y:y + c_h, x:x + c_w] - res).sum()
                if diff_abs_sum < eps:
                    res_assert = False
                    break
            if not res_assert:
                break
        assert not res_assert

    def test_transform(self):
        if False:
            while True:
                i = 10
        dy_res = self.dynamic_transform().numpy()
        st_res = self.static_transform()
        self.assert_test_random_equal(dy_res)
        self.assert_test_random_equal(st_res)

class TestRandomCrop_same(TestTransformUnitTestBase):

    def get_shape(self):
        if False:
            for i in range(10):
                print('nop')
        return (3, 224, 224)

    def set_trans_api(self):
        if False:
            for i in range(10):
                print('nop')
        self.crop_size = (224, 224)
        self.api = transforms.RandomCrop(self.crop_size)

class TestRandomRotation(TestTransformUnitTestBase):

    def set_trans_api(self):
        if False:
            for i in range(10):
                print('nop')
        degree = np.random.uniform(-180, 180)
        eps = 0.0001
        degree_tuple = (degree - eps, degree + eps)
        self.api = transforms.RandomRotation(degree_tuple)

class TestRandomRotation_expand_True(TestTransformUnitTestBase):

    def set_trans_api(self):
        if False:
            print('Hello World!')
        degree = np.random.uniform(-180, 180)
        eps = 0.0001
        degree_tuple = (degree - eps, degree + eps)
        self.api = transforms.RandomRotation(degree_tuple, expand=True, fill=3)

class TestRandomErasing(TestTransformUnitTestBase):

    def set_trans_api(self):
        if False:
            i = 10
            return i + 15
        self.value = 100
        self.scale = (0.02, 0.33)
        self.ratio = (0.3, 3.3)
        self.api = transforms.RandomErasing(prob=1, value=self.value, scale=self.scale, ratio=self.ratio)

    def test_transform(self):
        if False:
            i = 10
            return i + 15
        dy_res = self.dynamic_transform()
        if isinstance(dy_res, paddle.Tensor):
            dy_res = dy_res.numpy()
        st_res = self.static_transform()
        self.assert_test_erasing(dy_res)
        self.assert_test_erasing(st_res)

    def assert_test_erasing(self, arr):
        if False:
            return 10
        (_, h, w) = arr.shape
        area = h * w
        height = (arr[2] == self.value).cumsum(1)[:, -1].max()
        width = (arr[2] == self.value).cumsum(0)[-1].max()
        erasing_area = height * width
        assert self.ratio[0] < height / width < self.ratio[1]
        assert self.scale[0] < erasing_area / area < self.scale[1]

class TestRandomResizedCrop(TestTransformUnitTestBase):

    def set_trans_api(self, eps=0.0001):
        if False:
            while True:
                i = 10
        (c, h, w) = self.get_shape()
        size = (h, w)
        scale = (1 - eps, 1.0)
        ratio = (1 - eps, 1.0)
        self.api = transforms.RandomResizedCrop(size, scale=scale, ratio=ratio)
if __name__ == '__main__':
    unittest.main()