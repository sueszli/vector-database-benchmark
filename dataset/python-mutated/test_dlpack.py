import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase, run_tests, IS_JETSON
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCUDA, dtypes, skipMeta, skipCUDAIfRocm, onlyNativeDeviceTypes
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.utils.dlpack import from_dlpack, to_dlpack

class TestTorchDlPack(TestCase):
    exact_dtype = True

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_dlpack_capsule_conversion(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = make_tensor((5,), dtype=dtype, device=device)
        z = from_dlpack(to_dlpack(x))
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_dlpack_protocol_conversion(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = make_tensor((5,), dtype=dtype, device=device)
        z = from_dlpack(x)
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    def test_dlpack_shared_storage(self, device):
        if False:
            return 10
        x = make_tensor((5,), dtype=torch.float64, device=device)
        z = from_dlpack(to_dlpack(x))
        z[0] = z[0] + 20.0
        self.assertEqual(z, x)

    @skipMeta
    @onlyCUDA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_dlpack_conversion_with_streams(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = make_tensor((5,), dtype=dtype, device=device) + 1
        if IS_JETSON:
            stream.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            z = from_dlpack(x)
        stream.synchronize()
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_from_dlpack(self, device, dtype):
        if False:
            while True:
                i = 10
        x = make_tensor((5,), dtype=dtype, device=device)
        y = torch.from_dlpack(x)
        self.assertEqual(x, y)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_from_dlpack_noncontinguous(self, device, dtype):
        if False:
            return 10
        x = make_tensor((25,), dtype=dtype, device=device).reshape(5, 5)
        y1 = x[0]
        y1_dl = torch.from_dlpack(y1)
        self.assertEqual(y1, y1_dl)
        y2 = x[:, 0]
        y2_dl = torch.from_dlpack(y2)
        self.assertEqual(y2, y2_dl)
        y3 = x[1, :]
        y3_dl = torch.from_dlpack(y3)
        self.assertEqual(y3, y3_dl)
        y4 = x[1]
        y4_dl = torch.from_dlpack(y4)
        self.assertEqual(y4, y4_dl)
        y5 = x.t()
        y5_dl = torch.from_dlpack(y5)
        self.assertEqual(y5, y5_dl)

    @skipMeta
    @onlyCUDA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_dlpack_conversion_with_diff_streams(self, device, dtype):
        if False:
            return 10
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        with torch.cuda.stream(stream_a):
            x = make_tensor((5,), dtype=dtype, device=device) + 1
            z = torch.from_dlpack(x.__dlpack__(stream_b.cuda_stream))
            stream_a.synchronize()
        stream_b.synchronize()
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_from_dlpack_dtype(self, device, dtype):
        if False:
            while True:
                i = 10
        x = make_tensor((5,), dtype=dtype, device=device)
        y = torch.from_dlpack(x)
        assert x.dtype == y.dtype

    @skipMeta
    @onlyCUDA
    def test_dlpack_default_stream(self, device):
        if False:
            print('Hello World!')

        class DLPackTensor:

            def __init__(self, tensor):
                if False:
                    i = 10
                    return i + 15
                self.tensor = tensor

            def __dlpack_device__(self):
                if False:
                    i = 10
                    return i + 15
                return self.tensor.__dlpack_device__()

            def __dlpack__(self, stream=None):
                if False:
                    while True:
                        i = 10
                if torch.version.hip is None:
                    assert stream == 1
                else:
                    assert stream == 0
                capsule = self.tensor.__dlpack__(stream)
                return capsule
        with torch.cuda.stream(torch.cuda.default_stream()):
            x = DLPackTensor(make_tensor((5,), dtype=torch.float32, device=device))
            from_dlpack(x)

    @skipMeta
    @onlyCUDA
    @skipCUDAIfRocm
    def test_dlpack_convert_default_stream(self, device):
        if False:
            i = 10
            return i + 15
        torch.cuda.default_stream().synchronize()
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            x = torch.zeros(1, device=device)
            torch.cuda._sleep(2 ** 20)
            self.assertTrue(torch.cuda.default_stream().query())
            d = x.__dlpack__(1)
        self.assertFalse(torch.cuda.default_stream().query())

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_dlpack_tensor_invalid_stream(self, device, dtype):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            x = make_tensor((5,), dtype=dtype, device=device)
            x.__dlpack__(stream=object())

    @skipMeta
    def test_dlpack_export_requires_grad(self):
        if False:
            return 10
        x = torch.zeros(10, dtype=torch.float32, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, 'require gradient'):
            x.__dlpack__()

    @skipMeta
    def test_dlpack_export_is_conj(self):
        if False:
            print('Hello World!')
        x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        y = torch.conj(x)
        with self.assertRaisesRegex(RuntimeError, 'conjugate bit'):
            y.__dlpack__()

    @skipMeta
    def test_dlpack_export_non_strided(self):
        if False:
            return 10
        x = torch.sparse_coo_tensor([[0]], [1], size=(1,))
        y = torch.conj(x)
        with self.assertRaisesRegex(RuntimeError, 'strided'):
            y.__dlpack__()

    @skipMeta
    def test_dlpack_normalize_strides(self):
        if False:
            i = 10
            return i + 15
        x = torch.rand(16)
        y = x[::3][:1]
        self.assertEqual(y.shape, (1,))
        self.assertEqual(y.stride(), (3,))
        z = from_dlpack(y)
        self.assertEqual(z.shape, (1,))
        self.assertEqual(z.stride(), (1,))
instantiate_device_type_tests(TestTorchDlPack, globals())
if __name__ == '__main__':
    run_tests()