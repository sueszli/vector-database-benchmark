import pytest
import torch
from torch.autograd import gradcheck
from kornia.augmentation import RandomMotionBlur, RandomMotionBlur3D
from kornia.filters import motion_blur, motion_blur3d
from kornia.testing import assert_close, tensor_to_gradcheck_var

class TestRandomMotionBlur:

    @pytest.mark.xfail(reason='might fail under windows OS due to printing preicision.')
    def test_smoke(self, device):
        if False:
            print('Hello World!')
        f = RandomMotionBlur(kernel_size=(3, 5), angle=(10, 30), direction=0.5)
        repr = "RandomMotionBlur(kernel_size=(3, 5), angle=tensor([10., 30.]), direction=tensor([-0.5000, 0.5000]), border_type='constant', p=0.5, p_batch=1.0, same_on_batch=False, return_transform=None)"
        assert str(f) == repr

    @pytest.mark.parametrize('kernel_size', [(3, 5), (7, 21)])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    @pytest.mark.parametrize('p', [0.0, 1.0])
    def test_random_motion_blur(self, kernel_size, same_on_batch, p, device, dtype):
        if False:
            while True:
                i = 10
        f = RandomMotionBlur(kernel_size=kernel_size, angle=(10, 30), direction=0.5, same_on_batch=same_on_batch, p=p)
        torch.manual_seed(0)
        batch_size = 2
        input = torch.randn(1, 3, 5, 6, device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)
        output = f(input)
        if same_on_batch:
            assert_close(output[0], output[1], rtol=0.0001, atol=0.0001)
        elif p == 0:
            assert_close(output, input, rtol=0.0001, atol=0.0001)
        else:
            assert not torch.allclose(output[0], output[1], rtol=0.0001, atol=0.0001)
        assert output.shape == torch.Size([batch_size, 3, 5, 6])

    @pytest.mark.parametrize('input_shape', [(1, 1, 5, 5), (2, 1, 5, 5)])
    def test_against_functional(self, input_shape):
        if False:
            print('Hello World!')
        input = torch.randn(*input_shape)
        f = RandomMotionBlur(kernel_size=(3, 5), angle=(10, 30), direction=0.5, p=1.0)
        output = f(input)
        expected = motion_blur(input, f._params['ksize_factor'].unique().item(), f._params['angle_factor'], f._params['direction_factor'], f.flags['border_type'].name.lower())
        assert_close(output, expected, rtol=0.0001, atol=0.0001)

    @pytest.mark.slow
    def test_gradcheck(self, device):
        if False:
            print('Hello World!')
        torch.manual_seed(0)
        inp = torch.rand((1, 3, 11, 7)).to(device)
        inp = tensor_to_gradcheck_var(inp)
        params = {'batch_prob': torch.tensor([True]), 'ksize_factor': torch.tensor([31]), 'angle_factor': torch.tensor([30.0]), 'direction_factor': torch.tensor([-0.5]), 'border_type': torch.tensor([0]), 'idx': torch.tensor([0])}
        assert gradcheck(RandomMotionBlur(kernel_size=3, angle=(10, 30), direction=(-0.5, 0.5), p=1.0), (inp, params), raise_exception=True)

class TestRandomMotionBlur3D:

    @pytest.mark.xfail(reason='might fail under windows OS due to printing preicision.')
    def test_smoke(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        f = RandomMotionBlur3D(kernel_size=(3, 5), angle=(10, 30), direction=0.5)
        repr = "RandomMotionBlur3D(kernel_size=(3, 5), angle=tensor([[10., 30.],\n        [10., 30.],\n        [10., 30.]]), direction=tensor([-0.5000, 0.5000]), border_type='constant', p=0.5, p_batch=1.0, same_on_batch=False, return_transform=None)"
        assert str(f) == repr

    @pytest.mark.parametrize('same_on_batch', [True, False])
    @pytest.mark.parametrize('p', [0.0, 1.0])
    def test_random_motion_blur(self, same_on_batch, p, device, dtype):
        if False:
            return 10
        f = RandomMotionBlur3D(kernel_size=(3, 5), angle=(10, 30), direction=0.5, same_on_batch=same_on_batch, p=p)
        batch_size = 2
        input = torch.randn(1, 3, 5, 6, 7, device=device, dtype=dtype).repeat(batch_size, 1, 1, 1, 1)
        output = f(input)
        if same_on_batch:
            assert_close(output[0], output[1], rtol=0.0001, atol=0.0001)
        elif p == 0:
            assert_close(output, input, rtol=0.0001, atol=0.0001)
        else:
            assert not torch.allclose(output[0], output[1], rtol=0.0001, atol=0.0001)
        assert output.shape == torch.Size([batch_size, 3, 5, 6, 7])

    @pytest.mark.parametrize('input_shape', [(1, 1, 5, 6, 7), (2, 1, 5, 6, 7)])
    def test_against_functional(self, input_shape):
        if False:
            i = 10
            return i + 15
        input = torch.randn(*input_shape)
        f = RandomMotionBlur3D(kernel_size=(3, 5), angle=(10, 30), direction=0.5, p=1.0)
        output = f(input)
        expected = motion_blur3d(input, f._params['ksize_factor'].unique().item(), f._params['angle_factor'], f._params['direction_factor'], f.flags['border_type'].name.lower())
        assert_close(output, expected, rtol=0.0001, atol=0.0001)

    @pytest.mark.slow
    def test_gradcheck(self, device, dtype):
        if False:
            print('Hello World!')
        torch.manual_seed(0)
        inp = torch.rand((1, 3, 6, 7), device=device, dtype=dtype)
        inp = tensor_to_gradcheck_var(inp)
        params = {'batch_prob': torch.tensor([True]), 'ksize_factor': torch.tensor([31]), 'angle_factor': torch.tensor([[30.0, 30.0, 30.0]]), 'direction_factor': torch.tensor([-0.5]), 'border_type': torch.tensor([0]), 'idx': torch.tensor([0])}
        assert gradcheck(RandomMotionBlur3D(kernel_size=3, angle=(10, 30), direction=(-0.5, 0.5), p=1.0), (inp, params), raise_exception=True)