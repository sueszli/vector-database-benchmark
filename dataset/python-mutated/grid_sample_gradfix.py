"""Custom replacement for `torch.nn.functional.grid_sample` that
supports arbitrarily high order gradients between the input and output.
Only works on 2D images and assumes
`mode='bilinear'`, `padding_mode='zeros'`, `align_corners=False`."""
import torch
enabled = False

def grid_sample(input, grid):
    if False:
        return 10
    if _should_use_custom_op():
        return _GridSample2dForward.apply(input, grid)
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)

def _should_use_custom_op():
    if False:
        i = 10
        return i + 15
    return enabled

class _GridSample2dForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid):
        if False:
            print('Hello World!')
        assert input.ndim == 4
        assert grid.ndim == 4
        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            print('Hello World!')
        (input, grid) = ctx.saved_tensors
        (grad_input, grad_grid) = _GridSample2dBackward.apply(grad_output, input, grid)
        return (grad_input, grad_grid)

class _GridSample2dBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, grad_output, input, grid):
        if False:
            return 10
        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
        (grad_input, grad_grid) = op(grad_output, input, grid, 0, 0, False)
        ctx.save_for_backward(grid)
        return (grad_input, grad_grid)

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        if False:
            for i in range(10):
                print('nop')
        _ = grad2_grad_grid
        (grid,) = ctx.saved_tensors
        grad2_grad_output = None
        grad2_input = None
        grad2_grid = None
        if ctx.needs_input_grad[0]:
            grad2_grad_output = _GridSample2dForward.apply(grad2_grad_input, grid)
        assert not ctx.needs_input_grad[2]
        return (grad2_grad_output, grad2_input, grad2_grid)