import torch

class CustomFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        if False:
            for i in range(10):
                print('nop')
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            return 10
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input