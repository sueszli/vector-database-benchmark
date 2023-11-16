import torch

class LossScaler:

    def __init__(self, scale=1):
        if False:
            while True:
                i = 10
        self.cur_scale = scale

    def has_overflow(self, params):
        if False:
            i = 10
            return i + 15
        return False

    def _has_inf_or_nan(x):
        if False:
            print('Hello World!')
        return False

    def update_scale(self, overflow):
        if False:
            i = 10
            return i + 15
        pass

    @property
    def loss_scale(self):
        if False:
            return 10
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        if False:
            print('Hello World!')
        return tuple((self.loss_scale * g for g in grad_in))

    def backward(self, loss):
        if False:
            while True:
                i = 10
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward()

class DynamicLossScaler:

    def __init__(self, init_scale=2 ** 32, scale_factor=2.0, scale_window=1000):
        if False:
            for i in range(10):
                print('nop')
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    def has_overflow(self, params):
        if False:
            i = 10
            return i + 15
        for p in params:
            if p.grad is not None and DynamicLossScaler._has_inf_or_nan(p.grad.data):
                return True
        return False

    def _has_inf_or_nan(x):
        if False:
            i = 10
            return i + 15
        inf_count = torch.sum(x.abs() == float('inf'))
        if inf_count > 0:
            return True
        nan_count = torch.sum(x != x)
        return nan_count > 0

    def update_scale(self, overflow):
        if False:
            print('Hello World!')
        if overflow:
            self.cur_scale = max(self.cur_scale / self.scale_factor, 1)
            self.last_overflow_iter = self.cur_iter
        elif (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
            self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    @property
    def loss_scale(self):
        if False:
            return 10
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        if False:
            print('Hello World!')
        return tuple((self.loss_scale * g for g in grad_in))

    def backward(self, loss):
        if False:
            i = 10
            return i + 15
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward()