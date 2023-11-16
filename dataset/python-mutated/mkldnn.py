import torch

class MkldnnLinear(torch.jit.ScriptModule):

    def __init__(self, dense_module, dtype):
        if False:
            while True:
                i = 10
        super().__init__()
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))
        if dense_module.bias is not None:
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            self.register_buffer('bias', torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        if False:
            print('Hello World!')
        return (self.weight.to_dense(), self.bias.to_dense(), self.training)

    @torch.jit.script_method
    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.training = state[2]

    @torch.jit.script_method
    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
        y_mkldnn = torch._C._nn.mkldnn_linear(x_mkldnn, self.weight, self.bias)
        y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
        return y

class _MkldnnConvNd(torch.jit.ScriptModule):
    """Common base of MkldnnConv1d and MkldnnConv2d."""
    __constants__ = ['stride', 'padding', 'dilation', 'groups']

    def __init__(self, dense_module):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups
        if dense_module.bias is not None:
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            self.register_buffer('bias', torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.weight.to_dense(), self.bias.to_dense(), self.training)

    @torch.jit.script_method
    def forward(self, x):
        if False:
            return 10
        return torch.mkldnn_convolution(x, self.weight, self.bias, self.padding, self.stride, self.dilation, self.groups)

class MkldnnConv1d(_MkldnnConvNd):

    def __init__(self, dense_module, dtype):
        if False:
            print('Hello World!')
        super().__init__(dense_module)
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

    @torch.jit.script_method
    def __setstate__(self, state):
        if False:
            i = 10
            return i + 15
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.training = state[2]

class MkldnnConv2d(_MkldnnConvNd):

    def __init__(self, dense_module, dtype):
        if False:
            while True:
                i = 10
        super().__init__(dense_module)
        self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv2d_weight(dense_module.weight.to_mkldnn(dtype), self.padding, self.stride, self.dilation, self.groups))

    @torch.jit.script_method
    def __setstate__(self, state):
        if False:
            i = 10
            return i + 15
        self.weight = torch._C._nn.mkldnn_reorder_conv2d_weight(state[0].to_mkldnn(), self.padding, self.stride, self.dilation, self.groups)
        self.bias = state[1].to_mkldnn()
        self.training = state[2]

class MkldnnConv3d(_MkldnnConvNd):

    def __init__(self, dense_module, dtype):
        if False:
            i = 10
            return i + 15
        super().__init__(dense_module)
        self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv3d_weight(dense_module.weight.to_mkldnn(dtype), self.padding, self.stride, self.dilation, self.groups))

    @torch.jit.script_method
    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.weight = torch._C._nn.mkldnn_reorder_conv3d_weight(state[0].to_mkldnn(), self.padding, self.stride, self.dilation, self.groups)
        self.bias = state[1].to_mkldnn()
        self.training = state[2]

class MkldnnBatchNorm(torch.jit.ScriptModule):
    __constants__ = ['exponential_average_factor', 'eps']

    def __init__(self, dense_module):
        if False:
            i = 10
            return i + 15
        super().__init__()
        assert not dense_module.training
        assert dense_module.track_running_stats
        assert dense_module.affine
        if dense_module.momentum is None:
            self.exponential_average_factor = 0.0
        else:
            self.exponential_average_factor = dense_module.momentum
        self.eps = dense_module.eps
        self.register_buffer('weight', dense_module.weight.to_mkldnn())
        self.register_buffer('bias', dense_module.bias.to_mkldnn())
        self.register_buffer('running_mean', dense_module.running_mean.to_mkldnn())
        self.register_buffer('running_var', dense_module.running_var.to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        weight = self.weight.to_dense()
        bias = self.bias.to_dense()
        running_mean = self.running_mean.to_dense()
        running_var = self.running_var.to_dense()
        return (weight, bias, running_mean, running_var, self.training)

    @torch.jit.script_method
    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.running_mean = state[2].to_mkldnn()
        self.running_var = state[3].to_mkldnn()
        self.training = state[4]

    @torch.jit.script_method
    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return torch.batch_norm(x, self.weight, self.bias, self.running_mean, self.running_var, False, self.exponential_average_factor, self.eps, False)

class MkldnnPrelu(torch.jit.ScriptModule):

    def __init__(self, dense_module, dtype):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

    @torch.jit.script_method
    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.weight.to_dense(), self.training)

    @torch.jit.script_method
    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.weight = state[0].to_mkldnn()
        self.training = state[1]

    @torch.jit.script_method
    def forward(self, x):
        if False:
            return 10
        x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
        y_mkldnn = torch.prelu(x_mkldnn, self.weight)
        y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
        return y

def to_mkldnn(module, dtype=torch.float):
    if False:
        print('Hello World!')
    assert dtype in [torch.float, torch.bfloat16, torch.half], 'MKLDNN only support float, bfloat16, and half path now'

    def m_fn(m, d):
        if False:
            i = 10
            return i + 15
        if isinstance(m, torch.nn.Linear):
            return MkldnnLinear(m, d)
        elif isinstance(m, torch.nn.Conv1d):
            return MkldnnConv1d(m, d)
        elif isinstance(m, torch.nn.Conv2d):
            return MkldnnConv2d(m, d)
        elif isinstance(m, torch.nn.Conv3d):
            return MkldnnConv3d(m, d)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            return MkldnnBatchNorm(m)
        elif isinstance(m, torch.nn.PReLU):
            return MkldnnPrelu(m, d)
        else:
            return m

    def m_fn_rec(m, d):
        if False:
            for i in range(10):
                print('nop')
        new_m = m_fn(m, d)
        for (name, sub_m) in m.named_children():
            setattr(new_m, name, m_fn_rec(sub_m, d))
        return new_m
    return m_fn_rec(module, dtype)