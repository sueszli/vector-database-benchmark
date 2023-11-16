import torch
import torch.nn
import numpy as np
import torch.nn.functional as F

class Flatten(torch.nn.Module):

    def __init__(self):
        if False:
            return 10
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        if False:
            return 10
        return in_tensor.view((in_tensor.size()[0], -1))

def module_tracker(fwd_hook_func):
    if False:
        while True:
            i = 10
    '\n    Wrapper for tracking the layers throughout the forward pass.\n\n    Args:\n        fwd_hook_func: Forward hook function to be wrapped.\n\n    Returns:\n        Wrapped method.\n\n    '

    def hook_wrapper(relevance_propagator_instance, layer, *args):
        if False:
            while True:
                i = 10
        relevance_propagator_instance.module_list.append(layer)
        return fwd_hook_func(relevance_propagator_instance, layer, *args)
    return hook_wrapper

class RelevancePropagator:
    """
    Class for computing the relevance propagation and supplying
    the necessary forward hooks for all layers.
    """
    allowed_pass_layers = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.ReLU, torch.nn.ELU, Flatten, torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.Softmax, torch.nn.LogSoftmax)
    available_methods = ['e-rule', 'b-rule']

    def __init__(self, lrp_exponent, beta, method, epsilon, device):
        if False:
            return 10
        self.device = device
        self.layer = None
        self.p = lrp_exponent
        self.beta = beta
        self.eps = epsilon
        self.warned_log_softmax = False
        self.module_list = []
        if method not in self.available_methods:
            raise NotImplementedError('Only methods available are: ' + str(self.available_methods))
        self.method = method

    def reset_module_list(self):
        if False:
            i = 10
            return i + 15
        '\n        The module list is reset for every evaluation, in change the order or number\n        of layers changes dynamically.\n\n        Returns:\n            None\n\n        '
        self.module_list = []
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def compute_propagated_relevance(self, layer, relevance):
        if False:
            return 10
        '\n        This method computes the backward pass for the incoming relevance\n        for the specified layer.\n\n        Args:\n            layer: Layer to be reverted.\n            relevance: Incoming relevance from higher up in the network.\n\n        Returns:\n            The\n\n        '
        if isinstance(layer, (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
            return self.max_pool_nd_inverse(layer, relevance).detach()
        elif isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            return self.conv_nd_inverse(layer, relevance).detach()
        elif isinstance(layer, torch.nn.LogSoftmax):
            if relevance.sum() < 0:
                relevance[relevance == 0] = -1000000.0
                relevance = relevance.exp()
                if not self.warned_log_softmax:
                    print('WARNING: LogSoftmax layer was turned into probabilities.')
                    self.warned_log_softmax = True
            return relevance
        elif isinstance(layer, self.allowed_pass_layers):
            return relevance
        elif isinstance(layer, torch.nn.Linear):
            return self.linear_inverse(layer, relevance).detach()
        else:
            raise NotImplementedError('The network contains layers that are currently not supported {0:s}'.format(str(layer)))

    def get_layer_fwd_hook(self, layer):
        if False:
            return 10
        '\n        Each layer might need to save very specific data during the forward\n        pass in order to allow for relevance propagation in the backward\n        pass. For example, for max_pooling, we need to store the\n        indices of the max values. In convolutional layers, we need to calculate\n        the normalizations, to ensure the overall amount of relevance is conserved.\n\n        Args:\n            layer: Layer instance for which forward hook is needed.\n\n        Returns:\n            Layer-specific forward hook.\n\n        '
        if isinstance(layer, (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
            return self.max_pool_nd_fwd_hook
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            return self.conv_nd_fwd_hook
        if isinstance(layer, self.allowed_pass_layers):
            return self.silent_pass
        if isinstance(layer, torch.nn.Linear):
            return self.linear_fwd_hook
        else:
            raise NotImplementedError('The network contains layers that are currently not supported {0:s}'.format(str(layer)))

    @staticmethod
    def get_conv_method(conv_module):
        if False:
            i = 10
            return i + 15
        "\n        Get dimension-specific convolution.\n        The forward pass and inversion are made in a\n        'dimensionality-agnostic' manner and are the same for\n        all nd instances of the layer, except for the functional\n        that needs to be used.\n\n        Args:\n            conv_module: instance of convolutional layer.\n\n        Returns:\n            The correct functional used in the convolutional layer.\n\n        "
        conv_func_mapper = {torch.nn.Conv1d: F.conv1d, torch.nn.Conv2d: F.conv2d, torch.nn.Conv3d: F.conv3d}
        return conv_func_mapper[type(conv_module)]

    @staticmethod
    def get_inv_conv_method(conv_module):
        if False:
            print('Hello World!')
        "\n        Get dimension-specific convolution inversion layer.\n        The forward pass and inversion are made in a\n        'dimensionality-agnostic' manner and are the same for\n        all nd instances of the layer, except for the functional\n        that needs to be used.\n\n        Args:\n            conv_module: instance of convolutional layer.\n\n        Returns:\n            The correct functional used for inverting the convolutional layer.\n\n        "
        conv_func_mapper = {torch.nn.Conv1d: F.conv_transpose1d, torch.nn.Conv2d: F.conv_transpose2d, torch.nn.Conv3d: F.conv_transpose3d}
        return conv_func_mapper[type(conv_module)]

    @module_tracker
    def silent_pass(self, m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    def get_inv_max_pool_method(max_pool_instance):
        if False:
            i = 10
            return i + 15
        "\n        Get dimension-specific max_pooling layer.\n        The forward pass and inversion are made in a\n        'dimensionality-agnostic' manner and are the same for\n        all nd instances of the layer, except for the functional\n        that needs to be used.\n\n        Args:\n            max_pool_instance: instance of max_pool layer.\n\n        Returns:\n            The correct functional used in the max_pooling layer.\n\n        "
        conv_func_mapper = {torch.nn.MaxPool1d: F.max_unpool1d, torch.nn.MaxPool2d: F.max_unpool2d, torch.nn.MaxPool3d: F.max_unpool3d}
        return conv_func_mapper[type(max_pool_instance)]

    def linear_inverse(self, m, relevance_in):
        if False:
            print('Hello World!')
        if self.method == 'e-rule':
            m.in_tensor = m.in_tensor.pow(self.p)
            w = m.weight.pow(self.p)
            norm = F.linear(m.in_tensor, w, bias=None)
            norm = norm + torch.sign(norm) * self.eps
            relevance_in[norm == 0] = 0
            norm[norm == 0] = 1
            relevance_out = F.linear(relevance_in / norm, w.t(), bias=None)
            relevance_out *= m.in_tensor
            del m.in_tensor, norm, w, relevance_in
            return relevance_out
        if self.method == 'b-rule':
            (out_c, in_c) = m.weight.size()
            w = m.weight.repeat((4, 1))
            w[:out_c][w[:out_c] < 0] = 0
            w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0
            w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
            w[-out_c:][w[-out_c:] > 0] = 0
            m.in_tensor = m.in_tensor.repeat((1, 4))
            m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
            m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
            m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0
            norm_shape = m.out_shape
            norm_shape[1] *= 4
            norm = torch.zeros(norm_shape).to(self.device)
            for i in range(4):
                norm[:, out_c * i:(i + 1) * out_c] = F.linear(m.in_tensor[:, in_c * i:(i + 1) * in_c], w[out_c * i:(i + 1) * out_c], bias=None)
            norm_shape[1] = norm_shape[1] // 2
            new_norm = torch.zeros(norm_shape).to(self.device)
            new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
            new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
            norm = new_norm
            mask = norm == 0
            norm[mask] = 1
            rare_neurons = mask[:, :out_c] + mask[:, out_c:]
            norm[:, :out_c][rare_neurons] *= 1 if self.beta == -1 else 1 + self.beta
            norm[:, out_c:][rare_neurons] *= 1 if self.beta == 0 else -self.beta
            norm += self.eps * torch.sign(norm)
            input_relevance = relevance_in.squeeze(dim=-1).repeat(1, 4)
            input_relevance[:, :2 * out_c] *= (1 + self.beta) / norm[:, :out_c].repeat(1, 2)
            input_relevance[:, 2 * out_c:] *= -self.beta / norm[:, out_c:].repeat(1, 2)
            inv_w = w.t()
            relevance_out = torch.zeros_like(m.in_tensor)
            for i in range(4):
                relevance_out[:, i * in_c:(i + 1) * in_c] = F.linear(input_relevance[:, i * out_c:(i + 1) * out_c], weight=inv_w[:, i * out_c:(i + 1) * out_c], bias=None)
            relevance_out *= m.in_tensor
            sum_weights = torch.zeros([in_c, in_c * 4, 1]).to(self.device)
            for i in range(in_c):
                sum_weights[i, i::in_c] = 1
            relevance_out = F.conv1d(relevance_out[:, :, None], weight=sum_weights, bias=None)
            del sum_weights, input_relevance, norm, rare_neurons, mask, new_norm, m.in_tensor, w, inv_w
            return relevance_out

    @module_tracker
    def linear_fwd_hook(self, m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        setattr(m, 'in_tensor', in_tensor[0])
        setattr(m, 'out_shape', list(out_tensor.size()))
        return

    def max_pool_nd_inverse(self, layer_instance, relevance_in):
        if False:
            print('Hello World!')
        relevance_in = relevance_in.view(layer_instance.out_shape)
        invert_pool = self.get_inv_max_pool_method(layer_instance)
        inverted = invert_pool(relevance_in, layer_instance.indices, layer_instance.kernel_size, layer_instance.stride, layer_instance.padding, output_size=layer_instance.in_shape)
        del layer_instance.indices
        return inverted

    @module_tracker
    def max_pool_nd_fwd_hook(self, m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        if False:
            while True:
                i = 10
        _ = self
        tmp_return_indices = bool(m.return_indices)
        m.return_indices = True
        (_, indices) = m.forward(in_tensor[0])
        m.return_indices = tmp_return_indices
        setattr(m, 'indices', indices)
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_shape', in_tensor[0].size())

    def conv_nd_inverse(self, m, relevance_in):
        if False:
            i = 10
            return i + 15
        relevance_in = relevance_in.view(m.out_shape)
        inv_conv_nd = self.get_inv_conv_method(m)
        conv_nd = self.get_conv_method(m)
        if self.method == 'e-rule':
            with torch.no_grad():
                m.in_tensor = m.in_tensor.pow(self.p).detach()
                w = m.weight.pow(self.p).detach()
                norm = conv_nd(m.in_tensor, weight=w, bias=None, stride=m.stride, padding=m.padding, groups=m.groups)
                norm = norm + torch.sign(norm) * self.eps
                relevance_in[norm == 0] = 0
                norm[norm == 0] = 1
                relevance_out = inv_conv_nd(relevance_in / norm, weight=w, bias=None, padding=m.padding, stride=m.stride, groups=m.groups)
                relevance_out *= m.in_tensor
                del m.in_tensor, norm, w
                return relevance_out
        if self.method == 'b-rule':
            with torch.no_grad():
                w = m.weight
                (out_c, in_c) = (m.out_channels, m.in_channels)
                repeats = np.array(np.ones_like(w.size()).flatten(), dtype=int)
                repeats[0] *= 4
                w = w.repeat(tuple(repeats))
                w[:out_c][w[:out_c] < 0] = 0
                w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0
                w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
                w[-out_c:][w[-out_c:] > 0] = 0
                repeats = np.array(np.ones_like(m.in_tensor.size()).flatten(), dtype=int)
                repeats[1] *= 4
                m.in_tensor = m.in_tensor.repeat(tuple(repeats))
                m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
                m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
                m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0
                groups = 4
                norm = conv_nd(m.in_tensor, weight=w, bias=None, stride=m.stride, padding=m.padding, dilation=m.dilation, groups=groups * m.groups)
                new_shape = m.out_shape
                new_shape[1] *= 2
                new_norm = torch.zeros(new_shape).to(self.device)
                new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
                new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
                norm = new_norm
                mask = norm == 0
                norm[mask] = 1
                rare_neurons = mask[:, :out_c] + mask[:, out_c:]
                norm[:, :out_c][rare_neurons] *= 1 if self.beta == -1 else 1 + self.beta
                norm[:, out_c:][rare_neurons] *= 1 if self.beta == 0 else -self.beta
                norm += self.eps * torch.sign(norm)
                spatial_dims = [1] * len(relevance_in.size()[2:])
                input_relevance = relevance_in.repeat(1, 4, *spatial_dims)
                input_relevance[:, :2 * out_c] *= (1 + self.beta) / norm[:, :out_c].repeat(1, 2, *spatial_dims)
                input_relevance[:, 2 * out_c:] *= -self.beta / norm[:, out_c:].repeat(1, 2, *spatial_dims)
                relevance_out = torch.zeros_like(m.in_tensor)
                tmp_result = result = None
                for i in range(4):
                    tmp_result = inv_conv_nd(input_relevance[:, i * out_c:(i + 1) * out_c], weight=w[i * out_c:(i + 1) * out_c], bias=None, padding=m.padding, stride=m.stride, groups=m.groups)
                    result = torch.zeros_like(relevance_out[:, i * in_c:(i + 1) * in_c])
                    tmp_size = tmp_result.size()
                    slice_list = [slice(0, l) for l in tmp_size]
                    result[slice_list] += tmp_result
                    relevance_out[:, i * in_c:(i + 1) * in_c] = result
                relevance_out *= m.in_tensor
                sum_weights = torch.zeros([in_c, in_c * 4, *spatial_dims]).to(self.device)
                for i in range(m.in_channels):
                    sum_weights[i, i::in_c] = 1
                relevance_out = conv_nd(relevance_out, weight=sum_weights, bias=None)
                del sum_weights, m.in_tensor, result, mask, rare_neurons, norm, new_norm, input_relevance, tmp_result, w
                return relevance_out

    @module_tracker
    def conv_nd_fwd_hook(self, m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        if False:
            return 10
        setattr(m, 'in_tensor', in_tensor[0])
        setattr(m, 'out_shape', list(out_tensor.size()))
        return