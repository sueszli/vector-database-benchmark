import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F

def sample_mask_indices(input_dim, hidden_dim, simple=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Samples the indices assigned to hidden units during the construction of MADE masks\n\n    :param input_dim: the dimensionality of the input variable\n    :type input_dim: int\n    :param hidden_dim: the dimensionality of the hidden layer\n    :type hidden_dim: int\n    :param simple: True to space fractional indices by rounding to nearest int, false round randomly\n    :type simple: bool\n    '
    indices = torch.linspace(1, input_dim, steps=hidden_dim, device='cpu').to(torch.Tensor().device)
    if simple:
        return torch.round(indices)
    else:
        ints = indices.floor()
        ints += torch.bernoulli(indices - ints)
        return ints

def create_mask(input_dim, context_dim, hidden_dims, permutation, output_dim_multiplier):
    if False:
        i = 10
        return i + 15
    '\n    Creates MADE masks for a conditional distribution\n\n    :param input_dim: the dimensionality of the input variable\n    :type input_dim: int\n    :param context_dim: the dimensionality of the variable that is conditioned on (for conditional densities)\n    :type context_dim: int\n    :param hidden_dims: the dimensionality of the hidden layers(s)\n    :type hidden_dims: list[int]\n    :param permutation: the order of the input variables\n    :type permutation: torch.LongTensor\n    :param output_dim_multiplier: tiles the output (e.g. for when a separate mean and scale parameter are desired)\n    :type output_dim_multiplier: int\n    '
    var_index = torch.empty(permutation.shape, dtype=torch.get_default_dtype())
    var_index[permutation] = torch.arange(input_dim, dtype=torch.get_default_dtype())
    input_indices = torch.cat((torch.zeros(context_dim), 1 + var_index))
    if context_dim > 0:
        hidden_indices = [sample_mask_indices(input_dim, h) - 1 for h in hidden_dims]
    else:
        hidden_indices = [sample_mask_indices(input_dim - 1, h) for h in hidden_dims]
    output_indices = (var_index + 1).repeat(output_dim_multiplier)
    mask_skip = (output_indices.unsqueeze(-1) > input_indices.unsqueeze(0)).type_as(var_index)
    masks = [(hidden_indices[0].unsqueeze(-1) >= input_indices.unsqueeze(0)).type_as(var_index)]
    for i in range(1, len(hidden_dims)):
        masks.append((hidden_indices[i].unsqueeze(-1) >= hidden_indices[i - 1].unsqueeze(0)).type_as(var_index))
    masks.append((output_indices.unsqueeze(-1) > hidden_indices[-1].unsqueeze(0)).type_as(var_index))
    return (masks, mask_skip)

class MaskedLinear(nn.Linear):
    """
    A linear mapping with a given mask on the weights (arbitrary bias)

    :param in_features: the number of input features
    :type in_features: int
    :param out_features: the number of output features
    :type out_features: int
    :param mask: the mask to apply to the in_features x out_features weight matrix
    :type mask: torch.Tensor
    :param bias: whether or not `MaskedLinear` should include a bias term. defaults to `True`
    :type bias: bool
    """

    def __init__(self, in_features, out_features, mask, bias=True):
        if False:
            print('Hello World!')
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask.data)

    def forward(self, _input):
        if False:
            i = 10
            return i + 15
        masked_weight = self.weight * self.mask
        return F.linear(_input, masked_weight, self.bias)

class ConditionalAutoRegressiveNN(nn.Module):
    """
    An implementation of a MADE-like auto-regressive neural network that can input an additional context variable.
    (See Reference [2] Section 3.3 for an explanation of how the conditional MADE architecture works.)

    Example usage:

    >>> x = torch.randn(100, 10)
    >>> y = torch.randn(100, 5)
    >>> arn = ConditionalAutoRegressiveNN(10, 5, [50], param_dims=[1])
    >>> p = arn(x, context=y)  # 1 parameters of size (100, 10)
    >>> arn = ConditionalAutoRegressiveNN(10, 5, [50], param_dims=[1, 1])
    >>> m, s = arn(x, context=y) # 2 parameters of size (100, 10)
    >>> arn = ConditionalAutoRegressiveNN(10, 5, [50], param_dims=[1, 5, 3])
    >>> a, b, c = arn(x, context=y) # 3 parameters of sizes, (100, 1, 10), (100, 5, 10), (100, 3, 10)

    :param input_dim: the dimensionality of the input variable
    :type input_dim: int
    :param context_dim: the dimensionality of the context variable
    :type context_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n, input_dim) for p_n in param_dims
        when p_n > 1 and dimension (input_dim) when p_n == 1. The default is [1, 1], i.e. output two parameters
        of dimension (input_dim), which is useful for inverse autoregressive flow.
    :type param_dims: list[int]
    :param permutation: an optional permutation that is applied to the inputs and controls the order of the
        autoregressive factorization. in particular for the identity permutation the autoregressive structure
        is such that the Jacobian is upper triangular. By default this is chosen at random.
    :type permutation: torch.LongTensor
    :param skip_connections: Whether to add skip connections from the input to the output.
    :type skip_connections: bool
    :param nonlinearity: The nonlinearity to use in the feedforward network such as torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
    :type nonlinearity: torch.nn.module

    Reference:

    1. MADE: Masked Autoencoder for Distribution Estimation [arXiv:1502.03509]
    Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle

    2. Inference Networks for Sequential Monte Carlo in Graphical Models [arXiv:1602.06701]
    Brooks Paige, Frank Wood

    """

    def __init__(self, input_dim, context_dim, hidden_dims, param_dims=[1, 1], permutation=None, skip_connections=False, nonlinearity=nn.ReLU()):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if input_dim == 1:
            warnings.warn('ConditionalAutoRegressiveNN input_dim = 1. Consider using an affine transformation instead.')
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.param_dims = param_dims
        self.count_params = len(param_dims)
        self.output_multiplier = sum(param_dims)
        self.all_ones = (torch.tensor(param_dims) == 1).all().item()
        ends = torch.cumsum(torch.tensor(param_dims), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for (s, e) in zip(starts, ends)]
        for h in hidden_dims:
            if h < input_dim:
                raise ValueError('Hidden dimension must not be less than input dimension.')
        if permutation is None:
            P = torch.randperm(input_dim, device='cpu').to(torch.Tensor().device)
        else:
            P = permutation.type(dtype=torch.int64)
        self.register_buffer('permutation', P)
        (self.masks, self.mask_skip) = create_mask(input_dim=input_dim, context_dim=context_dim, hidden_dims=hidden_dims, permutation=self.permutation, output_dim_multiplier=self.output_multiplier)
        layers = [MaskedLinear(input_dim + context_dim, hidden_dims[0], self.masks[0])]
        for i in range(1, len(hidden_dims)):
            layers.append(MaskedLinear(hidden_dims[i - 1], hidden_dims[i], self.masks[i]))
        layers.append(MaskedLinear(hidden_dims[-1], input_dim * self.output_multiplier, self.masks[-1]))
        self.layers = nn.ModuleList(layers)
        if skip_connections:
            self.skip_layer = MaskedLinear(input_dim + context_dim, input_dim * self.output_multiplier, self.mask_skip, bias=False)
        else:
            self.skip_layer = None
        self.f = nonlinearity

    def get_permutation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the permutation applied to the inputs (by default this is chosen at random)\n        '
        return self.permutation

    def forward(self, x, context=None):
        if False:
            while True:
                i = 10
        if context is None:
            context = self.context
        context = context.expand(x.size()[:-1] + (context.size(-1),))
        x = torch.cat([context, x], dim=-1)
        return self._forward(x)

    def _forward(self, x):
        if False:
            while True:
                i = 10
        h = x
        for layer in self.layers[:-1]:
            h = self.f(layer(h))
        h = self.layers[-1](h)
        if self.skip_layer is not None:
            h = h + self.skip_layer(x)
        if self.output_multiplier == 1:
            return h
        else:
            h = h.reshape(list(x.size()[:-1]) + [self.output_multiplier, self.input_dim])
            if self.count_params == 1:
                return h
            elif self.all_ones:
                return torch.unbind(h, dim=-2)
            else:
                return tuple([h[..., s, :] for s in self.param_slices])

class AutoRegressiveNN(ConditionalAutoRegressiveNN):
    """
    An implementation of a MADE-like auto-regressive neural network.

    Example usage:

    >>> x = torch.randn(100, 10)
    >>> arn = AutoRegressiveNN(10, [50], param_dims=[1])
    >>> p = arn(x)  # 1 parameters of size (100, 10)
    >>> arn = AutoRegressiveNN(10, [50], param_dims=[1, 1])
    >>> m, s = arn(x) # 2 parameters of size (100, 10)
    >>> arn = AutoRegressiveNN(10, [50], param_dims=[1, 5, 3])
    >>> a, b, c = arn(x) # 3 parameters of sizes, (100, 1, 10), (100, 5, 10), (100, 3, 10)

    :param input_dim: the dimensionality of the input variable
    :type input_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n, input_dim) for p_n in param_dims
        when p_n > 1 and dimension (input_dim) when p_n == 1. The default is [1, 1], i.e. output two parameters
        of dimension (input_dim), which is useful for inverse autoregressive flow.
    :type param_dims: list[int]
    :param permutation: an optional permutation that is applied to the inputs and controls the order of the
        autoregressive factorization. in particular for the identity permutation the autoregressive structure
        is such that the Jacobian is upper triangular. By default this is chosen at random.
    :type permutation: torch.LongTensor
    :param skip_connections: Whether to add skip connections from the input to the output.
    :type skip_connections: bool
    :param nonlinearity: The nonlinearity to use in the feedforward network such as torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
    :type nonlinearity: torch.nn.module

    Reference:

    MADE: Masked Autoencoder for Distribution Estimation [arXiv:1502.03509]
    Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle

    """

    def __init__(self, input_dim, hidden_dims, param_dims=[1, 1], permutation=None, skip_connections=False, nonlinearity=nn.ReLU()):
        if False:
            while True:
                i = 10
        super(AutoRegressiveNN, self).__init__(input_dim, 0, hidden_dims, param_dims=param_dims, permutation=permutation, skip_connections=skip_connections, nonlinearity=nonlinearity)

    def forward(self, x):
        if False:
            return 10
        return self._forward(x)