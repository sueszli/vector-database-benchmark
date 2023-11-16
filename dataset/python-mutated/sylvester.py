import torch
import torch.nn as nn
from .. import constraints
from ..torch_transform import TransformModule
from ..util import copy_docs_from
from .householder import Householder

@copy_docs_from(TransformModule)
class Sylvester(Householder):
    """
    An implementation of the Sylvester bijective transform of the Householder
    variety (Van den Berg Et Al., 2018),

        :math:`\\mathbf{y} = \\mathbf{x} + QR\\tanh(SQ^T\\mathbf{x}+\\mathbf{b})`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs,
    :math:`R,S\\sim D\\times D` are upper triangular matrices for input dimension
    :math:`D`, :math:`Q\\sim D\\times D` is an orthogonal matrix, and
    :math:`\\mathbf{b}\\sim D` is learnable bias term.

    The Sylvester transform is a generalization of
    :class:`~pyro.distributions.transforms.Planar`. In the Householder type of the
    Sylvester transform, the orthogonality of :math:`Q` is enforced by representing
    it as the product of Householder transformations.

    Together with :class:`~pyro.distributions.TransformedDistribution` it provides a
    way to create richer variational approximations.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> transform = Sylvester(10, count_transforms=4)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    The inverse of this transform does not possess an analytical solution and is
    left unimplemented. However, the inverse is cached when the forward operation is
    called during sampling, and so samples drawn using the Sylvester transform can
    be scored.

    References:

    [1] Rianne van den Berg, Leonard Hasenclever, Jakub M. Tomczak, Max Welling.
    Sylvester Normalizing Flows for Variational Inference. UAI 2018.

    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim, count_transforms=1):
        if False:
            while True:
                i = 10
        super().__init__(input_dim, count_transforms)
        self.R_dense = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.S_dense = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.R_diag = nn.Parameter(torch.Tensor(input_dim))
        self.S_diag = nn.Parameter(torch.Tensor(input_dim))
        self.b = nn.Parameter(torch.Tensor(input_dim))
        triangular_mask = torch.triu(torch.ones(input_dim, input_dim), diagonal=1)
        self.register_buffer('triangular_mask', triangular_mask)
        self._cached_logDetJ = None
        self.tanh = nn.Tanh()
        self.reset_parameters2()

    def dtanh_dx(self, x):
        if False:
            return 10
        return 1.0 - self.tanh(x).pow(2)

    def R(self):
        if False:
            return 10
        return self.R_dense * self.triangular_mask + torch.diag(self.tanh(self.R_diag))

    def S(self):
        if False:
            while True:
                i = 10
        return self.S_dense * self.triangular_mask + torch.diag(self.tanh(self.S_diag))

    def Q(self, x):
        if False:
            print('Hello World!')
        u = self.u()
        partial_Q = torch.eye(self.input_dim, dtype=x.dtype, layout=x.layout, device=x.device) - 2.0 * torch.ger(u[0], u[0])
        for idx in range(1, self.u_unnormed.size(-2)):
            partial_Q = torch.matmul(partial_Q, torch.eye(self.input_dim) - 2.0 * torch.ger(u[idx], u[idx]))
        return partial_Q

    def reset_parameters2(self):
        if False:
            return 10
        for v in [self.b, self.R_diag, self.S_diag, self.R_dense, self.S_dense]:
            v.data.uniform_(-0.01, 0.01)

    def _call(self, x):
        if False:
            print('Hello World!')
        '\n        :param x: the input into the bijection\n        :type x: torch.Tensor\n\n        Invokes the bijection x=>y; in the prototypical context of a\n        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from\n        the base distribution (or the output of a previous transform)\n        '
        Q = self.Q(x)
        R = self.R()
        S = self.S()
        A = torch.matmul(Q, R)
        B = torch.matmul(S, Q.t())
        preactivation = torch.matmul(x, B) + self.b
        y = x + torch.matmul(self.tanh(preactivation), A)
        self._cached_logDetJ = torch.log1p(self.dtanh_dx(preactivation) * R.diagonal() * S.diagonal() + 1e-08).sum(-1)
        return y

    def _inverse(self, y):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param y: the output of the bijection\n        :type y: torch.Tensor\n        Inverts y => x. As noted above, this implementation is incapable of\n        inverting arbitrary values `y`; rather it assumes `y` is the result of a\n        previously computed application of the bijector to some `x` (which was\n        cached on the forward call)\n        '
        raise KeyError("Sylvester object expected to find key in intermediates cache but didn't")

    def log_abs_det_jacobian(self, x, y):
        if False:
            i = 10
            return i + 15
        '\n        Calculates the elementwise determinant of the log Jacobian\n        '
        (x_old, y_old) = self._cached_x_y
        if x is not x_old or y is not y_old:
            self(x)
        return self._cached_logDetJ

def sylvester(input_dim, count_transforms=None):
    if False:
        print('Hello World!')
    '\n    A helper function to create a :class:`~pyro.distributions.transforms.Sylvester`\n    object for consistency with other helpers.\n\n    :param input_dim: Dimension of input variable\n    :type input_dim: int\n    :param count_transforms: Number of Sylvester operations to apply. Defaults to\n        input_dim // 2 + 1. :type count_transforms: int\n\n    '
    if count_transforms is None:
        count_transforms = input_dim // 2 + 1
    return Sylvester(input_dim, count_transforms=count_transforms)