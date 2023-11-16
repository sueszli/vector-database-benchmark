from pyro.contrib.gp.parameterized import Parameterized

class Likelihood(Parameterized):
    """
    Base class for likelihoods used in Gaussian Process.

    Every inherited class should implement a forward pass which
    takes an input :math:`f` and returns a sample :math:`y`.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, f_loc, f_var, y=None):
        if False:
            while True:
                i = 10
        '\n        Samples :math:`y` given :math:`f_{loc}`, :math:`f_{var}`.\n\n        :param torch.Tensor f_loc: Mean of latent function output.\n        :param torch.Tensor f_var: Variance of latent function output.\n        :param torch.Tensor y: Training output tensor.\n        :returns: a tensor sampled from likelihood\n        :rtype: torch.Tensor\n        '
        raise NotImplementedError