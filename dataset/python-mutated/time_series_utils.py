"""
Time series distributional output classes and utilities.
"""
from typing import Callable, Dict, Optional, Tuple
import torch
from torch import nn
from torch.distributions import AffineTransform, Distribution, Independent, NegativeBinomial, Normal, StudentT, TransformedDistribution

class AffineTransformed(TransformedDistribution):

    def __init__(self, base_distribution: Distribution, loc=None, scale=None, event_dim=0):
        if False:
            for i in range(10):
                print('nop')
        self.scale = 1.0 if scale is None else scale
        self.loc = 0.0 if loc is None else loc
        super().__init__(base_distribution, [AffineTransform(loc=self.loc, scale=self.scale, event_dim=event_dim)])

    @property
    def mean(self):
        if False:
            while True:
                i = 10
        '\n        Returns the mean of the distribution.\n        '
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self):
        if False:
            return 10
        '\n        Returns the variance of the distribution.\n        '
        return self.base_dist.variance * self.scale ** 2

    @property
    def stddev(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the standard deviation of the distribution.\n        '
        return self.variance.sqrt()

class ParameterProjection(nn.Module):

    def __init__(self, in_features: int, args_dim: Dict[str, int], domain_map: Callable[..., Tuple[torch.Tensor]], **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.args_dim = args_dim
        self.proj = nn.ModuleList([nn.Linear(in_features, dim) for dim in args_dim.values()])
        self.domain_map = domain_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        if False:
            print('Hello World!')
        params_unbounded = [proj(x) for proj in self.proj]
        return self.domain_map(*params_unbounded)

class LambdaLayer(nn.Module):

    def __init__(self, function):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.function = function

    def forward(self, x, *args):
        if False:
            return 10
        return self.function(x, *args)

class DistributionOutput:
    distribution_class: type
    in_features: int
    args_dim: Dict[str, int]

    def __init__(self, dim: int=1) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.dim = dim
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}

    def _base_distribution(self, distr_args):
        if False:
            print('Hello World!')
        if self.dim == 1:
            return self.distribution_class(*distr_args)
        else:
            return Independent(self.distribution_class(*distr_args), 1)

    def distribution(self, distr_args, loc: Optional[torch.Tensor]=None, scale: Optional[torch.Tensor]=None) -> Distribution:
        if False:
            print('Hello World!')
        distr = self._base_distribution(distr_args)
        if loc is None and scale is None:
            return distr
        else:
            return AffineTransformed(distr, loc=loc, scale=scale, event_dim=self.event_dim)

    @property
    def event_shape(self) -> Tuple:
        if False:
            while True:
                i = 10
        '\n        Shape of each individual event contemplated by the distributions that this object constructs.\n        '
        return () if self.dim == 1 else (self.dim,)

    @property
    def event_dim(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Number of event dimensions, i.e., length of the `event_shape` tuple, of the distributions that this object\n        constructs.\n        '
        return len(self.event_shape)

    @property
    def value_in_support(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        '\n        A float that will have a valid numeric value when computing the log-loss of the corresponding distribution. By\n        default 0.0. This value will be used when padding data series.\n        '
        return 0.0

    def get_parameter_projection(self, in_features: int) -> nn.Module:
        if False:
            while True:
                i = 10
        '\n        Return the parameter projection layer that maps the input to the appropriate parameters of the distribution.\n        '
        return ParameterProjection(in_features=in_features, args_dim=self.args_dim, domain_map=LambdaLayer(self.domain_map))

    def domain_map(self, *args: torch.Tensor):
        if False:
            print('Hello World!')
        '\n        Converts arguments to the right shape and domain. The domain depends on the type of distribution, while the\n        correct shape is obtained by reshaping the trailing axis in such a way that the returned tensors define a\n        distribution of the right event_shape.\n        '
        raise NotImplementedError()

    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:\n        https://twitter.com/jon_barron/status/1387167648669048833\n        '
        return (x + torch.sqrt(torch.square(x) + 4.0)) / 2.0

class StudentTOutput(DistributionOutput):
    """
    Student-T distribution output class.
    """
    args_dim: Dict[str, int] = {'df': 1, 'loc': 1, 'scale': 1}
    distribution_class: type = StudentT

    @classmethod
    def domain_map(cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        scale = cls.squareplus(scale).clamp_min(torch.finfo(scale.dtype).eps)
        df = 2.0 + cls.squareplus(df)
        return (df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1))

class NormalOutput(DistributionOutput):
    """
    Normal distribution output class.
    """
    args_dim: Dict[str, int] = {'loc': 1, 'scale': 1}
    distribution_class: type = Normal

    @classmethod
    def domain_map(cls, loc: torch.Tensor, scale: torch.Tensor):
        if False:
            print('Hello World!')
        scale = cls.squareplus(scale).clamp_min(torch.finfo(scale.dtype).eps)
        return (loc.squeeze(-1), scale.squeeze(-1))

class NegativeBinomialOutput(DistributionOutput):
    """
    Negative Binomial distribution output class.
    """
    args_dim: Dict[str, int] = {'total_count': 1, 'logits': 1}
    distribution_class: type = NegativeBinomial

    @classmethod
    def domain_map(cls, total_count: torch.Tensor, logits: torch.Tensor):
        if False:
            while True:
                i = 10
        total_count = cls.squareplus(total_count)
        return (total_count.squeeze(-1), logits.squeeze(-1))

    def _base_distribution(self, distr_args) -> Distribution:
        if False:
            print('Hello World!')
        (total_count, logits) = distr_args
        if self.dim == 1:
            return self.distribution_class(total_count=total_count, logits=logits)
        else:
            return Independent(self.distribution_class(total_count=total_count, logits=logits), 1)

    def distribution(self, distr_args, loc: Optional[torch.Tensor]=None, scale: Optional[torch.Tensor]=None) -> Distribution:
        if False:
            for i in range(10):
                print('nop')
        (total_count, logits) = distr_args
        if scale is not None:
            logits += scale.log()
        return self._base_distribution((total_count, logits))