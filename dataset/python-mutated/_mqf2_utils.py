"""Classes and functions for the MQF2 metric."""
from typing import List, Optional, Tuple
from cpflows.flows import DeepConvexFlow, SequentialFlow
import torch
from torch.distributions import AffineTransform, Distribution, Normal, TransformedDistribution
import torch.nn.functional as F

class DeepConvexNet(DeepConvexFlow):
    """
    Class that takes a partially input convex neural network (picnn)
    as input and equips it with functions of logdet
    computation (both estimation and exact computation)
    This class is based on DeepConvexFlow of the CP-Flow
    repo (https://github.com/CW-Huang/CP-Flow)
    For details of the logdet estimator, see
    ``Convex potential flows: Universal probability distributions
    with optimal transport and convex optimization``
    Parameters
    ----------
    picnn
        A partially input convex neural network (picnn)
    dim
        Dimension of the input
    is_energy_score
        Indicates if energy score is used as the objective function
        If yes, the network is not required to be strictly convex,
        so we can just use the picnn
        otherwise, a quadratic term is added to the output of picnn
        to render it strictly convex
    m1
        Dimension of the Krylov subspace of the Lanczos tridiagonalization
        used in approximating H of logdet(H)
    m2
        Iteration number of the conjugate gradient algorithm
        used to approximate logdet(H)
    rtol
        relative tolerance of the conjugate gradient algorithm
    atol
        absolute tolerance of the conjugate gradient algorithm
    """

    def __init__(self, picnn: torch.nn.Module, dim: int, is_energy_score: bool=False, estimate_logdet: bool=False, m1: int=10, m2: Optional[int]=None, rtol: float=0.0, atol: float=0.001) -> None:
        if False:
            while True:
                i = 10
        super().__init__(picnn, dim, m1=m1, m2=m2, rtol=rtol, atol=atol)
        self.picnn = self.icnn
        self.is_energy_score = is_energy_score
        self.estimate_logdet = estimate_logdet

    def get_potential(self, x: torch.Tensor, context: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            print('Hello World!')
        n = x.size(0)
        output = self.picnn(x, context)
        if self.is_energy_score:
            return output
        else:
            return F.softplus(self.w1) * output + F.softplus(self.w0) * (x.view(n, -1) ** 2).sum(1, keepdim=True) / 2

    def forward_transform(self, x: torch.Tensor, logdet: Optional[torch.Tensor]=0, context: Optional[torch.Tensor]=None, extra: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            print('Hello World!')
        if self.estimate_logdet:
            return self.forward_transform_stochastic(x, logdet, context=context, extra=extra)
        else:
            return self.forward_transform_bruteforce(x, logdet, context=context)

class SequentialNet(SequentialFlow):
    """
    Class that combines a list of DeepConvexNet and ActNorm
    layers and provides energy score computation
    This class is based on SequentialFlow of the CP-Flow repo
    (https://github.com/CW-Huang/CP-Flow)
    Parameters
    ----------
    networks
        list of DeepConvexNet and/or ActNorm instances
    """

    def __init__(self, networks: List[torch.nn.Module]) -> None:
        if False:
            return 10
        super().__init__(networks)
        self.networks = self.flows

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            print('Hello World!')
        for network in self.networks:
            if isinstance(network, DeepConvexNet):
                x = network.forward(x, context=context)
            else:
                x = network.forward(x)
        return x

    def es_sample(self, hidden_state: torch.Tensor, dimension: int) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Auxiliary function for energy score computation\n        Drawing samples conditioned on the hidden state\n        Parameters\n        ----------\n        hidden_state\n            hidden_state which the samples conditioned\n            on (num_samples, hidden_size)\n        dimension\n            dimension of the input\n        Returns\n        -------\n        samples\n            samples drawn (num_samples, dimension)\n        '
        num_samples = hidden_state.shape[0]
        zero = torch.tensor(0, dtype=hidden_state.dtype, device=hidden_state.device)
        one = torch.ones_like(zero)
        standard_normal = Normal(zero, one)
        samples = self.forward(standard_normal.sample([num_samples * dimension]).view(num_samples, dimension), context=hidden_state)
        return samples

    def energy_score(self, z: torch.Tensor, hidden_state: torch.Tensor, es_num_samples: int=50, beta: float=1.0) -> torch.Tensor:
        if False:
            print('Hello World!')
        "\n        Computes the (approximated) energy score sum_i ES(g,z_i),\n        where ES(g,z_i) =\n        -1/(2*es_num_samples^2) * sum_{w,w'} ||w-w'||_2^beta\n        + 1/es_num_samples * sum_{w''} ||w''-z_i||_2^beta,\n        w's are samples drawn from the\n        quantile function g(., h_i) (gradient of picnn),\n        h_i is the hidden state associated with z_i,\n        and es_num_samples is the number of samples drawn\n        for each of w, w', w'' in energy score approximation\n        Parameters\n        ----------\n        z\n            Observations (numel_batch, dimension)\n        hidden_state\n            Hidden state (numel_batch, hidden_size)\n        es_num_samples\n            Number of samples drawn for each of w, w', w''\n            in energy score approximation\n        beta\n            Hyperparameter of the energy score, see the formula above\n        Returns\n        -------\n        loss\n            energy score (numel_batch)\n        "
        (numel_batch, dimension) = (z.shape[0], z.shape[1])
        hidden_state_repeat = hidden_state.repeat_interleave(repeats=es_num_samples, dim=0)
        w = self.es_sample(hidden_state_repeat, dimension)
        w_prime = self.es_sample(hidden_state_repeat, dimension)
        first_term = torch.norm(w.view(numel_batch, 1, es_num_samples, dimension) - w_prime.view(numel_batch, es_num_samples, 1, dimension), dim=-1) ** beta
        mean_first_term = torch.mean(first_term.view(numel_batch, -1), dim=-1)
        del w, w_prime
        z_repeat = z.repeat_interleave(repeats=es_num_samples, dim=0)
        w_bar = self.es_sample(hidden_state_repeat, dimension)
        second_term = torch.norm(w_bar.view(numel_batch, es_num_samples, dimension) - z_repeat.view(numel_batch, es_num_samples, dimension), dim=-1) ** beta
        mean_second_term = torch.mean(second_term.view(numel_batch, -1), dim=-1)
        loss = -0.5 * mean_first_term + mean_second_term
        return loss

class MQF2Distribution(Distribution):
    """
    Distribution class for the model MQF2 proposed in the paper
    ``Multivariate Quantile Function Forecaster``
    by Kan, Aubet, Januschowski, Park, Benidis, Ruthotto, Gasthaus
    Parameters
    ----------
    picnn
        A SequentialNet instance of a
        partially input convex neural network (picnn)
    hidden_state
        hidden_state obtained by unrolling the RNN encoder
        shape = (batch_size, context_length, hidden_size) in training
        shape = (batch_size, hidden_size) in inference
    prediction_length
        Length of the prediction horizon
    is_energy_score
        If True, use energy score as objective function
        otherwise use maximum likelihood as
        objective function (normalizing flows)
    es_num_samples
        Number of samples drawn to approximate the energy score
    beta
        Hyperparameter of the energy score (power of the two terms)
    threshold_input
        Clamping threshold of the (scaled) input when maximum
        likelihood is used as objective function
        this is used to make the forecaster more robust
        to outliers in training samples
    validate_args
        Sets whether validation is enabled or disabled
        For more details, refer to the descriptions in
        torch.distributions.distribution.Distribution
    """

    def __init__(self, picnn: torch.nn.Module, hidden_state: torch.Tensor, prediction_length: int, is_energy_score: bool=True, es_num_samples: int=50, beta: float=1.0, threshold_input: float=100.0, validate_args: bool=False) -> None:
        if False:
            while True:
                i = 10
        self.picnn = picnn
        self.hidden_state = hidden_state
        self.prediction_length = prediction_length
        self.is_energy_score = is_energy_score
        self.es_num_samples = es_num_samples
        self.beta = beta
        self.threshold_input = threshold_input
        super().__init__(batch_shape=self.batch_shape, validate_args=validate_args)
        self.context_length = self.hidden_state.shape[-2] if len(self.hidden_state.shape) > 2 else 1
        self.numel_batch = self.get_numel(self.batch_shape)
        mu = torch.tensor(0, dtype=hidden_state.dtype, device=hidden_state.device)
        sigma = torch.ones_like(mu)
        self.standard_normal = Normal(mu, sigma)

    def stack_sliding_view(self, z: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Auxiliary function for loss computation\n        Unfolds the observations by sliding a window of size prediction_length\n        over the observations z\n        Then, reshapes the observations into a 2-dimensional tensor for\n        further computation\n        Parameters\n        ----------\n        z\n            A batch of time series with shape\n            (batch_size, context_length + prediction_length - 1)\n        Returns\n        -------\n        Tensor\n            Unfolded time series with shape\n            (batch_size * context_length, prediction_length)\n        '
        z = z.unfold(dimension=-1, size=self.prediction_length, step=1)
        z = z.reshape(-1, z.shape[-1])
        return z

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        if self.is_energy_score:
            return self.energy_score(z)
        else:
            return -self.log_prob(z)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Computes the log likelihood  log(g(z)) + logdet(dg(z)/dz),\n        where g is the gradient of the picnn\n        Parameters\n        ----------\n        z\n            A batch of time series with shape\n            (batch_size, context_length + prediciton_length - 1)\n        Returns\n        -------\n        loss\n            Tesnor of shape (batch_size * context_length,)\n        '
        z = torch.clamp(z, min=-self.threshold_input, max=self.threshold_input)
        z = self.stack_sliding_view(z)
        loss = self.picnn.logp(z, self.hidden_state.reshape(-1, self.hidden_state.shape[-1]))
        return loss

    def energy_score(self, z: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        "\n        Computes the (approximated) energy score sum_i ES(g,z_i),\n        where ES(g,z_i) =\n        -1/(2*es_num_samples^2) * sum_{w,w'} ||w-w'||_2^beta\n        + 1/es_num_samples * sum_{w''} ||w''-z_i||_2^beta,\n        w's are samples drawn from the\n        quantile function g(., h_i) (gradient of picnn),\n        h_i is the hidden state associated with z_i,\n        and es_num_samples is the number of samples drawn\n        for each of w, w', w'' in energy score approximation\n        Parameters\n        ----------\n        z\n            A batch of time series with shape\n            (batch_size, context_length + prediction_length - 1)\n        Returns\n        -------\n        loss\n            Tensor of shape (batch_size * context_length,)\n        "
        es_num_samples = self.es_num_samples
        beta = self.beta
        z = self.stack_sliding_view(z)
        reshaped_hidden_state = self.hidden_state.reshape(-1, self.hidden_state.shape[-1])
        loss = self.picnn.energy_score(z, reshaped_hidden_state, es_num_samples=es_num_samples, beta=beta)
        return loss

    def rsample(self, sample_shape: torch.Size=torch.Size()) -> torch.Tensor:
        if False:
            return 10
        '\n        Generates the sample paths\n        Parameters\n        ----------\n        sample_shape\n            Shape of the samples\n        Returns\n        -------\n        sample_paths\n            Tesnor of shape (batch_size, *sample_shape, prediction_length)\n        '
        numel_batch = self.numel_batch
        prediction_length = self.prediction_length
        num_samples_per_batch = MQF2Distribution.get_numel(sample_shape)
        num_samples = num_samples_per_batch * numel_batch
        hidden_state_repeat = self.hidden_state.repeat_interleave(repeats=num_samples_per_batch, dim=0)
        alpha = torch.rand((num_samples, prediction_length), dtype=self.hidden_state.dtype, device=self.hidden_state.device, layout=self.hidden_state.layout).clamp(min=0.0001, max=1 - 0.0001)
        samples = self.quantile(alpha, hidden_state_repeat).reshape((numel_batch,) + sample_shape + (prediction_length,)).transpose(0, 1)
        return samples

    def quantile(self, alpha: torch.Tensor, hidden_state: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates the predicted paths associated with the quantile levels alpha\n        Parameters\n        ----------\n        alpha\n            quantile levels,\n            shape = (batch_shape, prediction_length)\n        hidden_state\n            hidden_state, shape = (batch_shape, hidden_size)\n        Returns\n        -------\n        results\n            predicted paths of shape = (batch_shape, prediction_length)\n        '
        if hidden_state is None:
            hidden_state = self.hidden_state
        normal_quantile = self.standard_normal.icdf(alpha)
        if self.is_energy_score:
            result = self.picnn(normal_quantile, context=hidden_state)
        else:
            result = self.picnn.reverse(normal_quantile, context=hidden_state)
        return result

    @staticmethod
    def get_numel(tensor_shape: torch.Size) -> int:
        if False:
            while True:
                i = 10
        return torch.prod(torch.tensor(tensor_shape)).item()

    @property
    def batch_shape(self) -> torch.Size:
        if False:
            while True:
                i = 10
        return self.hidden_state.shape[:-1]

    @property
    def event_shape(self) -> Tuple:
        if False:
            while True:
                i = 10
        return (self.prediction_length,)

    @property
    def event_dim(self) -> int:
        if False:
            i = 10
            return i + 15
        return 1

class TransformedMQF2Distribution(TransformedDistribution):

    def __init__(self, base_distribution: MQF2Distribution, transforms: List[AffineTransform], validate_args: bool=False) -> None:
        if False:
            return 10
        super().__init__(base_distribution, transforms, validate_args=validate_args)

    def scale_input(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            while True:
                i = 10
        scale = torch.tensor(1.0, device=y.device)
        for t in self.transforms[::-1]:
            y = t._inverse(y)
        for t in self.transforms:
            if isinstance(t, AffineTransform):
                scale = scale * t.scale
            else:
                scale = t(scale)
        return (y, scale)

    def repeat_scale(self, scale: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        return scale.squeeze(-1).repeat_interleave(self.base_dist.context_length, 0)

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        prediction_length = self.base_dist.prediction_length
        (z, scale) = self.scale_input(y)
        p = self.base_dist.log_prob(z)
        repeated_scale = self.repeat_scale(scale)
        return p - prediction_length * torch.log(repeated_scale)

    def energy_score(self, y: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        beta = self.base_dist.beta
        (z, scale) = self.scale_input(y)
        loss = self.base_dist.energy_score(z)
        repeated_scale = self.repeat_scale(scale)
        return loss * repeated_scale ** beta

    def quantile(self, alpha: torch.Tensor, hidden_state: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            while True:
                i = 10
        result = self.base_dist.quantile(alpha, hidden_state=hidden_state)
        result = result.reshape(self.base_dist.hidden_state.size(0), -1, self.base_dist.prediction_length).transpose(0, 1)
        for transform in self.transforms:
            result = transform(result)
        return result.transpose(0, 1).reshape_as(alpha)