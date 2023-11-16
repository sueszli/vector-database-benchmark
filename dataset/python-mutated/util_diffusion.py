import math
import os
import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from ..util import instantiate_from_config

def make_beta_schedule(schedule, n_timestep, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
    if False:
        while True:
            i = 10
    if schedule == 'linear':
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    elif schedule == 'cosine':
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule == 'sqrt_linear':
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == 'sqrt':
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if False:
        print('Hello World!')
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out

def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    if False:
        print('Hello World!')
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
    alpha_1 = (1 - alphas_prev) / (1 - alphas)
    alpha_2 = 1 - alphas / alphas_prev
    sigmas = eta * np.sqrt(alpha_1 * alpha_2)
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return (sigmas, alphas, alphas_prev)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    if False:
        while True:
            i = 10
    '\n    Create a beta schedule that discretizes the given alpha_t_bar function,\n    which defines the cumulative product of (1-beta) over time from t = [0,1].\n    :param num_diffusion_timesteps: the number of betas to produce.\n    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and\n                      produces the cumulative product of (1-beta) up to that\n                      part of the diffusion process.\n    :param max_beta: the maximum beta to use; use values lower than 1 to\n                     prevent singularities.\n    '
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def extract_into_tensor(a, t, x_shape):
    if False:
        while True:
            i = 10
    (b, *_) = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *(1,) * (len(x_shape) - 1))

def checkpoint(func, inputs, params, flag):
    if False:
        return 10
    '\n    Evaluate a function without caching intermediate activations, allowing for\n    reduced memory at the expense of extra compute in the backward pass.\n    :param func: the function to evaluate.\n    :param inputs: the argument sequence to pass to `func`.\n    :param params: a sequence of parameters `func` depends on but does not\n                   explicitly take as arguments.\n    :param flag: if False, disable gradient checkpointing.\n    '
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, length, *args):
        if False:
            return 10
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        if False:
            return 10
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params, output_grads, allow_unused=True)
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if False:
        i = 10
        return i + 15
    '\n    Create sinusoidal timestep embeddings.\n    :param timesteps: a 1-D Tensor of N indices, one per batch element.\n                      These may be fractional.\n    :param dim: the dimension of the output.\n    :param max_period: controls the minimum frequency of the embeddings.\n    :return: an [N x dim] Tensor of positional embeddings.\n    '
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

def zero_module(module):
    if False:
        for i in range(10):
            print('nop')
    '\n    Zero out the parameters of a module and return it.\n    '
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    if False:
        while True:
            i = 10
    '\n    Scale the parameters of a module and return it.\n    '
    for p in module.parameters():
        p.detach().mul_(scale)
    return module

def mean_flat(tensor):
    if False:
        i = 10
        return i + 15
    '\n    Take the mean over all non-batch dimensions.\n    '
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normalization(channels):
    if False:
        print('Hello World!')
    '\n    Make a standard normalization layer.\n    :param channels: number of input channels.\n    :return: an nn.Module for normalization.\n    '
    return GroupNorm32(32, channels)

class SiLU(nn.Module):

    def forward(self, x):
        if False:
            while True:
                i = 10
        return x * torch.sigmoid(x)

class GroupNorm32(nn.GroupNorm):

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Create a 1D, 2D, or 3D convolution module.\n    '
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')

def linear(*args, **kwargs):
    if False:
        return 10
    '\n    Create a linear module.\n    '
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    if False:
        print('Hello World!')
    '\n    Create a 1D, 2D, or 3D average pooling module.\n    '
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')

class HybridConditioner(nn.Module):

    def __init__(self, c_concat_config, c_crossattn_config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        if False:
            return 10
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}

def noise_like(shape, device, repeat=False):
    if False:
        return 10

    def repeat_noise():
        if False:
            return 10
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *(1,) * (len(shape) - 1))

    def noise():
        if False:
            for i in range(10):
                print('nop')
        return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()