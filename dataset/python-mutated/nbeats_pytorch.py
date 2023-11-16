import pickle
import random
from time import time
from typing import Union
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
from torch.optim import Optimizer
from .utils import PYTORCH_REGRESSION_LOSS_MAP
from pytorch_lightning import seed_everything

class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self, past_seq_len, future_seq_len, stack_types=(GENERIC_BLOCK, GENERIC_BLOCK), nb_blocks_per_stack=3, thetas_dim=(4, 8), share_weights_in_stack=False, hidden_layer_units=256, nb_harmonics=None, seed=None):
        if False:
            return 10
        super(NBeatsNet, self).__init__()
        seed_everything(seed, workers=True)
        self.future_seq_len = future_seq_len
        self.past_seq_len = past_seq_len
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = torch.nn.ModuleList()
        self.thetas_dim = thetas_dim
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))

    def create_stack(self, stack_id):
        if False:
            i = 10
            return i + 15
        stack_type = self.stack_types[stack_id]
        blocks = torch.nn.ModuleList()
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet._select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]
            else:
                block = block_init(self.hidden_layer_units, self.thetas_dim[stack_id], self.past_seq_len, self.future_seq_len, self.nb_harmonics)
            blocks.append(block)
        return blocks

    @staticmethod
    def _select_block(block_type):
        if False:
            i = 10
            return i + 15
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast):
        if False:
            print('Hello World!')
        backcast = backcast[..., 0]
        forecast = 0
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                (b, f) = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                forecast = forecast + f
        return torch.unsqueeze(forecast, 2)

def seasonality_model(thetas, t):
    if False:
        while True:
            i = 10
    p = thetas.size()[-1]
    from bigdl.nano.utils.common import invalidInputError
    invalidInputError(p <= thetas.shape[1], 'thetas_dim is too big.')
    (p1, p2) = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S)

def trend_model(thetas, t):
    if False:
        while True:
            i = 10
    p = thetas.size()[-1]
    from bigdl.nano.utils.common import invalidInputError
    invalidInputError(p <= 4, 'thetas_dim is too big.')
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T)

def linear_space(past_seq_len, future_seq_len):
    if False:
        for i in range(10):
            print('nop')
    ls = np.arange(-past_seq_len, future_seq_len, 1) / future_seq_len
    b_ls = np.abs(np.flip(ls[:past_seq_len]))
    f_ls = ls[past_seq_len:]
    return (b_ls, f_ls)

class Block(nn.Module):

    def __init__(self, units, thetas_dim, past_seq_len=10, future_seq_len=5, share_thetas=False, nb_harmonics=None):
        if False:
            print('Hello World!')
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.past_seq_len = past_seq_len
        self.future_seq_len = future_seq_len
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(past_seq_len, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        (self.backcast_linspace, self.forecast_linspace) = linear_space(past_seq_len, future_seq_len)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        if False:
            print('Hello World!')
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        if False:
            return 10
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, past_seq_len={self.past_seq_len}, future_seq_len={self.future_seq_len}, share_thetas={self.share_thetas}) at @{id(self)}'

class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, past_seq_len=10, future_seq_len=5, nb_harmonics=None):
        if False:
            for i in range(10):
                print('nop')
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, past_seq_len, future_seq_len, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, future_seq_len, past_seq_len, future_seq_len, share_thetas=True)

    def forward(self, x):
        if False:
            print('Hello World!')
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace)
        return (backcast, forecast)

class TrendBlock(Block):

    def __init__(self, units, thetas_dim, past_seq_len=10, future_seq_len=5, nb_harmonics=None):
        if False:
            return 10
        super(TrendBlock, self).__init__(units, thetas_dim, past_seq_len, future_seq_len, share_thetas=True)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace)
        return (backcast, forecast)

class GenericBlock(Block):

    def __init__(self, units, thetas_dim, past_seq_len=10, future_seq_len=5, nb_harmonics=None):
        if False:
            i = 10
            return i + 15
        super(GenericBlock, self).__init__(units, thetas_dim, past_seq_len, future_seq_len)
        self.backcast_fc = nn.Linear(thetas_dim, past_seq_len)
        self.forecast_fc = nn.Linear(thetas_dim, future_seq_len)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = super(GenericBlock, self).forward(x)
        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)
        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)
        return (backcast, forecast)

def model_creator(config):
    if False:
        i = 10
        return i + 15
    return NBeatsNet(past_seq_len=config['past_seq_len'], future_seq_len=config['future_seq_len'], stack_types=config.get('stack_types', ('generic', 'generic')), nb_blocks_per_stack=config.get('nb_blocks_per_stack', 3), thetas_dim=config.get('thetas_dim', (4, 8)), share_weights_in_stack=config.get('share_weights_in_stack', False), hidden_layer_units=config.get('hidden_layer_units', 256), nb_harmonics=config.get('nb_harmonics', None))

def optimizer_creator(model, config):
    if False:
        while True:
            i = 10
    return getattr(torch.optim, config.get('optim', 'Adam'))(model.parameters(), lr=config.get('lr', 0.001))

def loss_creator(config):
    if False:
        while True:
            i = 10
    loss_name = config.get('loss', 'mse')
    if loss_name in PYTORCH_REGRESSION_LOSS_MAP:
        loss_name = PYTORCH_REGRESSION_LOSS_MAP[loss_name]
    else:
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError(False, f"Got '{loss_name}' for loss name, where 'mse', 'mae' or 'huber_loss' is expected")
    return getattr(torch.nn, loss_name)()
try:
    from bigdl.orca.automl.model.base_pytorch_model import PytorchBaseModel

    class NBeatsPytorch(PytorchBaseModel):

        def __init__(self, check_optional_config=False):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(model_creator=model_creator, optimizer_creator=optimizer_creator, loss_creator=loss_creator, check_optional_config=check_optional_config)

        def _get_required_parameters(self):
            if False:
                i = 10
                return i + 15
            return {'past_seq_len', 'future_seq_len'}

        def _get_optional_parameters(self):
            if False:
                print('Hello World!')
            return {'stack_types', 'nb_blocks_per_stack', 'thetas_dim', 'share_weights_in_stack', 'hidden_layer_units', 'nb_harmonics'} | super()._get_optional_parameters()
except ImportError:
    pass