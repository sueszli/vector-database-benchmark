import torch
from torch.ao.quantization.observer import ObserverBase

class ModelReportObserver(ObserverBase):
    """This observer is used to record additional information regarding keeping track
    of S = average_batch_activation_range/epoch_activation_range.

    The purpose of this information is to prepare a report to present to users on whether
    Dynamic or Static Quantization is more appropriate for their model given the general
    distributions of their data.

    Args:
        ch_axis (int, optional): The channel axis for which the range and outlier stats are computed
            Default: 1
        comp_percentile (float, optional): The percentile to compare against 100 percentile to find outliers
            Should be between 0 and 1 exclusive
            Default: 0.9

    * :attr:`num_batches_tracked` specifies number of batches passed through the observer

    * :attr:`average_batch_activation_range` defines average across the ranges of each batch passed through

    * :attr:`epoch_activation_min` defines the minimum value passed through the observer

    * :attr:`epoch_activation_max` defines the maximum value passed through the observer

    * :attr:`ch_axis` defines the channel being used to compute per channel min max stats

    * :attr:`min_val` defines the per channel minimum values passed through

    * :attr:`max_val` defines the per channel maximum values passed through

    * :attr:`comp_percentile` defines comparison percentile to find outliers

    * :attr:`average_percentile_ratio` defines the per channel average percentile ratios

    * :attr:`percentile_batches_tracked` defines the number of percentile batches tracked for each channel

    * :attr:`constant_channels` defines the number of batches that aren't constant channels per channel

    Note: this tool is meant for FX Graph Mode Quantization
    """
    epoch_activation_min: torch.Tensor
    epoch_activation_max: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor
    comp_percentile: torch.Tensor
    average_percentile_ratio: torch.Tensor
    percentile_batches_tracked: torch.Tensor
    constant_channels: torch.Tensor

    def __init__(self, ch_axis: int=1, comp_percentile: float=0.9):
        if False:
            print('Hello World!')
        super().__init__(torch.qint8)
        self.num_batches_tracked = 0
        self.average_batch_activation_range: torch.Tensor = torch.tensor(float(0))
        self.register_buffer('epoch_activation_min', torch.tensor(float('inf')))
        self.register_buffer('epoch_activation_max', torch.tensor(float('-inf')))
        self.ch_axis: int = ch_axis
        self.register_buffer('min_val', torch.tensor([]))
        self.register_buffer('max_val', torch.tensor([]))
        self.register_buffer('comp_percentile', torch.tensor([comp_percentile]))
        self.register_buffer('average_percentile_ratio', torch.tensor([]))
        self.register_buffer('percentile_batches_tracked', torch.tensor([]))
        self.register_buffer('constant_channels', torch.tensor([]))

    def forward(self, x):
        if False:
            return 10
        x_copy = x.detach()
        x_copy = x_copy.to(self.epoch_activation_min.dtype)
        x_copy = self._calculate_range_stats(x_copy)
        x_copy = self._calculate_min_max_stats(x_copy)
        x_copy = self._calculate_percentile_stats(x_copy)
        return x

    def _calculate_range_stats(self, x_copy):
        if False:
            while True:
                i = 10
        'Calculates and stores range stats with forward values.\n\n        Args\n            x_copy: A copy of the forward data\n\n        Returns the passed in x_copy\n        '
        (min_val_cur, max_val_cur) = torch.aminmax(x_copy)
        epoch_min_val = torch.min(self.epoch_activation_min, min_val_cur)
        epoch_max_val = torch.max(self.epoch_activation_max, max_val_cur)
        self.epoch_activation_min.copy_(epoch_min_val)
        self.epoch_activation_max.copy_(epoch_max_val)
        current_batch_range = max_val_cur - min_val_cur
        new_range = (self.average_batch_activation_range * self.num_batches_tracked + current_batch_range) / (self.num_batches_tracked + 1)
        self.average_batch_activation_range = new_range
        self.num_batches_tracked += 1
        return x_copy

    def _calculate_min_max_stats(self, x_copy):
        if False:
            i = 10
            return i + 15
        'Calculates and stores the per_channel min, max stats with forward values.\n        Does calculation based on channel axis: self.ch_axis\n\n        Args\n            x_copy: A copy of the forward data\n\n        Returns the passed in x_copy\n        '
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x_copy.size()
        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x_copy.permute(new_axis_list)
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            (min_val, max_val) = torch.aminmax(y, dim=1)
        else:
            (min_val_cur, max_val_cur) = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_copy

    def _calculate_percentile_stats(self, x_copy):
        if False:
            i = 10
            return i + 15
        'Calculates and stores the per_channel percentile stats with forward values.\n        Does calculation based on channel axis: self.ch_axis\n\n        Args\n            x_copy: A copy of the forward data\n\n        Returns the passed in x_copy\n        '
        x_dim = x_copy.size()
        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x_copy.permute(new_axis_list)
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        y = y.to(dtype=self.min_val.dtype, device='cpu')
        quantiles_list = [0, self.comp_percentile, 1.0]
        quantiles_to_find = torch.tensor(quantiles_list, dtype=self.min_val.dtype)
        desired_quantiles = torch.quantile(y, quantiles_to_find, dim=self.ch_axis, interpolation='lower')
        zero_quantile = desired_quantiles[0]
        comp_quantile = desired_quantiles[1]
        hundreth_quartile = desired_quantiles[2]
        any_non_zero_quantile_value: torch.Tensor = (comp_quantile != torch.tensor([0])) | (hundreth_quartile != torch.tensor([0]))
        any_non_zero_quantile_value = any_non_zero_quantile_value.int()
        any_constant_channels: torch.Tensor = hundreth_quartile - zero_quantile == torch.tensor([0])
        any_constant_channels = any_constant_channels.int()
        quantile_ratios = hundreth_quartile / comp_quantile
        quantile_ratios = torch.nan_to_num(quantile_ratios)
        ratio_if_not_zero = any_non_zero_quantile_value * quantile_ratios
        if self.percentile_batches_tracked.shape[0] == 0 or self.average_percentile_ratio.shape[0] == 0:
            self.percentile_batches_tracked = torch.zeros_like(any_non_zero_quantile_value)
            self.average_percentile_ratio = torch.zeros_like(ratio_if_not_zero)
        if self.constant_channels.shape[0] == 0:
            self.constant_channels = torch.zeros_like(any_constant_channels)
        num_batches = self.percentile_batches_tracked
        average_ratio = self.average_percentile_ratio
        new_number_of_batches: torch.Tensor = num_batches + any_non_zero_quantile_value
        new_ratios: torch.Tensor = (average_ratio * num_batches + ratio_if_not_zero) / new_number_of_batches
        new_ratios = torch.nan_to_num(new_ratios)
        new_constant_count: torch.Tensor = self.constant_channels + any_constant_channels
        self.percentile_batches_tracked.copy_(new_number_of_batches)
        self.average_percentile_ratio.copy_(new_ratios)
        self.constant_channels.copy_(new_constant_count)
        return x_copy

    @torch.jit.export
    def get_batch_to_epoch_ratio(self):
        if False:
            for i in range(10):
                print('nop')
        epoch_activation_range = self.epoch_activation_max - self.epoch_activation_min
        if epoch_activation_range == torch.tensor(float(0)):
            raise ValueError('Range for Epoch is 0')
        elif epoch_activation_range == torch.tensor(float('inf')):
            raise ValueError('No data has been run through observer or infinity value present')
        else:
            return self.average_batch_activation_range / epoch_activation_range

    @torch.jit.export
    def reset_batch_and_epoch_values(self):
        if False:
            return 10
        device = self.max_val.device
        self.num_batches_tracked = 0
        self.average_batch_activation_range = torch.tensor(float(0), device=device)
        self.epoch_activation_min = torch.tensor(float('inf'), device=device)
        self.epoch_activation_max = torch.tensor(float('-inf'), device=device)
        self.min_val = torch.tensor([], device=device)
        self.max_val = torch.tensor([], device=device)
        self.average_percentile_ratio = torch.tensor([], device=device)
        self.percentile_batches_tracked = torch.tensor([], device=device)
        self.constant_channels = torch.tensor([], device=device)

    @torch.jit.export
    def calculate_qparams(self):
        if False:
            return 10
        raise Exception('calculate_qparams should not be called for ModelReportObserver')