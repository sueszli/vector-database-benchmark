import numpy as np
import torch
import torch.nn as nn
from neuralprophet.components.trend import Trend
from neuralprophet.utils_torch import init_parameter

class PiecewiseLinearTrend(Trend):

    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        if False:
            print('Hello World!')
        super().__init__(config=config, n_forecasts=n_forecasts, num_trends_modelled=num_trends_modelled, quantiles=quantiles, id_list=id_list, device=device)
        self.segmentwise_trend = self.config_trend.trend_reg == 0
        self.trend_k0 = init_parameter(dims=[len(self.quantiles)] + [self.num_trends_modelled] + [1])
        if self.config_trend.changepoints is None:
            linear_t = np.arange(self.config_trend.n_changepoints + 1).astype(float)
            linear_t = linear_t / (self.config_trend.n_changepoints + 1)
            self.config_trend.changepoints = self.config_trend.changepoints_range * linear_t
        else:
            self.config_trend.changepoints = np.insert(self.config_trend.changepoints, 0, 0.0)
        self.register_buffer('trend_changepoints_t', torch.tensor(self.config_trend.changepoints, requires_grad=False, dtype=torch.float))
        self.trend_deltas = init_parameter(dims=[len(self.quantiles)] + [self.num_trends_modelled] + [self.config_trend.n_changepoints + 1])
        if self.config_trend.growth == 'discontinuous':
            self.trend_m = init_parameter(dims=[len(self.quantiles)] + [self.num_trends_modelled] + [self.config_trend.n_changepoints + 1])

    def forward(self, t, meta):
        if False:
            return 10
        'Computes trend based on model configuration.\n\n        Parameters\n        ----------\n            t : torch.Tensor float\n                normalized time, dim: (batch, n_forecasts)\n            meta: dict\n                Metadata about the all the samples of the model input batch. Contains the following:\n                    * ``df_name`` (list, str), time series ID corresponding to each sample of the input batch.\n        Returns\n        -------\n            torch.Tensor\n                Trend component, same dimensions as input t\n\n        '
        if self.config_trend.trend_global_local == 'local':
            meta_name_tensor_one_hot = nn.functional.one_hot(meta, num_classes=len(self.id_list))
        else:
            meta_name_tensor_one_hot = None
        past_next_changepoint = t.unsqueeze(dim=2) >= self.trend_changepoints_t[1:].unsqueeze(dim=0)
        segment_id = past_next_changepoint.sum(dim=2)
        current_segment = nn.functional.one_hot(segment_id, num_classes=self.config_trend.n_changepoints + 1)
        k_t = self.compute_k_t(current_segment, past_next_changepoint, meta_name_tensor_one_hot)
        m_t = self.compute_m_t(current_segment, past_next_changepoint, meta_name_tensor_one_hot)
        trend = self.compute_trend(t, k_t, m_t, meta_name_tensor_one_hot)
        return self.bias.unsqueeze(dim=0).unsqueeze(dim=0) + trend

    @property
    def get_trend_deltas(self):
        if False:
            for i in range(10):
                print('nop')
        'trend deltas for regularization.\n\n        update if trend is modelled differently'
        if self.config_trend is None or self.config_trend.n_changepoints < 1:
            trend_delta = None
        elif self.segmentwise_trend:
            trend_delta = self.trend_deltas[:, :, :] - torch.cat((self.trend_k0, self.trend_deltas[:, :, 0:-1]), dim=2)
        else:
            trend_delta = self.trend_deltas
        return trend_delta

    def add_regularization(self):
        if False:
            i = 10
            return i + 15
        pass

    def compute_k_t(self, current_segment, past_next_changepoint, meta_name_tensor_one_hot):
        if False:
            i = 10
            return i + 15
        'For segmentwise, k_t is the model parameter representing the trend slope(actually, trend slope-k_0) in the\n        current_segment at time t (for each sample of the batch).\n\n        For not segmentwise, k_t is the model parameter representing the difference between trend slope in the\n        current_segment at time t and the trend slope in the previous segment (for each sample of the batch).\n\n        Parameters\n        ----------\n            current_segment : torch.Tensor, int\n                segment corresponding to time t (batch_size, n_forecasts, segments (+ 1))\n\n            past_next_changepoint : torch.Tensor, bool\n                whether the a changepoint >= time t (batch_size, n_forecasts, segments (+ 1))\n\n            meta_name_tensor_one_hot : torch.Tensor, float\n                Metadata about the all the samples of the model input batch.\n\n                Contains the following:\n                    * ``df_name`` (list, str), time series name ID corresponding to each sample of the input batch.\n        Returns\n        -------\n            torch.Tensor\n                k_t,  ( batch_size, n_forecasts, quantiles_size)\n        '
        pass

    def compute_m_t(self, current_segment, past_next_changepoint, meta_name_tensor_one_hot):
        if False:
            i = 10
            return i + 15
        'm_t represents the value at the origin(t=0) that we would need to have so that if we use (k_t + k_0) as\n        slope, we reach the same value at time = chagepoint_start_of_segment_i as we would reach by following the\n        segmented slope (having in each segment the slope trend_deltas(i) + k_0)\n\n        Parameters\n        ----------\n            current_segment : torch.Tensor, int\n                segment corresponding to time t (batch_size, n_forecasts, segments (+ 1))\n\n            past_next_changepoint : torch.Tensor, bool\n                whether the a changepoint >= time t (batch_size, n_forecasts, segments (+ 1))\n\n            meta_name_tensor_one_hot : torch.Tensor, float\n                Metadata about the all the samples of the model input batch.\n\n                Contains the following:\n                    * ``df_name`` (list, str), time series name ID corresponding to each sample of the input batch.\n        Returns\n        -------\n            torch.Tensor\n                m_t,  ( batch_size, n_forecasts, quantiles_size)\n        '
        pass

    def compute_trend(self, t, k_t, m_t, meta_name_tensor_one_hot=None):
        if False:
            return 10
        'This method computes the trend component of the model.\n\n\n        Parameters\n        ----------\n            t : torch.Tensor, float\n                time\n\n            k_t : torch.Tensor, int\n                see compute_k_t\n\n            m_t : torch.Tensor, bool\n                see compute_m_t\n\n            meta_name_tensor_one_hot : torch.Tensor, float\n                Metadata about the all the samples of the model input batch.\n\n                Contains the following:\n                    * ``df_name`` (list, str), time series name ID corresponding to each sample of the input batch.\n        Returns\n        -------\n            torch.Tensor\n                trend_component,  ( batch_size, n_forecasts, quantiles_size)\n        '
        pass

class GlobalPiecewiseLinearTrend(PiecewiseLinearTrend):

    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config=config, n_forecasts=n_forecasts, num_trends_modelled=num_trends_modelled, quantiles=quantiles, id_list=id_list, device=device)

    def compute_k_t(self, current_segment, past_next_changepoint, meta_name_tensor_one_hot=None):
        if False:
            i = 10
            return i + 15
        'This method overrides the method from the PiecewiseLinear class.'
        k_t = torch.sum(current_segment.unsqueeze(dim=2) * self.trend_deltas.permute(1, 0, 2).unsqueeze(1), dim=-1)
        if not self.segmentwise_trend:
            previous_deltas_t = torch.sum(past_next_changepoint.unsqueeze(dim=2) * self.trend_deltas.permute(1, 0, 2)[:, :, :-1].unsqueeze(dim=0), dim=-1)
            k_t = k_t + previous_deltas_t
        return k_t

    def compute_m_t(self, current_segment, past_next_changepoint, meta_name_tensor_one_hot=None):
        if False:
            i = 10
            return i + 15
        'This method overrides the method from the PiecewiseLinear class.'
        if self.config_trend.growth != 'discontinuous':
            if self.segmentwise_trend:
                deltas = self.trend_deltas[:, :, :] - torch.cat((self.trend_k0, self.trend_deltas[:, :, 0:-1]), dim=2)
            else:
                deltas = self.trend_deltas
            gammas = -self.trend_changepoints_t[1:] * deltas[:, :, 1:]
            m_t = torch.sum(past_next_changepoint.unsqueeze(dim=2) * gammas.permute(1, 0, 2).unsqueeze(1), dim=-1)
            if not self.segmentwise_trend:
                m_t = m_t.detach()
        else:
            m_t = torch.sum(current_segment.unsqueeze(dim=2) * self.trend_m.permute(1, 0, 2).unsqueeze(dim=0), dim=-1)
        return m_t

    def compute_trend(self, t, k_t, m_t, meta_name_tensor_one_hot=None):
        if False:
            i = 10
            return i + 15
        'This method overrides the method from the PiecewiseLinear class.'
        return (self.trend_k0.permute(1, 2, 0) + k_t) * torch.unsqueeze(t, dim=2) + m_t

class LocalPiecewiseLinearTrend(PiecewiseLinearTrend):

    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        if False:
            i = 10
            return i + 15
        super().__init__(config=config, n_forecasts=n_forecasts, num_trends_modelled=num_trends_modelled, quantiles=quantiles, id_list=id_list, device=device)

    def compute_k_t(self, current_segment, past_next_changepoint, meta_name_tensor_one_hot):
        if False:
            print('Hello World!')
        'This method overrides the method from the PiecewiseLinear class.'
        trend_deltas_by_sample = torch.sum(meta_name_tensor_one_hot.unsqueeze(dim=0).unsqueeze(dim=-1) * self.trend_deltas.unsqueeze(dim=1), dim=2)
        k_t = torch.sum(current_segment.unsqueeze(dim=2) * trend_deltas_by_sample.permute(1, 0, 2).unsqueeze(1), dim=-1)
        if not self.segmentwise_trend:
            previous_deltas_t = torch.sum(past_next_changepoint.unsqueeze(dim=2) * trend_deltas_by_sample.permute(1, 0, 2)[:, :, :-1].unsqueeze(dim=1), dim=-1)
            k_t = k_t + previous_deltas_t
        return k_t

    def compute_m_t(self, current_segment, past_next_changepoint, meta_name_tensor_one_hot=None):
        if False:
            for i in range(10):
                print('nop')
        'This method overrides the method from the PiecewiseLinear class.'
        if self.config_trend.growth != 'discontinuous':
            if self.segmentwise_trend:
                deltas = self.trend_deltas[:, :, :] - torch.cat((self.trend_k0, self.trend_deltas[:, :, 0:-1]), dim=2)
            else:
                deltas = self.trend_deltas
            gammas_0 = -self.trend_changepoints_t[1:] * deltas[:, :, 1:]
            gammas = torch.sum(torch.transpose(meta_name_tensor_one_hot, 1, 0).unsqueeze(dim=-2).unsqueeze(dim=0) * torch.unsqueeze(gammas_0, dim=-1), dim=1)
            m_t = torch.sum(past_next_changepoint.unsqueeze(2) * gammas.permute(2, 0, 1).unsqueeze(1), dim=-1)
            if not self.segmentwise_trend:
                m_t = m_t.detach()
        else:
            m_t_0 = torch.sum(meta_name_tensor_one_hot.unsqueeze(dim=0).unsqueeze(dim=-1) * self.trend_m.unsqueeze(dim=1), dim=2)
            m_t = torch.sum(current_segment.unsqueeze(dim=2) * m_t_0.permute(1, 0, 2).unsqueeze(dim=1), dim=-1)
        return m_t

    def compute_trend(self, t, k_t, m_t, meta_name_tensor_one_hot=None):
        if False:
            return 10
        'This method overrides the method from the PiecewiseLinear class.'
        trend_k_0 = torch.sum(meta_name_tensor_one_hot.unsqueeze(dim=0).unsqueeze(dim=-1) * self.trend_k0.unsqueeze(dim=1), dim=2).permute(1, 2, 0)
        return (trend_k_0 + k_t) * t.unsqueeze(dim=2) + m_t