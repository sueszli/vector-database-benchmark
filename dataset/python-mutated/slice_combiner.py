from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from snorkel.classification.utils import collect_flow_outputs_by_suffix

class SliceCombinerModule(nn.Module):
    """A module for combining the weighted representations learned by slices.

    Intended for use with the MultitaskClassifier including:
        * Indicator operations
        * Prediction operations
        * Prediction transform features

    NOTE: This module currently only handles binary labels.

    Parameters
    ----------
    slice_ind_key
        Suffix of operation corresponding to the slice indicator heads
    slice_pred_key
        Suffix of operation corresponding to the slice predictor heads
    slice_pred_feat_key
        Suffix of operation corresponding to the slice predictor features heads
    temperature
        Temperature constant for scaling the weighting between indicator prediction
        and predictor confidences: SoftMax(indicator_pred * predictor_confidence / tau)

    Attributes
    ----------
    slice_ind_key
        See above
    slice_pred_key
        See above
    slice_pred_feat_key
        See above
    """

    def __init__(self, slice_ind_key: str='_ind_head', slice_pred_key: str='_pred_head', slice_pred_feat_key: str='_pred_transform', temperature: float=1.0) -> None:
        if False:
            return 10
        super().__init__()
        self.slice_ind_key = slice_ind_key
        self.slice_pred_key = slice_pred_key
        self.slice_pred_feat_key = slice_pred_feat_key
        self.temperature = temperature

    def forward(self, output_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Reweight and combine predictor representations given output dict.\n\n        Parameters\n        ----------\n        output_dict\n            A dict of data fields containing operation outputs from indicator head,\n            predictor head, and predictor transform (corresponding to slice_ind_key,\n            slice_pred_key, slice_pred_feat_key, respectively).\n\n            NOTE: The output_dict outputs for the ind/pred heads must be raw logits.\n\n        Returns\n        -------\n        torch.Tensor\n            The reweighted predictor representation\n        '
        indicator_outputs = collect_flow_outputs_by_suffix(output_dict, self.slice_ind_key)
        indicator_preds = torch.cat([F.softmax(output, dim=1)[:, 1].unsqueeze(1) for output in indicator_outputs], dim=-1)
        predictor_outputs = collect_flow_outputs_by_suffix(output_dict, self.slice_pred_key)
        if predictor_outputs[0].shape[1] > 2:
            raise NotImplementedError('SliceCombiner does not support more than 2 classes yet.')
        elif predictor_outputs[0].shape[1] < 2:
            raise NotImplementedError('SliceCombiner currently requires output shape [..., 2] for predictor heads.')
        predictor_confidences = torch.cat([F.softmax(output, dim=1)[:, 1].unsqueeze(1) for output in predictor_outputs], dim=-1)
        predictor_feat_outputs = collect_flow_outputs_by_suffix(output_dict, self.slice_pred_feat_key)
        slice_representations = torch.stack(predictor_feat_outputs, dim=1)
        attention_weights = F.softmax(indicator_preds * predictor_confidences / self.temperature, dim=1)
        attention_weights = attention_weights.unsqueeze(-1).expand([-1, -1, slice_representations.size(-1)])
        reweighted_rep = torch.sum(attention_weights * slice_representations, dim=1)
        return reweighted_rep