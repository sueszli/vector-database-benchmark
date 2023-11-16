""" PyTorch OneFormer model."""
import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ... import AutoBackbone
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, is_scipy_available, logging, replace_return_docstrings, requires_backends
from .configuration_oneformer import OneFormerConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'OneFormerConfig'
_CHECKPOINT_FOR_DOC = 'shi-labs/oneformer_ade20k_swin_tiny'
ONEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ['shi-labs/oneformer_ade20k_swin_tiny']
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

def _get_clones(module, N):
    if False:
        return 10
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def multi_scale_deformable_attention(value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    (batch_size, _, num_heads, hidden_dim) = value.shape
    (_, num_queries, num_heads, num_levels, num_points, _) = sampling_locations.shape
    value_list = value.split([height.item() * width.item() for (height, width) in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for (level_id, (height, width)) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = nn.functional.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(batch_size * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(batch_size, num_heads * hidden_dim, num_queries)
    return output.transpose(1, 2).contiguous()

def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    if False:
        return 10
    '\n    Compute the DICE loss, similar to generalized IOU for masks as follows:\n\n    $$ \\mathcal{L}_{\\text{dice}(x, y) = 1 - \\frac{2 * x \\cap y }{x \\cup y + 1}} $$\n\n    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow\n\n    $$ \\mathcal{L}_{\\text{dice}(x, y) = 1 - \\frac{2 * x * y }{x + y + 1}} $$\n\n    Args:\n        inputs (`torch.Tensor`):\n            A tensor representing a mask.\n        labels (`torch.Tensor`):\n            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs\n            (0 for the negative class and 1 for the positive class).\n        num_masks (`int`):\n            The number of masks present in the current batch, used for normalization.\n\n    Returns:\n        `torch.Tensor`: The computed loss.\n    '
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss

def sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor, num_masks: int) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    Args:\n        inputs (`torch.Tensor`):\n            A float tensor of arbitrary shape.\n        labels (`torch.Tensor`):\n            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs\n            (0 for the negative class and 1 for the positive class).\n\n    Returns:\n        loss (`torch.Tensor`): The computed loss.\n    '
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    cross_entropy_loss = criterion(inputs, labels)
    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss

def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    if False:
        while True:
            i = 10
    '\n    A pair wise version of the dice loss, see `dice_loss` for usage.\n\n    Args:\n        inputs (`torch.Tensor`):\n            A tensor representing a mask\n        labels (`torch.Tensor`):\n            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs\n            (0 for the negative class and 1 for the positive class).\n\n    Returns:\n        `torch.Tensor`: The computed loss between each pairs.\n    '
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    A pair wise version of the cross entropy loss, see `sigmoid_cross_entropy_loss` for usage.\n\n    Args:\n        inputs (`torch.Tensor`):\n            A tensor representing a mask.\n        labels (`torch.Tensor`):\n            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs\n            (0 for the negative class and 1 for the positive class).\n\n    Returns:\n        loss (`torch.Tensor`): The computed loss between each pairs.\n    '
    height_and_width = inputs.shape[1]
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))
    loss_pos = torch.matmul(cross_entropy_loss_pos, labels.T)
    loss_neg = torch.matmul(cross_entropy_loss_neg, (1 - labels).T)
    loss = loss_pos + loss_neg
    loss = loss / height_and_width
    return loss

def sample_point(input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim=False, **kwargs) -> torch.Tensor:
    if False:
        return 10
    '\n    A wrapper around `torch.nn.functional.grid_sample` to support 3D point_coordinates tensors.\n\n    Args:\n        input_features (`torch.Tensor` of shape (batch_size, channels, height, width)):\n            A tensor that contains features map on a height * width grid\n        point_coordinates (`torch.Tensor` of shape (batch_size, num_points, 2) or (batch_size, grid_height, grid_width,:\n        2)):\n            A tensor that contains [0, 1] * [0, 1] normalized point coordinates\n        add_dim (`bool`):\n            boolean value to keep track of added dimension\n\n    Returns:\n        point_features (`torch.Tensor` of shape (batch_size, channels, num_points) or (batch_size, channels,\n        height_grid, width_grid):\n            A tensor that contains features for points in `point_coordinates`.\n    '
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)
    point_features = torch.nn.functional.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    if add_dim:
        point_features = point_features.squeeze(3)
    return point_features

class OneFormerHungarianMatcher(nn.Module):

    def __init__(self, cost_class: float=1.0, cost_mask: float=1.0, cost_dice: float=1.0, num_points: int=12544):
        if False:
            print('Hello World!')
        "This class computes an assignment between the labels and the predictions of the network.\n\n        For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more\n        predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are\n        un-matched (and thus treated as non-objects).\n\n        Params:\n            cost_class (float, *optional*, defaults to 1.0):\n                This is the relative weight of the classification error in the matching cost.\n            cost_mask (float, *optional*,  defaults to 1.0):\n                This is the relative weight of the sigmoid ce loss of the binary mask in the matching cost.\n            cost_dice (float, *optional*, defaults to 1.0):\n                This is the relative weight of the dice loss of the binary mask in the matching cost\n            num_points (int, *optional*, defaults to 12544):\n                Number of points to be sampled for dice and mask loss matching cost.\n        "
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and (cost_dice == 0):
            raise ValueError('All costs cant be 0')
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels) -> List[Tuple[Tensor]]:
        if False:
            return 10
        'Performs the matching\n\n        Params:\n            masks_queries_logits (`torch.Tensor`):\n                A tensor` of dim `batch_size, num_queries, num_labels` with the\n                  classification logits.\n            class_queries_logits (`torch.Tensor`):\n                A tensor` of dim `batch_size, num_queries, height, width` with the\n                  predicted masks.\n\n            class_labels (`torch.Tensor`):\n                A tensor` of dim `num_target_boxes` (where num_target_boxes is the number\n                  of ground-truth objects in the target) containing the class labels.\n            mask_labels (`torch.Tensor`):\n                A tensor` of dim `num_target_boxes, height, width` containing the target\n                  masks.\n\n        Returns:\n            `List[Tuple[Tensor]]`: A list of size batch_size, containing tuples of (index_i, index_j) where:\n                - index_i is the indices of the selected predictions (in order)\n                - index_j is the indices of the corresponding selected labels (in order)\n            For each batch element, it holds:\n                len(index_i) = len(index_j) = min(num_queries, num_targets).\n        '
        indices: List[Tuple[np.array]] = []
        num_queries = class_queries_logits.shape[1]
        preds_masks = masks_queries_logits
        preds_probs = class_queries_logits
        for (pred_probs, pred_mask, target_mask, labels) in zip(preds_probs, preds_masks, mask_labels, class_labels):
            pred_probs = pred_probs.softmax(-1)
            cost_class = -pred_probs[:, labels]
            pred_mask = pred_mask[:, None]
            target_mask = target_mask[:, None].to(pred_mask.device)
            point_coords = torch.rand(1, self.num_points, 2, device=pred_mask.device)
            target_mask = sample_point(target_mask, point_coords.repeat(target_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            pred_mask = sample_point(pred_mask, point_coords.repeat(pred_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            with autocast(enabled=False):
                pred_mask = pred_mask.float()
                target_mask = target_mask.float()
                cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
                cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
                cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
                cost_matrix = cost_matrix.reshape(num_queries, -1).cpu()
                assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix.cpu())
                indices.append(assigned_indices)
        matched_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for (i, j) in indices]
        return matched_indices

class OneFormerLoss(nn.Module):

    def __init__(self, num_classes: int, matcher: OneFormerHungarianMatcher, weight_dict: Dict[str, float], eos_coef: float, num_points: int, oversample_ratio: float, importance_sample_ratio: float, contrastive_temperature: float=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        This class computes the losses using the class predictions, mask predictions and the contrastive queries.\n\n        Oneformer calculates the classification CE loss on the class predictions. Mask predictions are used for\n        calculating the binary CE loss and dice loss. The contrastive queries are used for calculating the contrastive\n        loss.\n\n        Args:\n            num_labels (`int`):\n                The number of classes.\n            matcher (`OneFormerHungarianMatcher`):\n                A torch module that computes the assigments between the predictions and labels.\n            weight_dict (`Dict[str, float]`):\n                A dictionary of weights to be applied to the different losses.\n            eos_coef (`float`):\n                Weight to apply to the null class.\n            num_points (`int`):\n                Number of points to be sampled for dice and mask loss calculations.\n            oversample_ratio (`float`):\n                Required for pointwise loss calculation.\n            importance_sample_ratio (`float`):\n                Required for pointwise loss calculation.\n            contrastive_temperature (`float`):\n                Temperature for scaling the contrastive logits.\n        '
        requires_backends(self, ['scipy'])
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.contrastive_temperature = contrastive_temperature
        if self.contrastive_temperature is not None:
            self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / contrastive_temperature)))

    def _max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        if False:
            while True:
                i = 10
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for (index, item) in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        if False:
            print('Hello World!')
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_size = len(tensors)
        batch_shape = [batch_size] + max_size
        (b, _, h, w) = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for (tensor, padded_tensor, padding_mask) in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[:tensor.shape[0], :tensor.shape[1], :tensor.shape[2]].copy_(tensor)
            padding_mask[:tensor.shape[1], :tensor.shape[2]] = False
        return (padded_tensors, padding_masks)

    def loss_contrastive(self, contrastive_queries_logits: Tensor, text_queries: Tensor):
        if False:
            while True:
                i = 10
        'Compute the query-text contrastive loss.\n\n        Args:\n            contrastive_queries_logits (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, hidden_dim`\n            text_queries (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, hidden_dim`\n        Returns:\n            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:\n            - **loss_contrastive** -- The query-text contrastive loss computed using task-guided queries\n                                    and text queries derived from input text list.\n        '
        image_queries = contrastive_queries_logits.float()
        image_queries = nn.functional.normalize(image_queries.flatten(1), dim=-1)
        text_queries = nn.functional.normalize(text_queries.flatten(1), dim=-1)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        logits_per_text = torch.matmul(text_queries, image_queries.t()) * logit_scale
        logits_per_img = logits_per_text.t()
        loss_img = nn.functional.cross_entropy(logits_per_img, torch.arange(len(logits_per_img), device=logits_per_text.device))
        loss_text = nn.functional.cross_entropy(logits_per_text, torch.arange(len(logits_per_text), device=logits_per_text.device))
        loss_contrastive = loss_img + loss_text
        losses = {'loss_contrastive': loss_contrastive}
        return losses

    def loss_labels(self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]) -> Dict[str, Tensor]:
        if False:
            print('Hello World!')
        'Compute the losses related to the labels using cross entropy.\n\n        Args:\n            class_queries_logits (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, num_labels`\n            class_labels (`List[torch.Tensor]`):\n                List of class labels of shape `(labels)`.\n            indices (`Tuple[np.array])`:\n                The indices computed by the Hungarian matcher.\n\n        Returns:\n            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:\n            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.\n        '
        pred_logits = class_queries_logits
        (batch_size, num_queries, _) = pred_logits.shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)
        target_classes_o = torch.cat([target[j] for (target, (_, j)) in zip(class_labels, indices)])
        target_classes = torch.full((batch_size, num_queries), fill_value=self.num_classes, dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o
        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {'loss_cross_entropy': loss_ce}
        return losses

    def loss_masks(self, masks_queries_logits: Tensor, mask_labels: List[Tensor], indices: Tuple[np.array], num_masks: int) -> Dict[str, Tensor]:
        if False:
            return 10
        'Compute the losses related to the masks using focal and dice loss.\n\n        Args:\n            masks_queries_logits (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, height, width`\n            mask_labels (`torch.Tensor`):\n                List of mask labels of shape `(labels, height, width)`.\n            indices (`Tuple[np.array])`:\n                The indices computed by the Hungarian matcher.\n            num_masks (`int)`:\n                The number of masks, used for normalization.\n\n        Returns:\n            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing two keys:\n            - **loss_mask** -- The loss computed using sigmoid ce loss on the predicted and ground truth masks.\n            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth\n              masks.\n        '
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        pred_masks = masks_queries_logits[src_idx]
        (target_masks, _) = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            point_coords = self.sample_points_using_uncertainty(pred_masks, self.calculate_uncertainty, self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            point_labels = sample_point(target_masks, point_coords, align_corners=False).squeeze(1)
        point_logits = sample_point(pred_masks, point_coords, align_corners=False).squeeze(1)
        losses = {'loss_mask': sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks), 'loss_dice': dice_loss(point_logits, point_labels, num_masks)}
        del pred_masks
        del target_masks
        return losses

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        "\n        In Mask2Former paper, uncertainty is estimated as L1 distance between 0.0 and the logit prediction in 'logits'\n        for the foreground class in `classes`.\n\n        Args:\n            logits (`torch.Tensor`):\n            A tensor of shape (R, 1, ...) for class-specific or class-agnostic, where R is the total number of predicted masks in all images and C is:\n            the number of foreground classes. The values are logits.\n\n        Returns:\n            scores (`torch.Tensor`): A tensor of shape (R, 1, ...) that contains uncertainty scores with the most\n            uncertain locations having the highest uncertainty score.\n        "
        uncertainty_scores = -torch.abs(logits)
        return uncertainty_scores

    def sample_points_using_uncertainty(self, logits: torch.Tensor, uncertainty_function, num_points: int, oversample_ratio: int, importance_sample_ratio: float) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The\n        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit\n        prediction as input.\n\n        Args:\n            logits (`float`):\n                Logit predictions for P points.\n            uncertainty_function:\n                A function that takes logit predictions for P points and returns their uncertainties.\n            num_points (`int`):\n                The number of points P to sample.\n            oversample_ratio (`int`):\n                Oversampling parameter.\n            importance_sample_ratio (`float`):\n                Ratio of points that are sampled via importance sampling.\n\n        Returns:\n            point_coordinates (`torch.Tensor`):\n                Coordinates for P sampled points.\n        '
        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        point_uncertainties = uncertainty_function(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
        if num_random_points > 0:
            point_coordinates = torch.cat([point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)], dim=1)
        return point_coordinates

    def _get_predictions_permutation_indices(self, indices):
        if False:
            return 10
        batch_indices = torch.cat([torch.full_like(src, i) for (i, (src, _)) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return (batch_indices, predictions_indices)

    def _get_targets_permutation_indices(self, indices):
        if False:
            print('Hello World!')
        batch_indices = torch.cat([torch.full_like(tgt, i) for (i, (_, tgt)) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return (batch_indices, target_indices)

    def forward(self, masks_queries_logits: Tensor, class_queries_logits: Tensor, contrastive_queries_logits: Tensor, mask_labels: List[Tensor], class_labels: List[Tensor], text_queries: Tensor, auxiliary_predictions: Optional[Dict[str, Tensor]]=None, calculate_contrastive_loss: bool=True) -> Dict[str, Tensor]:
        if False:
            for i in range(10):
                print('nop')
        "\n        This performs the loss computation.\n\n        Args:\n            masks_queries_logits (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, height, width`\n            class_queries_logits (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, num_labels`\n            contrastive_queries_logits (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, hidden_dim`\n            mask_labels (`torch.Tensor`):\n                List of mask labels of shape `(labels, height, width)`.\n            class_labels (`List[torch.Tensor]`):\n                List of class labels of shape `(labels)`.\n            text_queries (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, hidden_dim`\n            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):\n                if `use_auxiliary_loss` was set to `true` in [`OneFormerConfig`], then it contains the logits from the\n                inner layers of the Detr's Decoder.\n            calculate_contrastive_loss (`bool`, *optional*, defaults to `True`):\n                Whether or not to calculate the contrastive loss.\n\n        Returns:\n            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing two keys:\n            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.\n            - **loss_mask** -- The loss computed using sigmoid ce loss on the predicted and ground truth masks.\n            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth\n              masks.\n            - **loss_contrastive** -- The query-text contrstive loss computed using object and text queries.\n            if `use_auxiliary_loss` was set to `true` in [`OneFormerConfig`], the dictionary contains addional losses\n            for each auxiliary predictions.\n        "
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        losses: Dict[str, Tensor] = {**self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks), **self.loss_labels(class_queries_logits, class_labels, indices)}
        if calculate_contrastive_loss:
            losses = {**losses, **self.loss_contrastive(contrastive_queries_logits, text_queries)}
        if auxiliary_predictions is not None:
            for (idx, aux_outputs) in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs['masks_queries_logits']
                class_queries_logits = aux_outputs['class_queries_logits']
                loss_dict = self.forward(masks_queries_logits, class_queries_logits, None, mask_labels, class_labels, None, calculate_contrastive_loss=False)
                loss_dict = {f'{key}_{idx}': value for (key, value) in loss_dict.items()}
                losses.update(loss_dict)
        return losses

    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Computes the average number of target masks across the batch, for normalization purposes.\n        '
        num_masks = sum([len(classes) for classes in class_labels])
        num_masks_pt = torch.as_tensor([num_masks], dtype=torch.float, device=device)
        return num_masks_pt

@dataclass
class OneFormerTransformerDecoderOutput(BaseModelOutput):
    """
    Base class for outputs of the Transformer decoder. This class adds attributes for class predictions, mask
    predictions and contrastive logits to BaseModelOutputWithCrossAttentions.

    Args:
        object_logits (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`):
            Queries representation for the region proposals.
        contrastive_logits (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`):
            Queries representation for the contrastive loss.
        prediction_masks (`torch.FloatTensor` of shape `(batch_size, num_queries, height, width)`):
            Mask predictions from last layer of the transformer decoder.
        prediction_class (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes+1)`):
            Class predictions from last layer of the transformer decoder.
        auxiliary_predictions (Tuple of Dict of `str, torch.FloatTensor`, *optional*):
            Tuple of class and mask predictions from each layer of the transformer decoder.
    """
    object_queries: torch.FloatTensor = None
    contrastive_logits: Optional[torch.FloatTensor] = None
    prediction_masks: torch.FloatTensor = None
    prediction_class: torch.FloatTensor = None
    auxiliary_predictions: Optional[Tuple[Dict[str, torch.FloatTensor]]] = None

@dataclass
class OneFormerPixelDecoderOutput(ModelOutput):
    """
    OneFormer's pixel decoder module output, practically a Multi-Scale Deformable Attention based decoder. It returns
    the mask features and the multiscale features.

    Args:
        multi_scale_features (`tuple(torch.FloatTensor)`):
            Tuple of multi-scale features of scales [1/8, 1/16, 1/32] and shape `(batch_size, num_channels, height,
            width)`from the Multi-Scale Deformable Attenntion based Pixel Decoder.
        mask_features (`torch.FloatTensor`):
            Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel Decoder
            Layer.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from pixel decoder. Returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`
    """
    multi_scale_features: Tuple[torch.FloatTensor] = None
    mask_features: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class OneFormerPixelLevelModuleOutput(ModelOutput):
    """
    OneFormer's pixel level module output. It returns both the last and (optionally) the hidden states from the
    `encoder` and `decoder`. By default, the `encoder` is a Swin/Dinat Backbone and the `decoder` is a Multi-Scale
    Deformable Attention based decoder.

    Args:
        encoder_features (List of `(torch.FloatTensor)`):
            List of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
        decoder_features (List of `(torch.FloatTensor)`):
            List of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
        decoder_last_feature (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)):
            1/4 scale features from the last Pixel Decoder Layer.
    """
    encoder_features: List[torch.FloatTensor] = None
    decoder_features: List[torch.FloatTensor] = None
    decoder_last_feature: torch.FloatTensor = None

@dataclass
class OneFormerModelOutput(ModelOutput):
    """
    Class for outputs of [`OneFormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage.
        transformer_decoder_object_queries (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`)
            Output object queries from the last layer in the transformer decoder.
        transformer_decoder_contrastive_queries (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`)
            Contrastive queries from the transformer decoder.
        transformer_decoder_mask_predictions (`torch.FloatTensor` of shape `(batch_size, num_queries, height, width)`)
            Mask Predictions from the last layer in the transformer decoder.
        transformer_decoder_class_predictions (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes+1)`):
            Class Predictions from the last layer in the transformer decoder.
        transformer_decoder_auxiliary_predictions (Tuple of Dict of `str, torch.FloatTensor`, *optional*):
            Tuple of class and mask predictions from each layer of the transformer decoder.
        text_queries (`torch.FloatTensor`, *optional* of shape `(batch_size, num_queries, hidden_dim)`)
            Text queries derived from the input text list used for calculating contrastive loss during training.
        task_token (`torch.FloatTensor` of shape `(batch_size, hidden_dim)`)
            1D task token to condition the queries.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self and Cross Attentions weights from transformer decoder.
    """
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    transformer_decoder_object_queries: torch.FloatTensor = None
    transformer_decoder_contrastive_queries: Optional[torch.FloatTensor] = None
    transformer_decoder_mask_predictions: torch.FloatTensor = None
    transformer_decoder_class_predictions: torch.FloatTensor = None
    transformer_decoder_auxiliary_predictions: Optional[Tuple[Dict[str, torch.FloatTensor]]] = None
    text_queries: Optional[torch.FloatTensor] = None
    task_token: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class OneFormerForUniversalSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`OneFormerForUniversalSegmentationOutput`].

    This output can be directly passed to [`~OneFormerImageProcessor.post_process_semantic_segmentation`] or
    [`~OneFormerImageProcessor.post_process_instance_segmentation`] or
    [`~OneFormerImageProcessor.post_process_panoptic_segmentation`] depending on the task. Please, see
    [`~OneFormerImageProcessor] for details regarding usage.

    Args:
        loss (`torch.Tensor`, *optional*):
            The computed loss, returned when labels are present.
        class_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        masks_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
        auxiliary_predictions (List of Dict of `str, torch.FloatTensor`, *optional*):
            List of class and mask predictions from each layer of the transformer decoder.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage.
        transformer_decoder_object_queries (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`)
            Output object queries from the last layer in the transformer decoder.
        transformer_decoder_contrastive_queries (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`)
            Contrastive queries from the transformer decoder.
        transformer_decoder_mask_predictions (`torch.FloatTensor` of shape `(batch_size, num_queries, height, width)`)
            Mask Predictions from the last layer in the transformer decoder.
        transformer_decoder_class_predictions (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes+1)`):
            Class Predictions from the last layer in the transformer decoder.
        transformer_decoder_auxiliary_predictions (List of Dict of `str, torch.FloatTensor`, *optional*):
            List of class and mask predictions from each layer of the transformer decoder.
        text_queries (`torch.FloatTensor`, *optional* of shape `(batch_size, num_queries, hidden_dim)`)
            Text queries derived from the input text list used for calculating contrastive loss during training.
        task_token (`torch.FloatTensor` of shape `(batch_size, hidden_dim)`)
            1D task token to condition the queries.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self and Cross Attentions weights from transformer decoder.
    """
    loss: Optional[torch.FloatTensor] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_predictions: List[Dict[str, torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[List[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    transformer_decoder_object_queries: torch.FloatTensor = None
    transformer_decoder_contrastive_queries: Optional[torch.FloatTensor] = None
    transformer_decoder_mask_predictions: torch.FloatTensor = None
    transformer_decoder_class_predictions: torch.FloatTensor = None
    transformer_decoder_auxiliary_predictions: Optional[List[Dict[str, torch.FloatTensor]]] = None
    text_queries: Optional[torch.FloatTensor] = None
    task_token: torch.FloatTensor = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

class OneFormerPixelDecoderFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if False:
            i = 10
            return i + 15
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-05
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias

class OneFormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim (d_model) must be divisible by num_heads, but got {embed_dim} and {num_heads}')
        dim_per_head = embed_dim // num_heads
        if not (dim_per_head & dim_per_head - 1 == 0 and dim_per_head != 0):
            warnings.warn("You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the dimension of each attention head a power of 2 which is more efficient in the authors' CUDA implementation.")
        self.im2col_step = 128
        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        if False:
            i = 10
            return i + 15
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states=None, encoder_attention_mask=None, position_embeddings: Optional[torch.Tensor]=None, reference_points=None, spatial_shapes=None, level_start_index=None, output_attentions: bool=False):
        if False:
            while True:
                i = 10
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)
        (batch_size, num_queries, _) = hidden_states.shape
        (batch_size, sequence_length, _) = encoder_hidden_states.shape
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError('Make sure to align the spatial shapes with the sequence length of the encoder hidden states')
        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(hidden_states).view(batch_size, num_queries, self.n_heads, self.n_levels * self.n_points)
        attention_weights = nn.functional.softmax(attention_weights, -1).view(batch_size, num_queries, self.n_heads, self.n_levels, self.n_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}')
        output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return (output, attention_weights)

class OneFormerPixelDecoderEncoderLayer(nn.Module):

    def __init__(self, config: OneFormerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.embed_dim = config.conv_dim
        self.self_attn = OneFormerPixelDecoderEncoderMultiscaleDeformableAttention(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, n_levels=3, n_points=4)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.dropout = config.dropout
        self.activation_fn = nn.functional.relu
        self.activation_dropout = config.dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_feedforward_dim)
        self.fc2 = nn.Linear(config.encoder_feedforward_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.is_training = config.is_training

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, position_embeddings: torch.Tensor=None, reference_points=None, spatial_shapes=None, level_start_index=None, output_attentions: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Input to the layer.\n            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):\n                Attention mask.\n            position_embeddings (`torch.FloatTensor`, *optional*):\n                Position embeddings, to be added to `hidden_states`.\n            reference_points (`torch.FloatTensor`, *optional*):\n                Reference points.\n            spatial_shapes (`torch.LongTensor`, *optional*):\n                Spatial shapes of the backbone feature maps.\n            level_start_index (`torch.LongTensor`, *optional*):\n                Level start index.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        (hidden_states, attn_weights) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, position_embeddings=position_embeddings, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.is_training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.is_training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.is_training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if self.is_training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class OneFormerPixelDecoderEncoderOnly(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`OneFormerPixelDecoderEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: OneFormerConfig
    """

    def __init__(self, config: OneFormerConfig):
        if False:
            return 10
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layers = nn.ModuleList([OneFormerPixelDecoderEncoderLayer(config) for _ in range(config.encoder_layers)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        if False:
            print('Hello World!')
        '\n        Get reference points for each feature map. Used in decoder.\n\n        Args:\n            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):\n                Spatial shapes of each feature map.\n            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):\n                Valid ratios of each feature map.\n            device (`torch.device`):\n                Device on which to create the tensors.\n        Returns:\n            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`\n        '
        reference_points_list = []
        for (lvl, (height, width)) in enumerate(spatial_shapes):
            (ref_y, ref_x) = torch.meshgrid(torch.linspace(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device), torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, inputs_embeds=None, attention_mask=None, position_embeddings=None, spatial_shapes=None, level_start_index=None, valid_ratios=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:\n                - 1 for pixel features that are real (i.e. **not masked**),\n                - 0 for pixel features that are padding (i.e. **masked**).\n                [What are attention masks?](../glossary#attention-mask)\n            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Position embeddings that are added to the queries and keys in each self-attention layer.\n            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):\n                Spatial shapes of each feature map.\n            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):\n                Starting index of each feature map.\n            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):\n                Ratio of valid area in each feature level.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = inputs_embeds
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=inputs_embeds.device)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for (i, encoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states, attention_mask, position_embeddings=position_embeddings, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

class OneFormerPixelDecoder(nn.Module):

    def __init__(self, config: OneFormerConfig, feature_channels):
        if False:
            return 10
        super().__init__()
        self.config = config
        self.position_embedding = OneFormerSinePositionEmbedding(num_pos_feats=config.conv_dim // 2, normalize=True)
        self.num_feature_levels = 3
        transformer_in_channels = feature_channels[-self.num_feature_levels:]
        self.transformer_feature_strides = config.strides[-self.num_feature_levels:]
        self.feature_channels = feature_channels
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, config.conv_dim))
        if self.num_feature_levels > 1:
            input_projections_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_projections_list.append(nn.Sequential(nn.Conv2d(in_channels, config.conv_dim, kernel_size=1), nn.GroupNorm(32, config.conv_dim)))
            self.input_projections = nn.ModuleList(input_projections_list)
        else:
            self.input_projections = nn.ModuleList([nn.Sequential(nn.Conv2d(transformer_in_channels[-1], config.conv_dim, kernel_size=1), nn.GroupNorm(32, config.conv_dim))])
        self.encoder = OneFormerPixelDecoderEncoderOnly(config)
        self.mask_projection = nn.Conv2d(config.conv_dim, config.mask_dim, kernel_size=1, stride=1, padding=0)
        self.common_stride = config.common_stride
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs = []
        output_convs = []
        for (idx, in_channels) in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_conv = nn.Sequential(nn.Conv2d(in_channels, config.conv_dim, kernel_size=1, bias=False), nn.GroupNorm(32, config.conv_dim))
            output_conv = nn.Sequential(nn.Conv2d(config.conv_dim, config.conv_dim, kernel_size=3, stride=1, padding=1, bias=False), nn.GroupNorm(32, config.conv_dim), nn.ReLU())
            self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
            self.add_module('layer_{}'.format(idx + 1), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def get_valid_ratio(self, mask, dtype=torch.float32):
        if False:
            i = 10
            return i + 15
        'Get the valid ratio of all feature maps.'
        (_, height, width) = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def forward(self, features, encoder_outputs=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            return 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        sources = []
        position_embeddings_list = []
        for (level, source) in enumerate(features[::-1][:self.num_feature_levels]):
            sources.append(self.input_projections[level](source))
            position_embeddings_list.append(self.position_embedding(source))
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in sources]
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for (level, (source, mask, pos_embed)) in enumerate(zip(sources, masks, position_embeddings_list)):
            (batch_size, num_channels, height, width) = source.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m, dtype=source_flatten.dtype) for m in masks], 1)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(inputs_embeds=source_flatten, attention_mask=mask_flatten, position_embeddings=lvl_pos_embed_flatten, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        y = encoder_outputs.last_hidden_state
        bs = y.shape[0]
        split_size_or_sections = [None] * self.num_feature_levels
        for i in range(self.num_feature_levels):
            if i < self.num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)
        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for (i, z) in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        for (idx, feats) in enumerate(features[:self.num_fpn_levels][::-1]):
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(feats)
            y = cur_fpn + nn.functional.interpolate(out[-1], size=cur_fpn.shape[-2:], mode='bilinear', align_corners=False)
            y = output_conv(y)
            out.append(y)
        for o in out:
            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        return OneFormerPixelDecoderOutput(mask_features=self.mask_projection(out[-1]), multi_scale_features=multi_scale_features, attentions=encoder_outputs.attentions)

class OneFormerPixelLevelModule(nn.Module):

    def __init__(self, config: OneFormerConfig):
        if False:
            while True:
                i = 10
        '\n        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image\n        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel\n        decoder, generating multi-scale feature maps and pixel embeddings.\n\n        Args:\n            config ([`OneFormerConfig`]):\n                The configuration used to instantiate this model.\n        '
        super().__init__()
        backbone_config = config.backbone_config
        self.encoder = AutoBackbone.from_config(backbone_config)
        self.decoder = OneFormerPixelDecoder(config, feature_channels=self.encoder.channels)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool=False) -> OneFormerPixelLevelModuleOutput:
        if False:
            return 10
        features: List[Tensor] = self.encoder(pixel_values).feature_maps
        decoder_output: OneFormerPixelDecoderOutput = self.decoder(features, output_hidden_states=output_hidden_states)
        return OneFormerPixelLevelModuleOutput(encoder_features=tuple(features), decoder_features=decoder_output.multi_scale_features, decoder_last_feature=decoder_output.mask_features)

class OneFormerAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Here, we add position embeddings to the queries and
    keys (as explained in the DETR paper).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True):
        if False:
            while True:
                i = 10
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        if False:
            return 10
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        if False:
            return 10
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_embeddings: Optional[torch.Tensor]=None, key_value_states: Optional[torch.Tensor]=None, key_value_position_embeddings: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            i = 10
            return i + 15
        'Input shape: Batch x Time x Channel'
        hidden_states = hidden_states.permute(1, 0, 2) if hidden_states is not None else None
        position_embeddings = position_embeddings.permute(1, 0, 2) if position_embeddings is not None else None
        key_value_states = key_value_states.permute(1, 0, 2) if key_value_states is not None else None
        key_value_position_embeddings = key_value_position_embeddings.permute(1, 0, 2) if key_value_position_embeddings is not None else None
        is_cross_attention = key_value_states is not None
        (batch_size, target_len, embed_dim) = hidden_states.size()
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)
        if key_value_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, key_value_position_embeddings)
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)
        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        source_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(f'Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (batch_size * self.num_heads, target_len, source_len):
                raise ValueError(f'Attention mask should be of size {(target_len, batch_size * self.num_heads, source_len)}, but is {attention_mask.size()}')
            attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)
        attn_output = self.out_proj(attn_output).permute(1, 0, 2)
        return (attn_output, attn_weights_reshaped)

class OneFormerTransformerDecoderSelfAttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, activation='relu', normalize_before=False, layer_norm_eps=1e-05):
        if False:
            while True:
                i = 10
        super().__init__()
        self.self_attn = OneFormerAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, is_decoder=True)
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if False:
            return 10
        return tensor if pos is None else tensor + pos

    def forward_post(self, output, output_mask: Optional[Tensor]=None, output_key_padding_mask: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if False:
            i = 10
            return i + 15
        (output2, attention_weights) = self.self_attn(hidden_states=output, position_embeddings=query_pos, attention_mask=output_mask, output_attentions=True)
        output = output + self.dropout(output2)
        output = self.norm(output)
        return (output, attention_weights)

    def forward_pre(self, output, output_mask: Optional[Tensor]=None, output_key_padding_mask: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if False:
            i = 10
            return i + 15
        output2 = self.norm(output)
        (output2, attention_weights) = self.self_attn(hidden_states=output2, position_embeddings=query_pos, attention_mask=output_mask, output_attentions=True)
        output = output + self.dropout(output2)
        return (output, attention_weights)

    def forward(self, output, output_mask: Optional[Tensor]=None, output_key_padding_mask: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if False:
            while True:
                i = 10
        if self.normalize_before:
            return self.forward_pre(output, output_mask, output_key_padding_mask, query_pos)
        return self.forward_post(output, output_mask, output_key_padding_mask, query_pos)

class OneFormerTransformerDecoderCrossAttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, activation='relu', normalize_before=False, layer_norm_eps=1e-05):
        if False:
            while True:
                i = 10
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if False:
            return 10
        return tensor if pos is None else tensor + pos

    def forward_post(self, output, memory, memory_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if False:
            i = 10
            return i + 15
        (output2, attention_weights) = self.multihead_attn(query=self.with_pos_embed(output, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        output = output + self.dropout(output2)
        output = self.norm(output)
        return (output, attention_weights)

    def forward_pre(self, output, memory, memory_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if False:
            return 10
        output2 = self.norm(output)
        (output2, attention_weights) = self.multihead_attn(query=self.with_pos_embed(output2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        output = output + self.dropout(output2)
        return (output, attention_weights)

    def forward(self, output, memory, memory_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if False:
            while True:
                i = 10
        if self.normalize_before:
            return self.forward_pre(output, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(output, memory, memory_mask, memory_key_padding_mask, pos, query_pos)

class OneFormerTransformerDecoderFFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation='relu', normalize_before=False, layer_norm_eps=1e-05):
        if False:
            while True:
                i = 10
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if False:
            for i in range(10):
                print('nop')
        return tensor if pos is None else tensor + pos

    def forward_post(self, output):
        if False:
            while True:
                i = 10
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout(output2)
        output = self.norm(output)
        return output

    def forward_pre(self, output):
        if False:
            i = 10
            return i + 15
        output2 = self.norm(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output2))))
        output = output + self.dropout(output2)
        return output

    def forward(self, output):
        if False:
            return 10
        if self.normalize_before:
            return self.forward_pre(output)
        return self.forward_post(output)

class OneFormerMLPPredictionHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int=3):
        if False:
            return 10
        '\n        A classic Multi Layer Perceptron (MLP).\n\n        Args:\n            input_dim (`int`):\n                The input dimensions.\n            hidden_dim (`int`):\n                The hidden dimensions.\n            output_dim (`int`):\n                The output dimensions.\n            num_layers (int, *optional*, defaults to 3):\n                The number of layers.\n        '
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        layers = []
        for (i, (in_dim, out_dim)) in enumerate(zip(in_dims, out_dims)):
            layers.append(PredictionBlock(in_dim, out_dim, activation=nn.ReLU() if i < num_layers - 1 else nn.Identity()))
        self.layers = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        if False:
            return 10
        return self.layers(input)

class OneFormerTransformerDecoderLayer(nn.Module):

    def __init__(self, config: OneFormerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.embed_dim = config.hidden_dim
        self.num_feature_levels = 3
        self.cross_attn = OneFormerTransformerDecoderCrossAttentionLayer(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, dropout=0.0, normalize_before=config.pre_norm, layer_norm_eps=config.layer_norm_eps)
        self.self_attn = OneFormerTransformerDecoderSelfAttentionLayer(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, dropout=0.0, normalize_before=config.pre_norm, layer_norm_eps=config.layer_norm_eps)
        self.ffn = OneFormerTransformerDecoderFFNLayer(d_model=self.embed_dim, dim_feedforward=config.dim_feedforward, dropout=0.0, normalize_before=config.pre_norm, layer_norm_eps=config.layer_norm_eps)

    def forward(self, index: int, output: torch.Tensor, multi_stage_features: List[torch.Tensor], multi_stage_positional_embeddings: List[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, query_embeddings: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            index (`int`): index of the layer in the Transformer decoder.\n            output (`torch.FloatTensor`): the object queries of shape `(N, batch, hidden_dim)`\n            multi_stage_features (`List[torch.Tensor]`): the multi-scale features from the pixel decoder.\n            multi_stage_positional_embeddings (`List[torch.Tensor]`):\n                positional embeddings for the multi_stage_features\n            attention_mask (`torch.FloatTensor`): attention mask for the masked cross attention layer\n            query_embeddings (`torch.FloatTensor`, *optional*):\n                position embeddings that are added to the queries and keys in the self-attention layer.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        level_index = index % self.num_feature_levels
        attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False
        (output, cross_attn_weights) = self.cross_attn(output, multi_stage_features[level_index], memory_mask=attention_mask, memory_key_padding_mask=None, pos=multi_stage_positional_embeddings[level_index], query_pos=query_embeddings)
        (output, self_attn_weights) = self.self_attn(output, output_mask=None, output_key_padding_mask=None, query_pos=query_embeddings)
        output = self.ffn(output)
        outputs = (output,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

class OneFormerTransformerDecoderQueryTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        if False:
            return 10
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, output, memory, output_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, output_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if False:
            print('Hello World!')
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, output_mask=output_mask, memory_mask=memory_mask, output_key_padding_mask=output_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)

class OneFormerTransformerDecoderQueryTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, layer_norm_eps=1e-05):
        if False:
            while True:
                i = 10
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if False:
            return 10
        return tensor if pos is None else tensor + pos

    def forward_post(self, output, memory, output_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, output_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if False:
            return 10
        q = k = self.with_pos_embed(output, query_pos)
        output2 = self.self_attn(q, k, value=output, attn_mask=output_mask, key_padding_mask=output_key_padding_mask)
        output2 = output2[0]
        output = output + self.dropout1(output2)
        output = self.norm1(output)
        output2 = self.multihead_attn(query=self.with_pos_embed(output, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        output2 = output2[0]
        output = output + self.dropout2(output2)
        output = self.norm2(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout3(output2)
        output = self.norm3(output)
        return output

    def forward_pre(self, output, memory, output_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, output_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if False:
            for i in range(10):
                print('nop')
        output2 = self.norm1(output)
        q = k = self.with_pos_embed(output2, query_pos)
        output2 = self.self_attn(q, k, value=output2, attn_mask=output_mask, key_padding_mask=output_key_padding_mask)
        output2 = output2[0]
        output = output + self.dropout1(output2)
        output2 = self.norm2(output)
        output2 = self.multihead_attn(query=self.with_pos_embed(output2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        output2 = output2[0]
        output = output + self.dropout2(output2)
        output2 = self.norm3(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output2))))
        output = output + self.dropout3(output2)
        return output

    def forward(self, output, memory, output_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, output_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if False:
            for i in range(10):
                print('nop')
        if self.normalize_before:
            return self.forward_pre(output, memory, output_mask, memory_mask, output_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(output, memory, output_mask, memory_mask, output_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class OneFormerTransformerDecoderQueryTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, return_intermediate_dec=False, layer_norm_eps=1e-05):
        if False:
            return 10
        super().__init__()
        decoder_layer = OneFormerTransformerDecoderQueryTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before, layer_norm_eps)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = OneFormerTransformerDecoderQueryTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, mask, query_embed, pos_embed, task_token=None):
        if False:
            print('Hello World!')
        batch_size = src.shape[0]
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        if mask is not None:
            mask = mask.flatten(1)
        if task_token is None:
            queries = torch.zeros_like(query_embed)
        else:
            queries = task_token.repeat(query_embed.shape[0], 1, 1)
        queries = self.decoder(queries, src, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return queries.transpose(1, 2)

class OneFormerTransformerDecoder(nn.Module):
    """
    Transformer decoder
    """

    def __init__(self, in_channels: int, config: OneFormerConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.num_heads = config.num_attention_heads
        self.is_training = config.is_training
        self.use_task_norm = config.use_task_norm
        self.use_auxiliary_loss = config.use_auxiliary_loss
        self.query_transformer = OneFormerTransformerDecoderQueryTransformer(d_model=config.hidden_dim, dropout=config.dropout, nhead=config.num_attention_heads, dim_feedforward=config.dim_feedforward, num_decoder_layers=config.query_dec_layers, normalize_before=config.pre_norm, return_intermediate_dec=False, layer_norm_eps=config.layer_norm_eps)
        self.decoder_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.num_feature_levels = 3
        self.layers = nn.ModuleList([OneFormerTransformerDecoderLayer(config) for _ in range(config.decoder_layers - 1)])
        self.query_input_projection = nn.Conv2d(in_channels, config.hidden_dim, kernel_size=1)
        self.class_embed = nn.Linear(config.hidden_dim, config.num_labels + 1)
        self.mask_embed = OneFormerMLPPredictionHead(config.hidden_dim, config.hidden_dim, config.mask_dim, 3)

    def forward(self, task_token=None, multi_stage_features=None, multi_stage_positional_embeddings=None, mask_features=None, query_features=None, query_embeddings=None, query_embedder=None, size_list=None, output_attentions=None):
        if False:
            for i in range(10):
                print('nop')
        if self.use_task_norm:
            task_token = self.decoder_norm(task_token)
        object_queries = self.query_transformer(query_features, None, query_embedder.weight[:-1], self.query_input_projection(mask_features), task_token if self.use_task_norm else None)
        object_queries = object_queries[0].permute(1, 0, 2)
        queries = torch.cat([object_queries, task_token], dim=0)
        output = queries.clone()
        intermediate_class_predictions = []
        intermediate_mask_predictions = []
        (outputs_class, outputs_mask, attention_mask) = self.forward_prediction_heads(output, mask_features, attention_mask_target_size=size_list[0])
        intermediate_class_predictions.append(outputs_class)
        intermediate_mask_predictions.append(outputs_mask)
        attentions = ()
        for (index, layer) in enumerate(self.layers):
            layer_outputs = layer(index=index, output=output, multi_stage_features=multi_stage_features, multi_stage_positional_embeddings=multi_stage_positional_embeddings, attention_mask=attention_mask, query_embeddings=query_embeddings, output_attentions=output_attentions)
            output = layer_outputs[0]
            attentions += (layer_outputs[1:],)
            (outputs_class, outputs_mask, attention_mask) = self.forward_prediction_heads(output, mask_features, attention_mask_target_size=size_list[(index + 1) % self.num_feature_levels])
            intermediate_class_predictions.append(outputs_class)
            intermediate_mask_predictions.append(outputs_mask)
        if not len(intermediate_mask_predictions) == len(self.layers) + 1:
            raise ValueError('Intermediate predictions in the transformer decoder must have the same number of elements as number of layers')
        object_queries = layer_outputs[0].permute(1, 0, 2)
        contrastive_logits = queries.permute(1, 0, 2)
        return OneFormerTransformerDecoderOutput(object_queries=object_queries, contrastive_logits=contrastive_logits, prediction_masks=intermediate_mask_predictions[-1], prediction_class=intermediate_class_predictions[-1], auxiliary_predictions=self._get_aux_predictions(intermediate_class_predictions, intermediate_mask_predictions) if self.use_auxiliary_loss else None, attentions=attentions)

    def forward_prediction_heads(self, output, mask_features, attention_mask_target_size):
        if False:
            for i in range(10):
                print('nop')
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
        attention_mask = nn.functional.interpolate(outputs_mask, size=attention_mask_target_size, mode='bilinear', align_corners=False)
        attention_mask = (attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()
        return (outputs_class, outputs_mask, attention_mask)

    @torch.jit.unused
    def _get_aux_predictions(self, outputs_class, outputs_seg_masks):
        if False:
            return 10
        aux_list = [{'class_queries_logits': a, 'masks_queries_logits': b} for (a, b) in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        return tuple(aux_list)

class OneFormerTransformerModule(nn.Module):
    """
    The OneFormer's transformer module.
    """

    def __init__(self, in_features: int, config: OneFormerConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        hidden_dim = config.hidden_dim
        self.num_feature_levels = 3
        self.position_embedder = OneFormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.input_projections = []
        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_proj:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())
        self.decoder = OneFormerTransformerDecoder(in_channels=in_features, config=config)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    def forward(self, multi_scale_features: List[Tensor], mask_features: Tensor, task_token: Tensor, output_attentions: bool=False) -> OneFormerTransformerDecoderOutput:
        if False:
            i = 10
            return i + 15
        if not len(multi_scale_features) == self.num_feature_levels:
            raise ValueError(f'Number of elements in multi_scale_features ({len(multi_scale_features)}) and num_feature_levels ({self.num_feature_levels}) do not match!')
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []
        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
            multi_stage_features.append(self.input_projections[i](multi_scale_features[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)
        (_, batch_size, _) = multi_stage_features[0].shape
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        task_token = task_token.unsqueeze(0)
        query_features = self.position_embedder(mask_features, None)
        return self.decoder(task_token=task_token, multi_stage_features=multi_stage_features, multi_stage_positional_embeddings=multi_stage_positional_embeddings, mask_features=mask_features, query_features=query_features, query_embeddings=query_embeddings, query_embedder=self.queries_embedder, size_list=size_list, output_attentions=output_attentions)

class OneFormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats: int=64, temperature: int=10000, normalize: bool=False, scale: Optional[float]=None):
        if False:
            while True:
                i = 10
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        if False:
            return 10
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = (~mask).to(x.dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PredictionBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        if False:
            return 10
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        for (i, layer) in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state

class OneFormerTextMapperAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        if False:
            while True:
                i = 10
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        if False:
            print('Hello World!')
        (batch_size, q_sequence_length, num_channels) = q.shape
        if not k.shape == v.shape:
            raise ValueError(f'keys ({list(k.shape)}) and values ({list(v.shape)}) have different shapes!')
        (batch_size, k_sequence_length, num_channels) = k.shape
        q = self.q_proj(q).reshape(batch_size, q_sequence_length, self.num_heads, num_channels // self.num_heads)
        k = self.k_proj(k).reshape(batch_size, k_sequence_length, self.num_heads, num_channels // self.num_heads)
        v = self.v_proj(v).reshape(batch_size, k_sequence_length, self.num_heads, num_channels // self.num_heads)
        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        output = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(batch_size, q_sequence_length, num_channels)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output

class OneFormerTextTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1, layer_norm_eps=1e-05):
        if False:
            while True:
                i = 10
        super().__init__()
        self.self_attn = OneFormerTextMapperAttention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = OneFormerTextMapperAttention(d_model, nhead, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))

    def forward(self, hidden_state, mem):
        if False:
            for i in range(10):
                print('nop')
        q = k = v = self.norm1(hidden_state)
        hidden_state = hidden_state + self.self_attn(q, k, v)
        q = self.norm2(hidden_state)
        hidden_state = hidden_state + self.cross_attn(q, mem, mem)
        hidden_state = hidden_state + self.dropout(self.mlp(self.norm3(hidden_state)))
        return hidden_state

class OneFormerTextContextDecoder(nn.Module):

    def __init__(self, transformer_width=256, transformer_heads=4, transformer_layers=6, visual_dim=1024, dropout=0.1, layer_norm_eps=1e-05, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.memory_proj = nn.Sequential(nn.LayerNorm(visual_dim, eps=layer_norm_eps), nn.Linear(visual_dim, transformer_width), nn.LayerNorm(transformer_width, eps=layer_norm_eps))
        self.text_proj = nn.Sequential(nn.LayerNorm(visual_dim, eps=layer_norm_eps), nn.Linear(visual_dim, transformer_width))
        self.decoder = nn.ModuleList([OneFormerTextTransformerDecoderLayer(transformer_width, transformer_heads, dropout, layer_norm_eps) for _ in range(transformer_layers)])
        self.out_proj = nn.Sequential(nn.LayerNorm(transformer_width, eps=layer_norm_eps), nn.Linear(transformer_width, visual_dim))

    def forward(self, text, visual):
        if False:
            print('Hello World!')
        visual = self.memory_proj(visual)
        hidden_state = self.text_proj(text)
        for layer in self.decoder:
            hidden_state = layer(hidden_state, visual)
        return self.out_proj(hidden_state)

class OneFormerTextMLP(nn.Module):

    def __init__(self, hidden_size: Optional[int]=None, intermediate_size: Optional[int]=None, output_size: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.activation_fn = ACT2FN['quick_gelu']
        hidden_size = hidden_size
        intermediate_size = intermediate_size
        output_size = output_size
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, output_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class OneFormerTextTransformerLayer(nn.Module):

    def __init__(self, width: int, heads: int, attn_mask: torch.Tensor, layer_norm_eps=1e-05):
        if False:
            while True:
                i = 10
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads)
        self.layer_norm1 = nn.LayerNorm(width, eps=layer_norm_eps)
        self.mlp = OneFormerTextMLP(width, width * 4, width)
        self.layer_norm2 = nn.LayerNorm(width, eps=layer_norm_eps)
        self.attn_mask = attn_mask

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None) -> torch.FloatTensor:
        if False:
            print('Hello World!')
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, hidden_states, hidden_states, need_weights=False, key_padding_mask=key_padding_mask)[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class OneFormerTextTransformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor=None, use_checkpoint=False, layer_norm_eps=1e-05):
        if False:
            while True:
                i = 10
        super().__init__()
        self.width = width
        self.num_layers = layers
        self.layers = nn.Sequential(*[OneFormerTextTransformerLayer(width, heads, attn_mask, layer_norm_eps) for _ in range(layers)])
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states: torch.Tensor):
        if False:
            return 10
        for layer in self.layers:
            if self.use_checkpoint:
                hidden_states = self._gradient_checkpointing_func(layer, hidden_states)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states

class OneFormerTextEncoder(nn.Module):

    def __init__(self, context_length: int, width: int, layers: int, vocab_size, use_checkpoint=False, layer_norm_eps=1e-05):
        if False:
            while True:
                i = 10
        super().__init__()
        heads = width // 64
        self.context_length = context_length
        self.width = width
        self.transformer = OneFormerTextTransformer(width=width, layers=layers, heads=heads, attn_mask=self.build_attention_mask(), use_checkpoint=use_checkpoint, layer_norm_eps=layer_norm_eps)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.ln_final = nn.LayerNorm(width, eps=layer_norm_eps)
        self.token_embedding = nn.Embedding(vocab_size, width)

    def build_attention_mask(self):
        if False:
            print('Hello World!')
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    def forward(self, text):
        if False:
            print('Hello World!')
        hidden_state = self.token_embedding(text)
        hidden_state = hidden_state + self.positional_embedding
        hidden_state = hidden_state.permute(1, 0, 2)
        hidden_state = self.transformer(hidden_state)
        hidden_state = hidden_state.permute(1, 0, 2)
        hidden_state = self.ln_final(hidden_state)
        hidden_state = hidden_state[torch.arange(hidden_state.shape[0]), text.argmax(dim=-1)]
        return hidden_state

class OneFormerTextMapper(nn.Module):

    def __init__(self, config: OneFormerConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.text_encoder = OneFormerTextEncoder(context_length=config.text_encoder_context_length, width=config.text_encoder_width, layers=config.text_encoder_num_layers, vocab_size=config.text_encoder_vocab_size, layer_norm_eps=config.layer_norm_eps)
        self.text_projector = OneFormerMLPPredictionHead(config.text_encoder_width, config.hidden_dim, config.hidden_dim, config.text_encoder_proj_layers)
        if config.text_encoder_n_ctx > 0:
            self.prompt_ctx = nn.Embedding(config.text_encoder_n_ctx, config.text_encoder_width)
        else:
            self.prompt_ctx = None

    def forward(self, inputs: Tensor) -> Tensor:
        if False:
            return 10
        text_queries = self.encode_text(inputs)
        return text_queries

    def encode_text(self, text):
        if False:
            return 10
        if text.ndim is None:
            raise ValueError('text must not be NoneType')
        if text.ndim not in [2, 3]:
            raise ValueError('Number of dimensions in text must be 2 or 3')
        squeeze_dim = False
        num_text = 1
        if text.ndim == 3:
            num_text = text.shape[1]
            (batch_size, num_text, hidden_dim) = text.shape
            text = text.reshape(batch_size * num_text, hidden_dim)
            squeeze_dim = True
        encoded_text = self.text_encoder(text)
        text_queries = self.text_projector(encoded_text)
        if squeeze_dim:
            (_, hidden_dim) = text_queries.shape
            text_queries = text_queries.reshape(batch_size, num_text, hidden_dim)
            if self.prompt_ctx is not None:
                text_queries_ctx = self.prompt_ctx.weight.unsqueeze(0).repeat(text_queries.shape[0], 1, 1)
                text_queries = torch.cat([text_queries, text_queries_ctx], dim=1)
        return text_queries

class OneFormerTaskModel(nn.Module):

    def __init__(self, config: OneFormerConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.task_mlp = OneFormerMLPPredictionHead(config.task_seq_len, config.hidden_dim, config.hidden_dim, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        task_tokens = self.task_mlp(inputs)
        return task_tokens
ONEFORMER_START_DOCSTRING = '\n    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a\n    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.\n\n    Parameters:\n        config ([`OneFormerConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
ONEFORMER_INPUTS_DOCSTRING = "\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`OneFormerProcessor`]. See\n            [`OneFormerProcessor.__call__`] for details.\n        task_inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):\n            Task inputs. Task inputs can be obtained using [`AutoImageProcessor`]. See [`OneFormerProcessor.__call__`]\n            for details.\n        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):\n            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:\n\n            - 1 for pixels that are real (i.e. **not masked**),\n            - 0 for pixels that are padding (i.e. **masked**).\n\n            [What are attention masks?](../glossary#attention-mask)\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of Detr's decoder attention layers.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~OneFormerModelOutput`] instead of a plain tuple.\n"

class OneFormerPreTrainedModel(PreTrainedModel):
    config_class = OneFormerConfig
    base_model_prefix = 'model'
    main_input_name = 'pixel_values'

    def _init_weights(self, module: nn.Module):
        if False:
            i = 10
            return i + 15
        xavier_std = self.config.init_xavier_std
        std = self.config.init_std
        if isinstance(module, OneFormerTransformerModule):
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        nn.init.xavier_uniform_(input_projection.weight, gain=xavier_std)
                        nn.init.constant_(input_projection.bias, 0)
        elif isinstance(module, OneFormerTransformerDecoder):
            nn.init.xavier_uniform_(module.query_input_projection.weight, gain=xavier_std)
            nn.init.constant_(module.query_input_projection.bias, 0)
            module.query_input_projection._is_hf_initialized = True
        elif isinstance(module, OneFormerPixelDecoderEncoderMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            thetas = torch.arange(module.n_heads, dtype=torch.float32) * (2.0 * math.pi / module.n_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(module.n_heads, 1, 1, 2).repeat(1, module.n_levels, module.n_points, 1)
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)
        elif isinstance(module, OneFormerPixelDecoderEncoderOnly):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        elif isinstance(module, OneFormerPixelDecoder):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.normal_(module.level_embed, std=0)
        elif isinstance(module, OneFormerTransformerDecoderSelfAttentionLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)
        elif isinstance(module, OneFormerTransformerDecoderCrossAttentionLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)
        elif isinstance(module, OneFormerTransformerDecoderFFNLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)
        elif isinstance(module, OneFormerTransformerDecoderQueryTransformer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)
        elif isinstance(module, OneFormerPixelLevelModule):
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                    submodule.weight.data.normal_(mean=0.0, std=std)
                    if submodule.bias is not None:
                        submodule.bias.data.zero_()
        elif isinstance(module, OneFormerTextContextDecoder):
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.trunc_normal_(submodule.weight, std=0.02)
                    if isinstance(submodule, nn.Linear) and submodule.bias is not None:
                        nn.init.constant_(submodule.bias, 0)
                elif isinstance(submodule, nn.LayerNorm):
                    nn.init.constant_(submodule.bias, 0)
                    nn.init.constant_(submodule.weight, 1.0)
        elif isinstance(module, OneFormerTextTransformer):
            proj_std = module.width ** (-0.5) * (2 * module.num_layers) ** (-0.5)
            attn_std = module.width ** (-0.5)
            fc_std = (2 * module.width) ** (-0.5)
            for layer in module.layers:
                nn.init.normal_(layer.self_attn.in_proj_weight, std=attn_std)
                nn.init.normal_(layer.self_attn.out_proj.weight, std=proj_std)
                nn.init.normal_(layer.mlp.fc1.weight, std=fc_std)
                nn.init.normal_(layer.mlp.fc2.weight, std=proj_std)
        elif isinstance(module, OneFormerTextEncoder):
            nn.init.normal_(module.token_embedding.weight, std=0.02)
            nn.init.normal_(module.positional_embedding, std=0.01)
        if hasattr(module, 'reference_points'):
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        elif isinstance(module, OneFormerTaskModel):
            for submodule in module.modules():
                if isinstance(module, OneFormerMLPPredictionHead):
                    for submodule in module.modules():
                        if isinstance(submodule, nn.Linear):
                            nn.init.xavier_uniform_(submodule.weight, gain=xavier_std)
                            nn.init.constant_(submodule.bias, 0)
                        elif isinstance(module, nn.LayerNorm):
                            module.bias.data.zero_()
                            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=std)
            module.in_proj_bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

@add_start_docstrings('The bare OneFormer Model outputting raw hidden-states without any specific head on top.', ONEFORMER_START_DOCSTRING)
class OneFormerModel(OneFormerPreTrainedModel):
    main_input_name = ['pixel_values', 'task_inputs']

    def __init__(self, config: OneFormerConfig):
        if False:
            return 10
        super().__init__(config)
        self.pixel_level_module = OneFormerPixelLevelModule(config)
        self.transformer_module = OneFormerTransformerModule(in_features=config.conv_dim, config=config)
        self.task_encoder = OneFormerTaskModel(config)
        self.is_training = config.is_training
        if self.is_training:
            self.text_mapper = OneFormerTextMapper(config)
        else:
            self.text_mapper = None
        self.post_init()

    @add_start_docstrings_to_model_forward(ONEFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OneFormerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, task_inputs: Tensor, text_inputs: Optional[Tensor]=None, pixel_mask: Optional[Tensor]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> OneFormerModelOutput:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n            `OneFormerModelOutput`\n        Example:\n\n        ```python\n        >>> import torch\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import OneFormerProcessor, OneFormerModel\n\n        >>> # download texting image\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> # load processor for preprocessing the inputs\n        >>> processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")\n        >>> model = OneFormerModel.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")\n        >>> inputs = processor(image, ["semantic"], return_tensors="pt")\n\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n\n        >>> mask_predictions = outputs.transformer_decoder_mask_predictions\n        >>> class_predictions = outputs.transformer_decoder_class_predictions\n\n        >>> f"👉 Mask Predictions Shape: {list(mask_predictions.shape)}, Class Predictions Shape: {list(class_predictions.shape)}"\n        \'👉 Mask Predictions Shape: [1, 150, 128, 171], Class Predictions Shape: [1, 150, 151]\'\n        ```'
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        (batch_size, _, height, width) = pixel_values.shape
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)
        pixel_level_module_output = self.pixel_level_module(pixel_values, output_hidden_states)
        multi_scale_features = pixel_level_module_output.decoder_features
        mask_features = pixel_level_module_output.decoder_last_feature
        task_token = self.task_encoder(task_inputs.to(self.dtype))
        if self.is_training:
            text_queries = self.text_mapper(text_inputs)
        else:
            text_queries = None
        transformer_module_output = self.transformer_module(multi_scale_features=multi_scale_features, mask_features=mask_features, task_token=task_token, output_attentions=output_attentions)
        queries = transformer_module_output.object_queries
        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_features
            pixel_decoder_hidden_states = (pixel_level_module_output.decoder_last_feature,)
            for f in pixel_level_module_output.decoder_features:
                pixel_decoder_hidden_states += (f,)
            transformer_decoder_hidden_states = transformer_module_output.auxiliary_predictions
        output = OneFormerModelOutput(encoder_hidden_states=encoder_hidden_states, pixel_decoder_hidden_states=pixel_decoder_hidden_states, transformer_decoder_hidden_states=transformer_decoder_hidden_states, transformer_decoder_object_queries=queries, transformer_decoder_contrastive_queries=transformer_module_output.contrastive_logits, transformer_decoder_mask_predictions=transformer_module_output.prediction_masks, transformer_decoder_class_predictions=transformer_module_output.prediction_class, transformer_decoder_auxiliary_predictions=transformer_module_output.auxiliary_predictions, text_queries=text_queries, task_token=task_token, attentions=transformer_module_output.attentions)
        if not return_dict:
            output = tuple((v for v in output.values()))
        return output

@add_start_docstrings('OneFormer Model for instance, semantic and panoptic image segmentation.', ONEFORMER_START_DOCSTRING)
class OneFormerForUniversalSegmentation(OneFormerPreTrainedModel):
    main_input_name = ['pixel_values', 'task_inputs']

    def __init__(self, config: OneFormerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.model = OneFormerModel(config)
        self.matcher = OneFormerHungarianMatcher(cost_class=config.class_weight, cost_dice=config.dice_weight, cost_mask=config.mask_weight, num_points=config.train_num_points)
        self.weight_dict: Dict[str, float] = {'loss_cross_entropy': config.class_weight, 'loss_mask': config.mask_weight, 'loss_dice': config.dice_weight, 'loss_contrastive': config.contrastive_weight}
        self.criterion = OneFormerLoss(num_classes=config.num_labels, matcher=self.matcher, weight_dict=self.weight_dict, eos_coef=config.no_object_weight, num_points=config.train_num_points, oversample_ratio=config.oversample_ratio, importance_sample_ratio=config.importance_sample_ratio, contrastive_temperature=config.contrastive_temperature)
        self.post_init()

    def get_loss_dict(self, masks_queries_logits: Tensor, class_queries_logits: Tensor, contrastive_queries_logits: Tensor, mask_labels: Tensor, class_labels: Tensor, text_queries: Tensor, auxiliary_predictions: Dict[str, Tensor], calculate_contrastive_loss: bool) -> Dict[str, Tensor]:
        if False:
            print('Hello World!')
        loss_dict: Dict[str, Tensor] = self.criterion(masks_queries_logits=masks_queries_logits, class_queries_logits=class_queries_logits, contrastive_queries_logits=contrastive_queries_logits, mask_labels=mask_labels, class_labels=class_labels, text_queries=text_queries, auxiliary_predictions=auxiliary_predictions, calculate_contrastive_loss=calculate_contrastive_loss)
        for (key, weight) in self.weight_dict.items():
            for (loss_key, loss) in loss_dict.items():
                if key in loss_key:
                    loss *= weight
        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        return sum(loss_dict.values())

    @add_start_docstrings_to_model_forward(ONEFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OneFormerForUniversalSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, task_inputs: Tensor, text_inputs: Optional[Tensor]=None, mask_labels: Optional[List[Tensor]]=None, class_labels: Optional[List[Tensor]]=None, pixel_mask: Optional[Tensor]=None, output_auxiliary_logits: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> OneFormerForUniversalSegmentationOutput:
        if False:
            for i in range(10):
                print('nop')
        '\n        text_inputs (`List[torch.Tensor]`, *optional*):\n            Tensor fof shape `(num_queries, sequence_length)` to be fed to a model\n        mask_labels (`List[torch.Tensor]`, *optional*):\n            List of mask labels of shape `(num_labels, height, width)` to be fed to a model\n        class_labels (`List[torch.LongTensor]`, *optional*):\n            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the\n            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.\n\n        Returns:\n            `OneFormerUniversalSegmentationOutput`\n        Example:\n\n        Universal segmentation example:\n\n        ```python\n        >>> from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation\n        >>> from PIL import Image\n        >>> import requests\n        >>> import torch\n\n        >>> # load OneFormer fine-tuned on ADE20k for universal segmentation\n        >>> processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")\n        >>> model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")\n\n        >>> url = (\n        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"\n        ... )\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> # Semantic Segmentation\n        >>> inputs = processor(image, ["semantic"], return_tensors="pt")\n\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`\n        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`\n        >>> class_queries_logits = outputs.class_queries_logits\n        >>> masks_queries_logits = outputs.masks_queries_logits\n\n        >>> # you can pass them to processor for semantic postprocessing\n        >>> predicted_semantic_map = processor.post_process_semantic_segmentation(\n        ...     outputs, target_sizes=[image.size[::-1]]\n        ... )[0]\n        >>> f"👉 Semantic Predictions Shape: {list(predicted_semantic_map.shape)}"\n        \'👉 Semantic Predictions Shape: [512, 683]\'\n\n        >>> # Instance Segmentation\n        >>> inputs = processor(image, ["instance"], return_tensors="pt")\n\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`\n        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`\n        >>> class_queries_logits = outputs.class_queries_logits\n        >>> masks_queries_logits = outputs.masks_queries_logits\n\n        >>> # you can pass them to processor for instance postprocessing\n        >>> predicted_instance_map = processor.post_process_instance_segmentation(\n        ...     outputs, target_sizes=[image.size[::-1]]\n        ... )[0]["segmentation"]\n        >>> f"👉 Instance Predictions Shape: {list(predicted_instance_map.shape)}"\n        \'👉 Instance Predictions Shape: [512, 683]\'\n\n        >>> # Panoptic Segmentation\n        >>> inputs = processor(image, ["panoptic"], return_tensors="pt")\n\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`\n        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`\n        >>> class_queries_logits = outputs.class_queries_logits\n        >>> masks_queries_logits = outputs.masks_queries_logits\n\n        >>> # you can pass them to processor for panoptic postprocessing\n        >>> predicted_panoptic_map = processor.post_process_panoptic_segmentation(\n        ...     outputs, target_sizes=[image.size[::-1]]\n        ... )[0]["segmentation"]\n        >>> f"👉 Panoptic Predictions Shape: {list(predicted_panoptic_map.shape)}"\n        \'👉 Panoptic Predictions Shape: [512, 683]\'\n        ```\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(pixel_values=pixel_values, task_inputs=task_inputs, text_inputs=text_inputs, pixel_mask=pixel_mask, output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss, output_attentions=output_attentions, return_dict=True)
        (loss, loss_dict, auxiliary_predictions) = (None, None, None)
        class_queries_logits = outputs.transformer_decoder_class_predictions
        masks_queries_logits = outputs.transformer_decoder_mask_predictions
        contrastive_queries_logits = outputs.transformer_decoder_contrastive_queries
        auxiliary_predictions = outputs.transformer_decoder_auxiliary_predictions
        text_queries = outputs.text_queries
        if mask_labels is not None and class_labels is not None:
            loss_dict: Dict[str, Tensor] = self.get_loss_dict(masks_queries_logits=masks_queries_logits, class_queries_logits=class_queries_logits, contrastive_queries_logits=contrastive_queries_logits, mask_labels=mask_labels, class_labels=class_labels, text_queries=text_queries, auxiliary_predictions=auxiliary_predictions, calculate_contrastive_loss=self.config.contrastive_temperature is not None)
            loss = self.get_loss(loss_dict)
        output_auxiliary_logits = self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        if not output_auxiliary_logits:
            auxiliary_predictions = None
        output = OneFormerForUniversalSegmentationOutput(class_queries_logits=class_queries_logits, masks_queries_logits=masks_queries_logits, auxiliary_predictions=auxiliary_predictions, loss=loss, **outputs)
        if not return_dict:
            output = tuple((v for v in output.values()))
            if loss is not None:
                output = loss + output
        return output