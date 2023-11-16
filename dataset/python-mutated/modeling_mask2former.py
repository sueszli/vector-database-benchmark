""" PyTorch Mask2Former model."""
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ... import AutoBackbone
from ...activations import ACT2FN
from ...file_utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, is_scipy_available, replace_return_docstrings, requires_backends
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_mask2former import Mask2FormerConfig
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'Mask2FormerConfig'
_CHECKPOINT_FOR_DOC = 'facebook/mask2former-swin-small-coco-instance'
_IMAGE_PROCESSOR_FOR_DOC = 'Mask2FormerImageProcessor'
MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/mask2former-swin-small-coco-instance']

@dataclass
class Mask2FormerPixelDecoderOutput(ModelOutput):
    """
    Mask2Former's pixel decoder module output, practically a Multi-Scale Deformable Attention based decoder. It returns
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
class Mask2FormerMaskedAttentionDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the Transformer decoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions for mask predictions logits and a tuple of intermediate decoder activations,
    i.e. the output of each decoder layer, each of them gone through a layernorm.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs. Returned when `output_hidden_states=True`.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads. Returned when `output_attentions=True`.
        masks_queries_logits (`tuple(torch.FloatTensor)` of shape `(batch_size, num_queries, height, width)`):
            Tuple of mask predictions from all layers of the transformer decoder.
        intermediate_hidden_states (`tuple(torch.FloatTensor)` of shape `(num_queries, 1, hidden_size)`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[torch.FloatTensor] = None
    masks_queries_logits: Tuple[torch.FloatTensor] = None
    intermediate_hidden_states: Tuple[torch.FloatTensor] = None

@dataclass
class Mask2FormerPixelLevelModuleOutput(ModelOutput):
    """
    Mask2Former's pixel level module output. It returns the output of the encoder (optional) and all hidden states
    (multi-scale features) from the `decoder`. By default, the `encoder` is a Swin Backbone and the `decoder` is a
    Multi-Scale Deformable Attention based decoder.

    The `decoder_last_hidden_state` are the **per-pixel embeddings** while `decoder_hidden_states` refer to multi-scale
    feature maps produced using **multi-scaling strategy** defined in the paper.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor`):
            Last hidden states (final feature map of shape `(batch_size, num_channels, height, width)`) of the last
            stage of the encoder.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden states (also
            called feature maps) of the model at the output of each stage. Returned if output_hidden_states is set to
            True.
        decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)):
            1/4 scale features from the last Pixel Decoder Layer.
        decoder_hidden_states (`tuple(torch.FloatTensor)`):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden states (also
            called feature maps) of the model at the output of each stage.
    """
    encoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_last_hidden_state: torch.FloatTensor = None
    decoder_hidden_states: Tuple[torch.FloatTensor] = None

@dataclass
class Mask2FormerModelOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone). Returned when
            `output_hidden_states=True` is passed.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage. Returned when `output_hidden_states=True` is passed.
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Last hidden states (final feature map) of the last stage of the pixel decoder model.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, , *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage. Returned when `output_hidden_states=True` is passed.
        transformer_decoder_last_hidden_state (`tuple(torch.FloatTensor)`):
            Final output of the transformer decoder `(batch_size, sequence_length, hidden_size)`.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage. Returned when `output_hidden_states=True` is passed.
        transformer_decoder_intermediate_states (`tuple(torch.FloatTensor)` of shape `(num_queries, 1, hidden_size)`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
        masks_queries_logits (`tuple(torch.FloatTensor)` of shape `(batch_size, num_queries, height, width)`)
            Mask Predictions from each layer in the transformer decoder.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self attentions weights from transformer decoder.
    """
    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_intermediate_states: Tuple[torch.FloatTensor] = None
    masks_queries_logits: Tuple[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class Mask2FormerForUniversalSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerForUniversalSegmentationOutput`].

    This output can be directly passed to [`~Mask2FormerImageProcessor.post_process_semantic_segmentation`] or
    [`~Mask2FormerImageProcessor.post_process_instance_segmentation`] or
    [`~Mask2FormerImageProcessor.post_process_panoptic_segmentation`] to compute final segmentation maps. Please, see
    [`~Mask2FormerImageProcessor] for details regarding usage.

    Args:
        loss (`torch.Tensor`, *optional*):
            The computed loss, returned when labels are present.
        class_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        masks_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
        auxiliary_logits (`List[Dict(str, torch.FloatTensor)]`, *optional*):
            List of class and mask predictions from each layer of the transformer decoder.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_last_hidden_state (`tuple(torch.FloatTensor)`):
            Final output of the transformer decoder `(batch_size, sequence_length, hidden_size)`.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self and Cross Attentions weights from transformer decoder.
    """
    loss: Optional[torch.FloatTensor] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_logits: Optional[List[Dict[str, torch.FloatTensor]]] = None
    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

def sample_point(input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim=False, **kwargs) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    '\n    A wrapper around `torch.nn.functional.grid_sample` to support 3D point_coordinates tensors.\n\n    Args:\n        input_features (`torch.Tensor` of shape (batch_size, channels, height, width)):\n            A tensor that contains features map on a height * width grid\n        point_coordinates (`torch.Tensor` of shape (batch_size, num_points, 2) or (batch_size, grid_height, grid_width,:\n        2)):\n            A tensor that contains [0, 1] * [0, 1] normalized point coordinates\n        add_dim (`bool`):\n            boolean value to keep track of added dimension\n\n    Returns:\n        point_features (`torch.Tensor` of shape (batch_size, channels, num_points) or (batch_size, channels,\n        height_grid, width_grid):\n            A tensor that contains features for points in `point_coordinates`.\n    '
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)
    point_features = torch.nn.functional.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    if add_dim:
        point_features = point_features.squeeze(3)
    return point_features

def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
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
        return 10
    '\n    A pair wise version of the dice loss, see `dice_loss` for usage.\n\n    Args:\n        inputs (`torch.Tensor`):\n            A tensor representing a mask\n        labels (`torch.Tensor`):\n            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs\n            (0 for the negative class and 1 for the positive class).\n\n    Returns:\n        `torch.Tensor`: The computed loss between each pairs.\n    '
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
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

class Mask2FormerHungarianMatcher(nn.Module):
    """This class computes an assignment between the labels and the predictions of the network.

    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float=1.0, cost_mask: float=1.0, cost_dice: float=1.0, num_points: int=12544):
        if False:
            print('Hello World!')
        'Creates the matcher\n\n        Params:\n            cost_class (`float`, *optional*, defaults to 1.0):\n                Relative weight of the classification error in the matching cost.\n            cost_mask (`float`, *optional*,  defaults to 1.0):\n                This is the relative weight of the focal loss of the binary mask in the matching cost.\n            cost_dice (`float`, *optional*, defaults to 1.0):\n                This is the relative weight of the dice loss of the binary mask in the matching cost.\n            num_points (`int`, *optional*, defaults to 12544):\n                No. of points to sample on which the mask loss will be calculated. The same set of K points are\n                uniformly sampled for all prediction and ground truth masks to construct the cost matrix for bipartite\n                matching.\n        '
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and (cost_dice == 0):
            raise ValueError('All costs cant be 0')
        self.num_points = num_points
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, masks_queries_logits: torch.Tensor, class_queries_logits: torch.Tensor, mask_labels: torch.Tensor, class_labels: torch.Tensor) -> List[Tuple[Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Params:\n            masks_queries_logits (`torch.Tensor`):\n                A tensor of dim `batch_size, num_queries, num_labels` with the classification logits.\n            class_queries_logits (`torch.Tensor`):\n                A tensor of dim `batch_size, num_queries, height, width` with the predicted masks.\n            class_labels (`torch.Tensor`):\n                A tensor of dim `num_target_boxes` (where num_target_boxes is the number of ground-truth objects in the\n                target) containing the class labels.\n            mask_labels (`torch.Tensor`):\n                A tensor of dim `num_target_boxes, height, width` containing the target masks.\n\n        Returns:\n            matched_indices (`List[Tuple[Tensor]]`): A list of size batch_size, containing tuples of (index_i, index_j)\n            where:\n                - index_i is the indices of the selected predictions (in order)\n                - index_j is the indices of the corresponding selected labels (in order)\n            For each batch element, it holds:\n                len(index_i) = len(index_j) = min(num_queries, num_target_boxes).\n        '
        indices: List[Tuple[np.array]] = []
        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            pred_probs = class_queries_logits[i].softmax(-1)
            pred_mask = masks_queries_logits[i]
            cost_class = -pred_probs[:, class_labels[i]]
            target_mask = mask_labels[i].to(pred_mask)
            target_mask = target_mask[:, None]
            pred_mask = pred_mask[:, None]
            point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)
            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            target_mask = sample_point(target_mask, target_coordinates, align_corners=False).squeeze(1)
            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            pred_mask = sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)
            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix.cpu())
            indices.append(assigned_indices)
        matched_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for (i, j) in indices]
        return matched_indices

class Mask2FormerLoss(nn.Module):

    def __init__(self, config: Mask2FormerConfig, weight_dict: Dict[str, float]):
        if False:
            while True:
                i = 10
        '\n        The Mask2Former Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we\n        compute hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair\n        of matched ground-truth / prediction (supervise class and mask)\n\n        Args:\n            config (`Mask2FormerConfig`):\n                The configuration for Mask2Former model also containing loss calculation specific parameters.\n            weight_dict (`Dict[str, float]`):\n                A dictionary of weights to be applied to the different losses.\n        '
        super().__init__()
        requires_backends(self, ['scipy'])
        self.num_labels = config.num_labels
        self.weight_dict = weight_dict
        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.num_points = config.train_num_points
        self.oversample_ratio = config.oversample_ratio
        self.importance_sample_ratio = config.importance_sample_ratio
        self.matcher = Mask2FormerHungarianMatcher(cost_class=1.0, cost_dice=config.dice_weight, cost_mask=config.mask_weight, num_points=self.num_points)

    def _max_by_axis(self, sizes: List[List[int]]) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for (index, item) in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        if False:
            return 10
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_shape = [len(tensors)] + max_size
        (batch_size, _, height, width) = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for (tensor, padded_tensor, padding_mask) in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[:tensor.shape[0], :tensor.shape[1], :tensor.shape[2]].copy_(tensor)
            padding_mask[:tensor.shape[1], :tensor.shape[2]] = False
        return (padded_tensors, padding_masks)

    def loss_labels(self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]) -> Dict[str, Tensor]:
        if False:
            print('Hello World!')
        'Compute the losses related to the labels using cross entropy.\n\n        Args:\n            class_queries_logits (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, num_labels`\n            class_labels (`List[torch.Tensor]`):\n                List of class labels of shape `(labels)`.\n            indices (`Tuple[np.array])`:\n                The indices computed by the Hungarian matcher.\n\n        Returns:\n            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:\n            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.\n        '
        pred_logits = class_queries_logits
        (batch_size, num_queries, _) = pred_logits.shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)
        target_classes_o = torch.cat([target[j] for (target, (_, j)) in zip(class_labels, indices)])
        target_classes = torch.full((batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o
        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {'loss_cross_entropy': loss_ce}
        return losses

    def loss_masks(self, masks_queries_logits: torch.Tensor, mask_labels: List[torch.Tensor], indices: Tuple[np.array], num_masks: int) -> Dict[str, torch.Tensor]:
        if False:
            return 10
        'Compute the losses related to the masks using sigmoid_cross_entropy_loss and dice loss.\n\n        Args:\n            masks_queries_logits (`torch.Tensor`):\n                A tensor of shape `(batch_size, num_queries, height, width)`.\n            mask_labels (`torch.Tensor`):\n                List of mask labels of shape `(labels, height, width)`.\n            indices (`Tuple[np.array])`:\n                The indices computed by the Hungarian matcher.\n            num_masks (`int)`:\n                The number of masks, used for normalization.\n\n        Returns:\n            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing two keys:\n            - **loss_mask** -- The loss computed using sigmoid cross entropy loss on the predicted and ground truth.\n              masks.\n            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth,\n              masks.\n        '
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        pred_masks = masks_queries_logits[src_idx]
        (target_masks, _) = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(pred_masks, lambda logits: self.calculate_uncertainty(logits), self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)
        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)
        losses = {'loss_mask': sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks), 'loss_dice': dice_loss(point_logits, point_labels, num_masks)}
        del pred_masks
        del target_masks
        return losses

    def _get_predictions_permutation_indices(self, indices):
        if False:
            while True:
                i = 10
        batch_indices = torch.cat([torch.full_like(src, i) for (i, (src, _)) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return (batch_indices, predictions_indices)

    def _get_targets_permutation_indices(self, indices):
        if False:
            while True:
                i = 10
        batch_indices = torch.cat([torch.full_like(tgt, i) for (i, (_, tgt)) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return (batch_indices, target_indices)

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        "\n        In Mask2Former paper, uncertainty is estimated as L1 distance between 0.0 and the logit prediction in 'logits'\n        for the foreground class in `classes`.\n\n        Args:\n            logits (`torch.Tensor`):\n            A tensor of shape (R, 1, ...) for class-specific or class-agnostic, where R is the total number of predicted masks in all images and C is:\n            the number of foreground classes. The values are logits.\n\n        Returns:\n            scores (`torch.Tensor`): A tensor of shape (R, 1, ...) that contains uncertainty scores with the most\n            uncertain locations having the highest uncertainty score.\n        "
        uncertainty_scores = -torch.abs(logits)
        return uncertainty_scores

    def sample_points_using_uncertainty(self, logits: torch.Tensor, uncertainty_function, num_points: int, oversample_ratio: int, importance_sample_ratio: float) -> torch.Tensor:
        if False:
            print('Hello World!')
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

    def forward(self, masks_queries_logits: torch.Tensor, class_queries_logits: torch.Tensor, mask_labels: List[torch.Tensor], class_labels: List[torch.Tensor], auxiliary_predictions: Optional[Dict[str, torch.Tensor]]=None) -> Dict[str, torch.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        '\n        This performs the loss computation.\n\n        Args:\n            masks_queries_logits (`torch.Tensor`):\n                A tensor of shape `(batch_size, num_queries, height, width)`.\n            class_queries_logits (`torch.Tensor`):\n                A tensor of shape `(batch_size, num_queries, num_labels)`.\n            mask_labels (`torch.Tensor`):\n                List of mask labels of shape `(labels, height, width)`.\n            class_labels (`List[torch.Tensor]`):\n                List of class labels of shape `(labels)`.\n            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):\n                if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], then it contains the logits from\n                the inner layers of the Mask2FormerMaskedAttentionDecoder.\n\n        Returns:\n            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing three keys:\n            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.\n            - **loss_mask** -- The loss computed using sigmoid cross_entropy loss on the predicted and ground truth\n              masks.\n            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth\n              masks.\n            if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], the dictionary contains additional\n            losses for each auxiliary predictions.\n        '
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        losses: Dict[str, Tensor] = {**self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks), **self.loss_labels(class_queries_logits, class_labels, indices)}
        if auxiliary_predictions is not None:
            for (idx, aux_outputs) in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs['masks_queries_logits']
                class_queries_logits = aux_outputs['class_queries_logits']
                loss_dict = self.forward(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
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

def multi_scale_deformable_attention(value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor) -> Tensor:
    if False:
        return 10
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

class Mask2FormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats: int=64, temperature: int=10000, normalize: bool=False, scale: Optional[float]=None):
        if False:
            return 10
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
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

class Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Module):
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
            return 10
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

class Mask2FormerPixelDecoderEncoderLayer(nn.Module):

    def __init__(self, config: Mask2FormerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.embed_dim = config.feature_size
        self.self_attn = Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, n_levels=3, n_points=4)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = nn.functional.relu
        self.activation_dropout = config.dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_feedforward_dim)
        self.fc2 = nn.Linear(config.encoder_feedforward_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, position_embeddings: torch.Tensor=None, reference_points=None, spatial_shapes=None, level_start_index=None, output_attentions: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Input to the layer.\n            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):\n                Attention mask.\n            position_embeddings (`torch.FloatTensor`, *optional*):\n                Position embeddings, to be added to `hidden_states`.\n            reference_points (`torch.FloatTensor`, *optional*):\n                Reference points.\n            spatial_shapes (`torch.LongTensor`, *optional*):\n                Spatial shapes of the backbone feature maps.\n            level_start_index (`torch.LongTensor`, *optional*):\n                Level start index.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        (hidden_states, attn_weights) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, position_embeddings=position_embeddings, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights.transpose(1, 0),)
        return outputs

class Mask2FormerPixelDecoderEncoderOnly(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`Mask2FormerPixelDecoderEncoderLayer`]. The encoder updates the flattened multi-scale feature maps through
    multiple deformable attention layers.

    Args:
        config: Mask2FormerConfig
    """

    def __init__(self, config: Mask2FormerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layers = nn.ModuleList([Mask2FormerPixelDecoderEncoderLayer(config) for _ in range(config.encoder_layers)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        if False:
            i = 10
            return i + 15
        '\n        Get reference points for each feature map. Used in decoder.\n\n        Args:\n            spatial_shapes (`torch.LongTensor`):\n                Spatial shapes of each feature map, has shape of `(num_feature_levels, 2)`.\n            valid_ratios (`torch.FloatTensor`):\n                Valid ratios of each feature map, has shape of `(batch_size, num_feature_levels, 2)`.\n            device (`torch.device`):\n                Device on which to create the tensors.\n        Returns:\n            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`\n        '
        reference_points_list = []
        for (lvl, (height, width)) in enumerate(spatial_shapes):
            (ref_y, ref_x) = torch.meshgrid(torch.linspace(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device), torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device), indexing='ij')
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
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for (i, encoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states.transpose(1, 0),)
            layer_outputs = encoder_layer(hidden_states, attention_mask, position_embeddings=position_embeddings, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states += (hidden_states.transpose(1, 0),)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

class Mask2FormerPixelDecoder(nn.Module):

    def __init__(self, config: Mask2FormerConfig, feature_channels):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        feature_dim = config.feature_size
        mask_dim = config.mask_feature_size
        num_pos_features = feature_dim // 2
        self.position_embedding = Mask2FormerSinePositionEmbedding(num_pos_feats=num_pos_features, normalize=True)
        self.num_feature_levels = 3
        transformer_in_channels = feature_channels[-self.num_feature_levels:]
        self.transformer_feature_strides = config.feature_strides[-self.num_feature_levels:]
        self.feature_channels = feature_channels
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, feature_dim))
        if self.num_feature_levels > 1:
            input_projections_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_projections_list.append(nn.Sequential(nn.Conv2d(in_channels, feature_dim, kernel_size=1), nn.GroupNorm(32, feature_dim)))
            self.input_projections = nn.ModuleList(input_projections_list)
        else:
            self.input_projections = nn.ModuleList([nn.Sequential(nn.Conv2d(transformer_in_channels[-1], feature_dim, kernel_size=1), nn.GroupNorm(32, feature_dim))])
        self.encoder = Mask2FormerPixelDecoderEncoderOnly(config)
        self.mask_projection = nn.Conv2d(feature_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        stride = min(self.transformer_feature_strides)
        self.common_stride = config.common_stride
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs = []
        output_convs = []
        for (idx, in_channels) in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_conv = nn.Sequential(nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False), nn.GroupNorm(32, feature_dim))
            output_conv = nn.Sequential(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=False), nn.GroupNorm(32, feature_dim), nn.ReLU())
            self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
            self.add_module('layer_{}'.format(idx + 1), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convolutions = lateral_convs[::-1]
        self.output_convolutions = output_convs[::-1]

    def get_valid_ratio(self, mask, dtype=torch.float32):
        if False:
            print('Hello World!')
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
            i = 10
            return i + 15
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        input_embeds = []
        position_embeddings = []
        for (level, x) in enumerate(features[::-1][:self.num_feature_levels]):
            input_embeds.append(self.input_projections[level](x))
            position_embeddings.append(self.position_embedding(x))
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in input_embeds]
        spatial_shapes = [(embed.shape[2], embed.shape[3]) for embed in input_embeds]
        input_embeds_flat = torch.cat([embed.flatten(2).transpose(1, 2) for embed in input_embeds], 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=input_embeds_flat.device)
        masks_flat = torch.cat([mask.flatten(1) for mask in masks], 1)
        position_embeddings = [embed.flatten(2).transpose(1, 2) for embed in position_embeddings]
        level_pos_embed_flat = [x + self.level_embed[i].view(1, 1, -1) for (i, x) in enumerate(position_embeddings)]
        level_pos_embed_flat = torch.cat(level_pos_embed_flat, 1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(mask, dtype=input_embeds_flat.dtype) for mask in masks], 1)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(inputs_embeds=input_embeds_flat, attention_mask=masks_flat, position_embeddings=level_pos_embed_flat, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs.last_hidden_state
        batch_size = last_hidden_state.shape[0]
        split_sizes = [None] * self.num_feature_levels
        for i in range(self.num_feature_levels):
            if i < self.num_feature_levels - 1:
                split_sizes[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_sizes[i] = last_hidden_state.shape[1] - level_start_index[i]
        encoder_output = torch.split(last_hidden_state, [size.item() for size in split_sizes], dim=1)
        outputs = [x.transpose(1, 2).view(batch_size, -1, spatial_shapes[i][0], spatial_shapes[i][1]) for (i, x) in enumerate(encoder_output)]
        for (idx, feature) in enumerate(features[:self.num_fpn_levels][::-1]):
            lateral_conv = self.lateral_convolutions[idx]
            output_conv = self.output_convolutions[idx]
            current_fpn = lateral_conv(feature)
            out = current_fpn + nn.functional.interpolate(outputs[-1], size=current_fpn.shape[-2:], mode='bilinear', align_corners=False)
            out = output_conv(out)
            outputs.append(out)
        num_cur_levels = 0
        multi_scale_features = []
        for out in outputs:
            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(out)
                num_cur_levels += 1
        return Mask2FormerPixelDecoderOutput(mask_features=self.mask_projection(outputs[-1]), multi_scale_features=tuple(multi_scale_features), attentions=encoder_outputs.attentions)

class Mask2FormerPixelLevelModule(nn.Module):

    def __init__(self, config: Mask2FormerConfig):
        if False:
            for i in range(10):
                print('nop')
        '\n        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image\n        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel\n        decoder, generating multi-scale feature maps and pixel embeddings.\n\n        Args:\n            config ([`Mask2FormerConfig`]):\n                The configuration used to instantiate this model.\n        '
        super().__init__()
        self.encoder = AutoBackbone.from_config(config.backbone_config)
        self.decoder = Mask2FormerPixelDecoder(config, feature_channels=self.encoder.channels)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool=False) -> Mask2FormerPixelLevelModuleOutput:
        if False:
            while True:
                i = 10
        backbone_features = self.encoder(pixel_values).feature_maps
        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)
        return Mask2FormerPixelLevelModuleOutput(encoder_last_hidden_state=backbone_features[-1], encoder_hidden_states=tuple(backbone_features) if output_hidden_states else None, decoder_last_hidden_state=decoder_output.mask_features, decoder_hidden_states=decoder_output.multi_scale_features)

class Mask2FormerAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Here, we add position embeddings to the queries and
    keys (as explained in the DETR paper).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True):
        if False:
            print('Hello World!')
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
            print('Hello World!')
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        if False:
            print('Hello World!')
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_embeddings: Optional[torch.Tensor]=None, key_value_states: Optional[torch.Tensor]=None, key_value_position_embeddings: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            while True:
                i = 10
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

class Mask2FormerMaskedAttentionDecoderLayer(nn.Module):
    """
    The Mask2FormerMaskedAttentionDecoderLayer is made up of self-attention, cross (masked) attention as well as FFN
    blocks. The cross attention block used as part of `Mask2FormerMaskedAttentionDecoderLayer` is actually a `masked
    attention` block that restricts the attention to localized features centered around predicted segments which leads
    to faster convergence and improved performance. The order of self and cross (i.e. masked) attention blocks have
    also been swapped in Mask2FormerMaskedAttentionDecoder compared to a standard DetrDecoder as an optimization
    improvement.

    Args:
        config (`Mask2FormerConfig`):
            The configuration used to initialize the Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config: Mask2FormerConfig):
        if False:
            return 10
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_dim
        self.pre_norm = self.config.pre_norm
        self.self_attn = Mask2FormerAttention(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, dropout=config.dropout, is_decoder=True)
        self.dropout = self.config.dropout
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout = self.config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.config.num_attention_heads, self.config.dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.config.dim_feedforward)
        self.fc2 = nn.Linear(self.config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if False:
            i = 10
            return i + 15
        return tensor if pos is None else tensor + pos

    def forward_post(self, hidden_states: torch.Tensor, level_index: int=None, attention_mask: Optional[torch.Tensor]=None, position_embeddings: Optional[torch.Tensor]=None, query_position_embeddings: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False):
        if False:
            while True:
                i = 10
        cross_attn_weights = None
        self_attn_weights = None
        residual = hidden_states
        (hidden_states, cross_attn_weights) = self.cross_attn(query=self.with_pos_embed(hidden_states, query_position_embeddings), key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]), value=encoder_hidden_states[level_index], attn_mask=encoder_attention_mask, key_padding_mask=None)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        residual = hidden_states
        (hidden_states, self_attn_weights) = self.self_attn(hidden_states=hidden_states, position_embeddings=query_position_embeddings, attention_mask=None, output_attentions=True)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

    def forward_pre(self, hidden_states: torch.Tensor, level_index: int=None, attention_mask: Optional[torch.Tensor]=None, position_embeddings: Optional[torch.Tensor]=None, query_position_embeddings: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False):
        if False:
            for i in range(10):
                print('nop')
        cross_attn_weights = None
        self_attn_weights = None
        residual = hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        (hidden_states, cross_attn_weights) = self.cross_attn(query=self.with_pos_embed(hidden_states, query_position_embeddings), key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]), value=encoder_hidden_states[level_index], attn_mask=encoder_attention_mask, key_padding_mask=None)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        (hidden_states, self_attn_weights) = self.self_attn(hidden_states=hidden_states, position_embeddings=query_position_embeddings, attention_mask=None, output_attentions=True)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

    def forward(self, hidden_states: torch.Tensor, level_index: int=None, attention_mask: Optional[torch.Tensor]=None, position_embeddings: Optional[torch.Tensor]=None, query_position_embeddings: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False):
        if False:
            return 10
        '\n        Args:\n            hidden_states (`torch.FloatTensor`):\n                Input to the layer of shape `(seq_len, batch, embed_dim)`.\n            attention_mask (`torch.FloatTensor`):\n                Attention mask of shape `(1, seq_len, tgt_len, src_len)`.\n            position_embeddings (`torch.FloatTensor`, *optional*):\n                Position embeddings that are added to the keys in the masked-attention layer.\n            query_position_embeddings (`torch.FloatTensor`, *optional*):\n                Position embeddings that are added to the queries and keys in the self-attention layer.\n            encoder_hidden_states (`torch.FloatTensor`):\n                Cross attention input to the layer of shape `(seq_len, batch, embed_dim)`.\n            encoder_attention_mask (`torch.FloatTensor`):\n                Encoder attention mask of size`(1, seq_len, tgt_len, src_len)`.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        if self.pre_norm:
            outputs = self.forward_pre(hidden_states=hidden_states, level_index=level_index, position_embeddings=position_embeddings, query_position_embeddings=query_position_embeddings, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions)
        else:
            outputs = self.forward_post(hidden_states=hidden_states, level_index=level_index, position_embeddings=position_embeddings, query_position_embeddings=query_position_embeddings, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions)
        return outputs

class Mask2FormerMaskedAttentionDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a
    [`Mask2FormerMaskedAttentionDecoderLayer`]. The decoder updates the query embeddings through multiple cross
    (masked) and self-attention layers. The decoder uses a new **masked attention** mechanism instead of the standard
    cross-attention, which extracts localized features by constraining cross-attention to within the foreground region
    of the predicted mask for each query, instead of attending to the full feature map.

    Args:
        config (`Mask2FormerConfig`):
            Configuration used to instantiate Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config: Mask2FormerConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.mask_feature_size = config.mask_feature_size
        self.dropout = config.dropout
        self.layerdrop = config.dropout
        self.num_feature_levels = 3
        self.decoder_layers = config.decoder_layers - 1
        self.layers = nn.ModuleList([Mask2FormerMaskedAttentionDecoderLayer(self.config) for _ in range(self.decoder_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_dim)
        self.mask_predictor = Mask2FormerMaskPredictor(hidden_size=config.hidden_dim, num_heads=config.num_attention_heads, mask_feature_size=self.mask_feature_size)
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds: torch.Tensor=None, multi_stage_positional_embeddings: torch.Tensor=None, pixel_embeddings: torch.Tensor=None, encoder_hidden_states: torch.Tensor=None, query_position_embeddings: torch.Tensor=None, feature_size_list: List=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            inputs_embeds (`torch.FloatTensor` of shape `(num_queries, batch_size, hidden_size)`):\n                The query embeddings that are passed into the decoder.\n            multi_stage_positional_embeddings (`torch.FloatTensor` of shape `(height*width, batch_size, num_channels)`):\n                Position embeddings that are added to the keys in each cross(masked)-attention layer.\n            pixel_embeddings (`torch.FloatTensor`):\n                Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel\n                Decoder.\n            query_position_embeddings (`torch.FloatTensor` of shape `(num_queries, batch_size, hidden_size)`):\n                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.\n            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`):\n                Sequence of hidden-states at the output of the last layer of the encoder. Used in the\n                cross(masked)-attention of the decoder.\n            feature_size_list (`List[torch.Size]` ):\n                This is a list containing shapes (height & width) of multi-scale features from the Pixel Decoder.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        intermediate = ()
        all_hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        intermediate_mask_predictions = ()
        intermediate_hidden_states = self.layernorm(inputs_embeds)
        intermediate += (intermediate_hidden_states,)
        (predicted_mask, attention_mask) = self.mask_predictor(intermediate_hidden_states, pixel_embeddings, feature_size_list[0])
        intermediate_mask_predictions += (predicted_mask,)
        for (idx, decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = torch.rand([])
            if self.training and dropout_probability < self.layerdrop:
                continue
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, attention_mask, encoder_hidden_states, None, None, output_attentions)
            else:
                level_index = idx % self.num_feature_levels
                attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False
                layer_outputs = decoder_layer(hidden_states, level_index=level_index, position_embeddings=multi_stage_positional_embeddings, query_position_embeddings=query_position_embeddings, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask, output_attentions=output_attentions)
                intermediate_hidden_states = self.layernorm(layer_outputs[0])
                (predicted_mask, attention_mask) = self.mask_predictor(intermediate_hidden_states, pixel_embeddings, feature_size_list[(idx + 1) % self.num_feature_levels])
                intermediate_mask_predictions += (predicted_mask,)
                intermediate += (intermediate_hidden_states,)
            hidden_states = layer_outputs[0]
            if output_attentions:
                attentions += (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        hidden_states = hidden_states.transpose(1, 0)
        if not return_dict:
            outputs = [hidden_states, all_hidden_states, attentions, intermediate, intermediate_mask_predictions]
            return tuple((v for v in outputs if v is not None))
        return Mask2FormerMaskedAttentionDecoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=attentions, intermediate_hidden_states=intermediate, masks_queries_logits=intermediate_mask_predictions)

class Mask2FormerPredictionBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        for (i, layer) in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        if False:
            return 10
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state

class Mask2FormerMLPPredictionHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int=3):
        if False:
            for i in range(10):
                print('nop')
        '\n        A classic Multi Layer Perceptron (MLP).\n\n        Args:\n            input_dim (`int`):\n                The input dimensions.\n            hidden_dim (`int`):\n                The hidden dimensions.\n            output_dim (`int`):\n                The output dimensions.\n            num_layers (int, *optional*, defaults to 3):\n                The number of layers.\n        '
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = []
        for (i, (in_dim, out_dim)) in enumerate(zip(in_dims, out_dims)):
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            layer = Mask2FormerPredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        if False:
            return 10
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state

class Mask2FormerMaskPredictor(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, mask_feature_size: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        '\n        This class is used to get the predicted mask for a given Mask2FormerMaskedAttentionDecoder layer. It also\n        generates the binarized attention mask associated with the given predicted mask. The attention mask obtained\n        using predicted mask of the (l-1)th decoder layer is fed to the cross(masked)-attention block of the next\n        decoder layer as input.\n\n        Args:\n            hidden_size (`int`):\n                The feature dimension of the Mask2FormerMaskedAttentionDecoder\n            num_heads (`int`):\n                The number of heads used in the Mask2FormerMaskedAttentionDecoder\n            mask_feature_size (`torch.Tensor`):\n                one of the output dimensions of the predicted masks for each query\n        '
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mask_embedder = Mask2FormerMLPPredictionHead(self.hidden_size, self.hidden_size, mask_feature_size)

    def forward(self, outputs: torch.Tensor, pixel_embeddings: torch.Tensor, attention_mask_target_size: int=None):
        if False:
            while True:
                i = 10
        mask_embeddings = self.mask_embedder(outputs.transpose(0, 1))
        (batch_size, num_queries, num_channels) = mask_embeddings.shape
        (_, _, height, width) = pixel_embeddings.shape
        outputs_mask = torch.zeros((batch_size, num_queries, height, width), device=mask_embeddings.device)
        for c in range(num_channels):
            outputs_mask += mask_embeddings[..., c][..., None, None] * pixel_embeddings[:, None, c]
        attention_mask = nn.functional.interpolate(outputs_mask, size=attention_mask_target_size, mode='bilinear', align_corners=False)
        attention_mask = attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = (attention_mask.flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()
        return (outputs_mask, attention_mask)

class Mask2FormerTransformerModule(nn.Module):
    """
    The Mask2Former's transformer module.
    """

    def __init__(self, in_features: int, config: Mask2FormerConfig):
        if False:
            print('Hello World!')
        super().__init__()
        hidden_dim = config.hidden_dim
        self.num_feature_levels = 3
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.queries_features = nn.Embedding(config.num_queries, hidden_dim)
        self.input_projections = []
        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_projection:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())
        self.decoder = Mask2FormerMaskedAttentionDecoder(config=config)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    def forward(self, multi_scale_features: List[Tensor], mask_features: Tensor, output_hidden_states: bool=False, output_attentions: bool=False) -> Mask2FormerMaskedAttentionDecoderOutput:
        if False:
            while True:
                i = 10
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
        query_features = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)
        decoder_output = self.decoder(inputs_embeds=query_features, multi_stage_positional_embeddings=multi_stage_positional_embeddings, pixel_embeddings=mask_features, encoder_hidden_states=multi_stage_features, query_position_embeddings=query_embeddings, feature_size_list=size_list, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=True)
        return decoder_output
MASK2FORMER_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`Mask2FormerConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
MASK2FORMER_INPUTS_DOCSTRING = "\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`AutoImageProcessor.preprocess`] for details.\n        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):\n            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:\n\n            - 1 for pixels that are real (i.e. **not masked**),\n            - 0 for pixels that are padding (i.e. **masked**).\n\n            [What are attention masks?](../glossary#attention-mask)\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of Detr's decoder attention layers.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~Mask2FormerModelOutput`] instead of a plain tuple.\n"

class Mask2FormerPreTrainedModel(PreTrainedModel):
    config_class = Mask2FormerConfig
    base_model_prefix = 'model'
    main_input_name = 'pixel_values'

    def _init_weights(self, module: nn.Module):
        if False:
            while True:
                i = 10
        xavier_std = self.config.init_xavier_std
        std = self.config.init_std
        if isinstance(module, Mask2FormerTransformerModule):
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        nn.init.xavier_uniform_(input_projection.weight, gain=xavier_std)
                        nn.init.constant_(input_projection.bias, 0)
        elif isinstance(module, Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention):
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
        elif isinstance(module, Mask2FormerMaskedAttentionDecoderLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)
        elif isinstance(module, Mask2FormerPixelLevelModule):
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                    submodule.weight.data.normal_(mean=0.0, std=std)
                    if submodule.bias is not None:
                        submodule.bias.data.zero_()
        elif isinstance(module, Mask2FormerPixelDecoder):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.normal_(module.level_embed, std=0)
        elif isinstance(module, Mask2FormerPixelDecoderEncoderOnly):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if hasattr(module, 'reference_points'):
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)

@add_start_docstrings('The bare Mask2Former Model outputting raw hidden-states without any specific head on top.', MASK2FORMER_START_DOCSTRING)
class Mask2FormerModel(Mask2FormerPreTrainedModel):
    main_input_name = 'pixel_values'

    def __init__(self, config: Mask2FormerConfig):
        if False:
            return 10
        super().__init__(config)
        self.pixel_level_module = Mask2FormerPixelLevelModule(config)
        self.transformer_module = Mask2FormerTransformerModule(in_features=config.feature_size, config=config)
        self.post_init()

    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Mask2FormerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, pixel_mask: Optional[Tensor]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Mask2FormerModelOutput:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n            `Mask2FormerModelOutput`\n\n        Examples:\n        ```python\n        >>> import torch\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoImageProcessor, Mask2FormerModel\n\n        >>> # load image\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> # load image preprocessor and Mask2FormerModel trained on COCO instance segmentation dataset\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")\n        >>> model = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-small-coco-instance")\n        >>> inputs = image_processor(image, return_tensors="pt")\n\n        >>> # forward pass\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n\n        >>> # model outputs last hidden states of shape (batch_size, num_queries, hidden_size)\n        >>> print(outputs.transformer_decoder_last_hidden_state.shape)\n        torch.Size([1, 100, 256])\n        ```\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        (batch_size, _, height, width) = pixel_values.shape
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)
        pixel_level_module_output = self.pixel_level_module(pixel_values=pixel_values, output_hidden_states=output_hidden_states)
        transformer_module_output = self.transformer_module(multi_scale_features=pixel_level_module_output.decoder_hidden_states, mask_features=pixel_level_module_output.decoder_last_hidden_state, output_hidden_states=True, output_attentions=output_attentions)
        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        transformer_decoder_intermediate_states = None
        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_hidden_states
            pixel_decoder_hidden_states = pixel_level_module_output.decoder_hidden_states
            transformer_decoder_hidden_states = transformer_module_output.hidden_states
            transformer_decoder_intermediate_states = transformer_module_output.intermediate_hidden_states
        output = Mask2FormerModelOutput(encoder_last_hidden_state=pixel_level_module_output.encoder_last_hidden_state, pixel_decoder_last_hidden_state=pixel_level_module_output.decoder_last_hidden_state, transformer_decoder_last_hidden_state=transformer_module_output.last_hidden_state, encoder_hidden_states=encoder_hidden_states, pixel_decoder_hidden_states=pixel_decoder_hidden_states, transformer_decoder_hidden_states=transformer_decoder_hidden_states, transformer_decoder_intermediate_states=transformer_decoder_intermediate_states, attentions=transformer_module_output.attentions, masks_queries_logits=transformer_module_output.masks_queries_logits)
        if not return_dict:
            output = tuple((v for v in output.values() if v is not None))
        return output

@add_start_docstrings('The Mask2Former Model with heads on top for instance/semantic/panoptic segmentation.', MASK2FORMER_START_DOCSTRING)
class Mask2FormerForUniversalSegmentation(Mask2FormerPreTrainedModel):
    main_input_name = 'pixel_values'

    def __init__(self, config: Mask2FormerConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.model = Mask2FormerModel(config)
        self.weight_dict: Dict[str, float] = {'loss_cross_entropy': config.class_weight, 'loss_mask': config.mask_weight, 'loss_dice': config.dice_weight}
        self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)
        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)
        self.post_init()

    def get_loss_dict(self, masks_queries_logits: Tensor, class_queries_logits: Tensor, mask_labels: Tensor, class_labels: Tensor, auxiliary_predictions: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if False:
            return 10
        loss_dict: Dict[str, Tensor] = self.criterion(masks_queries_logits=masks_queries_logits, class_queries_logits=class_queries_logits, mask_labels=mask_labels, class_labels=class_labels, auxiliary_predictions=auxiliary_predictions)
        for (key, weight) in self.weight_dict.items():
            for (loss_key, loss) in loss_dict.items():
                if key in loss_key:
                    loss *= weight
        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        if False:
            print('Hello World!')
        return sum(loss_dict.values())

    def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        auxiliary_logits: List[Dict(str, Tensor)] = []
        for (aux_binary_masks, aux_classes) in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append({'masks_queries_logits': aux_binary_masks, 'class_queries_logits': aux_classes})
        return auxiliary_logits

    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Mask2FormerForUniversalSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, mask_labels: Optional[List[Tensor]]=None, class_labels: Optional[List[Tensor]]=None, pixel_mask: Optional[Tensor]=None, output_hidden_states: Optional[bool]=None, output_auxiliary_logits: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Mask2FormerForUniversalSegmentationOutput:
        if False:
            return 10
        '\n        mask_labels (`List[torch.Tensor]`, *optional*):\n            List of mask labels of shape `(num_labels, height, width)` to be fed to a model\n        class_labels (`List[torch.LongTensor]`, *optional*):\n            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the\n            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.\n\n        Returns:\n            `Mask2FormerUniversalSegmentationOutput`\n\n        Examples:\n\n        Instance segmentation example:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation\n        >>> from PIL import Image\n        >>> import requests\n        >>> import torch\n\n        >>> # Load Mask2Former trained on COCO instance segmentation dataset\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")\n        >>> model = Mask2FormerForUniversalSegmentation.from_pretrained(\n        ...     "facebook/mask2former-swin-small-coco-instance"\n        ... )\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> inputs = image_processor(image, return_tensors="pt")\n\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n\n        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`\n        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`\n        >>> class_queries_logits = outputs.class_queries_logits\n        >>> masks_queries_logits = outputs.masks_queries_logits\n\n        >>> # Perform post-processing to get instance segmentation map\n        >>> pred_instance_map = image_processor.post_process_semantic_segmentation(\n        ...     outputs, target_sizes=[image.size[::-1]]\n        ... )[0]\n        >>> print(pred_instance_map.shape)\n        torch.Size([480, 640])\n        ```\n\n        Semantic segmentation example:\n        ```python\n        >>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation\n        >>> from PIL import Image\n        >>> import requests\n        >>> import torch\n\n        >>> # Load Mask2Former trained on ADE20k semantic segmentation dataset\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")\n        >>> model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")\n\n        >>> url = (\n        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"\n        ... )\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> inputs = image_processor(image, return_tensors="pt")\n\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n\n        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`\n        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`\n        >>> class_queries_logits = outputs.class_queries_logits\n        >>> masks_queries_logits = outputs.masks_queries_logits\n\n        >>> # Perform post-processing to get semantic segmentation map\n        >>> pred_semantic_map = image_processor.post_process_semantic_segmentation(\n        ...     outputs, target_sizes=[image.size[::-1]]\n        ... )[0]\n        >>> print(pred_semantic_map.shape)\n        torch.Size([512, 683])\n        ```\n\n        Panoptic segmentation example:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation\n        >>> from PIL import Image\n        >>> import requests\n        >>> import torch\n\n        >>> # Load Mask2Former trained on CityScapes panoptic segmentation dataset\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")\n        >>> model = Mask2FormerForUniversalSegmentation.from_pretrained(\n        ...     "facebook/mask2former-swin-small-cityscapes-panoptic"\n        ... )\n\n        >>> url = "https://cdn-media.huggingface.co/Inference-API/Sample-results-on-the-Cityscapes-dataset-The-above-images-show-how-our-method-can-handle.png"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> inputs = image_processor(image, return_tensors="pt")\n\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n\n        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`\n        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`\n        >>> class_queries_logits = outputs.class_queries_logits\n        >>> masks_queries_logits = outputs.masks_queries_logits\n\n        >>> # Perform post-processing to get panoptic segmentation map\n        >>> pred_panoptic_map = image_processor.post_process_panoptic_segmentation(\n        ...     outputs, target_sizes=[image.size[::-1]]\n        ... )[0]["segmentation"]\n        >>> print(pred_panoptic_map.shape)\n        torch.Size([338, 676])\n        ```\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss, output_attentions=output_attentions, return_dict=True)
        (loss, loss_dict, auxiliary_logits) = (None, None, None)
        class_queries_logits = ()
        for decoder_output in outputs.transformer_decoder_intermediate_states:
            class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
            class_queries_logits += (class_prediction,)
        masks_queries_logits = outputs.masks_queries_logits
        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)
        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(masks_queries_logits=masks_queries_logits[-1], class_queries_logits=class_queries_logits[-1], mask_labels=mask_labels, class_labels=class_labels, auxiliary_predictions=auxiliary_logits)
            loss = self.get_loss(loss_dict)
        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        if output_hidden_states:
            encoder_hidden_states = outputs.encoder_hidden_states
            pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states
            transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states
        output_auxiliary_logits = self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        if not output_auxiliary_logits:
            auxiliary_logits = None
        output = Mask2FormerForUniversalSegmentationOutput(loss=loss, class_queries_logits=class_queries_logits[-1], masks_queries_logits=masks_queries_logits[-1], auxiliary_logits=auxiliary_logits, encoder_last_hidden_state=outputs.encoder_last_hidden_state, pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state, transformer_decoder_last_hidden_state=outputs.transformer_decoder_last_hidden_state, encoder_hidden_states=encoder_hidden_states, pixel_decoder_hidden_states=pixel_decoder_hidden_states, transformer_decoder_hidden_states=transformer_decoder_hidden_states, attentions=outputs.attentions)
        if not return_dict:
            output = tuple((v for v in output.values() if v is not None))
            if loss is not None:
                output = loss + output
        return output