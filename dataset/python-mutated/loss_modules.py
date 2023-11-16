from typing import Type
import torch
from torch import nn, Tensor
from torch.nn import HuberLoss as _HuberLoss
from torch.nn import L1Loss
from torch.nn import MSELoss as _MSELoss
from torchmetrics.functional import mean_absolute_percentage_error
import ludwig.utils.loss_utils as utils
from ludwig.constants import LOGITS
from ludwig.modules.loss_implementations.corn import corn_loss
from ludwig.schema.features.loss.loss import BaseLossConfig, BWCEWLossConfig, CORNLossConfig, HuberLossConfig, MAELossConfig, MAPELossConfig, MSELossConfig, NextTokenSoftmaxCrossEntropyLossConfig, RMSELossConfig, RMSPELossConfig, SequenceSoftmaxCrossEntropyLossConfig, SigmoidCrossEntropyLossConfig, SoftmaxCrossEntropyLossConfig
from ludwig.utils import strings_utils
from ludwig.utils.registry import Registry
EPSILON = 1e-10
loss_impl_registry = Registry[Type[nn.Module]]()

def register_loss(config_cls: Type[BaseLossConfig]):
    if False:
        while True:
            i = 10

    def wrap(cls: Type[nn.Module]):
        if False:
            i = 10
            return i + 15
        loss_impl_registry[config_cls] = cls
        return cls
    return wrap

def create_loss(config: BaseLossConfig) -> nn.Module:
    if False:
        print('Hello World!')
    return loss_impl_registry[type(config)](config)

class LogitsInputsMixin:

    @classmethod
    def get_loss_inputs(cls):
        if False:
            print('Hello World!')
        'Maps loss to the desired predicted input type.'
        return LOGITS

@register_loss(MSELossConfig)
class MSELoss(_MSELoss, LogitsInputsMixin):
    """Mean squared error."""

    def __init__(self, config: MSELossConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()

@register_loss(MAELossConfig)
class MAELoss(L1Loss, LogitsInputsMixin):
    """Mean absolute error."""

    def __init__(self, config: MAELossConfig):
        if False:
            return 10
        super().__init__()

@register_loss(MAPELossConfig)
class MAPELoss(nn.Module, LogitsInputsMixin):
    """Mean absolute error."""

    def __init__(self, config: MAPELossConfig):
        if False:
            while True:
                i = 10
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        return mean_absolute_percentage_error(preds, target)

@register_loss(RMSELossConfig)
class RMSELoss(nn.Module, LogitsInputsMixin):
    """Root mean square error."""

    def __init__(self, config: RMSELossConfig):
        if False:
            return 10
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        return torch.sqrt(self.mse(preds, target))

@register_loss(RMSPELossConfig)
class RMSPELoss(nn.Module, LogitsInputsMixin):
    """Root mean square percentage error."""

    def __init__(self, config: RMSPELossConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        loss = utils.rmspe_loss(target, preds)
        return loss

@register_loss(BWCEWLossConfig)
class BWCEWLoss(nn.Module, LogitsInputsMixin):
    """Binary weighted cross entropy loss."""

    def __init__(self, config: BWCEWLossConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if config.positive_class_weight:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([config.positive_class_weight]))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=config.positive_class_weight)
        self.robust_lambda = config.robust_lambda
        self.confidence_penalty = config.confidence_penalty

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        train_loss = self.loss_fn(preds, target.float())
        if self.robust_lambda > 0:
            train_loss = (1 - self.robust_lambda) * train_loss + self.robust_lambda / 2
        train_mean_loss = torch.mean(train_loss)
        if self.confidence_penalty > 0:
            probabilities = torch.sigmoid(preds)
            mean_penalty = utils.mean_confidence_penalty(probabilities, 2)
            train_mean_loss += self.confidence_penalty * mean_penalty
        return train_mean_loss

@register_loss(SoftmaxCrossEntropyLossConfig)
class SoftmaxCrossEntropyLoss(nn.Module, LogitsInputsMixin):

    def __init__(self, config: SoftmaxCrossEntropyLossConfig):
        if False:
            i = 10
            return i + 15
        '\n        Params:\n            class_weights: List or 1D tensor of length equal to number of classes.\n        '
        super().__init__()
        if config.class_weights:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(config.class_weights))
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        '\n        Params:\n            preds: Tensor of shape [batch x num_classes]\n            target: Tensor of shape [batch], where each element is integral\n                between 0 and num_classes.\n        '
        if len(target.shape) == 1:
            target = target.long()
        return self.loss_fn(preds, target)

@register_loss(SequenceSoftmaxCrossEntropyLossConfig)
class SequenceSoftmaxCrossEntropyLoss(nn.Module, LogitsInputsMixin):

    def __init__(self, config: SequenceSoftmaxCrossEntropyLossConfig):
        if False:
            return 10
        '\n        Params:\n            class_weights: List or 1D tensor of length equal to number of classes.\n        '
        super().__init__()
        if config.class_weights:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(config.class_weights), ignore_index=strings_utils.SpecialSymbol.PADDING.value)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=strings_utils.SpecialSymbol.PADDING.value)

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        '\n        Params:\n            preds: Tensor of shape [batch x sequence_length x vocab_size]\n            target: Tensor of shape [batch x sequence_length], where each element is integral between 0 and vocab_size.\n        '
        target = target.long()
        return self.loss_fn(preds[1:].view(-1, preds.size(-1)), target[1:].view(-1))

@register_loss(NextTokenSoftmaxCrossEntropyLossConfig)
class NextTokenSoftmaxCrossEntropyLoss(nn.Module, LogitsInputsMixin):

    def __init__(self, config: NextTokenSoftmaxCrossEntropyLossConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        '\n        Params:\n            preds: Tensor of shape [batch x sequence_length x vocab_size]\n            target: Tensor of shape [batch x sequence_length], where each element is integral between 0 and vocab_size.\n\n        Reference implementation:\n        https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/bert/modeling_bert.py#LL1253C1-L1260C1 # noqa\n        '
        target = target.long()
        (_, _, vocab_size) = preds.shape
        shifted_predictions = preds[:, :-1, :]
        shifted_targets = target[:, 1:]
        return self.loss_fn(shifted_predictions.reshape(-1, vocab_size), shifted_targets.reshape(-1))

@register_loss(SigmoidCrossEntropyLossConfig)
class SigmoidCrossEntropyLoss(nn.Module, LogitsInputsMixin):

    def __init__(self, config: SigmoidCrossEntropyLossConfig):
        if False:
            print('Hello World!')
        '\n        Params:\n            class_weights: List or 1D tensor of length equal to number of classes.\n        '
        super().__init__()
        if config.class_weights:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(config.class_weights))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if False:
            return 10
        if preds.ndim != 2:
            raise RuntimeError('SigmoidCrossEntropyLoss currently only supported for 2D tensors.')
        return self.loss_fn(preds.type(torch.float32), target.type(torch.float32))

@register_loss(HuberLossConfig)
class HuberLoss(_HuberLoss, LogitsInputsMixin):
    """Huber loss."""

    def __init__(self, config: HuberLossConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(delta=config.delta)

@register_loss(CORNLossConfig)
class CORNLoss(nn.Module, LogitsInputsMixin):
    """CORN loss."""

    def __init__(self, config: CORNLossConfig):
        if False:
            while True:
                i = 10
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        num_classes = preds.shape[1]
        return corn_loss(preds, target, num_classes=num_classes)