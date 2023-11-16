from functools import partial
from torch.nn.modules.loss import _Loss
from catalyst import metrics

class FocalLossBinary(_Loss):
    """Compute focal loss for binary classification problem.

    It has been proposed in `Focal Loss for Dense Object Detection`_ paper.

    .. _Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, ignore: int=None, reduced: bool=False, gamma: float=2.0, alpha: float=0.25, threshold: float=0.5, reduction: str='mean'):
        if False:
            i = 10
            return i + 15
        '@TODO: Docs. Contribution is welcome.'
        super().__init__()
        self.ignore = ignore
        if reduced:
            self.loss_fn = partial(metrics.reduced_focal_loss, gamma=gamma, threshold=threshold, reduction=reduction)
        else:
            self.loss_fn = partial(metrics.sigmoid_focal_loss, gamma=gamma, alpha=alpha, reduction=reduction)

    def forward(self, logits, targets):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            logits: [bs; ...]\n            targets: [bs; ...]\n\n        Returns:\n            computed loss\n        '
        targets = targets.view(-1)
        logits = logits.view(-1)
        if self.ignore is not None:
            not_ignored = targets != self.ignore
            logits = logits[not_ignored]
            targets = targets[not_ignored]
        loss = self.loss_fn(logits, targets)
        return loss

class FocalLossMultiClass(FocalLossBinary):
    """Compute focal loss for multiclass problem. Ignores targets having -1 label.

    It has been proposed in `Focal Loss for Dense Object Detection`_ paper.

    .. _Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002
    """

    def forward(self, logits, targets):
        if False:
            print('Hello World!')
        '\n        Args:\n            logits: [bs; num_classes; ...]\n            targets: [bs; ...]\n\n        Returns:\n            computed loss\n        '
        num_classes = logits.size(1)
        loss = 0
        targets = targets.view(-1)
        logits = logits.view(-1, num_classes)
        if self.ignore is not None:
            not_ignored = targets != self.ignore
        for class_id in range(num_classes):
            cls_label_target = (targets == class_id + 0).long()
            cls_label_input = logits[..., class_id]
            if self.ignore is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]
            loss += self.loss_fn(cls_label_input, cls_label_target)
        return loss
__all__ = ['FocalLossBinary', 'FocalLossMultiClass']