from __future__ import print_function, division
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse

def lovasz_grad(gt_sorted):
    if False:
        i = 10
        return i + 15
    '\n    Computes gradient of the Lovasz extension w.r.t sorted errors\n    See Alg. 1 in paper\n    '
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def iou_binary(preds, labels, EMPTY=1.0, ignore=None, per_image=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    IoU for foreground class\n    binary: 1 foreground, 0 background\n    '
    if not per_image:
        (preds, labels) = ((preds,), (labels,))
    ious = []
    for (pred, label) in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | (pred == 1) & (label != ignore)).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)
    return 100 * iou

def iou(preds, labels, C, EMPTY=1.0, ignore=None, per_image=False):
    if False:
        i = 10
        return i + 15
    '\n    Array of IoU for each (non ignored) class\n    '
    if not per_image:
        (preds, labels) = ((preds,), (labels,))
    ious = []
    for (pred, label) in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | (pred == i) & (label != ignore)).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious))
    return 100 * np.array(ious)

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    if False:
        while True:
            i = 10
    '\n    Binary Lovasz hinge loss\n      logits: [B, H, W] Variable, logits at each pixel (between -\\infty and +\\infty)\n      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)\n      per_image: compute the loss per image instead of per batch\n      ignore: void class id\n    '
    if per_image:
        loss = mean((lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)) for (log, lab) in zip(logits, labels)))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    if False:
        return 10
    '\n    Binary Lovasz hinge loss\n      logits: [P] Variable, logits at each prediction (between -\\infty and +\\infty)\n      labels: [P] Tensor, binary ground truth labels (0 or 1)\n      ignore: label to ignore\n    '
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    (errors_sorted, perm) = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
    return loss

def flatten_binary_scores(scores, labels, ignore=None):
    if False:
        print('Hello World!')
    "\n    Flattens predictions in the batch (binary case)\n    Remove labels equal to 'ignore'\n    "
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return (scores, labels)
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return (vscores, vlabels)

class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        if False:
            i = 10
            return i + 15
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()

def binary_xloss(logits, labels, ignore=None):
    if False:
        return 10
    '\n    Binary Cross entropy loss\n      logits: [B, H, W] Variable, logits at each pixel (between -\\infty and +\\infty)\n      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)\n      ignore: void class id\n    '
    (logits, labels) = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss

def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    if False:
        return 10
    '\n    Multi-class Lovasz-Softmax loss\n      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)\n      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)\n      only_present: average only on classes present in ground truth\n      per_image: compute the loss per image instead of per batch\n      ignore: void class labels\n    '
    if per_image:
        loss = mean((lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present) for (prob, lab) in zip(probas, labels)))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss

def lovasz_softmax_flat(probas, labels, only_present=False):
    if False:
        while True:
            i = 10
    '\n    Multi-class Lovasz-Softmax loss\n      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)\n      labels: [P] Tensor, ground truth labels (between 0 and C - 1)\n      only_present: average only on classes present in ground truth\n    '
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        (errors_sorted, perm) = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

def flatten_probas(probas, labels, ignore=None):
    if False:
        i = 10
        return i + 15
    '\n    Flattens predictions in the batch\n    '
    (B, C, H, W) = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return (probas, labels)
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return (vprobas, vlabels)

def xloss(logits, labels, ignore=None):
    if False:
        print('Hello World!')
    '\n    Cross entropy loss\n    '
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)

def mean(l, ignore_nan=False, empty=0):
    if False:
        while True:
            i = 10
    '\n    nanmean compatible with generators.\n    '
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for (n, v) in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n