"""This file is an incomplete implementation of `Single-path NAS <https://arxiv.org/abs/1904.02877>`__.
These are merely some components of the algorithm. The complete support is an undergoing work item.

Keep this file here so that it can be "blamed".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.nas.nn.pytorch import ValueChoice

class DifferentiableSuperConv2d(nn.Conv2d):
    """
    Only ``kernel_size`` ``in_channels`` and ``out_channels`` are supported. Kernel size candidates should be larger or smaller
    than each other in both candidates. See examples below:
    the following example is not allowed:
        >>> ValueChoice(candidates = [(5, 3), (3, 5)])
            □ ■ ■ ■ □   □ □ □ □ □
            □ ■ ■ ■ □   ■ ■ ■ ■ ■    # candidates are not bigger or smaller on both dimension
            □ ■ ■ ■ □   ■ ■ ■ ■ ■
            □ ■ ■ ■ □   ■ ■ ■ ■ ■
            □ ■ ■ ■ □   □ □ □ □ □
    the following 3 examples are valid:
        >>> ValueChoice(candidates = [5, 3, 1])
            ■ ■ ■ ■ ■   □ □ □ □ □   □ □ □ □ □
            ■ ■ ■ ■ ■   □ ■ ■ ■ □   □ □ □ □ □
            ■ ■ ■ ■ ■   □ ■ ■ ■ □   □ □ ■ □ □
            ■ ■ ■ ■ ■   □ ■ ■ ■ □   □ □ □ □ □
            ■ ■ ■ ■ ■   □ □ □ □ □   □ □ □ □ □
        >>> ValueChoice(candidates = [(5, 7), (3, 5), (1, 3)])
            ■ ■ ■ ■ ■ ■ ■  □ □ □ □ □ □ □   □ □ □ □ □ □ □
            ■ ■ ■ ■ ■ ■ ■  □ ■ ■ ■ ■ ■ □   □ □ □ □ □ □ □
            ■ ■ ■ ■ ■ ■ ■  □ ■ ■ ■ ■ ■ □   □ □ ■ ■ ■ □ □
            ■ ■ ■ ■ ■ ■ ■  □ ■ ■ ■ ■ ■ □   □ □ □ □ □ □ □
            ■ ■ ■ ■ ■ ■ ■  □ □ □ □ □ □ □   □ □ □ □ □ □ □
        >>> # when the difference between any two candidates is not even, the left upper will be picked:
        >>> ValueChoice(candidates = [(5, 5), (4, 4), (3, 3)])
            ■ ■ ■ ■ ■   ■ ■ ■ ■ □   □ □ □ □ □
            ■ ■ ■ ■ ■   ■ ■ ■ ■ □   □ ■ ■ ■ □
            ■ ■ ■ ■ ■   ■ ■ ■ ■ □   □ ■ ■ ■ □
            ■ ■ ■ ■ ■   ■ ■ ■ ■ □   □ ■ ■ ■ □
            ■ ■ ■ ■ ■   □ □ □ □ □   □ □ □ □ □
    """

    def __init__(self, module, name):
        if False:
            i = 10
            return i + 15
        self.label = name
        args = module.trace_kwargs
        if isinstance(args['in_channels'], ValueChoice):
            args['in_channels'] = max(args['in_channels'].candidates)
        self.out_channel_candidates = None
        if isinstance(args['out_channels'], ValueChoice):
            self.out_channel_candidates = sorted(args['out_channels'].candidates, reverse=True)
            args['out_channels'] = self.out_channel_candidates[0]
        self.kernel_size_candidates = None
        if isinstance(args['kernel_size'], ValueChoice):
            candidates = args['kernel_size'].candidates
            if not isinstance(candidates[0], tuple):
                candidates = [(k, k) for k in candidates]
            self.kernel_size_candidates = sorted(candidates, key=lambda t: t[0], reverse=True)
            for i in range(0, len(self.kernel_size_candidates) - 1):
                bigger = self.kernel_size_candidates[i]
                smaller = self.kernel_size_candidates[i + 1]
                assert bigger[1] > smaller[1] or (bigger[1] == smaller[1] and bigger[0] > smaller[0]), f'Kernel_size candidates should be larger or smaller than each other on both dimensions, but found {bigger} and {smaller}.'
            args['kernel_size'] = self.kernel_size_candidates[0]
        super().__init__(**args)
        self.generate_architecture_params()

    def forward(self, input):
        if False:
            print('Hello World!')
        weight = self.weight

        def sum_weight(input_weight, masks, thresholds, indicator):
            if False:
                print('Hello World!')
            '\n            This is to get the weighted sum of weight.\n\n            Parameters\n            ----------\n            input_weight : Tensor\n                the weight to be weighted summed\n            masks : list[Tensor]\n                weight masks.\n            thresholds : list[float]\n                thresholds, should have a length of ``len(masks) - 1``\n            indicator : Callable[[Tensor, float], float]\n                take a tensor and a threshold as input, and output the weight\n\n            Returns\n            ----------\n            weight : Tensor\n                weighted sum of ``input_weight``. this is of the same shape as ``input_sum``\n            '
            weight = torch.zeros_like(input_weight)
            for (mask, t) in zip(masks[:-1], thresholds):
                cur_part = input_weight * mask
                alpha = indicator(cur_part, t)
                weight = (weight + cur_part) * alpha
            weight += input_weight * masks[-1]
            return weight
        if self.kernel_size_candidates is not None:
            weight = sum_weight(weight, self.kernel_masks, self.t_kernel, self.Lasso_sigmoid)
        if self.out_channel_candidates is not None:
            weight = sum_weight(weight, self.channel_masks, self.t_expansion, self.Lasso_sigmoid)
        output = self._conv_forward(input, weight, self.bias)
        return output

    def parameters(self):
        if False:
            return 10
        for (_, p) in self.named_parameters():
            yield p

    def named_parameters(self):
        if False:
            i = 10
            return i + 15
        for (name, p) in super().named_parameters():
            if name == 'alpha':
                continue
            yield (name, p)

    def export(self):
        if False:
            print('Hello World!')
        "\n        result = {\n            'kernel_size': i,\n            'out_channels': j\n        }\n        which means the best candidate for an argument is the i-th one if candidates are sorted in descending order\n        "
        result = {}
        eps = 1e-05
        with torch.no_grad():
            if self.kernel_size_candidates is not None:
                weight = torch.zeros_like(self.weight)
                for i in range(len(self.kernel_size_candidates) - 2, -1, -1):
                    mask = self.kernel_masks[i]
                    t = self.t_kernel[i]
                    cur_part = self.weight * mask
                    alpha = self.Lasso_sigmoid(cur_part, t)
                    if alpha <= eps:
                        result['kernel_size'] = self.kernel_size_candidates[i + 1]
                        break
                    weight = (weight + cur_part) * alpha
                if 'kernel_size' not in result:
                    result['kernel_size'] = self.kernel_size_candidates[0]
            else:
                weight = self.weight
            if self.out_channel_candidates is not None:
                for i in range(len(self.out_channel_candidates) - 2, -1, -1):
                    mask = self.channel_masks[i]
                    t = self.t_expansion[i]
                    alpha = self.Lasso_sigmoid(weight * mask, t)
                    if alpha <= eps:
                        result['out_channels'] = self.out_channel_candidates[i + 1]
                if 'out_channels' not in result:
                    result['out_channels'] = self.out_channel_candidates[0]
        return result

    @staticmethod
    def Lasso_sigmoid(matrix, t):
        if False:
            for i in range(10):
                print('nop')
        '\n        A trick that can make use of both the value of bool(lasso > t) and the gradient of sigmoid(lasso - t)\n\n        Parameters\n        ----------\n        matrix : Tensor\n            the matrix to calculate lasso norm\n        t : float\n            the threshold\n        '
        lasso = torch.norm(matrix) - t
        indicator = (lasso > 0).float()
        with torch.no_grad():
            indicator -= F.sigmoid(lasso)
        indicator += F.sigmoid(lasso)
        return indicator

    def generate_architecture_params(self):
        if False:
            i = 10
            return i + 15
        self.alpha = {}
        if self.kernel_size_candidates is not None:
            self.t_kernel = nn.Parameter(torch.rand(len(self.kernel_size_candidates) - 1))
            self.alpha['kernel_size'] = self.t_kernel
            self.kernel_masks = []
            for i in range(0, len(self.kernel_size_candidates) - 1):
                big_size = self.kernel_size_candidates[i]
                small_size = self.kernel_size_candidates[i + 1]
                mask = torch.zeros_like(self.weight)
                mask[:, :, :big_size[0], :big_size[1]] = 1
                mask[:, :, :small_size[0], :small_size[1]] = 0
                self.kernel_masks.append(mask)
            mask = torch.zeros_like(self.weight)
            mask[:, :, :self.kernel_size_candidates[-1][0], :self.kernel_size_candidates[-1][1]] = 1
            self.kernel_masks.append(mask)
        if self.out_channel_candidates is not None:
            self.t_expansion = nn.Parameter(torch.rand(len(self.out_channel_candidates) - 1))
            self.alpha['out_channels'] = self.t_expansion
            self.channel_masks = []
            for i in range(0, len(self.out_channel_candidates) - 1):
                (big_channel, small_channel) = (self.out_channel_candidates[i], self.out_channel_candidates[i + 1])
                mask = torch.zeros_like(self.weight)
                mask[:big_channel] = 1
                mask[:small_channel] = 0
                self.channel_masks.append(mask)
            mask = torch.zeros_like(self.weight)
            mask[:self.out_channel_candidates[-1]] = 1
            self.channel_masks.append(mask)

class DifferentiableBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, module, name):
        if False:
            for i in range(10):
                print('nop')
        self.label = name
        args = module.trace_kwargs
        if isinstance(args['num_features'], ValueChoice):
            args['num_features'] = max(args['num_features'].candidates)
        super().__init__(**args)
        self.alpha = nn.Parameter(torch.tensor([]))

    def export(self):
        if False:
            return 10
        '\n        No need to export ``BatchNorm2d``. Refer to the ``Conv2d`` layer that has the ``ValueChoice`` as ``out_channels``.\n        '
        return -1