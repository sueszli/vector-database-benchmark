from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from ludwig.modules.normalization_modules import GhostBatchNormalization
from ludwig.utils.entmax import Entmax15, EntmaxBisect, Sparsemax
from ludwig.utils.torch_utils import LudwigModule

class TabNet(LudwigModule):

    def __init__(self, input_size: int, size: int, output_size: int, num_steps: int=1, num_total_blocks: int=4, num_shared_blocks: int=2, relaxation_factor: float=1.5, bn_momentum: float=0.3, bn_epsilon: float=0.001, bn_virtual_bs: Optional[int]=None, sparsity: float=1e-05, entmax_mode: str='sparsemax', entmax_alpha: float=1.5):
        if False:
            return 10
        'TabNet Will output a vector of size output_dim.\n\n        Args:\n            input_size: concatenated size of input feature encoder outputs\n            size: Embedding feature dimension\n            output_size: Output dimension for TabNet\n            num_steps: Total number of steps.\n            num_total_blocks: Total number of feature transformer blocks.\n            num_shared_blocks: Number of shared feature transformer blocks.\n            relaxation_factor: >1 will allow features to be used more than once.\n            bn_momentum: Batch normalization, momentum.\n            bn_epsilon: Batch normalization, epsilon.\n            bn_virtual_bs: Virtual batch ize for ghost batch norm.\n            entmax_mode: Entmax is a sparse family of probability mapping which generalizes softmax and sparsemax.\n                         entmax_mode controls the sparsity.  One of {"sparsemax", "entmax15", "constant", "adaptive"}.\n            entmax_alpha: Must be a number between 1.0 and 2.0.  If entmax_mode is "adaptive", entmax_alpha is used\n                          as the initial value for the learnable parameter.\n        '
        super().__init__()
        self.input_size = input_size
        self.size = size
        self.output_size = output_size
        self.num_steps = num_steps
        self.bn_virtual_bs = bn_virtual_bs
        self.relaxation_factor = relaxation_factor
        self.sparsity = torch.tensor(sparsity)
        self.batch_norm = nn.BatchNorm1d(input_size, momentum=bn_momentum, eps=bn_epsilon)
        kargs = {'num_total_blocks': num_total_blocks, 'num_shared_blocks': num_shared_blocks, 'bn_momentum': bn_momentum, 'bn_epsilon': bn_epsilon, 'bn_virtual_bs': bn_virtual_bs}
        self.feature_transforms = nn.ModuleList([FeatureTransformer(input_size, size + output_size, **kargs)])
        self.attentive_transforms = nn.ModuleList([None])
        for i in range(num_steps):
            self.feature_transforms.append(FeatureTransformer(input_size, size + output_size, **kargs, shared_fc_layers=self.feature_transforms[0].shared_fc_layers))
            self.attentive_transforms.append(AttentiveTransformer(size, input_size, bn_momentum, bn_epsilon, bn_virtual_bs, entmax_mode, entmax_alpha))
        self.final_projection = nn.Linear(output_size, output_size)
        self.register_buffer('out_accumulator', torch.zeros(output_size))
        self.register_buffer('aggregated_mask', torch.zeros(input_size))
        self.register_buffer('prior_scales', torch.ones(input_size))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if False:
            i = 10
            return i + 15
        if features.dim() != 2:
            raise ValueError(f'Expecting incoming tensor to be dim 2, instead dim={features.dim()}')
        batch_size = features.shape[0]
        out_accumulator = torch.tile(self.out_accumulator, (batch_size, 1))
        aggregated_mask = torch.tile(self.aggregated_mask, (batch_size, 1))
        prior_scales = torch.tile(self.prior_scales, (batch_size, 1))
        masks = []
        total_entropy = 0.0
        if batch_size != 1 or not self.training:
            features = self.batch_norm(features)
        elif batch_size == 1:
            self.batch_norm.eval()
            features = self.batch_norm(features)
            self.batch_norm.train()
        masked_features = features
        x = self.feature_transforms[0](masked_features)
        for step_i in range(1, self.num_steps + 1):
            mask_values = self.attentive_transforms[step_i](x[:, self.output_size:], prior_scales)
            prior_scales = prior_scales * (self.relaxation_factor - mask_values)
            if self.sparsity.item() != 0.0:
                total_entropy += torch.mean(torch.sum(-mask_values * torch.log(mask_values + 1e-05), dim=1)) / self.num_steps
            masks.append(torch.unsqueeze(torch.unsqueeze(mask_values, 0), 3))
            masked_features = torch.multiply(mask_values, features)
            x = self.feature_transforms[step_i](masked_features)
            out = nn.functional.relu(x[:, :self.output_size])
            out_accumulator += out
            scale = torch.sum(out, dim=1, keepdim=True) / self.num_steps
            aggregated_mask += mask_values * scale
        final_output = self.final_projection(out_accumulator)
        sparsity_loss = torch.multiply(self.sparsity, total_entropy)
        self.update_loss('sparsity_loss', sparsity_loss)
        return (final_output, aggregated_mask, masks)

    @property
    def input_shape(self) -> torch.Size:
        if False:
            print('Hello World!')
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        if False:
            while True:
                i = 10
        return torch.Size([self.output_size])

class FeatureBlock(LudwigModule):

    def __init__(self, input_size: int, size: int, apply_glu: bool=True, bn_momentum: float=0.1, bn_epsilon: float=0.001, bn_virtual_bs: int=None, shared_fc_layer: LudwigModule=None):
        if False:
            return 10
        super().__init__()
        self.input_size = input_size
        self.apply_glu = apply_glu
        self.size = size
        units = size * 2 if apply_glu else size
        self.fc_layer = nn.Linear(input_size, units, bias=False)
        if shared_fc_layer is not None:
            assert shared_fc_layer.weight.shape == self.fc_layer.weight.shape
            self.fc_layer = shared_fc_layer
        self.batch_norm = GhostBatchNormalization(units, virtual_batch_size=bn_virtual_bs, momentum=bn_momentum, epsilon=bn_epsilon)

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        hidden = self.fc_layer(inputs)
        hidden = self.batch_norm(hidden)
        if self.apply_glu:
            hidden = nn.functional.glu(hidden, dim=-1)
        return hidden

    @property
    def input_shape(self) -> torch.Size:
        if False:
            for i in range(10):
                print('nop')
        return torch.Size([self.input_size])

class AttentiveTransformer(LudwigModule):

    def __init__(self, input_size: int, size: int, bn_momentum: float=0.1, bn_epsilon: float=0.001, bn_virtual_bs: int=None, entmax_mode: str='sparsemax', entmax_alpha: float=1.5):
        if False:
            print('Hello World!')
        super().__init__()
        self.input_size = input_size
        self.size = size
        self.entmax_mode = entmax_mode
        if entmax_mode == 'adaptive':
            self.register_buffer('trainable_alpha', torch.tensor(entmax_alpha, requires_grad=True))
        else:
            self.trainable_alpha = entmax_alpha
        if self.entmax_mode == 'sparsemax':
            self.entmax_module = Sparsemax()
        elif self.entmax_mode == 'entmax15':
            self.entmax_module = Entmax15()
        else:
            self.entmax_module = EntmaxBisect(alpha=self.trainable_alpha)
        self.feature_block = FeatureBlock(input_size, size, bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, bn_virtual_bs=bn_virtual_bs, apply_glu=False)

    def forward(self, inputs, prior_scales):
        if False:
            i = 10
            return i + 15
        hidden = self.feature_block(inputs)
        hidden = hidden * prior_scales
        return self.entmax_module(hidden)

    @property
    def input_shape(self) -> torch.Size:
        if False:
            return 10
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        if False:
            for i in range(10):
                print('nop')
        return torch.Size([self.size])

class FeatureTransformer(LudwigModule):

    def __init__(self, input_size: int, size: int, shared_fc_layers: Optional[List]=None, num_total_blocks: int=4, num_shared_blocks: int=2, bn_momentum: float=0.1, bn_epsilon: float=0.001, bn_virtual_bs: int=None):
        if False:
            return 10
        super().__init__()
        if shared_fc_layers is None:
            shared_fc_layers = []
        self.input_size = input_size
        self.num_total_blocks = num_total_blocks
        self.num_shared_blocks = num_shared_blocks
        self.size = size
        kwargs = {'bn_momentum': bn_momentum, 'bn_epsilon': bn_epsilon, 'bn_virtual_bs': bn_virtual_bs}
        self.blocks = nn.ModuleList()
        for n in range(num_total_blocks):
            if n == 0:
                in_features = input_size
            else:
                in_features = size
            if shared_fc_layers and n < len(shared_fc_layers):
                self.blocks.append(FeatureBlock(in_features, size, **kwargs, shared_fc_layer=shared_fc_layers[n]))
            else:
                self.blocks.append(FeatureBlock(in_features, size, **kwargs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        hidden = self.blocks[0](inputs)
        for n in range(1, self.num_total_blocks):
            hidden = (self.blocks[n](hidden) + hidden) * 0.5 ** 0.5
        return hidden

    @property
    def shared_fc_layers(self):
        if False:
            i = 10
            return i + 15
        return [self.blocks[i].fc_layer for i in range(self.num_shared_blocks)]

    @property
    def input_shape(self) -> torch.Size:
        if False:
            for i in range(10):
                print('nop')
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        if False:
            print('Hello World!')
        return torch.Size([self.size])