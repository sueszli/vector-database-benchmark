import torch
from torch.nn import functional as F
from functools import reduce
from typing import Any, List, Optional, Tuple
from .base_data_sparsifier import BaseDataSparsifier
__all__ = ['DataNormSparsifier']

class DataNormSparsifier(BaseDataSparsifier):
    """L1-Norm Sparsifier
    This sparsifier computes the *L1-norm* of every sparse block and "zeroes-out" the
    ones with the lowest norm. The level of sparsity defines how many of the
    blocks is removed.
    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of *sparse blocks* that are zeroed-out
    2. `sparse_block_shape` defines the shape of the sparse blocks. Note that
        the sparse blocks originate at the zero-index of the tensor.
    3. `zeros_per_block` is the number of zeros that we are expecting in each
        sparse block. By default we assume that all elements within a block are
        zeroed-out. However, setting this variable sets the target number of
        zeros per block. The zeros within each block are chosen as the *smallest
        absolute values*.
    Args:
        sparsity_level: The target level of sparsity
        sparse_block_shape: The shape of a sparse block
        zeros_per_block: Number of zeros in a sparse block
    Note::
        All arguments to the DataNormSparsifier constructor are "default"
        arguments and could be overriden by the configuration provided in the
        `add_data` step.
    """

    def __init__(self, data_list: Optional[List[Tuple[str, Any]]]=None, sparsity_level: float=0.5, sparse_block_shape: Tuple[int, int]=(1, 4), zeros_per_block: Optional[int]=None, norm: str='L1'):
        if False:
            print('Hello World!')
        if zeros_per_block is None:
            zeros_per_block = reduce(lambda x, y: x * y, sparse_block_shape)
        assert norm in ['L1', 'L2'], 'only L1 and L2 norm supported at the moment'
        defaults = {'sparsity_level': sparsity_level, 'sparse_block_shape': sparse_block_shape, 'zeros_per_block': zeros_per_block}
        self.norm = norm
        super().__init__(data_list=data_list, **defaults)

    def __get_scatter_folded_mask(self, data, dim, indices, output_size, sparse_block_shape):
        if False:
            i = 10
            return i + 15
        mask = torch.ones_like(data)
        mask.scatter_(dim=dim, index=indices, value=0)
        mask = F.fold(mask, output_size=output_size, kernel_size=sparse_block_shape, stride=sparse_block_shape)
        mask = mask.to(torch.int8)
        return mask

    def __get_block_level_mask(self, data, sparse_block_shape, zeros_per_block):
        if False:
            print('Hello World!')
        (height, width) = (data.shape[-2], data.shape[-1])
        (block_height, block_width) = sparse_block_shape
        values_per_block = block_height * block_width
        if values_per_block == zeros_per_block:
            return torch.zeros_like(data, dtype=torch.int8)
        dh = (block_height - height % block_height) % block_height
        dw = (block_width - width % block_width) % block_width
        padded_data = torch.ones(height + dh, width + dw, dtype=data.dtype, device=data.device)
        padded_data = padded_data * torch.nan
        padded_data[0:height, 0:width] = data
        unfolded_data = F.unfold(padded_data[None, None, :], kernel_size=sparse_block_shape, stride=sparse_block_shape)
        (_, sorted_idx) = torch.sort(unfolded_data, dim=1)
        sorted_idx = sorted_idx[:, :zeros_per_block, :]
        mask = self.__get_scatter_folded_mask(data=unfolded_data, dim=1, indices=sorted_idx, output_size=padded_data.shape, sparse_block_shape=sparse_block_shape)
        mask = mask.squeeze(0).squeeze(0)[:height, :width].contiguous()
        return mask

    def __get_data_level_mask(self, data, sparsity_level, sparse_block_shape):
        if False:
            for i in range(10):
                print('nop')
        (height, width) = (data.shape[-2], data.shape[-1])
        (block_height, block_width) = sparse_block_shape
        dh = (block_height - height % block_height) % block_height
        dw = (block_width - width % block_width) % block_width
        data_norm = F.avg_pool2d(data[None, None, :], kernel_size=sparse_block_shape, stride=sparse_block_shape, ceil_mode=True)
        values_per_block = reduce(lambda x, y: x * y, sparse_block_shape)
        data_norm = data_norm.flatten()
        num_blocks = len(data_norm)
        data_norm = data_norm.repeat(1, values_per_block, 1)
        (_, sorted_idx) = torch.sort(data_norm, dim=2)
        threshold_idx = round(sparsity_level * num_blocks)
        sorted_idx = sorted_idx[:, :, :threshold_idx]
        mask = self.__get_scatter_folded_mask(data=data_norm, dim=2, indices=sorted_idx, output_size=(height + dh, width + dw), sparse_block_shape=sparse_block_shape)
        mask = mask.squeeze(0).squeeze(0)[:height, :width]
        return mask

    def update_mask(self, name, data, sparsity_level, sparse_block_shape, zeros_per_block, **kwargs):
        if False:
            while True:
                i = 10
        values_per_block = reduce(lambda x, y: x * y, sparse_block_shape)
        if zeros_per_block > values_per_block:
            raise ValueError('Number of zeros per block cannot be more than the total number of elements in that block.')
        if zeros_per_block < 0:
            raise ValueError('Number of zeros per block should be positive.')
        if self.norm == 'L1':
            data_norm = torch.abs(data).squeeze()
        else:
            data_norm = (data * data).squeeze()
        if len(data_norm.shape) > 2:
            raise ValueError('only supports 2-D at the moment')
        elif len(data_norm.shape) == 1:
            data_norm = data_norm[None, :]
        mask = self.get_mask(name)
        if sparsity_level <= 0 or zeros_per_block == 0:
            mask.data = torch.ones_like(mask)
        elif sparsity_level >= 1.0 and zeros_per_block == values_per_block:
            mask.data = torch.zeros_like(mask)
        data_lvl_mask = self.__get_data_level_mask(data=data_norm, sparsity_level=sparsity_level, sparse_block_shape=sparse_block_shape)
        block_lvl_mask = self.__get_block_level_mask(data=data_norm, sparse_block_shape=sparse_block_shape, zeros_per_block=zeros_per_block)
        mask.data = torch.where(data_lvl_mask == 1, data_lvl_mask, block_lvl_mask)