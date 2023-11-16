import jittor as jt
import jittor.nn as nn
cpu_src = '\n    for (int bs = 0; bs < in0_shape0; ++bs)\n        for (int i = 0; i < in0_shape1; ++i) {\n            float min_dis = (@in0(bs, i, 0) - @in1(bs, 0, 0)) * (@in0(bs, i, 0) - @in1(bs, 0, 0)) +\n                            (@in0(bs, i, 1) - @in1(bs, 0, 1)) * (@in0(bs, i, 1) - @in1(bs, 0, 1)) +\n                            (@in0(bs, i, 2) - @in1(bs, 0, 2)) * (@in0(bs, i, 2) - @in1(bs, 0, 2));\n            @out(bs, i) = 0;\n            for (int j = 1; j < in1_shape1; ++j) {\n                float dis = (@in0(bs, i, 0) - @in1(bs, j, 0)) * (@in0(bs, i, 0) - @in1(bs, j, 0)) +\n                            (@in0(bs, i, 1) - @in1(bs, j, 1)) * (@in0(bs, i, 1) - @in1(bs, j, 1)) +\n                            (@in0(bs, i, 2) - @in1(bs, j, 2)) * (@in0(bs, i, 2) - @in1(bs, j, 2));\n                if (dis < min_dis) {\n                    min_dis = dis;\n                    @out(bs, i) = j;\n                }\n            }\n        }\n'
cuda_src = '\n    __global__ void chamfer_loss_min_idx_kernel(@ARGS_DEF) {\n        @PRECALC\n        int bs = blockIdx.x;\n        int n = in0_shape1;\n        int m = in1_shape1;\n\n        for (int i = threadIdx.x; i < n; i += blockDim.x) {\n            float min_dis = (@in0(bs, i, 0) - @in1(bs, 0, 0)) * (@in0(bs, i, 0) - @in1(bs, 0, 0)) +\n                            (@in0(bs, i, 1) - @in1(bs, 0, 1)) * (@in0(bs, i, 1) - @in1(bs, 0, 1)) +\n                            (@in0(bs, i, 2) - @in1(bs, 0, 2)) * (@in0(bs, i, 2) - @in1(bs, 0, 2));\n            @out(bs, i) = 0;\n            for (int j = 1; j < m; ++j) {\n                float dis = (@in0(bs, i, 0) - @in1(bs, j, 0)) * (@in0(bs, i, 0) - @in1(bs, j, 0)) +\n                            (@in0(bs, i, 1) - @in1(bs, j, 1)) * (@in0(bs, i, 1) - @in1(bs, j, 1)) +\n                            (@in0(bs, i, 2) - @in1(bs, j, 2)) * (@in0(bs, i, 2) - @in1(bs, j, 2));\n                if (dis < min_dis) {\n                    min_dis = dis;\n                    @out(bs, i) = j;\n                }\n            }\n        }\n    }\n\n    chamfer_loss_min_idx_kernel<<<in0_shape0, 512>>>(@ARGS);\n'

def chamfer_loss(pc1, pc2, reduction='mean', dims='BNC', bidirectional=False):
    if False:
        print('Hello World!')
    " return the chamfer loss from pc1 to pc2.\n\n    :param pc1:  input point cloud\n    :type pc1: jittor array\n\n    :param pc2:  input point cloud\n    :type pc2: jittor array\n\n    :param reduction: reduction method in batches, can be 'mean', 'sum', or None. Default: 'mean'.\n    :type reduction: str, optional\n            \n    :param dims: a string that represents each dimension, can be\n            '[BNC]' ([batch, number of points, xyz]), or\n            '[BCN]' ([batch, xyz, number of points]). Default: 'BNC'.\n    :type dims: str, optional\n\n    Example:\n\n    >>> import jittor as jt\n    >>> from jittor.loss3d import chamfer_loss\n    >>> jt.flags.use_cuda = True\n    >>> pc1 = jt.rand([10, 100, 3], dtype=jt.float32)\n    >>> pc2 = jt.rand([10, 100, 3], dtype=jt.float32)\n    >>> cf = chamfer_loss(pc1, pc2, dims='BNC', bidirectional=True)\n    >>> print('chamfer loss =', cf.item())\n    "
    if bidirectional:
        return chamfer_loss(pc1, pc2, reduction, dims) + chamfer_loss(pc2, pc1, reduction, dims)
    assert dims in ['BNC', 'BCN']
    if dims == 'BCN':
        (pc1, pc2) = (pc1.permute(0, 2, 1), pc2.permute(0, 2, 1))
    (batch_size_1, N, _) = pc1.shape
    (batch_size_2, M, _) = pc2.shape
    assert batch_size_1 == batch_size_2
    batch_size = batch_size_1
    idx = jt.code([batch_size, N], 'int32', [pc1, pc2], cpu_src=cpu_src, cuda_src=cuda_src)
    nearest_pts = pc2.reindex([batch_size, idx.shape[1], 3], ['i0', '@e0(i0, i1)', 'i2'], extras=[idx])
    chamfer_distance = ((pc1 - nearest_pts) ** 2).sum(dim=-1).sqrt()
    if reduction is None:
        return chamfer_distance
    elif reduction == 'sum':
        return jt.sum(chamfer_distance)
    elif reduction == 'mean':
        return jt.mean(chamfer_distance)

class ChamferLoss(nn.Module):
    """ A loss layer that computes the chamfer loss from pc1 to pc2.

    :param pc1:  input point cloud
    :type pc1: jittor array

    :param pc2:  input point cloud
    :type pc2: jittor array

    :param reduction: reduction method in batches, can be 'mean', 'sum', or None. Default: 'mean'.
    :type reduction: str, optional
            
    :param dims: a string that represents each dimension, can be
            '[BNC]' ([batch, number of points, xyz]), or
            '[BCN]' ([batch, xyz, number of points]). Default: 'BNC'.
    :type dims: str, optional

    Example:

    >>> import jittor as jt
    >>> from jittor.loss3d import ChamferLoss
    >>> jt.flags.use_cuda = True
    >>> pc1 = jt.rand([10, 100, 3], dtype=jt.float32)
    >>> pc2 = jt.rand([10, 100, 3], dtype=jt.float32)
    >>> CF = ChamferLoss(dims='BNC', bidirectional=True)
    >>> cf = CF(pc1, pc2)
    >>> print('chamfer loss =', cf.item())
    """

    def __init__(self, reduction='mean', dims='BNC', bidirectional=False):
        if False:
            return 10
        ' see function @chamfer_loss\n        '
        super().__init__()
        self.reduction = reduction
        self.dims = dims
        self.bidirectional = bidirectional

    def execute(self, pc1, pc2):
        if False:
            while True:
                i = 10
        return chamfer_loss(pc1, pc2, self.reduction, self.dims, self.bidirectional)