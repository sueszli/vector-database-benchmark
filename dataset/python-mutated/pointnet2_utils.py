from typing import Tuple
import pointnet2_cuda as pointnet2
import torch
import torch.nn as nn
from torch.autograd import Function, Variable

class FurthestPointSampling(Function):

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Uses iterative furthest point sampling to select a set of npoint features that have the largest\n        minimum distance\n        :param ctx:\n        :param xyz: (B, N, 3) where N > npoint\n        :param npoint: int, number of features in the sampled set\n        :return:\n             output: (B, npoint) tensor containing the set\n        '
        assert xyz.is_contiguous()
        (B, N, _) = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(10000000000.0)
        pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        if False:
            return 10
        return (None, None)
furthest_point_sample = FurthestPointSampling.apply

class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        :param ctx:\n        :param features: (B, C, N)\n        :param idx: (B, npoint) index tensor of the features to gather\n        :return:\n            output: (B, C, npoint)\n        '
        assert features.is_contiguous()
        assert idx.is_contiguous()
        (B, npoint) = idx.size()
        (_, C, N) = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)
        pointnet2.gather_points_wrapper(B, C, N, npoint, features, idx, output)
        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        if False:
            while True:
                i = 10
        (idx, C, N) = ctx.for_backwards
        (B, npoint) = idx.size()
        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return (grad_features, None)
gather_operation = GatherOperation.apply

class KNN(Function):

    @staticmethod
    def forward(ctx, k: int, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Find the three nearest neighbors of unknown in known\n        :param ctx:\n        :param unknown: (B, N, 3)\n        :param known: (B, M, 3)\n        :return:\n            dist: (B, N, k) l2 distance to the three nearest neighbors\n            idx: (B, N, k) index of 3 nearest neighbors\n        '
        assert unknown.is_contiguous()
        assert known.is_contiguous()
        (B, N, _) = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, k)
        idx = torch.cuda.IntTensor(B, N, k)
        pointnet2.knn_wrapper(B, N, m, k, unknown, known, dist2, idx)
        return (torch.sqrt(dist2), idx)

    @staticmethod
    def backward(ctx, a=None, b=None):
        if False:
            return 10
        return (None, None, None)
knn = KNN.apply

class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            print('Hello World!')
        '\n        Find the three nearest neighbors of unknown in known\n        :param ctx:\n        :param unknown: (B, N, 3)\n        :param known: (B, M, 3)\n        :return:\n            dist: (B, N, 3) l2 distance to the three nearest neighbors\n            idx: (B, N, 3) index of 3 nearest neighbors\n        '
        assert unknown.is_contiguous()
        assert known.is_contiguous()
        (B, N, _) = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)
        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return (torch.sqrt(dist2), idx)

    @staticmethod
    def backward(ctx, a=None, b=None):
        if False:
            for i in range(10):
                print('nop')
        return (None, None)
three_nn = ThreeNN.apply

class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Performs weight linear interpolation on 3 features\n        :param ctx:\n        :param features: (B, C, M) Features descriptors to be interpolated from\n        :param idx: (B, n, 3) three nearest neighbors of the target features in features\n        :param weight: (B, n, 3) weights\n        :return:\n            output: (B, C, N) tensor of the interpolated features\n        '
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()
        (B, c, m) = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)
        pointnet2.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if False:
            while True:
                i = 10
        '\n        :param ctx:\n        :param grad_out: (B, C, N) tensor with gradients of outputs\n        :return:\n            grad_features: (B, C, M) tensor with gradients of features\n            None:\n            None:\n        '
        (idx, weight, m) = ctx.three_interpolate_for_backward
        (B, c, n) = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return (grad_features, None, None)
three_interpolate = ThreeInterpolate.apply

class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        :param ctx:\n        :param features: (B, C, N) tensor of features to group\n        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with\n        :return:\n            output: (B, C, npoint, nsample) tensor\n        '
        assert features.is_contiguous()
        assert idx.is_contiguous()
        idx = idx.int()
        (B, nfeatures, nsample) = idx.size()
        (_, C, N) = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)
        pointnet2.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)
        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            print('Hello World!')
        '\n        :param ctx:\n        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward\n        :return:\n            grad_features: (B, C, N) gradient of the features\n        '
        (idx, N) = ctx.for_backwards
        (B, C, npoint, nsample) = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return (grad_features, None)
grouping_operation = GroupingOperation.apply

class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        :param ctx:\n        :param radius: float, radius of the balls\n        :param nsample: int, maximum number of features in the balls\n        :param xyz: (B, N, 3) xyz coordinates of the features\n        :param new_xyz: (B, npoint, 3) centers of the ball query\n        :return:\n            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls\n        '
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()
        (B, N, _) = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()
        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        if False:
            while True:
                i = 10
        return (None, None, None, None)
ball_query = BallQuery.apply

class QueryAndGroup(nn.Module):

    def __init__(self, radius: float, nsample: int, use_xyz: bool=True):
        if False:
            while True:
                i = 10
        '\n        :param radius: float, radius of ball\n        :param nsample: int, maximum number of features to gather in the ball\n        :param use_xyz:\n        '
        super().__init__()
        (self.radius, self.nsample, self.use_xyz) = (radius, nsample, use_xyz)

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None) -> Tuple[torch.Tensor]:
        if False:
            return 10
        '\n        :param xyz: (B, N, 3) xyz coordinates of the features\n        :param new_xyz: (B, npoint, 3) centroids\n        :param features: (B, C, N) descriptors of the features\n        :return:\n            new_features: (B, 3 + C, npoint, nsample)\n        '
        (B, N, C) = new_xyz.shape
        (dist, idx) = knn(self.nsample, new_xyz, xyz)
        if self.radius is not None:
            tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, self.nsample).to(idx.device)
            idx[dist > self.radius] = tmp_idx[dist > self.radius]
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz
        return (new_features, grouped_xyz)

class GroupAll(nn.Module):

    def __init__(self, use_xyz: bool=True):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None):
        if False:
            print('Hello World!')
        '\n        :param xyz: (B, N, 3) xyz coordinates of the features\n        :param new_xyz: ignored\n        :param features: (B, C, N) descriptors of the features\n        :return:\n            new_features: (B, C + 3, 1, N)\n        '
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        return (new_features, grouped_xyz)