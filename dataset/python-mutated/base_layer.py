import torch.nn as nn
import torch
import sys
from fairseq import utils
from fairseq.distributed import utils as distributed_utils
from fairseq.modules.layer_norm import LayerNorm

class BaseLayer(nn.Module):

    def __init__(self, args):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.num_workers = distributed_utils.get_data_parallel_world_size()
        expert_centroids = torch.empty(self.num_workers, args.decoder_embed_dim)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        self.register_parameter('expert_centroids', torch.nn.Parameter(expert_centroids))
        self.expert_network = nn.Sequential(*[BaseSublayer(args) for _ in range(args.base_sublayers)])
        self.expert_id = distributed_utils.get_data_parallel_rank()
        self.shuffle = args.base_shuffle
        self.cpp = self.load_assignment()
        for param in self.expert_network.parameters():
            param.expert = True

    def forward(self, input_features, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        features = input_features.reshape(-1, input_features.size(-1))
        is_training = input_features.requires_grad
        if self.shuffle and is_training:
            shuffle_sort = torch.randperm(features.size(0), device=features.device)
            features = All2All.apply(features[shuffle_sort])
        with torch.no_grad():
            token_expert_affinities = features.matmul(self.expert_centroids.transpose(0, 1))
        (sort_by_expert, input_splits, output_splits) = self.balanced_assignment(token_expert_affinities) if is_training else self.greedy_assignment(token_expert_affinities)
        routed_features = All2All.apply(features[sort_by_expert], output_splits, input_splits)
        if routed_features.size(0) > 0:
            alpha = torch.sigmoid(routed_features.mv(self.expert_centroids[self.expert_id])).unsqueeze(1)
            routed_features = alpha * self.expert_network(routed_features) + (1 - alpha) * routed_features
        result = All2All.apply(routed_features, input_splits, output_splits)[self.inverse_sort(sort_by_expert)]
        if self.shuffle and is_training:
            result = All2All.apply(result)[self.inverse_sort(shuffle_sort)]
        return (result.view(input_features.size()), None, None)

    def inverse_sort(self, order):
        if False:
            while True:
                i = 10
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

    def balanced_assignment(self, scores):
        if False:
            for i in range(10):
                print('nop')
        ok = scores.isfinite()
        if not ok.all():
            scores[~ok] = scores[ok].min()
        return (self.cpp.balanced_assignment(scores), None, None)

    def greedy_assignment(self, scores, k=1):
        if False:
            i = 10
            return i + 15
        token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
        (token_to_workers, sort_ordering) = torch.sort(token_to_workers)
        worker2token = sort_ordering // k
        output_splits = torch.zeros((self.num_workers,), dtype=torch.long, device=scores.device)
        (workers, counts) = torch.unique_consecutive(token_to_workers, return_counts=True)
        output_splits[workers] = counts
        input_splits = All2All.apply(output_splits)
        return (worker2token, input_splits.tolist(), output_splits.tolist())

    def load_assignment(self):
        if False:
            i = 10
            return i + 15
        try:
            from fairseq import libbase
            return libbase
        except ImportError as e:
            sys.stderr.write('ERROR: missing libbase. run `python setup.py build_ext --inplace`\n')
            raise e

class BaseSublayer(nn.Module):

    def __init__(self, args):
        if False:
            print('Hello World!')
        super().__init__()
        self.activation_fn = utils.get_activation_fn(activation=getattr(args, 'activation_fn', 'relu') or 'relu')
        self.norm = LayerNorm(args.decoder_embed_dim, export=False)
        self.ff1 = torch.nn.Linear(args.decoder_embed_dim, args.decoder_ffn_embed_dim)
        self.ff2 = torch.nn.Linear(args.decoder_ffn_embed_dim, args.decoder_embed_dim)
        self.ff2.weight.data.zero_()

    def forward(self, xs):
        if False:
            for i in range(10):
                print('nop')
        return xs + self.ff2(self.activation_fn(self.ff1(self.norm(xs))))

class All2All(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xs, input_splits=None, output_splits=None):
        if False:
            print('Hello World!')
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ys = torch.empty_like(xs) if output_splits is None else xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        torch.distributed.all_to_all_single(ys, xs, output_split_sizes=output_splits, input_split_sizes=input_splits)
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            for i in range(10):
                print('nop')
        result = torch.empty_like(grad_output) if ctx.input_splits is None else grad_output.new_empty(size=[sum(ctx.input_splits)] + list(grad_output.size()[1:]))
        torch.distributed.all_to_all_single(result, grad_output, output_split_sizes=ctx.input_splits, input_split_sizes=ctx.output_splits)
        return (result, None, None)