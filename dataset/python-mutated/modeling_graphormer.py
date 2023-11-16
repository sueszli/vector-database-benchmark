""" PyTorch Graphormer model."""
import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithNoAttention, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_graphormer import GraphormerConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'graphormer-base-pcqm4mv1'
_CONFIG_FOR_DOC = 'GraphormerConfig'
GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ['clefourrier/graphormer-base-pcqm4mv1', 'clefourrier/graphormer-base-pcqm4mv2']

def quant_noise(module: nn.Module, p: float, block_size: int):
    if False:
        return 10
    '\n    From:\n    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/quant_noise.py\n\n    Wraps modules and applies quantization noise to the weights for subsequent quantization with Iterative Product\n    Quantization as described in "Training with Quantization Noise for Extreme Model Compression"\n\n    Args:\n        - module: nn.Module\n        - p: amount of Quantization Noise\n        - block_size: size of the blocks for subsequent quantization with iPQ\n\n    Remarks:\n        - Module weights must have the right sizes wrt the block size\n        - Only Linear, Embedding and Conv2d modules are supported for the moment\n        - For more detail on how to quantize by blocks with convolutional weights, see "And the Bit Goes Down:\n          Revisiting the Quantization of Neural Networks"\n        - We implement the simplest form of noise here as stated in the paper which consists in randomly dropping\n          blocks\n    '
    if p <= 0:
        return module
    if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
        raise NotImplementedError('Module unsupported for quant_noise.')
    is_conv = module.weight.ndim == 4
    if not is_conv:
        if module.weight.size(1) % block_size != 0:
            raise AssertionError('Input features must be a multiple of block sizes')
    elif module.kernel_size == (1, 1):
        if module.in_channels % block_size != 0:
            raise AssertionError('Input channels must be a multiple of block sizes')
    else:
        k = module.kernel_size[0] * module.kernel_size[1]
        if k % block_size != 0:
            raise AssertionError('Kernel size must be a multiple of block size')

    def _forward_pre_hook(mod, input):
        if False:
            return 10
        if mod.training:
            if not is_conv:
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
            else:
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
            mask = mask.to(torch.bool)
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)
    module.register_forward_pre_hook(_forward_pre_hook)
    return module

class LayerDropModuleList(nn.ModuleList):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/layer_drop.py
    A LayerDrop implementation based on [`torch.nn.ModuleList`]. LayerDrop as described in
    https://arxiv.org/abs/1909.11556.

    We refresh the choice of which layers to drop every time we iterate over the LayerDropModuleList instance. During
    evaluation we always iterate over all layers.

    Usage:

    ```python
    layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
    for layer in layers:  # this might iterate over layers 1 and 3
        x = layer(x)
    for layer in layers:  # this might iterate over all layers
        x = layer(x)
    for layer in layers:  # this might not iterate over any layers
        x = layer(x)
    ```

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p: float, modules: Optional[Iterable[nn.Module]]=None):
        if False:
            i = 10
            return i + 15
        super().__init__(modules)
        self.p = p

    def __iter__(self) -> Iterator[nn.Module]:
        if False:
            print('Hello World!')
        dropout_probs = torch.empty(len(self)).uniform_()
        for (i, m) in enumerate(super().__iter__()):
            if not self.training or dropout_probs[i] > self.p:
                yield m

class GraphormerGraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, config: GraphormerConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_atoms = config.num_atoms
        self.atom_encoder = nn.Embedding(config.num_atoms + 1, config.hidden_size, padding_idx=config.pad_token_id)
        self.in_degree_encoder = nn.Embedding(config.num_in_degree, config.hidden_size, padding_idx=config.pad_token_id)
        self.out_degree_encoder = nn.Embedding(config.num_out_degree, config.hidden_size, padding_idx=config.pad_token_id)
        self.graph_token = nn.Embedding(1, config.hidden_size)

    def forward(self, input_nodes: torch.LongTensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        (n_graph, n_node) = input_nodes.size()[:2]
        node_feature = self.atom_encoder(input_nodes).sum(dim=-2) + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        return graph_node_feature

class GraphormerGraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(self, config: GraphormerConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.multi_hop_max_dist = config.multi_hop_max_dist
        self.edge_encoder = nn.Embedding(config.num_edges + 1, config.num_attention_heads, padding_idx=0)
        self.edge_type = config.edge_type
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(config.num_edge_dis * config.num_attention_heads * config.num_attention_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(config.num_spatial, config.num_attention_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, config.num_attention_heads)

    def forward(self, input_nodes: torch.LongTensor, attn_bias: torch.Tensor, spatial_pos: torch.LongTensor, input_edges: torch.LongTensor, attn_edge_type: torch.LongTensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        (n_graph, n_node) = input_nodes.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        if self.edge_type == 'multi_hop':
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                input_edges = input_edges[:, :, :, :self.multi_hop_max_dist, :]
            input_edges = self.edge_encoder(input_edges).mean(-2)
            max_dist = input_edges.size(-2)
            edge_input_flat = input_edges.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, :, :])
            input_edges = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            input_edges = (input_edges.sum(-2) / spatial_pos_.float().unsqueeze(-1)).permute(0, 3, 1, 2)
        else:
            input_edges = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + input_edges
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)
        return graph_attn_bias

class GraphormerMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, config: GraphormerConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.kdim = config.kdim if config.kdim is not None else config.embedding_dim
        self.vdim = config.vdim if config.vdim is not None else config.embedding_dim
        self.qkv_same_dim = self.kdim == config.embedding_dim and self.vdim == config.embedding_dim
        self.num_heads = config.num_attention_heads
        self.attention_dropout_module = torch.nn.Dropout(p=config.attention_dropout, inplace=False)
        self.head_dim = config.embedding_dim // config.num_attention_heads
        if not self.head_dim * config.num_attention_heads == self.embedding_dim:
            raise AssertionError('The embedding_dim must be divisible by num_heads.')
        self.scaling = self.head_dim ** (-0.5)
        self.self_attention = True
        if not self.self_attention:
            raise NotImplementedError('The Graphormer model only supports self attention for now.')
        if self.self_attention and (not self.qkv_same_dim):
            raise AssertionError('Self-attention requires query, key and value to be of the same size.')
        self.k_proj = quant_noise(nn.Linear(self.kdim, config.embedding_dim, bias=config.bias), config.q_noise, config.qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, config.embedding_dim, bias=config.bias), config.q_noise, config.qn_block_size)
        self.q_proj = quant_noise(nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias), config.q_noise, config.qn_block_size)
        self.out_proj = quant_noise(nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias), config.q_noise, config.qn_block_size)
        self.onnx_trace = False

    def reset_parameters(self):
        if False:
            return 10
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query: torch.LongTensor, key: Optional[torch.Tensor], value: Optional[torch.Tensor], attn_bias: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]=None, need_weights: bool=True, attn_mask: Optional[torch.Tensor]=None, before_softmax: bool=False, need_head_weights: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if False:
            return 10
        '\n        Args:\n            key_padding_mask (Bytetorch.Tensor, optional): mask to exclude\n                keys that are pads, of shape `(batch, src_len)`, where padding elements are indicated by 1s.\n            need_weights (bool, optional): return the attention weights,\n                averaged over heads (default: False).\n            attn_mask (Bytetorch.Tensor, optional): typically used to\n                implement causal attention, where the mask prevents the attention from looking forward in time\n                (default: None).\n            before_softmax (bool, optional): return the raw attention\n                weights and values before the attention softmax.\n            need_head_weights (bool, optional): return the attention\n                weights for each head. Implies *need_weights*. Default: return the average attention weights over all\n                heads.\n        '
        if need_head_weights:
            need_weights = True
        (tgt_len, bsz, embedding_dim) = query.size()
        src_len = tgt_len
        if not embedding_dim == self.embedding_dim:
            raise AssertionError(f'The query embedding dimension {embedding_dim} is not equal to the expected embedding_dim {self.embedding_dim}.')
        if not list(query.size()) == [tgt_len, bsz, embedding_dim]:
            raise AssertionError('Query size incorrect in Graphormer, compared to model dimensions.')
        if key is not None:
            (src_len, key_bsz, _) = key.size()
            if not torch.jit.is_scripting():
                if key_bsz != bsz or value is None or (not (src_len, bsz == value.shape[:2])):
                    raise AssertionError('The batch shape does not match the key or value shapes provided to the attention.')
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is None or not k.size(1) == src_len:
            raise AssertionError('The shape of the key generated in the attention is incorrect')
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz or key_padding_mask.size(1) != src_len:
                raise AssertionError('The shape of the generated padding mask for the key does not match expected dimensions.')
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        if list(attn_weights.size()) != [bsz * self.num_heads, tgt_len, src_len]:
            raise AssertionError('The attention weights generated do not match the expected dimensions.')
        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if before_softmax:
            return (attn_weights, v)
        attn_weights_float = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.attention_dropout_module(attn_weights)
        if v is None:
            raise AssertionError('No value generated')
        attn = torch.bmm(attn_probs, v)
        if list(attn.size()) != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise AssertionError('The attention generated do not match the expected dimensions.')
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embedding_dim)
        attn: torch.Tensor = self.out_proj(attn)
        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.contiguous().view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        return (attn, attn_weights)

    def apply_sparse_mask(self, attn_weights: torch.Tensor, tgt_len: int, src_len: int, bsz: int) -> torch.Tensor:
        if False:
            print('Hello World!')
        return attn_weights

class GraphormerGraphEncoderLayer(nn.Module):

    def __init__(self, config: GraphormerConfig) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_attention_heads = config.num_attention_heads
        self.q_noise = config.q_noise
        self.qn_block_size = config.qn_block_size
        self.pre_layernorm = config.pre_layernorm
        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)
        self.activation_dropout_module = torch.nn.Dropout(p=config.activation_dropout, inplace=False)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.self_attn = GraphormerMultiheadAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = self.build_fc(self.embedding_dim, config.ffn_embedding_dim, q_noise=config.q_noise, qn_block_size=config.qn_block_size)
        self.fc2 = self.build_fc(config.ffn_embedding_dim, self.embedding_dim, q_noise=config.q_noise, qn_block_size=config.qn_block_size)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def build_fc(self, input_dim: int, output_dim: int, q_noise: float, qn_block_size: int) -> Union[nn.Module, nn.Linear, nn.Embedding, nn.Conv2d]:
        if False:
            while True:
                i = 10
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def forward(self, input_nodes: torch.Tensor, self_attn_bias: Optional[torch.Tensor]=None, self_attn_mask: Optional[torch.Tensor]=None, self_attn_padding_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        nn.LayerNorm is applied either before or after the self-attention/ffn modules similar to the original\n        Transformer implementation.\n        '
        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)
        (input_nodes, attn) = self.self_attn(query=input_nodes, key=input_nodes, value=input_nodes, attn_bias=self_attn_bias, key_padding_mask=self_attn_padding_mask, need_weights=False, attn_mask=self_attn_mask)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)
        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        input_nodes = self.activation_fn(self.fc1(input_nodes))
        input_nodes = self.activation_dropout_module(input_nodes)
        input_nodes = self.fc2(input_nodes)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        return (input_nodes, attn)

class GraphormerGraphEncoder(nn.Module):

    def __init__(self, config: GraphormerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)
        self.layerdrop = config.layerdrop
        self.embedding_dim = config.embedding_dim
        self.apply_graphormer_init = config.apply_graphormer_init
        self.traceable = config.traceable
        self.graph_node_feature = GraphormerGraphNodeFeature(config)
        self.graph_attn_bias = GraphormerGraphAttnBias(config)
        self.embed_scale = config.embed_scale
        if config.q_noise > 0:
            self.quant_noise = quant_noise(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), config.q_noise, config.qn_block_size)
        else:
            self.quant_noise = None
        if config.encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None
        if config.pre_layernorm:
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([GraphormerGraphEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        if config.freeze_embeddings:
            raise NotImplementedError('Freezing embeddings is not implemented yet.')
        for layer in range(config.num_trans_layers_to_freeze):
            m = self.layers[layer]
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, input_nodes: torch.LongTensor, input_edges: torch.LongTensor, attn_bias: torch.Tensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor, spatial_pos: torch.LongTensor, attn_edge_type: torch.LongTensor, perturb=None, last_state_only: bool=False, token_embeddings: Optional[torch.Tensor]=None, attn_mask: Optional[torch.Tensor]=None) -> Tuple[Union[torch.Tensor, List[torch.LongTensor]], torch.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        data_x = input_nodes
        (n_graph, n_node) = data_x.size()[:2]
        padding_mask = data_x[:, :, 0].eq(0)
        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        attn_bias = self.graph_attn_bias(input_nodes, attn_bias, spatial_pos, input_edges, attn_edge_type)
        if token_embeddings is not None:
            input_nodes = token_embeddings
        else:
            input_nodes = self.graph_node_feature(input_nodes, in_degree, out_degree)
        if perturb is not None:
            input_nodes[:, 1:, :] += perturb
        if self.embed_scale is not None:
            input_nodes = input_nodes * self.embed_scale
        if self.quant_noise is not None:
            input_nodes = self.quant_noise(input_nodes)
        if self.emb_layer_norm is not None:
            input_nodes = self.emb_layer_norm(input_nodes)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = input_nodes.transpose(0, 1)
        inner_states = []
        if not last_state_only:
            inner_states.append(input_nodes)
        for layer in self.layers:
            (input_nodes, _) = layer(input_nodes, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=attn_bias)
            if not last_state_only:
                inner_states.append(input_nodes)
        graph_rep = input_nodes[0, :, :]
        if last_state_only:
            inner_states = [input_nodes]
        if self.traceable:
            return (torch.stack(inner_states), graph_rep)
        else:
            return (inner_states, graph_rep)

class GraphormerDecoderHead(nn.Module):

    def __init__(self, embedding_dim: int, num_classes: int):
        if False:
            print('Hello World!')
        super().__init__()
        'num_classes should be 1 for regression, or the number of classes for classification'
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.num_classes = num_classes

    def forward(self, input_nodes: torch.Tensor, **unused) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        input_nodes = self.classifier(input_nodes)
        input_nodes = input_nodes + self.lm_output_learned_bias
        return input_nodes

class GraphormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GraphormerConfig
    base_model_prefix = 'graphormer'
    main_input_name_nodes = 'input_nodes'
    main_input_name_edges = 'input_edges'

    def normal_(self, data: torch.Tensor):
        if False:
            while True:
                i = 10
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    def init_graphormer_params(self, module: Union[nn.Linear, nn.Embedding, GraphormerMultiheadAttention]):
        if False:
            print('Hello World!')
        '\n        Initialize the weights specific to the Graphormer Model.\n        '
        if isinstance(module, nn.Linear):
            self.normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            self.normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, GraphormerMultiheadAttention):
            self.normal_(module.q_proj.weight.data)
            self.normal_(module.k_proj.weight.data)
            self.normal_(module.v_proj.weight.data)

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, GraphormerMultiheadAttention, GraphormerGraphEncoder]):
        if False:
            print('Hello World!')
        '\n        Initialize the weights\n        '
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GraphormerMultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.reset_parameters()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, GraphormerGraphEncoder):
            if module.apply_graphormer_init:
                module.apply(self.init_graphormer_params)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class GraphormerModel(GraphormerPreTrainedModel):
    """The Graphormer model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    GraphormerForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in GraphormerForGraphClassification.
    """

    def __init__(self, config: GraphormerConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.max_nodes = config.max_nodes
        self.graph_encoder = GraphormerGraphEncoder(config)
        self.share_input_output_embed = config.share_input_output_embed
        self.lm_output_learned_bias = None
        self.load_softmax = not getattr(config, 'remove_head', False)
        self.lm_head_transform_weight = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.post_init()

    def reset_output_layer_parameters(self):
        if False:
            print('Hello World!')
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

    def forward(self, input_nodes: torch.LongTensor, input_edges: torch.LongTensor, attn_bias: torch.Tensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor, spatial_pos: torch.LongTensor, attn_edge_type: torch.LongTensor, perturb: Optional[torch.FloatTensor]=None, masked_tokens: None=None, return_dict: Optional[bool]=None, **unused) -> Union[Tuple[torch.LongTensor], BaseModelOutputWithNoAttention]:
        if False:
            for i in range(10):
                print('nop')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        (inner_states, graph_rep) = self.graph_encoder(input_nodes, input_edges, attn_bias, in_degree, out_degree, spatial_pos, attn_edge_type, perturb=perturb)
        input_nodes = inner_states[-1].transpose(0, 1)
        if masked_tokens is not None:
            raise NotImplementedError
        input_nodes = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(input_nodes)))
        if self.share_input_output_embed and hasattr(self.graph_encoder.embed_tokens, 'weight'):
            input_nodes = torch.nn.functional.linear(input_nodes, self.graph_encoder.embed_tokens.weight)
        if not return_dict:
            return tuple((x for x in [input_nodes, inner_states] if x is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=input_nodes, hidden_states=inner_states)

    def max_nodes(self):
        if False:
            i = 10
            return i + 15
        'Maximum output length supported by the encoder.'
        return self.max_nodes

class GraphormerForGraphClassification(GraphormerPreTrainedModel):
    """
    This model can be used for graph-level classification or regression tasks.

    It can be trained on
    - regression (by setting config.num_classes to 1); there should be one float-type label per graph
    - one task classification (by setting config.num_classes to the number of classes); there should be one integer
      label per graph
    - binary multi-task classification (by setting config.num_classes to the number of labels); there should be a list
      of integer labels for each graph.
    """

    def __init__(self, config: GraphormerConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.encoder = GraphormerModel(config)
        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.classifier = GraphormerDecoderHead(self.embedding_dim, self.num_classes)
        self.is_encoder_decoder = True
        self.post_init()

    def forward(self, input_nodes: torch.LongTensor, input_edges: torch.LongTensor, attn_bias: torch.Tensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor, spatial_pos: torch.LongTensor, attn_edge_type: torch.LongTensor, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, **unused) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if False:
            return 10
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs = self.encoder(input_nodes, input_edges, attn_bias, in_degree, out_degree, spatial_pos, attn_edge_type, return_dict=True)
        (outputs, hidden_states) = (encoder_outputs['last_hidden_state'], encoder_outputs['hidden_states'])
        head_outputs = self.classifier(outputs)
        logits = head_outputs[:, 0, :].contiguous()
        loss = None
        if labels is not None:
            mask = ~torch.isnan(labels)
            if self.num_classes == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits[mask].squeeze(), labels[mask].squeeze().float())
            elif self.num_classes > 1 and len(labels.shape) == 1:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[mask].view(-1, self.num_classes), labels[mask].view(-1))
            else:
                loss_fct = BCEWithLogitsLoss(reduction='sum')
                loss = loss_fct(logits[mask], labels[mask])
        if not return_dict:
            return tuple((x for x in [loss, logits, hidden_states] if x is not None))
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=None)