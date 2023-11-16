import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.modules import FairseqDropout, LayerDropModuleList, LayerNorm, PositionalEmbedding
from .speech_dlm_decoder_layer import CrossChannelTransformerDecoderLayer, StandardTransformerDecoderLayer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

class CrossChannelTransformerDecoder(FairseqIncrementalDecoder):
    """
    Cross-channel Transformer Decoder Block for parallel spoken dialogue units
    as described in the paper: https://arxiv.org/pdf/2203.16502.pdf;
    consisting of *args.decoder_layers* layers. Each layer is a
    :class:`StandardTransformerDecoderLayer` or
    :class:`CrossChannelTransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        channels (list): list of channel names (string)
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, channels, no_encoder_attn=False):
        if False:
            i = 10
            return i + 15
        self.args = args
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))
        self._future_mask = torch.empty(0)
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.channels = channels
        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)
        if args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(nn.Linear(embed_dim, embed_dim, bias=False), args.quant_noise_pq, args.quant_noise_pq_block_size)
        else:
            self.quant_noise = None
        self.project_in_dim = nn.Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None
        self.embed_positions = PositionalEmbedding(self.max_target_positions, embed_dim, self.padding_idx, learned=args.decoder_learned_pos) if not args.no_token_positional_embeddings else None
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None
        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        assert 0 <= args.decoder_cross_layers <= args.decoder_layers, f'The number of cross-channel attention decoder layers must be non-negativeand not exceeds the number of decoder layers (found {args.decoder_cross_layers})'
        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([self.build_decoder_layer(args, no_encoder_attn) if i < args.decoder_layers - args.decoder_cross_layers else self.build_cross_decoder_layer(args, no_encoder_attn) for i in range(args.decoder_layers)])
        self.num_layers = len(self.layers)
        self.non_cross_layers = args.decoder_layers - args.decoder_cross_layers
        if args.decoder_normalize_before and (not getattr(args, 'no_decoder_final_norm', False)):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        self.project_out_dim = nn.Linear(embed_dim, self.output_embed_dim, bias=False) if embed_dim != self.output_embed_dim else None
        self.output_projection = None
        self.is_cross_prediction = bool(float(args.main_and_cross_weights.split(',')[1]) != 0)
        self.n_output_projections = 1 if not self.is_cross_prediction else len(self.channels)
        if self.share_input_output_embed:
            self.output_projection = nn.ModuleList([nn.Linear(embed_tokens.weight.shape[1], embed_tokens.weight.shape[0], bias=False) for _ in range(self.n_output_projections)])
            self.output_projection[0].weight = embed_tokens.weight
            for i in range(1, self.n_output_projections):
                nn.init.normal_(self.output_projection[i].weight, mean=0, std=embed_tokens.weight.shape[1] ** (-0.5))
        else:
            self.output_projection = nn.ModuleList([nn.Linear(self.output_embed_dim, len(dictionary), bias=False) for _ in range(self.n_output_projections)])
            for i in range(self.n_output_projections):
                nn.init.normal_(self.output_projection[i].weight, mean=0, std=self.output_embed_dim ** (-0.5))
        self.output_duration_prediction = None if str(args.duration_prediction).lower() == 'false' else nn.ModuleList([nn.Linear(self.output_embed_dim, 1) for _ in range(self.n_output_projections)])

    def build_decoder_layer(self, args, no_encoder_attn=False):
        if False:
            return 10
        layer = StandardTransformerDecoderLayer(args, no_encoder_attn)
        if getattr(args, 'checkpoint_activations', False):
            offload_to_cpu = getattr(args, 'offload_activations', False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    def build_cross_decoder_layer(self, args, no_encoder_attn=False):
        if False:
            for i in range(10):
                print('nop')
        layer = CrossChannelTransformerDecoderLayer(args, no_encoder_attn)
        if getattr(args, 'checkpoint_activations', False):
            offload_to_cpu = getattr(args, 'offload_activations', False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    def forward(self, prev_output_tokens: Dict[str, Tensor], encoder_out: Optional[Dict[str, List[Tensor]]]=None, incremental_state: Optional[List[Dict[str, Dict[str, Optional[Tensor]]]]]=None, features_only: bool=False, full_context_alignment: bool=False, alignment_layer: Optional[int]=None, alignment_heads: Optional[int]=None, src_lengths: Optional[Any]=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Args:\n            prev_output_tokens (dict[str, LongTensor]): previous decoder outputs,\n                dictionary over all channels with the values being the tensors\n                of shape `(batch, tgt_len)`, for teacher forcing\n            encoder_out (optional): output from the encoder, used for\n                encoder-side attention\n            incremental_state (dict): list of dictionaries used for storing state\n                during :ref:`Incremental decoding`\n            features_only (bool, optional): only return features without\n                applying output layer (default: False).\n            full_context_alignment (bool, optional): don't apply\n                auto-regressive mask to self-attention (default: False).\n\n        Returns:\n            tuple:\n                - the decoder's output, dict over channels of tensors\n                    of shape `(batch, tgt_len, vocab)`\n                - a dictionary with any model-specific outputs\n        "
        (x, extra) = self.extract_features(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        if not features_only:
            x = self.output_layer(x)
        return (x, extra)

    def extract_features(self, prev_output_tokens: Dict[str, Tensor], encoder_out: Optional[Dict[str, List[Tensor]]], incremental_state: Optional[List[Dict[str, Dict[str, Optional[Tensor]]]]]=None, full_context_alignment: bool=False, alignment_layer: Optional[int]=None, alignment_heads: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
        return self.extract_features_scriptable(prev_output_tokens, encoder_out, incremental_state, full_context_alignment, alignment_layer, alignment_heads)
    '\n    A scriptable subclass of this class has an extract_features method and calls\n    super().extract_features, but super() is not supported in torchscript. A copy of\n    this function is made to be used in the subclass instead.\n    '

    def extract_features_scriptable(self, prev_output_tokens: Dict[str, Tensor], encoder_out: Optional[Dict[str, List[Tensor]]], incremental_state: Optional[List[Dict[str, Dict[str, Optional[Tensor]]]]]=None, full_context_alignment: bool=False, alignment_layer: Optional[int]=None, alignment_heads: Optional[int]=None):
        if False:
            while True:
                i = 10
        "\n        The core function of *forward* but only return features.\n\n        The input (prev_output_tokens) is a dictionary over all channels,\n        expected to have the following form:\n            {\n                'channel1' : Tensor((batch x tgt_len)),\n                'channel2' : Tensor((batch x tgt_len)),\n            }\n\n        Args:\n            full_context_alignment (bool, optional): don't apply\n                auto-regressive mask to self-attention (default: False).\n            alignment_layer (int, optional): return mean alignment over\n                heads at this layer (default: last layer).\n            alignment_heads (int, optional): only average alignment over\n                this many heads (default: all heads).\n\n        Returns:\n            tuple:\n                - the decoder's features, dict over channels of tensors\n                    of shape `(batch, tgt_len, embed_dim)`\n                - a dictionary with any model-specific outputs\n        "
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1
        x_list = []
        for (i, channel) in enumerate(self.channels):
            positions = None
            if self.embed_positions is not None:
                positions = self.embed_positions(prev_output_tokens[channel], incremental_state=incremental_state[i] if incremental_state is not None else None)
            if incremental_state is not None:
                prev_output_tokens[channel] = prev_output_tokens[channel][:, -1:]
                if positions is not None:
                    positions = positions[:, -1:]
            x = self.embed_tokens(prev_output_tokens[channel])
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
            x = self.embed_scale * x
            if self.quant_noise is not None:
                x = self.quant_noise(x)
            if positions is not None:
                x += positions
            if self.layernorm_embedding is not None:
                x = self.layernorm_embedding(x)
            x = self.dropout_module(x)
            x = x.transpose(0, 1)
            x_list.append(x)
        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens[self.channels[0]].eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens[self.channels[0]].eq(self.padding_idx)
        attn: Optional[Dict[Tensor]] = None
        inner_states: List[Optional[Dict[str, Tensor]]] = [{channel: x_list[i] for (i, channel) in enumerate(self.channels)}]
        for (idx, layer) in enumerate(self.layers):
            if incremental_state is None and (not full_context_alignment):
                self_attn_mask = self.buffered_future_mask(x_list[0])
            else:
                self_attn_mask = None
            if isinstance(x_list, list):
                x_list = torch.stack(x_list)
            (x_list, layer_attn_list, _) = layer(x_list, encoder_out['encoder_out'][0] if encoder_out is not None and len(encoder_out['encoder_out']) > 0 else None, encoder_out['encoder_padding_mask'][0] if encoder_out is not None and len(encoder_out['encoder_padding_mask']) > 0 else None, incremental_state, self_attn_mask=self_attn_mask, self_attn_padding_mask=self_attn_padding_mask, need_attn=bool(idx == alignment_layer), need_head_weights=bool(idx == alignment_layer))
            inner_states.append({channel: x_list[i] for (i, channel) in enumerate(self.channels)})
            if idx == alignment_layer and all((layer_attn is not None for layer_attn in layer_attn_list)):
                attn = {channel: layer_attn_list[i].float().to(x_list[0]) for (i, channel) in enumerate(self.channels)}
        if not isinstance(x_list, list):
            x_list = list(torch.unbind(x_list))
        if attn is not None:
            for channel in attn:
                if alignment_heads is not None:
                    attn[channel] = attn[channel][:alignment_heads]
                attn[channel] = attn[channel].mean(dim=0)
        for (i, x) in enumerate(x_list):
            if self.layer_norm is not None:
                x = self.layer_norm(x)
            x = x.transpose(0, 1)
            if self.project_out_dim is not None:
                x = self.project_out_dim(x)
            x_list[i] = x
        x = {channel: x_list[i] for (i, channel) in enumerate(self.channels)}
        return (x, {'attn': [attn], 'inner_states': inner_states})

    def output_layer(self, features):
        if False:
            while True:
                i = 10
        "Project features to the vocabulary size.\n        Return a dictionary of the form:\n            {\n                'input-channel': {\n                    'predicted-channel': token prediction tensor of shape `(batch, tgt_len, vocab)`,\n                }\n            }\n\n        if duration_prediction is enabled\n            {\n                'input-channel': {\n                    'predicted-channel': {\n                        'pred_token': token prediction tensor of shape `(batch, tgt_len, vocab)`,\n                        'pred_duration': duration prediction tensor\n                    }\n                }\n            }\n        "
        if self.output_duration_prediction is None:
            if self.is_cross_prediction:
                return {channel: {pred_channel: self.output_projection[j - i](features[channel]) for (j, pred_channel) in enumerate(self.channels)} for (i, channel) in enumerate(self.channels)}
            else:
                return {channel: {channel: self.output_projection[0](features[channel])} for (i, channel) in enumerate(self.channels)}
        elif self.is_cross_prediction:
            return {channel: {pred_channel: {'pred_token': self.output_projection[j - i](features[channel]), 'pred_duration': self.output_duration_prediction[j - i](features[channel])} for (j, pred_channel) in enumerate(self.channels)} for (i, channel) in enumerate(self.channels)}
        else:
            return {channel: {channel: {'pred_token': self.output_projection[0](features[channel]), 'pred_duration': self.output_duration_prediction[0](features[channel])}} for (i, channel) in enumerate(self.channels)}

    def max_positions(self):
        if False:
            while True:
                i = 10
        'Maximum output length supported by the decoder.'
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        if False:
            while True:
                i = 10
        dim = tensor.size(0)
        if self._future_mask.size(0) == 0 or not self._future_mask.device == tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1)
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def get_normalized_probs_scriptable(self, net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]], log_probs: bool, sample: Optional[Dict[str, Tensor]]=None):
        if False:
            i = 10
            return i + 15
        "Get normalized probabilities (or log probs) from a net's output."
        logits_dict = net_output[0]
        out_dict = {}
        for channel in logits_dict:
            out_dict[channel] = {}
            for pred_channel in logits_dict[channel]:
                if isinstance(logits_dict[channel][pred_channel], dict):
                    pred_token_logits = logits_dict[channel][pred_channel]['pred_token']
                else:
                    pred_token_logits = logits_dict[channel][pred_channel]
                if log_probs:
                    out = utils.log_softmax(pred_token_logits, dim=-1, onnx_trace=self.onnx_trace)
                else:
                    out = utils.softmax(pred_token_logits, dim=-1, onnx_trace=self.onnx_trace)
                if isinstance(logits_dict[channel][pred_channel], dict):
                    out_dict[channel][pred_channel] = {'pred_token': out, 'pred_duration': logits_dict[channel][pred_channel]['pred_duration'].float()}
                else:
                    out_dict[channel][pred_channel] = out
        return out_dict

    def reorder_incremental_state_scripting(self, incremental_state: List[Dict[str, Dict[str, Optional[Tensor]]]], new_order: Tensor):
        if False:
            return 10
        'Main entry point for reordering the incremental state.\n\n        Due to limitations in TorchScript, we call this function in\n        :class:`fairseq.sequence_generator.SequenceGenerator` instead of\n        calling :func:`reorder_incremental_state` directly.\n        '
        for module in self.modules():
            if hasattr(module, 'reorder_incremental_state'):
                for (i, incremental_state_channel) in enumerate(incremental_state):
                    result = module.reorder_incremental_state(incremental_state_channel, new_order)
                    if result is not None:
                        incremental_state[i] = result