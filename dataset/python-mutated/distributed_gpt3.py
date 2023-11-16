import math
import os
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Union
import torch
from megatron_util import get_args, mpu
from megatron_util.global_vars import get_global_memory_buffer
from megatron_util.model import AttnMaskType, Float16Module, LayerNorm, bias_gelu_impl
from megatron_util.model.fused_softmax import FusedScaleMaskSoftmax
from torch import nn
from torch.nn import functional as F
from transformers.modeling_utils import PreTrainedModel
from modelscope.models import TorchModel
from modelscope.models.nlp.gpt3 import GPT3Config
from modelscope.outputs import TextGenerationModelOutput, TokenGeneratorOutput
from modelscope.utils.megatron_utils import init_megatron_util
from modelscope.utils.nlp.load_checkpoint import pre_load
from modelscope.utils.streaming_output import StreamingOutputMixin

class GPT3ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, init_method, output_layer_init_method):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense_h_to_4h = mpu.ColumnParallelLinear(config.hidden_size, config.ffn_hidden_size, gather_output=False, init_method=init_method, skip_bias_add=True)
        self.bias_gelu_fusion = config.bias_gelu_fusion
        self.activation_func = F.gelu
        self.dense_4h_to_h = mpu.RowParallelLinear(config.ffn_hidden_size, config.hidden_size, input_is_parallel=True, init_method=output_layer_init_method, skip_bias_add=True)

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        (intermediate_parallel, bias_parallel) = self.dense_h_to_4h(hidden_states)
        if self.bias_gelu_fusion:
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)
        (output, output_bias) = self.dense_4h_to_h(intermediate_parallel)
        return (output, output_bias)

class GPT3Embedding(nn.Module):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self, config, init_method):
        if False:
            while True:
                i = 10
        super().__init__()
        self.hidden_size = config.hidden_size
        self.init_method = init_method
        self.word_embeddings = mpu.VocabParallelEmbedding(config.vocab_size, self.hidden_size, init_method=self.init_method)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.hidden_size)
        self.init_method(self.position_embeddings.weight)
        self.fp32_residual_connection = config.fp32_residual_connection
        self.sequence_parallel = config.sequence_parallel
        self.embedding_dropout = nn.Dropout(config.hidden_dropout)

    def zero_parameters(self):
        if False:
            while True:
                i = 10
        'Zero out all parameters in embedding.'
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True

    def forward(self, input_ids, position_ids):
        if False:
            for i in range(10):
                print('nop')
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        embeddings = embeddings.transpose(0, 1).contiguous()
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        if self.sequence_parallel:
            embeddings = mpu.scatter_to_sequence_parallel_region(embeddings)
            with mpu.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)
        return embeddings

class NoopTransformerLayer(nn.Module):

    def __init__(self, layer_number):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask, encoder_output=None, enc_dec_attn_mask=None, inference_params=None):
        if False:
            for i in range(10):
                print('nop')
        return hidden_states.clone()

def attention_mask_func(attention_scores, attention_mask):
    if False:
        i = 10
        return i + 15
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores

class GPT3CoreAttention(nn.Module):

    def __init__(self, config, layer_number, attn_mask_type=AttnMaskType.padding):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fp16 = config.fp16
        self.bf16 = config.bf16
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = config.sequence_parallel
        projection_size = config.kv_channels * config.num_attention_heads
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(projection_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(config.num_attention_heads, world_size)
        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.scale_mask_softmax = FusedScaleMaskSoftmax(self.fp16, self.bf16, self.attn_mask_type, config.masked_softmax_fusion, attention_mask_func, self.attention_softmax_in_fp32, coeff)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        if False:
            while True:
                i = 10
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        matmul_input_buffer = get_global_memory_buffer().get_tensor((output_size[0] * output_size[1], output_size[2], output_size[3]), query_layer.dtype, 'mpu')
        matmul_result = torch.baddbmm(matmul_input_buffer, query_layer.transpose(0, 1), key_layer.transpose(0, 1).transpose(1, 2), beta=0.0, alpha=1.0 / self.norm_factor)
        attention_scores = matmul_result.view(*output_size)
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        if not self.sequence_parallel:
            with mpu.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
        context_layer = context_layer.view(*output_size)
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class GPT3ParallelAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config, init_method, output_layer_init_method, layer_number):
        if False:
            return 10
        super().__init__()
        self.layer_number = max(1, layer_number)
        self.params_dtype = config.params_dtype
        projection_size = config.kv_channels * config.num_attention_heads
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = mpu.divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(config.num_attention_heads, world_size)
        self.query_key_value = mpu.ColumnParallelLinear(config.hidden_size, 3 * projection_size, gather_output=False, init_method=init_method)
        self.core_attention = GPT3CoreAttention(config, self.layer_number)
        self.dense = mpu.RowParallelLinear(projection_size, config.hidden_size, input_is_parallel=True, init_method=output_layer_init_method, skip_bias_add=True)

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        if False:
            i = 10
            return i + 15
        return torch.empty(inference_max_sequence_len, batch_size, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head, dtype=self.params_dtype, device=torch.cuda.current_device())

    def forward(self, hidden_states, attention_mask, inference_params=None):
        if False:
            for i in range(10):
                print('nop')
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (inference_key_memory, inference_value_memory)
            else:
                (inference_key_memory, inference_value_memory) = inference_params.key_value_memory_dict[self.layer_number]
        (mixed_x_layer, _) = self.query_key_value(hidden_states)
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition, 3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)
        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[:sequence_end, batch_start:batch_end, ...]
        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        (output, bias) = self.dense(context_layer)
        return (output, bias)

class nullcontext:

    def __init__(self, enter_result=None):
        if False:
            return 10
        self.enter_result = enter_result

    def __enter__(self):
        if False:
            print('Hello World!')
        return self.enter_result

    def __exit__(self, *excinfo):
        if False:
            while True:
                i = 10
        pass

def bias_dropout_add(x, bias, residual, prob, training):
    if False:
        print('Hello World!')
    out = F.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out

def get_bias_dropout_add(training):
    if False:
        print('Hello World!')

    def _bias_dropout_add(x, bias, residual, prob):
        if False:
            print('Hello World!')
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add

@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float) -> torch.Tensor:
    if False:
        while True:
            i = 10
    return bias_dropout_add(x, bias, residual, prob, True)

@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    return bias_dropout_add(x, bias, residual, prob, False)

class GPT3ParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config, init_method, output_layer_init_method, layer_number):
        if False:
            print('Hello World!')
        super().__init__()
        self.layer_number = layer_number
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layernorm_epsilon, no_persist_layer_norm=config.no_persist_layer_norm, sequence_parallel=config.sequence_parallel)
        self.self_attention = GPT3ParallelAttention(config, init_method, output_layer_init_method, layer_number)
        self.hidden_dropout = config.hidden_dropout
        self.bias_dropout_fusion = config.bias_dropout_fusion
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.layernorm_epsilon, no_persist_layer_norm=config.no_persist_layer_norm, sequence_parallel=config.sequence_parallel)
        self.mlp = GPT3ParallelMLP(config, init_method, output_layer_init_method)
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, attention_mask, inference_params=None):
        if False:
            for i in range(10):
                print('nop')
        layernorm_output = self.input_layernorm(hidden_states)
        (attention_output, attention_bias) = self.self_attention(layernorm_output, attention_mask, inference_params=inference_params)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)
        with self.bias_dropout_add_exec_handler():
            layernorm_input = bias_dropout_add_func(attention_output, attention_bias.expand_as(residual), residual, self.hidden_dropout)
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        (mlp_output, mlp_bias) = self.mlp(layernorm_output)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input
        with self.bias_dropout_add_exec_handler():
            output = bias_dropout_add_func(mlp_output, mlp_bias.expand_as(residual), residual, self.hidden_dropout)
        output = mpu.make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
        return output

class GPT3ParallelTransformer(nn.Module):
    """Transformer class."""

    def __init__(self, config, init_method, output_layer_init_method, post_layer_norm=True, pre_process=True, post_process=True):
        if False:
            while True:
                i = 10
        super().__init__()
        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.sequence_parallel = config.sequence_parallel
        self.num_layers = config.num_hidden_layers

        def build_layer(layer_number):
            if False:
                while True:
                    i = 10
            return GPT3ParallelTransformerLayer(config, init_method, output_layer_init_method, layer_number)
        if self.num_layers == 0:
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([NoopTransformerLayer(1)])
        else:
            self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])
        if self.post_process and self.post_layer_norm:
            self.final_layernorm = LayerNorm(config.hidden_size, eps=config.layernorm_epsilon, no_persist_layer_norm=config.no_persist_layer_norm, sequence_parallel=config.sequence_parallel)

    def _get_layer(self, layer_number):
        if False:
            return 10
        return self.layers[layer_number]

    def forward(self, hidden_states, attention_mask, inference_params=None):
        if False:
            while True:
                i = 10
        if not self.pre_process:
            hidden_states = self.input_tensor
        hidden_states = mpu.make_viewless_tensor(hidden_states, requires_grad=True, keep_graph=True)
        if self.sequence_parallel:
            rng_context = mpu.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        with rng_context:
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                hidden_states = layer(hidden_states, attention_mask, inference_params=inference_params)
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

class GPT3TransformerLanguageModel(nn.Module):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self, config, init_method, output_layer_init_method):
        if False:
            while True:
                i = 10
        super().__init__()
        self.hidden_size = config.hidden_size
        self.init_method = init_method
        self.encoder_hidden_state = None
        self.embedding = GPT3Embedding(config, self.init_method)
        self.encoder = GPT3ParallelTransformer(config, self.init_method, output_layer_init_method)

    def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask, inference_params=None, enc_hidden_states=None):
        if False:
            print('Hello World!')
        encoder_input = self.embedding(enc_input_ids, enc_position_ids)
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(encoder_input, enc_attn_mask, inference_params=inference_params)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)
        return encoder_output

def init_method_normal(sigma):
    if False:
        for i in range(10):
            print('nop')
    'Init method based on N(0, sigma).'

    def init_(tensor):
        if False:
            return 10
        return nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_

def scaled_init_method_normal(sigma, num_layers):
    if False:
        while True:
            i = 10
    'Init method based on N(0, sigma/sqrt(2*num_layers).'
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        if False:
            print('Hello World!')
        return nn.init.normal_(tensor, mean=0.0, std=std)
    return init_

class GPT3Model(PreTrainedModel):
    config_class = GPT3Config

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.language_model = GPT3TransformerLanguageModel(config, init_method_normal(config.init_method_std), scaled_init_method_normal(config.init_method_std, config.num_hidden_layers))

    def word_embeddings_weight(self):
        if False:
            return 10
        return self.language_model.embedding.word_embeddings.weight

    @staticmethod
    def build_attention_mask_and_position_ids(tokens):
        if False:
            while True:
                i = 10
        seq_length = tokens.size(1)
        attention_mask = torch.tril(torch.ones((1, 1, seq_length, seq_length), device=tokens.device))
        attention_mask = attention_mask < 0.5
        position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)
        return (attention_mask, position_ids)

    def forward(self, input_ids, attention_mask=None, position_ids=None, inference_params=None, labels=None, **kwargs):
        if False:
            print('Hello World!')
        if attention_mask is None and position_ids is None:
            (attention_mask, position_ids) = self.build_attention_mask_and_position_ids(input_ids)
        lm_output = self.language_model(input_ids, position_ids, attention_mask, inference_params=inference_params)
        logits_parallel = mpu.LinearWithGradAccumulationAndAsyncCommunication.apply(lm_output, self.word_embeddings_weight(), None, False, True, self.config.sequence_parallel)
        losses = None
        if labels is not None:
            labels = labels.transpose(0, 1).contiguous()
            losses = mpu.vocab_parallel_cross_entropy(logits_parallel.clone().float(), labels)
            losses = losses.transpose(0, 1).contiguous()
        logits = mpu.gather_from_tensor_model_parallel_region(logits_parallel)
        logits = logits.transpose(0, 1).contiguous()
        return (logits, losses)

def modify_logits_for_top_k_filtering(logits, top_k):
    if False:
        print('Hello World!')
    'Set the logits for none top-k values to -inf.'
    filter_ = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(filter_, float('-Inf'))

def modify_logits_for_top_p_filtering(logits, top_p):
    if False:
        return 10
    'Set the logits for none top-p values to -inf.'
    (sorted_logits, sorted_indices) = torch.sort(logits, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    filter_ = cumulative_probs > top_p
    filter_[:, 1:] = filter_[:, :-1].clone()
    filter_[..., 0] = 0
    filter_ = filter_.scatter(1, sorted_indices, filter_)
    logits.masked_fill_(filter_, float('-Inf'))

def sample(logits, top_k=0, top_p=0.0, temperature=1.0, vocab_size=None):
    if False:
        return 10
    ' Sample and generate a token.\n    Note: logits has the dimension [b, v] where b is the batch size\n          and v is the vocabulary size.\n    If vocab_size is provided, we will make sure the sample that is\n    generated is in [0, vocab-size). This will avoid out of vocabulary\n    generations due to padding.\n    '
    assert logits.ndim == 2, 'expected the logits to be of [b, v] shape.'
    if top_k == 1:
        assert top_p == 0.0, 'cannot set both greedy and top-p samplings.'
        samples = torch.argmax(logits, dim=-1)
    else:
        logits = logits.clone()
        if temperature != 1.0:
            logits.div_(temperature)
        if top_k > 1:
            assert top_p == 0.0, 'cannot set both top-k and top-p samplings.'
            assert top_k <= logits.size(1), 'top-k is larger than logit size.'
            if vocab_size:
                assert top_k < vocab_size, 'top-k is larger than vocab size.'
            modify_logits_for_top_k_filtering(logits, top_k)
        elif top_p > 0.0:
            assert top_p <= 1.0, 'top-p should be in (0, 1].'
            modify_logits_for_top_p_filtering(logits, top_p)
        probs = logits.softmax(dim=-1)
        samples = torch.multinomial(probs, num_samples=1).view(-1)
    if vocab_size:
        samples = torch.clamp(samples, min=0, max=vocab_size - 1)
    return samples

class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    def __init__(self, max_batch_size, max_sequence_len):
        if False:
            i = 10
            return i + 15
        'Note that offsets are set to zero and we always set the\n        flag to allocate memory. After the first call, make sure to\n        set this flag to False.'
        self.max_sequence_len = max_sequence_len
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}

    def swap_key_value_dict(self, batch_idx):
        if False:
            print('Hello World!')
        'swap between batches'
        if len(self.key_value_memory_dict) == 0:
            raise ValueError('should not swap when dict in empty')
        for layer_number in self.key_value_memory_dict.keys():
            (inference_key_memory, inference_value_memory) = self.key_value_memory_dict[layer_number]
            assert len(batch_idx) == inference_key_memory.shape[1]
            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (new_inference_key_memory, new_inference_value_memory)

def split_into_partitions(tensor, num_partitions, partition_dim, stride):
    if False:
        i = 10
        return i + 15
    per_partition_size = mpu.utils.divide(tensor.size(partition_dim), num_partitions)
    per_partition_per_stride_size = mpu.utils.divide(per_partition_size, stride)
    partitions_list = torch.split(tensor, per_partition_per_stride_size, dim=partition_dim)
    partitions = []
    for i in range(num_partitions):
        partition = torch.cat(partitions_list[i::num_partitions], dim=partition_dim)
        partitions.append(partition)
    return partitions

def split_state_dict(state_dict: Dict[str, torch.Tensor], model: GPT3Model, partitions: int) -> Dict[str, torch.Tensor]:
    if False:
        i = 10
        return i + 15
    if partitions == 1:
        return state_dict
    rank: int = mpu.get_tensor_model_parallel_rank()
    for (name, parameters) in model.named_parameters():
        if parameters.shape == state_dict[name].shape:
            continue
        dim = max(parameters.partition_dim, 0)
        stride = parameters.partition_stride
        state_dict[name] = split_into_partitions(state_dict[name], partitions, dim, stride)[rank]
    return state_dict

class DistributedGPT3(TorchModel, StreamingOutputMixin):

    def __init__(self, model_dir, rank, path_load_tag='model', *args, megatron_cfg=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(model_dir, *args, **kwargs)
        init_megatron_util(megatron_cfg, model_dir, rank=rank)
        self.config = GPT3Config.from_pretrained(model_dir)
        model = GPT3Model(self.config)
        for param in model.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        model.cuda(torch.cuda.current_device())
        if self.config.fp16 or self.config.bf16:
            model = Float16Module(model, self.config)
        self.dist_model = model
        tensor_ws = mpu.get_tensor_model_parallel_world_size()
        ckpt_ws = get_args().get('checkpoint_tensor_model_parallel_size', None)
        ckpt_ws = tensor_ws if ckpt_ws is None else ckpt_ws
        ckpt_rank = mpu.get_tensor_model_parallel_rank() * ckpt_ws // tensor_ws
        load_model = pre_load(ckpt_rank, model_dir, tag=path_load_tag)
        load_model = split_state_dict(load_model, model, tensor_ws // ckpt_ws)
        self.dist_model.load_state_dict(load_model, strict=kwargs.get('strict', True))
        self.inference_params = None

    def train(self, mode: bool=True):
        if False:
            return 10
        if mode:
            self.inference_params = None
        return super().train(mode)

    def forward(self, tokens, attention_mask=None, position_ids=None, labels=None, prompts_len=None, inputs_len=None):
        if False:
            return 10
        (logits, losses) = self.dist_model(tokens, attention_mask, position_ids, inference_params=self.inference_params, labels=labels)
        loss = None
        if labels is None:
            self.inference_params.sequence_len_offset += tokens.size(1)
        else:
            loss_mask = torch.ones(labels.size(), dtype=torch.float, device=tokens.device)
            if inputs_len is None:
                for (i, l) in enumerate(prompts_len):
                    loss_mask[i, l:] = 0
            else:
                for (i, l) in enumerate(inputs_len):
                    loss_mask[i, l - 1:] = 0
                for (i, l) in enumerate(prompts_len):
                    loss_mask[i, :l - 1] = 0
            losses = losses.float()
            loss_mask = loss_mask.view(-1).float()
            mask_sum = loss_mask.sum()
            if mask_sum == 0:
                loss = torch.sum(losses.view(-1)).zero_()
            else:
                loss = torch.sum(losses.view(-1) * loss_mask) / mask_sum
        return TextGenerationModelOutput(logits=logits, loss=loss)

    def sample(self, tokens, prompts_len=None, use_eod_token_for_early_termination=True, stop_on_double_eol=False, stop_on_eol=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        top_k = kwargs.pop('top_k', self.config.top_k)
        top_p = kwargs.pop('top_p', self.config.top_p)
        temperature = kwargs.pop('temperature', self.config.temperature)
        max_length = kwargs.pop('max_length', tokens.size(1) + self.config.tokens_to_generate)
        batch_size = tokens.size(0)
        lengths = prompts_len
        if lengths is None:
            lengths = torch.tensor([tokens.size(1)], device=tokens.device)
        min_prompt_length = lengths.min().item()
        max_sequence_length = min(max_length, self.config.max_position_embeddings)
        if min_prompt_length >= max_sequence_length:
            raise ValueError('context length + tokens_to_generate too large')
        pad_length = max_sequence_length - tokens.size(1)
        if pad_length > 0:
            pads = torch.zeros(batch_size, pad_length, device=tokens.device).long()
            tokens = torch.cat((tokens, pads), dim=-1)
        self.inference_params = InferenceParams(batch_size, max_sequence_length)
        termination_id = self.config.eod_id
        is_generation_done = torch.zeros(batch_size, dtype=torch.uint8, device=torch.cuda.current_device())
        (attention_mask, position_ids) = GPT3Model.build_attention_mask_and_position_ids(tokens)
        prev_context_length = 0
        for context_length in range(min_prompt_length, max_sequence_length):
            tokens2use = tokens[:, prev_context_length:context_length]
            positions2use = position_ids[:, prev_context_length:context_length]
            attention_mask2use = attention_mask[..., prev_context_length:context_length, :context_length]
            logits = self(tokens2use, attention_mask2use, positions2use).logits
            last_token_logits = logits[:, -1, :]
            new_sample = sample(last_token_logits, top_k=top_k, top_p=top_p, temperature=temperature, vocab_size=self.config.vocab_size)
            started = lengths <= context_length
            tokens[started, context_length] = new_sample[started]
            yield TokenGeneratorOutput(sequences=tokens[:, :context_length + 1])
            prev_context_length = context_length
            if stop_on_double_eol:
                hit_double_eol = (new_sample == 628).byte() & started.byte()
                hit_two_eols = (new_sample == 198).byte() & (tokens[:, context_length - 1] == 198).byte() & started.byte()
                done_token = hit_double_eol | hit_two_eols
            elif stop_on_eol:
                hit_double_eol = (new_sample == 628).byte() & started.byte()
                hit_eol = (new_sample == 198).byte() & started.byte()
                done_token = hit_double_eol | hit_eol
            else:
                done_token = (new_sample == termination_id).byte() & started.byte()
            is_generation_done = is_generation_done | done_token
            done = torch.all(is_generation_done)
            if use_eod_token_for_early_termination and done:
                break

    def beam_search(self, tokens, beam_size=5, num_return_gen=1, **kwargs):
        if False:
            print('Hello World!')
        batch_size = tokens.size(0)
        assert batch_size == 1
        prompt_length = kwargs.pop('prompt_length', torch.tensor([tokens.size(1)], device=tokens.device)).item()
        stop_token = self.config.eod_id
        pads = torch.ones(1, self.config.tokens_to_generate, device=tokens.device).long() * stop_token
        tokens = torch.cat((tokens, pads), dim=-1)
        final_sequence_length = tokens.size(1)
        final_sequence_length = min(final_sequence_length, self.config.max_position_embeddings)
        if prompt_length >= final_sequence_length:
            raise ValueError('context length + tokens_to_generate too large')
        self.inference_params = InferenceParams(beam_size, final_sequence_length)
        beam_hyp = BeamHypotheses(beam_size)
        done = False
        scores = torch.zeros(beam_size, dtype=torch.float32, device=torch.cuda.current_device()).unsqueeze(1)
        tokens = tokens.repeat(beam_size, 1)
        (attention_mask, position_ids) = GPT3Model.build_attention_mask_and_position_ids(tokens)
        prev_context_length = 0
        for context_length in range(prompt_length, final_sequence_length):
            tokens2use = tokens[:, prev_context_length:context_length]
            positions2use = position_ids[:, prev_context_length:context_length]
            attention_mask2use = attention_mask[..., prev_context_length:context_length, :context_length]
            logits = self(tokens2use, attention_mask2use, positions2use).logits
            vocab_size = logits.size(2)
            log_probs = F.log_softmax(logits, dim=2)
            new_scores = log_probs[:, -1, :] + scores
            if context_length == prompt_length:
                (sorted_scores, indices) = torch.sort(new_scores[0, :], descending=True)
            else:
                (sorted_scores, indices) = torch.sort(new_scores.view(-1), descending=True)
            best_beam_ids = torch.div(indices[:2 * beam_size], vocab_size).trunc().long()
            best_words = indices[:2 * beam_size] % vocab_size
            best_scores = sorted_scores[:2 * beam_size]
            next_beams = []
            for (beam_token_rank, (token_id, beam_score, beam_id)) in enumerate(zip(best_words, best_scores, best_beam_ids)):
                if token_id.item() == stop_token:
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= beam_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(tokens[beam_id].clone(), beam_score, context_length + 1 - prompt_length)
                else:
                    next_beams.append((token_id, beam_score, beam_id))
                if len(next_beams) == beam_size:
                    break
            if beam_hyp.is_done(best_scores.max().item(), context_length + 1 - prompt_length):
                done = True
                break
            best_batches = tokens.new([item[2] for item in next_beams])
            tokens = tokens[best_batches, :]
            tokens[:, context_length] = tokens.new([item[0] for item in next_beams])
            scores = scores.new([item[1] for item in next_beams]).unsqueeze(1)
            self.inference_params.swap_key_value_dict(best_batches)
            prev_context_length = context_length
        if not done:
            for beam_id in range(beam_size):
                beam_hyp.add(tokens[beam_id].clone(), scores[beam_id], context_length + 1 - prompt_length)
        sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0], reverse=True)
        num_return_gen = min(num_return_gen, len(sorted_hyps))
        scores = [sorted_hyps[i][0] for i in range(num_return_gen)]
        tokens = [sorted_hyps[i][1] for i in range(num_return_gen)]
        scores = torch.stack(scores, dim=0)
        tokens = torch.stack(tokens, dim=0)
        return TokenGeneratorOutput(sequences=tokens, scores=scores)

    @torch.no_grad()
    def generate(self, tokens, do_sample=True, *args, **kwargs):
        if False:
            print('Hello World!')
        if do_sample:
            last_output = None
            for output in self.sample(tokens, *args, **kwargs):
                last_output = output
            return last_output
        else:
            return self.beam_search(tokens, *args, **kwargs)

    @torch.no_grad()
    def stream_generate(self, tokens, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.sample(tokens, *args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if False:
            print('Hello World!')
        return self.dist_model.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool=True):
        if False:
            return 10
        return self.dist_model.load_state_dict(state_dict, strict)

    def save_pretrained(self, target_folder: Union[str, os.PathLike], save_checkpoint_names: Union[str, List[str]]=None, save_function: Callable=None, config: Optional[dict]=None, **kwargs):
        if False:
            return 10
        config['pipeline']['type'] = 'gpt3-generation'
        config['model'].pop('rank', None)
        config['model'].pop('megatron_cfg', None)
        config['megatron'].pop('rank', None)
        config['megatron'].pop('checkpoint_tensor_model_parallel_size', None)
        tp_size = get_args().tensor_model_parallel_size
        pp_size = get_args().pipeline_model_parallel_size
        config['megatron']['world_size'] = tp_size * pp_size
        return super().save_pretrained(target_folder, save_checkpoint_names, save_function, config, **kwargs)

class BeamHypotheses:

    def __init__(self, num_beams: int, length_penalty: float=1.0, early_stopping: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize n-best list of hypotheses.\n        '
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1000000000.0

    def __len__(self):
        if False:
            return 10
        '\n        Number of hypotheses in the list.\n        '
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float, beam_indices: Optional[torch.LongTensor]=None):
        if False:
            return 10
        '\n        Add a new hypothesis to the list.\n        '
        score = sum_logprobs / hyp.shape[-1] ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for (idx, (s, _, _)) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst\n        one in the heap, then we are done with this sentence.\n        '
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret