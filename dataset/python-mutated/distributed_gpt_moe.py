import math
import torch
from megatron_util import mpu
from megatron_util.global_vars import get_global_memory_buffer
from megatron_util.model import AttnMaskType, Float16Module, LayerNorm, bias_gelu_impl
from megatron_util.model.fused_softmax import FusedScaleMaskSoftmax
from torch import nn
from torch.nn import functional as F
from transformers.modeling_utils import PreTrainedModel
from modelscope.models import TorchModel
from modelscope.models.nlp.gpt_moe import GPTMoEConfig
from modelscope.outputs import TextGenerationModelOutput, TokenGeneratorOutput
from modelscope.utils.megatron_utils import init_megatron_util
from .checkpointing import load_checkpoint
from .moe.layer import MoE

class GPTMoEParallelMLP(nn.Module):

    def __init__(self, config, init_method, output_layer_init_method, moe=False, enable_expert_tensor_parallelism=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense_h_to_4h = mpu.ColumnParallelLinear(config.hidden_size, config.ffn_hidden_size, gather_output=False, init_method=init_method, skip_bias_add=True, moe=moe, enable_expert_tensor_parallelism=enable_expert_tensor_parallelism)
        self.bias_gelu_fusion = config.bias_gelu_fusion
        self.activation_func = F.gelu
        self.dense_4h_to_h = mpu.RowParallelLinear(config.ffn_hidden_size, config.hidden_size, input_is_parallel=True, init_method=output_layer_init_method, skip_bias_add=True, moe=moe, enable_expert_tensor_parallelism=enable_expert_tensor_parallelism)

    def forward(self, hidden_states):
        if False:
            i = 10
            return i + 15
        (intermediate_parallel, bias_parallel) = self.dense_h_to_4h(hidden_states)
        if self.bias_gelu_fusion:
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)
        (output, output_bias) = self.dense_4h_to_h(intermediate_parallel)
        return (output, output_bias)

class GPTMoEEmbedding(nn.Module):
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
        self._word_embeddings_key = 'word_embeddings'
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.hidden_size)
        self._position_embeddings_key = 'position_embeddings'
        self.init_method(self.position_embeddings.weight)
        self.fp32_residual_connection = config.fp32_residual_connection
        self.sequence_parallel = config.sequence_parallel
        self.embedding_dropout = nn.Dropout(config.hidden_dropout)

    def zero_parameters(self):
        if False:
            return 10
        'Zero out all parameters in embedding.'
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True

    def forward(self, input_ids, position_ids):
        if False:
            return 10
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

    def load_state_dict(self, state_dict, strict=True):
        if False:
            while True:
                i = 10
        'Customized load.'
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)
        if self._position_embeddings_key in state_dict:
            state_dict_ = state_dict[self._position_embeddings_key]
        else:
            state_dict_ = {}
            for key in state_dict.keys():
                if 'position_embeddings' in key:
                    state_dict_[key.split('position_embeddings.')[1]] = state_dict[key]
        self.position_embeddings.load_state_dict(state_dict_, strict=strict)

class NoopTransformerLayer(nn.Module):

    def __init__(self, layer_number):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask, encoder_output=None, enc_dec_attn_mask=None, inference_params=None):
        if False:
            while True:
                i = 10
        return hidden_states.clone()

def attention_mask_func(attention_scores, attention_mask):
    if False:
        while True:
            i = 10
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores

class GPTMoECoreAttention(nn.Module):

    def __init__(self, config, layer_number, attn_mask_type=AttnMaskType.padding):
        if False:
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
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

class GPTMoEParallelAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config, init_method, output_layer_init_method, layer_number):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.layer_number = max(1, layer_number)
        self.params_dtype = config.params_dtype
        projection_size = config.kv_channels * config.num_attention_heads
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = mpu.divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(config.num_attention_heads, world_size)
        self.query_key_value = mpu.ColumnParallelLinear(config.hidden_size, 3 * projection_size, gather_output=False, init_method=init_method)
        self.core_attention = GPTMoECoreAttention(config, self.layer_number)
        self.dense = mpu.RowParallelLinear(projection_size, config.hidden_size, input_is_parallel=True, init_method=output_layer_init_method, skip_bias_add=True)

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        if False:
            while True:
                i = 10
        return torch.empty(inference_max_sequence_len, batch_size, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head, dtype=self.params_dtype, device=torch.cuda.current_device())

    def forward(self, hidden_states, attention_mask, inference_params=None):
        if False:
            i = 10
            return i + 15
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
            print('Hello World!')
        self.enter_result = enter_result

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self.enter_result

    def __exit__(self, *excinfo):
        if False:
            for i in range(10):
                print('nop')
        pass

def bias_dropout_add(x, bias, residual, prob, training):
    if False:
        for i in range(10):
            print('nop')
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out

def get_bias_dropout_add(training):
    if False:
        return 10

    def _bias_dropout_add(x, bias, residual, prob):
        if False:
            i = 10
            return i + 15
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add

@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float) -> torch.Tensor:
    if False:
        return 10
    return bias_dropout_add(x, bias, residual, prob, True)

@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    return bias_dropout_add(x, bias, residual, prob, False)

class GPTMoEParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config, init_method, output_layer_init_method, layer_number, num_experts=1):
        if False:
            print('Hello World!')
        super().__init__()
        self.layer_number = layer_number
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layernorm_epsilon, no_persist_layer_norm=config.no_persist_layer_norm, sequence_parallel=config.sequence_parallel)
        self.self_attention = GPTMoEParallelAttention(config, init_method, output_layer_init_method, layer_number)
        self.hidden_dropout = config.hidden_dropout
        self.bias_dropout_fusion = config.bias_dropout_fusion
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.layernorm_epsilon, no_persist_layer_norm=config.no_persist_layer_norm, sequence_parallel=config.sequence_parallel)
        self.num_experts = num_experts
        if self.num_experts == 1:
            self.mlp = GPTMoEParallelMLP(config, init_method, output_layer_init_method)
        else:
            enable_expert_tensor_parallelism = config.enable_expert_tensor_parallelism
            self.mlp = MoE(config.hidden_size, GPTMoEParallelMLP(config, init_method, output_layer_init_method=output_layer_init_method, moe=True, enable_expert_tensor_parallelism=enable_expert_tensor_parallelism), num_experts=self.num_experts, ep_size=config.moe_expert_parallel_size, k=1, use_residual=False, capacity_factor=1.0, eval_capacity_factor=1.0, noisy_gate_policy=None, min_capacity=1, drop_tokens=True, use_tutel=config.use_tutel, top_k_linear_strategy=config.top_k_linear_strategy, use_expert_residual_network=config.use_expert_residual_network)
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
        moe_loss = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)
        mlp_bias = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)
        if self.num_experts == 1:
            (mlp_output, mlp_bias) = self.mlp(layernorm_output)
        else:
            (mlp_output, moe_loss, _) = self.mlp(layernorm_output)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input
        with self.bias_dropout_add_exec_handler():
            output = bias_dropout_add_func(mlp_output, mlp_bias.expand_as(residual), residual, self.hidden_dropout)
        output = mpu.make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
        return (output, moe_loss)

class GPTMoEParallelTransformer(nn.Module):
    """Transformer class."""

    def __init__(self, config, init_method, output_layer_init_method, post_layer_norm=True, pre_process=True, post_process=True, num_experts=[0]):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.sequence_parallel = config.sequence_parallel
        self.num_layers = config.num_hidden_layers

        def build_layer(layer_number, n_e=1):
            if False:
                while True:
                    i = 10
            return GPTMoEParallelTransformerLayer(config, init_method, output_layer_init_method, layer_number, num_experts=n_e)
        offset = 0
        if len(num_experts) == 1 and num_experts[0] > 0:
            num_experts = num_experts * (self.num_layers // 2)
        if self.num_layers == 0:
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([NoopTransformerLayer(1)])
        elif num_experts[0] == 0:
            self.layers = torch.nn.ModuleList([build_layer(i + 1 + offset) for i in range(self.num_layers)])
        else:
            self.layers = []
            for i in range(self.num_layers):
                layer_num = i + 1 + offset
                if layer_num % 2 == 0:
                    n_e = num_experts[(layer_num - 1) // 2]
                else:
                    n_e = 1
                self.layers.append(build_layer(layer_num, n_e))
            self.layers = torch.nn.ModuleList(self.layers)
        if self.post_process and self.post_layer_norm:
            self.final_layernorm = LayerNorm(config.hidden_size, eps=config.layernorm_epsilon, no_persist_layer_norm=config.no_persist_layer_norm, sequence_parallel=config.sequence_parallel)

    def _get_layer(self, layer_number):
        if False:
            print('Hello World!')
        return self.layers[layer_number]

    def forward(self, hidden_states, attention_mask, inference_params=None):
        if False:
            return 10
        if not self.pre_process:
            hidden_states = self.input_tensor
        hidden_states = mpu.make_viewless_tensor(hidden_states, requires_grad=True, keep_graph=True)
        if self.sequence_parallel:
            rng_context = mpu.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        with rng_context:
            moe_losses = []
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                (hidden_states, moe_loss) = layer(hidden_states, attention_mask, inference_params=inference_params)
                moe_losses.append(moe_loss)
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
        return (hidden_states, *moe_losses)

class GPTMoETransformerLanguageModel(nn.Module):
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

    def __init__(self, config, init_method, output_layer_init_method, num_experts=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.hidden_size = config.hidden_size
        self.init_method = init_method
        self.encoder_hidden_state = None
        self.num_experts = num_experts
        self.embedding = GPTMoEEmbedding(config, self.init_method)
        self.encoder = GPTMoEParallelTransformer(config, self.init_method, output_layer_init_method, num_experts=self.num_experts)

    def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask, inference_params=None, enc_hidden_states=None):
        if False:
            print('Hello World!')
        encoder_input = self.embedding(enc_input_ids, enc_position_ids)
        if enc_hidden_states is None:
            if self.encoder is not None:
                (encoder_output, *moe_losses) = self.encoder(encoder_input, enc_attn_mask, inference_params=inference_params)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)
        return (encoder_output, *moe_losses)

    def load_state_dict(self, state_dict, strict=True):
        if False:
            return 10
        'Customized load.'
        if 'embedding' in state_dict:
            state_dict_ = state_dict['embedding']
        else:
            state_dict_ = {}
            for key in state_dict.keys():
                if '_embeddings' in key:
                    state_dict_[key] = state_dict[key]
        self.embedding.load_state_dict(state_dict_, strict=strict)
        if True:
            if 'encoder' in state_dict:
                state_dict_ = state_dict['encoder']
            elif 'transformer' in state_dict:
                state_dict_ = state_dict['transformer']
            else:
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'transformer.' in key:
                        state_dict_[key.split('transformer.')[1]] = state_dict[key]
            state_dict_self_attention = {}
            encoder_state_dict_keys = list(self.encoder.state_dict().keys())
            for key in state_dict_.keys():
                if '.attention.' in key and key not in encoder_state_dict_keys:
                    state_dict_self_attention[key.replace('.attention.', '.self_attention.')] = state_dict_[key]
                elif '.self_attention.' in key and key not in encoder_state_dict_keys:
                    state_dict_self_attention[key.replace('.self_attention.', '.attention.')] = state_dict_[key]
                else:
                    state_dict_self_attention[key] = state_dict_[key]
            state_dict_ = state_dict_self_attention
            if 'moe_state_dict' in state_dict:
                for key in list(state_dict['moe_state_dict'].keys()):
                    if 'encoder' in key:
                        key_list = key.split('.')
                        while key_list[0] != 'encoder':
                            key_list.pop(0)
                        key_list.pop(0)
                        actual_key = '.'.join(key_list)
                        state_dict_[actual_key] = state_dict['moe_state_dict'].pop(key)
                if len(state_dict['moe_state_dict']) == 0:
                    del state_dict['moe_state_dict']
            self.encoder.load_state_dict(state_dict_, strict=strict)

def init_method_normal(sigma):
    if False:
        for i in range(10):
            print('nop')
    'Init method based on N(0, sigma).'

    def init_(tensor):
        if False:
            while True:
                i = 10
        return nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_

def scaled_init_method_normal(sigma, num_layers):
    if False:
        for i in range(10):
            print('nop')
    'Init method based on N(0, sigma/sqrt(2*num_layers).'
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        if False:
            return 10
        return nn.init.normal_(tensor, mean=0.0, std=std)
    return init_

class GPTMoEModel(PreTrainedModel):
    config_class = GPTMoEConfig

    def __init__(self, config, parallel_output=False):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.parallel_output = parallel_output
        self.language_model = GPTMoETransformerLanguageModel(config, init_method_normal(config.init_method_std), scaled_init_method_normal(config.init_method_std, config.num_hidden_layers), num_experts=config.num_experts)

    def word_embeddings_weight(self):
        if False:
            i = 10
            return i + 15
        return self.language_model.embedding.word_embeddings.weight

    @staticmethod
    def build_attention_mask_and_position_ids(tokens):
        if False:
            for i in range(10):
                print('nop')
        seq_length = tokens.size(1)
        attention_mask = torch.tril(torch.ones((1, 1, seq_length, seq_length), dtype=torch.long, device=tokens.device))
        attention_mask = attention_mask < 0.5
        position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)
        return (attention_mask, position_ids)

    @staticmethod
    def post_language_model_processing(input_, labels, word_embeddings_weight, sequence_parallel):
        if False:
            return 10
        input_parallel = input_
        logits_parallel = mpu.LinearWithGradAccumulationAndAsyncCommunication.apply(input_parallel, word_embeddings_weight, None, False, False, sequence_parallel)
        output = logits_parallel
        if labels is None:
            return output.transpose(0, 1).contiguous()
        else:
            labels = labels.transpose(0, 1).contiguous()
            loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
            loss = loss.transpose(0, 1).contiguous()
            return loss

    def forward(self, input_ids, attention_mask=None, position_ids=None, inference_params=None, labels=None, **kwargs):
        if False:
            while True:
                i = 10
        if attention_mask is None and position_ids is None:
            (attention_mask, position_ids) = self.build_attention_mask_and_position_ids(input_ids)
        (lm_output, *moe_losses) = self.language_model(input_ids, position_ids, attention_mask, inference_params=inference_params)
        lm_output = self.post_language_model_processing(lm_output, labels, self.word_embeddings_weight(), self.config.sequence_parallel)
        return (lm_output, *moe_losses)

    def load_state_dict(self, state_dict, strict=True):
        if False:
            for i in range(10):
                print('nop')
        'Customized load.'
        moe_state_dict = {}
        for key in list(state_dict.keys()):
            if 'expert' in key and 'moe.gate.wg.weight' not in key:
                moe_state_dict[key] = state_dict.pop(key)
        if 'language_model' in state_dict:
            state_dict = state_dict['language_model']
        if len(moe_state_dict) > 0:
            state_dict['moe_state_dict'] = moe_state_dict
        self.language_model.load_state_dict(state_dict, strict=strict)

def modify_logits_for_top_k_filtering(logits, top_k):
    if False:
        i = 10
        return i + 15
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
        print('Hello World!')
    ' Sample and generate a token.\n    Note: logits has the dimension [b, v] where b is the batch size\n          and v is the vocabulary size.\n    If vocab_size is provided, we will make sure the sample that is\n    generated is in [0, vocab-size). This will avoid out of vocabulary\n    generations due to padding.\n    '
    assert logits.ndim == 2, 'expected the logits to be of [b, v] shape.'
    assert logits.type() == 'torch.cuda.FloatTensor', 'input logits should be floats.'
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
            while True:
                i = 10
        'Note that offsets are set to zero and we always set the\n        flag to allocate memory. After the first call, make sure to\n        set this flag to False.'
        self.max_sequence_len = max_sequence_len
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}

    def swap_key_value_dict(self, batch_idx):
        if False:
            return 10
        'swap between batches'
        if len(self.key_value_memory_dict) == 0:
            raise ValueError('should not swap when dict in empty')
        for layer_number in self.key_value_memory_dict.keys():
            (inference_key_memory, inference_value_memory) = self.key_value_memory_dict[layer_number]
            assert len(batch_idx) == inference_key_memory.shape[1]
            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (new_inference_key_memory, new_inference_value_memory)

class DistributedGPTMoE(TorchModel):

    def __init__(self, model_dir, rank, path_load_tag='model', *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(model_dir, *args, **kwargs)
        init_megatron_util(model_dir=model_dir, rank=rank)
        self.config = GPTMoEConfig.from_pretrained(model_dir)
        if self.config.num_experts[0] > 0:
            mpu.create_expert_and_data_parallel(self.config.moe_expert_parallel_size)
        model = GPTMoEModel(self.config)
        for param in model.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        model.cuda(torch.cuda.current_device())
        if self.config.fp16 or self.config.bf16:
            model = Float16Module(model, self.config)
        self.dist_model = model
        if self.config.model_dir is not None:
            model_dir = self.config.model_dir
        load_checkpoint(self.dist_model, model_dir, num_experts=self.config.num_experts, path_load_tag=path_load_tag, load_ds_ckpts=self.config.load_ds_ckpts)
        self.inference_params = None

    def train(self, mode: bool=True):
        if False:
            return 10
        if mode:
            self.inference_params = None
        return super().train(mode)

    def forward(self, tokens, attention_mask=None, position_ids=None, labels=None, prompt_length=None, is_pair=(False,)):
        if False:
            return 10
        (outputs, *other_losses) = self.dist_model(tokens, attention_mask, position_ids, inference_params=self.inference_params, labels=labels)
        if labels is None:
            self.inference_params.sequence_len_offset += tokens.size(1)
            return TextGenerationModelOutput(logits=outputs)
        else:
            moe_losses = []
            for moe_loss in other_losses:
                if moe_loss is not None:
                    moe_losses.append(moe_loss)
            moe_loss = sum(moe_losses) * 0.01
            loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)
            losses = outputs.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
            loss = loss + moe_loss
            return TextGenerationModelOutput(loss=loss)

    def generate(self, tokens, temperature=1.0, use_eod_token_for_early_termination=True, stop_on_double_eol=False, stop_on_eol=False, **kwargs):
        if False:
            return 10
        batch_size = tokens.size(0)
        lengths = kwargs.pop('prompt_length', torch.tensor([tokens.size(1)], device=tokens.device))
        pads = torch.ones(batch_size, self.config.tokens_to_generate, device=tokens.device).long() * self.config.eod_id
        tokens = torch.cat((tokens, pads), dim=-1)
        min_prompt_length = lengths.min().item()
        max_sequence_length = tokens.size(1)
        max_sequence_length = min(max_sequence_length, self.config.max_position_embeddings)
        if min_prompt_length >= max_sequence_length:
            raise ValueError('context length + tokens_to_generate too large')
        self.inference_params = InferenceParams(batch_size, max_sequence_length)
        termination_id = self.config.eod_id
        is_generation_done = torch.zeros(batch_size, dtype=torch.uint8, device=torch.cuda.current_device())
        with torch.no_grad():
            (attention_mask, position_ids) = GPTMoEModel.build_attention_mask_and_position_ids(tokens)
            prev_context_length = 0
            for context_length in range(min_prompt_length, max_sequence_length):
                tokens2use = tokens[:, prev_context_length:context_length]
                positions2use = position_ids[:, prev_context_length:context_length]
                attention_mask2use = attention_mask[..., prev_context_length:context_length, :context_length]
                logits = self(tokens2use, attention_mask2use, positions2use).logits
                last_token_logits = logits[:, -1, :]
                new_sample = sample(last_token_logits, top_k=self.config.top_k, top_p=self.config.top_p, temperature=temperature, vocab_size=self.config.vocab_size)
                started = lengths <= context_length
                tokens[started, context_length] = new_sample[started]
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
        tokens = tokens[:, :context_length + 1]
        return TokenGeneratorOutput(sequences=tokens)