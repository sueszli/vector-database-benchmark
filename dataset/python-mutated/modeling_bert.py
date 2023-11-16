"""PyTorch BERT model."""
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import logging
import math
import os
import shutil
import tarfile
import tempfile
import json
import torch
import torch.nn.functional as F
from data_utils.file_utils import cached_path
from megatron_util import mpu
from torch import nn
from torch.nn import CrossEntropyLoss

def normal_init_method(mean, std):
    if False:
        i = 10
        return i + 15

    def init_(tensor):
        if False:
            while True:
                i = 10
        return torch.nn.init.normal_(tensor, mean=mean, std=std)
    return init_

def scaled_init_method(mean, std, num_layers):
    if False:
        for i in range(10):
            print('nop')
    'Init method based on N(0, sigma/sqrt(2*num_layers).'
    std = std / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        if False:
            while True:
                i = 10
        return torch.nn.init.normal_(tensor, mean=mean, std=std)
    return init_

def bert_extended_attention_mask(attention_mask):
    if False:
        return 10
    attention_mask_b1s = attention_mask.unsqueeze(1)
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    extended_attention_mask = attention_mask_bss.unsqueeze(1)
    return extended_attention_mask
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
PRETRAINED_MODEL_ARCHIVE_MAP = {'bert-base-uncased': '/root/data/bert-base-uncased.tar.gz', 'bert-large-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz', 'bert-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz', 'bert-large-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz', 'bert-base-multilingual-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz', 'bert-base-multilingual-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz', 'bert-base-chinese': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz'}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'

def load_tf_weights_in_bert(model, tf_checkpoint_path):
    if False:
        while True:
            i = 10
    ' Load tf checkpoints in a pytorch model\n    '
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print('Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for (name, shape) in init_vars:
        print('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for (name, array) in zip(names, arrays):
        name = name.split('/')
        if any((n in ['adam_v', 'adam_m'] for n in name)):
            print('Skipping {}'.format('/'.join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                l = re.split('_(\\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array)
    return model

def gelu(x):
    if False:
        i = 10
        return i + 15
    "Implementation of the gelu activation function.\n        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):\n        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))\n    "
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    if False:
        print('Hello World!')
    return x * torch.sigmoid(x)
ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self, vocab_size_or_config_json_file, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, deep_init=False, fp32_layernorm=False, fp32_embedding=False, fp32_tokentypes=False, layernorm_epsilon=1e-12):
        if False:
            return 10
        'Constructs BertConfig.\n\n        Args:\n            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.\n            hidden_size: Size of the encoder layers and the pooler layer.\n            num_hidden_layers: Number of hidden layers in the Transformer encoder.\n            num_attention_heads: Number of attention heads for each attention layer in\n                the Transformer encoder.\n            intermediate_size: The size of the "intermediate" (i.e., feed-forward)\n                layer in the Transformer encoder.\n            hidden_act: The non-linear activation function (function or string) in the\n                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.\n            hidden_dropout_prob: The dropout probabilitiy for all fully connected\n                layers in the embeddings, encoder, and pooler.\n            attention_probs_dropout_prob: The dropout ratio for the attention\n                probabilities.\n            max_position_embeddings: The maximum sequence length that this model might\n                ever be used with. Typically set this to something large just in case\n                (e.g., 512 or 1024 or 2048).\n            type_vocab_size: The vocabulary size of the `token_type_ids` passed into\n                `BertModel`.\n            initializer_range: The sttdev of the truncated_normal_initializer for\n                initializing all weight matrices.\n        '
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for (key, value) in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.deep_init = deep_init
            self.fp32_layernorm = fp32_layernorm
            self.fp32_embedding = fp32_embedding
            self.layernorm_epsilon = layernorm_epsilon
            self.fp32_tokentypes = fp32_tokentypes
        else:
            raise ValueError('First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)')

    @classmethod
    def from_dict(cls, json_object):
        if False:
            i = 10
            return i + 15
        'Constructs a `BertConfig` from a Python dictionary of parameters.'
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for (key, value) in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        if False:
            i = 10
            return i + 15
        'Constructs a `BertConfig` from a json file of parameters.'
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return str(self.to_json_string())

    def to_dict(self):
        if False:
            print('Hello World!')
        'Serializes this instance to a Python dictionary.'
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        if False:
            return 10
        'Serializes this instance to a JSON string.'
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print('Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.')

    class BertLayerNorm(nn.Module):

        def __init__(self, hidden_size, eps=1e-12):
            if False:
                for i in range(10):
                    print('nop')
            'Construct a layernorm module in the TF style (epsilon inside the square root).\n            '
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            if False:
                while True:
                    i = 10
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.fp32_layernorm = config.fp32_layernorm
        self.fp32_embedding = config.fp32_embedding
        self.fp32_tokentypes = config.fp32_tokentypes
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        if False:
            print('Hello World!')
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if not self.fp32_tokentypes:
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
            if self.fp32_embedding and (not self.fp32_layernorm):
                embeddings = embeddings.half()
            previous_type = embeddings.type()
            if self.fp32_layernorm:
                embeddings = embeddings.float()
            embeddings = self.LayerNorm(embeddings)
            if self.fp32_layernorm:
                if self.fp32_embedding:
                    embeddings = embeddings.half()
                else:
                    embeddings = embeddings.type(previous_type)
        else:
            embeddings = words_embeddings.float() + position_embeddings.float() + token_type_embeddings.float()
            if self.fp32_tokentypes and (not self.fp32_layernorm):
                embeddings = embeddings.half()
            previous_type = embeddings.type()
            if self.fp32_layernorm:
                embeddings = embeddings.float()
            embeddings = self.LayerNorm(embeddings)
            if self.fp32_layernorm:
                if self.fp32_tokentypes:
                    embeddings = embeddings.half()
                else:
                    embeddings = embeddings.type(previous_type)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        if False:
            for i in range(10):
                print('nop')
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        if False:
            while True:
                i = 10
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        previous_type = attention_probs.type()
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super(BertSelfOutput, self).__init__()
        if hasattr(config, 'deep_init') and config.deep_init:
            init_method = scaled_init_method(mean=0.0, std=config.initializer_range, num_layers=config.num_hidden_layers)
        else:
            init_method = normal_init_method(mean=0.0, std=config.initializer_range)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.fp32_layernorm = config.fp32_layernorm
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        ln_input = hidden_states + input_tensor
        previous_type = ln_input.type()
        if self.fp32_layernorm:
            ln_input = ln_input.float()
        hidden_states = self.LayerNorm(ln_input)
        if self.fp32_layernorm:
            hidden_states = hidden_states.type(previous_type)
        return hidden_states

class BertAttention(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        if False:
            for i in range(10):
                print('nop')
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super(BertOutput, self).__init__()
        if hasattr(config, 'deep_init') and config.deep_init:
            init_method = scaled_init_method(mean=0.0, std=config.initializer_range, num_layers=config.num_hidden_layers)
        else:
            init_method = normal_init_method(mean=0.0, std=config.initializer_range)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.fp32_layernorm = config.fp32_layernorm
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        ln_input = hidden_states + input_tensor
        previous_type = ln_input.type()
        if self.fp32_layernorm:
            ln_input = ln_input.float()
        hidden_states = self.LayerNorm(ln_input)
        if self.fp32_layernorm:
            hidden_states = hidden_states.type(previous_type)
        return hidden_states

class BertLayer(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        if False:
            return 10
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, checkpoint_activations=False):
        if False:
            i = 10
            return i + 15
        all_encoder_layers = []

        def custom(start, end):
            if False:
                while True:
                    i = 10

            def custom_forward(*inputs):
                if False:
                    for i in range(10):
                        print('nop')
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_
            return custom_forward
        if checkpoint_activations:
            l = 0
            num_layers = len(self.layer)
            chunk_length = 1
            while l < num_layers:
                hidden_states = mpu.checkpoint(custom(l, l + chunk_length), hidden_states, attention_mask * 1)
                l += chunk_length
        else:
            for (i, layer_module) in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers or checkpoint_activations:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertPooler(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        if False:
            return 10
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.fp32_layernorm = config.fp32_layernorm

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        previous_type = hidden_states.type()
        if self.fp32_layernorm:
            hidden_states = hidden_states.float()
        hidden_states = self.LayerNorm(hidden_states)
        if self.fp32_layernorm:
            hidden_states = hidden_states.type(previous_type)
        return hidden_states

class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        if False:
            return 10
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False)
        self.fp32_embedding = config.fp32_embedding
        self.fp32_layernorm = config.fp32_layernorm

        def convert_to_type(tensor):
            if False:
                i = 10
                return i + 15
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor
        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
                if self.fp32_layernorm:
                    self.transform.LayerNorm.float()
        hidden_states = self.transform(self.type_converter(hidden_states))
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertOnlyMLMHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        if False:
            print('Hello World!')
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        if False:
            print('Hello World!')
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        if False:
            i = 10
            return i + 15
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class BertPreTrainingHeads(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        if False:
            print('Hello World!')
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        if False:
            for i in range(10):
                print('nop')
        prediction_scores = self.predictions(sequence_output)
        for p in self.seq_relationship.parameters():
            if p is None:
                continue
            pooled_output = pooled_output.type_as(p)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return (prediction_scores, seq_relationship_score)

class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        if False:
            return 10
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `BertConfig`. To create a model from a Google pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_bert_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        ' Initialize the weights.\n        '
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, fp32_layernorm=False, fp32_embedding=False, layernorm_epsilon=1e-12, fp32_tokentypes=False, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.\n        Download and cache the pre-trained model file if needed.\n\n        Params:\n            pretrained_model_name: either:\n                - a str with the name of a pre-trained model to load selected in the list of:\n                    . `bert-base-uncased`\n                    . `bert-large-uncased`\n                    . `bert-base-cased`\n                    . `bert-large-cased`\n                    . `bert-base-multilingual-uncased`\n                    . `bert-base-multilingual-cased`\n                    . `bert-base-chinese`\n                - a path or url to a pretrained model archive containing:\n                    . `bert_config.json` a configuration file for the model\n                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance\n            cache_dir: an optional path to a folder in which the pre-trained models will be cached.\n            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models\n            *inputs, **kwargs: additional input for the specific Bert class\n                (ex: num_labels for BertForSequenceClassification)\n        '
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url.".format(pretrained_model_name, ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()), archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info('loading archive file {}'.format(archive_file))
        else:
            logger.info('loading archive file {} from cache at {}'.format(archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            logger.info('extracting archive file {} to temp dir {}'.format(resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        config.fp32_layernorm = fp32_layernorm
        config.fp32_embedding = fp32_embedding
        config.layernorm_epsilon = layernorm_epsilon
        config.fp32_tokentypes = fp32_tokentypes
        logger.info('Model config {}'.format(config))
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for (old_key, new_key) in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            if False:
                print('Hello World!')
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for (name, child) in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            print('Weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print('Weights from pretrained model not used in {}: {}'.format(model.__class__.__name__, unexpected_keys))
        if tempdir:
            shutil.rmtree(tempdir)
        return model

class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    Examples:
        >>> # Already been converted into WordPiece token ids
        >>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        >>> input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        >>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        >>> config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        >>>     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        >>> model = modeling.BertModel(config=config)
        >>> all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config):
        if False:
            return 10
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, checkpoint_activations=False):
        if False:
            print('Hello World!')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.encoder.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=output_all_encoded_layers, checkpoint_activations=checkpoint_activations)
        sequence_output = encoded_layers[-1]
        for p in self.pooler.parameters():
            if p is None:
                continue
            sequence_output = sequence_output.type_as(p)
            break
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers or checkpoint_activations:
            encoded_layers = encoded_layers[-1]
        return (encoded_layers, pooled_output)

class BertForPreTraining(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Examples:
        >>> # Already been converted into WordPiece token ids
        >>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        >>> input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        >>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        >>> config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        >>>     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        >>> model = BertForPreTraining(config)
        >>> masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, checkpoint_activations=False):
        if False:
            i = 10
            return i + 15
        (sequence_output, pooled_output) = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
        (prediction_scores, seq_relationship_score) = self.cls(sequence_output, pooled_output)
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size).float(), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2).float(), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return (prediction_scores, seq_relationship_score)

class BertForMaskedLM(PreTrainedBertModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Examples:
        >>> # Already been converted into WordPiece token ids
        >>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        >>> input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        >>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        >>> config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        >>>     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        >>> model = BertForMaskedLM(config)
        >>> masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, checkpoint_activations=False):
        if False:
            print('Hello World!')
        (sequence_output, _) = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
        prediction_scores = self.cls(sequence_output)
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores

class BertForNextSentencePrediction(PreTrainedBertModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Examples:
        >>> # Already been converted into WordPiece token ids
        >>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        >>> input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        >>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        >>> config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        >>>     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        >>> model = BertForNextSentencePrediction(config)
        >>> seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None, checkpoint_activations=False):
        if False:
            for i in range(10):
                print('nop')
        (_, pooled_output) = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
        seq_relationship_score = self.cls(pooled_output)
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score

class BertForSequenceClassification(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Examples:
        >>> # Already been converted into WordPiece token ids
        >>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        >>> input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        >>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        >>> config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        >>>     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        >>> num_labels = 2

        >>> model = BertForSequenceClassification(config, num_labels)
        >>> logits = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config, num_labels=2):
        if False:
            while True:
                i = 10
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, checkpoint_activations=False):
        if False:
            i = 10
            return i + 15
        (_, pooled_output) = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertForMultipleChoice(PreTrainedBertModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Examples:
        >>> # Already been converted into WordPiece token ids
        >>> input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
        >>> input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
        >>> token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
        >>> config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        >>>     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        >>> num_choices = 2

        >>> model = BertForMultipleChoice(config, num_choices)
        >>> logits = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config):
        if False:
            return 10
        super(BertForMultipleChoice, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, checkpoint_activations=False):
        if False:
            while True:
                i = 10
        (batch_size, num_choices) = input_ids.shape[:2]
        flat_input_ids = input_ids.reshape(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
        (_, pooled_output) = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits

class BertForTokenClassification(PreTrainedBertModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Examples:
        >>> # Already been converted into WordPiece token ids
        >>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        >>> input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        >>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        >>> config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        >>>     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        >>> num_labels = 2

        >>> model = BertForTokenClassification(config, num_labels)
        >>> logits = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config, num_labels=2):
        if False:
            return 10
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, checkpoint_activations=False):
        if False:
            i = 10
            return i + 15
        (sequence_output, _) = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
        with mpu.get_cuda_rng_tracker().fork():
            sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertForQuestionAnswering(PreTrainedBertModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Examples:
        >>> # Already been converted into WordPiece token ids
        >>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        >>> input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        >>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        >>> config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        >>>     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        >>> model = BertForQuestionAnswering(config)
        >>> start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None, checkpoint_activations=False):
        if False:
            i = 10
            return i + 15
        (sequence_output, _) = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
        logits = self.qa_outputs(sequence_output)
        (start_logits, end_logits) = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return (start_logits, end_logits)