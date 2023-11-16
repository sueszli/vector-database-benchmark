"""
A version of the BaseModel which uses LSTMs to predict the correct next transition
based on the current known state.

The primary purpose of this class is to implement the prediction of the next
transition, which is done by concatenating the output of an LSTM operated over
previous transitions, the words, and the partially built constituents.

A complete processing of a sentence is as follows:
  1) Run the input words through an encoder.
     The encoder includes some or all of the following:
       pretrained word embedding
       finetuned word embedding for training set words - "delta_embedding"
       POS tag embedding
       pretrained charlm representation
       BERT or similar large language model representation
       attention transformer over the previous inputs
       labeled attention transformer over the first attention layer
     The encoded input is then put through a bi-lstm, giving a word representation
  2) Transitions are put in an embedding, and transitions already used are tracked
     in an LSTM
  3) Constituents already built are also processed in an LSTM
  4) Every transition is chosen by taking the output of the current word position,
     the transition LSTM, and the constituent LSTM, and classifying the next
     transition
  5) Transitions are repeated (with constraints) until the sentence is completed
"""
from collections import namedtuple
from enum import Enum
import logging
import math
import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from stanza.models.common.bert_embedding import extract_bert_embeddings
from stanza.models.common.maxout_linear import MaxoutLinear
from stanza.models.common.utils import unsort
from stanza.models.common.vocab import PAD_ID, UNK_ID
from stanza.models.constituency.base_model import BaseModel
from stanza.models.constituency.label_attention import LabelAttentionModule
from stanza.models.constituency.lstm_tree_stack import LSTMTreeStack
from stanza.models.constituency.parse_transitions import TransitionScheme
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.partitioned_transformer import PartitionedTransformerModule
from stanza.models.constituency.positional_encoding import ConcatSinusoidalEncoding
from stanza.models.constituency.transformer_tree_stack import TransformerTreeStack
from stanza.models.constituency.tree_stack import TreeStack
from stanza.models.constituency.utils import build_nonlinearity, initialize_linear
logger = logging.getLogger('stanza')
WordNode = namedtuple('WordNode', ['value', 'hx'])
Constituent = namedtuple('Constituent', ['value', 'tree_hx', 'tree_cx'])

class SentenceBoundary(Enum):
    NONE = 1
    WORDS = 2
    EVERYTHING = 3

class StackHistory(Enum):
    LSTM = 1
    ATTN = 2

class ConstituencyComposition(Enum):
    BILSTM = 1
    MAX = 2
    TREE_LSTM = 3
    BILSTM_MAX = 4
    BIGRAM = 5
    ATTN = 6
    TREE_LSTM_CX = 7
    UNTIED_MAX = 8
    KEY = 9
    UNTIED_KEY = 10

class LSTMModel(BaseModel, nn.Module):

    def __init__(self, pretrain, forward_charlm, backward_charlm, bert_model, bert_tokenizer, force_bert_saved, transitions, constituents, tags, words, rare_words, root_labels, constituent_opens, unary_limit, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        pretrain: a Pretrain object\n        transitions: a list of all possible transitions which will be\n          used to build trees\n        constituents: a list of all possible constituents in the treebank\n        tags: a list of all possible tags in the treebank\n        words: a list of all known words, used for a delta word embedding.\n          note that there will be an attempt made to learn UNK words as well,\n          and tags by themselves may help UNK words\n        rare_words: a list of rare words, used to occasionally replace with UNK\n        root_labels: probably ROOT, although apparently some treebanks like TOP\n        constituent_opens: a list of all possible open nodes which will go on the stack\n          - this might be different from constituents if there are nodes\n            which represent multiple constituents at once\n        args: hidden_size, transition_hidden_size, etc as gotten from\n          constituency_parser.py\n\n        Note that it might look like a hassle to pass all of this in\n        when it can be collected directly from the trees themselves.\n        However, that would only work at train time.  At eval or\n        pipeline time we will load the lists from the saved model.\n        '
        super().__init__(transition_scheme=args['transition_scheme'], unary_limit=unary_limit, reverse_sentence=args.get('reversed', False))
        self.args = args
        self.unsaved_modules = []
        emb_matrix = pretrain.emb
        self.add_unsaved_module('embedding', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
        self.vocab_map = {word.replace('\xa0', ' '): i for (i, word) in enumerate(pretrain.vocab)}
        self.register_buffer('vocab_tensors', torch.tensor(range(len(pretrain.vocab)), requires_grad=False))
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]
        self.root_labels = sorted(list(root_labels))
        self.constituents = sorted(list(constituents))
        self.hidden_size = self.args['hidden_size']
        self.constituency_composition = self.args.get('constituency_composition', ConstituencyComposition.BILSTM)
        if self.constituency_composition in (ConstituencyComposition.ATTN, ConstituencyComposition.KEY, ConstituencyComposition.UNTIED_KEY):
            self.reduce_heads = self.args['reduce_heads']
            if self.hidden_size % self.reduce_heads != 0:
                self.hidden_size = self.hidden_size + self.reduce_heads - self.hidden_size % self.reduce_heads
        if args['constituent_stack'] == StackHistory.ATTN:
            self.reduce_heads = self.args['reduce_heads']
            if self.hidden_size % args['constituent_heads'] != 0:
                self.hidden_size = self.hidden_size + args['constituent_heads'] - hidden_size % args['constituent_heads']
                if self.constituency_composition == ConstituencyComposition.ATTN and self.hidden_size % self.reduce_heads != 0:
                    raise ValueError('--reduce_heads and --constituent_heads not compatible!')
        self.transition_hidden_size = self.args['transition_hidden_size']
        if args['transition_stack'] == StackHistory.ATTN:
            if self.transition_hidden_size % args['transition_heads'] > 0:
                logger.warning('transition_hidden_size %d %% transition_heads %d != 0.  reconfiguring', transition_hidden_size, args['transition_heads'])
                self.transition_hidden_size = self.transition_hidden_size + args['transition_heads'] - self.transition_hidden_size % args['transition_heads']
        self.tag_embedding_dim = self.args['tag_embedding_dim']
        self.transition_embedding_dim = self.args['transition_embedding_dim']
        self.delta_embedding_dim = self.args['delta_embedding_dim']
        self.word_input_size = self.embedding_dim + self.tag_embedding_dim + self.delta_embedding_dim
        if forward_charlm is not None:
            self.add_unsaved_module('forward_charlm', forward_charlm)
            self.word_input_size += self.forward_charlm.hidden_dim()
            if not forward_charlm.is_forward_lm:
                raise ValueError('Got a backward charlm as a forward charlm!')
        else:
            self.forward_charlm = None
        if backward_charlm is not None:
            self.add_unsaved_module('backward_charlm', backward_charlm)
            self.word_input_size += self.backward_charlm.hidden_dim()
            if backward_charlm.is_forward_lm:
                raise ValueError('Got a forward charlm as a backward charlm!')
        else:
            self.backward_charlm = None
        self.delta_words = sorted(set(words))
        self.delta_word_map = {word: i + 2 for (i, word) in enumerate(self.delta_words)}
        assert PAD_ID == 0
        assert UNK_ID == 1
        self.delta_embedding = nn.Embedding(num_embeddings=len(self.delta_words) + 2, embedding_dim=self.delta_embedding_dim, padding_idx=0)
        nn.init.normal_(self.delta_embedding.weight, std=0.05)
        self.register_buffer('delta_tensors', torch.tensor(range(len(self.delta_words) + 2), requires_grad=False))
        self.rare_words = set(rare_words)
        self.tags = sorted(list(tags))
        if self.tag_embedding_dim > 0:
            self.tag_map = {t: i + 2 for (i, t) in enumerate(self.tags)}
            self.tag_embedding = nn.Embedding(num_embeddings=len(tags) + 2, embedding_dim=self.tag_embedding_dim, padding_idx=0)
            nn.init.normal_(self.tag_embedding.weight, std=0.25)
            self.register_buffer('tag_tensors', torch.tensor(range(len(self.tags) + 2), requires_grad=False))
        self.num_lstm_layers = self.args['num_lstm_layers']
        self.num_tree_lstm_layers = self.args['num_tree_lstm_layers']
        self.lstm_layer_dropout = self.args['lstm_layer_dropout']
        self.word_dropout = nn.Dropout(self.args['word_dropout'])
        self.predict_dropout = nn.Dropout(self.args['predict_dropout'])
        self.lstm_input_dropout = nn.Dropout(self.args['lstm_input_dropout'])
        self.register_buffer('word_zeros', torch.zeros(self.hidden_size * self.num_tree_lstm_layers))
        self.register_buffer('constituent_zeros', torch.zeros(self.num_lstm_layers, 1, self.hidden_size))
        self.sentence_boundary_vectors = self.args['sentence_boundary_vectors']
        if self.sentence_boundary_vectors is not SentenceBoundary.NONE:
            self.register_parameter('word_start_embedding', torch.nn.Parameter(0.2 * torch.randn(self.word_input_size, requires_grad=True)))
            self.register_parameter('word_end_embedding', torch.nn.Parameter(0.2 * torch.randn(self.word_input_size, requires_grad=True)))
        self.force_bert_saved = force_bert_saved
        if self.args['bert_finetune'] or self.args['stage1_bert_finetune'] or force_bert_saved:
            self.bert_model = bert_model
        else:
            self.add_unsaved_module('bert_model', bert_model)
        self.add_unsaved_module('bert_tokenizer', bert_tokenizer)
        if bert_model is not None:
            if bert_tokenizer is None:
                raise ValueError('Cannot have a bert model without a tokenizer')
            self.bert_dim = self.bert_model.config.hidden_size
            if args['bert_hidden_layers']:
                if args['bert_hidden_layers'] > bert_model.config.num_hidden_layers:
                    args['bert_hidden_layers'] = bert_model.config.num_hidden_layers + 1
                self.bert_layer_mix = nn.Linear(args['bert_hidden_layers'], 1, bias=False)
                nn.init.zeros_(self.bert_layer_mix.weight)
            else:
                self.bert_layer_mix = None
            self.word_input_size = self.word_input_size + self.bert_dim
        self.partitioned_transformer_module = None
        self.pattn_d_model = 0
        if LSTMModel.uses_pattn(self.args):
            self.pattn_d_model = self.args['pattn_d_model'] // 2 * 2
            self.partitioned_transformer_module = PartitionedTransformerModule(self.args['pattn_num_layers'], d_model=self.pattn_d_model, n_head=self.args['pattn_num_heads'], d_qkv=self.args['pattn_d_kv'], d_ff=self.args['pattn_d_ff'], ff_dropout=self.args['pattn_relu_dropout'], residual_dropout=self.args['pattn_residual_dropout'], attention_dropout=self.args['pattn_attention_dropout'], word_input_size=self.word_input_size, bias=self.args['pattn_bias'], morpho_emb_dropout=self.args['pattn_morpho_emb_dropout'], timing=self.args['pattn_timing'], encoder_max_len=self.args['pattn_encoder_max_len'])
            self.word_input_size += self.pattn_d_model
        self.label_attention_module = None
        if LSTMModel.uses_lattn(self.args):
            if self.partitioned_transformer_module is None:
                logger.error('Not using Labeled Attention, as the Partitioned Attention module is not used')
            else:
                if self.args['lattn_combined_input']:
                    self.lattn_d_input = self.word_input_size
                else:
                    self.lattn_d_input = self.pattn_d_model
                self.label_attention_module = LabelAttentionModule(self.lattn_d_input, self.args['lattn_d_input_proj'], self.args['lattn_d_kv'], self.args['lattn_d_kv'], self.args['lattn_d_l'], self.args['lattn_d_proj'], self.args['lattn_combine_as_self'], self.args['lattn_resdrop'], self.args['lattn_q_as_matrix'], self.args['lattn_residual_dropout'], self.args['lattn_attention_dropout'], self.pattn_d_model // 2, self.args['lattn_d_ff'], self.args['lattn_relu_dropout'], self.args['lattn_partitioned'])
                self.word_input_size = self.word_input_size + self.args['lattn_d_proj'] * self.args['lattn_d_l']
        self.word_lstm = nn.LSTM(input_size=self.word_input_size, hidden_size=self.hidden_size, num_layers=self.num_lstm_layers, bidirectional=True, dropout=self.lstm_layer_dropout)
        self.word_to_constituent = nn.Linear(self.hidden_size * 2, self.hidden_size * self.num_tree_lstm_layers)
        initialize_linear(self.word_to_constituent, self.args['nonlinearity'], self.hidden_size * 2)
        self.transitions = sorted(list(transitions))
        self.transition_map = {t: i for (i, t) in enumerate(self.transitions)}
        self.register_buffer('transition_tensors', torch.tensor(range(len(transitions)), requires_grad=False))
        self.transition_embedding = nn.Embedding(num_embeddings=len(transitions), embedding_dim=self.transition_embedding_dim)
        nn.init.normal_(self.transition_embedding.weight, std=0.25)
        if args['transition_stack'] == StackHistory.LSTM:
            self.transition_stack = LSTMTreeStack(input_size=self.transition_embedding_dim, hidden_size=self.transition_hidden_size, num_lstm_layers=self.num_lstm_layers, dropout=self.lstm_layer_dropout, uses_boundary_vector=self.sentence_boundary_vectors is SentenceBoundary.EVERYTHING, input_dropout=self.lstm_input_dropout)
        elif args['transition_stack'] == StackHistory.ATTN:
            self.transition_stack = TransformerTreeStack(input_size=self.transition_embedding_dim, output_size=self.transition_hidden_size, input_dropout=self.lstm_input_dropout, use_position=True, num_heads=args['transition_heads'])
        else:
            raise ValueError('Unhandled transition_stack StackHistory: {}'.format(args['transition_stack']))
        self.constituent_opens = sorted(list(constituent_opens))
        self.constituent_open_map = {x: i for (i, x) in enumerate(self.constituent_opens)}
        self.constituent_open_embedding = nn.Embedding(num_embeddings=len(self.constituent_open_map), embedding_dim=self.hidden_size)
        nn.init.normal_(self.constituent_open_embedding.weight, std=0.2)
        if args['constituent_stack'] == StackHistory.LSTM:
            self.constituent_stack = LSTMTreeStack(input_size=self.hidden_size, hidden_size=self.hidden_size, num_lstm_layers=self.num_lstm_layers, dropout=self.lstm_layer_dropout, uses_boundary_vector=self.sentence_boundary_vectors is SentenceBoundary.EVERYTHING, input_dropout=self.lstm_input_dropout)
        elif args['constituent_stack'] == StackHistory.ATTN:
            self.constituent_stack = TransformerTreeStack(input_size=self.hidden_size, output_size=self.hidden_size, input_dropout=self.lstm_input_dropout, use_position=True, num_heads=args['constituent_heads'])
        else:
            raise ValueError('Unhandled constituent_stack StackHistory: {}'.format(args['transition_stack']))
        if args['combined_dummy_embedding']:
            self.dummy_embedding = self.constituent_open_embedding
        else:
            self.dummy_embedding = nn.Embedding(num_embeddings=len(self.constituent_open_map), embedding_dim=self.hidden_size)
            nn.init.normal_(self.dummy_embedding.weight, std=0.2)
        self.register_buffer('constituent_open_tensors', torch.tensor(range(len(constituent_opens)), requires_grad=False))
        if self.constituency_composition == ConstituencyComposition.BILSTM or self.constituency_composition == ConstituencyComposition.BILSTM_MAX:
            self.constituent_reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_lstm_layers, bidirectional=True, dropout=self.lstm_layer_dropout)
            if self.constituency_composition == ConstituencyComposition.BILSTM:
                self.reduce_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
                initialize_linear(self.reduce_linear, self.args['nonlinearity'], self.hidden_size * 2)
            else:
                self.reduce_forward = nn.Linear(self.hidden_size, self.hidden_size)
                self.reduce_backward = nn.Linear(self.hidden_size, self.hidden_size)
                initialize_linear(self.reduce_forward, self.args['nonlinearity'], self.hidden_size)
                initialize_linear(self.reduce_backward, self.args['nonlinearity'], self.hidden_size)
        elif self.constituency_composition == ConstituencyComposition.MAX:
            self.reduce_linear = nn.Linear(self.hidden_size, self.hidden_size)
            initialize_linear(self.reduce_linear, self.args['nonlinearity'], self.hidden_size)
        elif self.constituency_composition == ConstituencyComposition.UNTIED_MAX:
            self.register_parameter('reduce_linear_weight', torch.nn.Parameter(torch.randn(len(constituent_opens), self.hidden_size, self.hidden_size, requires_grad=True)))
            self.register_parameter('reduce_linear_bias', torch.nn.Parameter(torch.randn(len(constituent_opens), self.hidden_size, requires_grad=True)))
            for layer_idx in range(len(constituent_opens)):
                nn.init.kaiming_normal_(self.reduce_linear_weight[layer_idx], nonlinearity=self.args['nonlinearity'])
            nn.init.uniform_(self.reduce_linear_bias, 0, 1 / (self.hidden_size * 2) ** 0.5)
        elif self.constituency_composition == ConstituencyComposition.BIGRAM:
            self.reduce_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.reduce_bigram = nn.Linear(self.hidden_size * 2, self.hidden_size)
            initialize_linear(self.reduce_linear, self.args['nonlinearity'], self.hidden_size)
            initialize_linear(self.reduce_bigram, self.args['nonlinearity'], self.hidden_size)
        elif self.constituency_composition == ConstituencyComposition.ATTN:
            self.reduce_attn = nn.MultiheadAttention(self.hidden_size, self.reduce_heads)
        elif self.constituency_composition == ConstituencyComposition.KEY or self.constituency_composition == ConstituencyComposition.UNTIED_KEY:
            if self.args['reduce_position']:
                self.add_unsaved_module('reduce_position', ConcatSinusoidalEncoding(self.args['reduce_position'], 50))
            else:
                self.add_unsaved_module('reduce_position', nn.Identity())
            self.reduce_query = nn.Linear(self.hidden_size + self.args['reduce_position'], self.hidden_size, bias=False)
            self.reduce_value = nn.Linear(self.hidden_size + self.args['reduce_position'], self.hidden_size)
            if self.constituency_composition == ConstituencyComposition.KEY:
                self.register_parameter('reduce_key', torch.nn.Parameter(torch.randn(self.reduce_heads, self.hidden_size // self.reduce_heads, 1, requires_grad=True)))
            else:
                self.register_parameter('reduce_key', torch.nn.Parameter(torch.randn(len(constituent_opens), self.reduce_heads, self.hidden_size // self.reduce_heads, 1, requires_grad=True)))
        elif self.constituency_composition == ConstituencyComposition.TREE_LSTM:
            self.constituent_reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_tree_lstm_layers, dropout=self.lstm_layer_dropout)
        elif self.constituency_composition == ConstituencyComposition.TREE_LSTM_CX:
            self.constituent_reduce_embedding = nn.Embedding(num_embeddings=len(tags) + 2, embedding_dim=self.num_tree_lstm_layers * self.hidden_size)
            self.constituent_reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_tree_lstm_layers, dropout=self.lstm_layer_dropout)
        else:
            raise ValueError('Unhandled ConstituencyComposition: {}'.format(self.constituency_composition))
        self.nonlinearity = build_nonlinearity(self.args['nonlinearity'])
        self.maxout_k = self.args.get('maxout_k', 0)
        self.output_layers = self.build_output_layers(self.args['num_output_layers'], len(transitions), self.maxout_k)

    def reverse_sentence(self):
        if False:
            i = 10
            return i + 15
        return self._reverse_sentence

    @staticmethod
    def uses_lattn(args):
        if False:
            while True:
                i = 10
        return args.get('use_lattn', True) and args.get('lattn_d_proj', 0) > 0 and (args.get('lattn_d_l', 0) > 0)

    @staticmethod
    def uses_pattn(args):
        if False:
            return 10
        return args['pattn_num_heads'] > 0 and args['pattn_num_layers'] > 0

    def copy_with_new_structure(self, other):
        if False:
            while True:
                i = 10
        "\n        Copy parameters from the other model to this model\n\n        word_lstm can change size if the other model didn't use pattn / lattn and this one does.\n        In that case, the new values are initialized to 0.\n        This will rebuild the model in such a way that the outputs will be\n        exactly the same as the previous model.\n        "
        if self.constituency_composition != other.constituency_composition and self.constituency_composition != ConstituencyComposition.UNTIED_MAX:
            raise ValueError('Models are incompatible: self.constituency_composition == {}, other.constituency_composition == {}'.format(self.constituency_composition, other.constituency_composition))
        for (name, other_parameter) in other.named_parameters():
            if name.startswith('reduce_linear.') and self.constituency_composition == ConstituencyComposition.UNTIED_MAX:
                if name == 'reduce_linear.weight':
                    my_parameter = self.reduce_linear_weight
                elif name == 'reduce_linear.bias':
                    my_parameter = self.reduce_linear_bias
                else:
                    raise ValueError('Unexpected other parameter name {}'.format(name))
                for idx in range(len(self.constituent_opens)):
                    my_parameter[idx].data.copy_(other_parameter.data)
            elif name.startswith('word_lstm.weight_ih_l0'):
                my_parameter = self.get_parameter(name)
                copy_size = min(other_parameter.data.shape[-1], my_parameter.data.shape[-1])
                new_values = torch.zeros_like(my_parameter.data)
                new_values[..., :copy_size] = other_parameter.data[..., :copy_size]
                my_parameter.data.copy_(new_values)
            else:
                self.get_parameter(name).data.copy_(other_parameter.data)

    def build_output_layers(self, num_output_layers, final_layer_size, maxout_k):
        if False:
            print('Hello World!')
        '\n        Build a ModuleList of Linear transformations for the given num_output_layers\n\n        The final layer size can be specified.\n        Initial layer size is the combination of word, constituent, and transition vectors\n        Middle layer sizes are self.hidden_size\n        '
        middle_layers = num_output_layers - 1
        predict_input_size = [self.hidden_size + self.hidden_size * self.num_tree_lstm_layers + self.transition_hidden_size] + [self.hidden_size] * middle_layers
        predict_output_size = [self.hidden_size] * middle_layers + [final_layer_size]
        if not maxout_k:
            output_layers = nn.ModuleList([nn.Linear(input_size, output_size) for (input_size, output_size) in zip(predict_input_size, predict_output_size)])
            for (output_layer, input_size) in zip(output_layers, predict_input_size):
                initialize_linear(output_layer, self.args['nonlinearity'], input_size)
        else:
            output_layers = nn.ModuleList([MaxoutLinear(input_size, output_size, maxout_k) for (input_size, output_size) in zip(predict_input_size, predict_output_size)])
        return output_layers

    def num_words_known(self, words):
        if False:
            while True:
                i = 10
        return sum((word in self.vocab_map or word.lower() in self.vocab_map for word in words))

    def uses_xpos(self):
        if False:
            print('Hello World!')
        return self.args['retag_package'] is not None and self.args['retag_method'] == 'xpos'

    def add_unsaved_module(self, name, module):
        if False:
            while True:
                i = 10
        '\n        Adds a module which will not be saved to disk\n\n        Best used for large models such as pretrained word embeddings\n        '
        self.unsaved_modules += [name]
        setattr(self, name, module)
        if module is not None and name in ('bert_model', 'forward_charlm', 'backward_charlm'):
            for (_, parameter) in module.named_parameters():
                parameter.requires_grad = False

    def is_unsaved_module(self, name):
        if False:
            while True:
                i = 10
        return name.split('.')[0] in self.unsaved_modules

    def get_root_labels(self):
        if False:
            while True:
                i = 10
        return self.root_labels

    def get_norms(self):
        if False:
            return 10
        lines = []
        skip = set()
        if self.constituency_composition == ConstituencyComposition.UNTIED_MAX:
            skip = {'reduce_linear_weight', 'reduce_linear_bias'}
            lines.append('reduce_linear:')
            for (c_idx, c_open) in enumerate(self.constituent_opens):
                lines.append('  %s weight %.6g bias %.6g' % (c_open, torch.norm(self.reduce_linear_weight[c_idx]).item(), torch.norm(self.reduce_linear_bias[c_idx]).item()))
        max_name_len = max((len(name) for (name, param) in self.named_parameters() if param.requires_grad and name not in skip))
        max_norm_len = max((len('%.6g' % torch.norm(param).item()) for (name, param) in self.named_parameters() if param.requires_grad and name not in skip))
        format_string = '%-' + str(max_name_len) + 's   norm %' + str(max_norm_len) + 's  zeros %d / %d'
        for (name, param) in self.named_parameters():
            if param.requires_grad and name not in skip:
                zeros = torch.sum(param.abs() < 1e-06).item()
                norm = '%.6g' % torch.norm(param).item()
                lines.append(format_string % (name, norm, zeros, param.nelement()))
        return lines

    def log_norms(self):
        if False:
            print('Hello World!')
        lines = ['NORMS FOR MODEL PARAMTERS']
        lines.extend(self.get_norms())
        logger.info('\n'.join(lines))

    def log_shapes(self):
        if False:
            print('Hello World!')
        lines = ['NORMS FOR MODEL PARAMTERS']
        for (name, param) in self.named_parameters():
            if param.requires_grad:
                lines.append('{} {}'.format(name, param.shape))
        logger.info('\n'.join(lines))

    def initial_word_queues(self, tagged_word_lists):
        if False:
            print('Hello World!')
        "\n        Produce initial word queues out of the model's LSTMs for use in the tagged word lists.\n\n        Operates in a batched fashion to reduce the runtime for the LSTM operations\n        "
        device = next(self.parameters()).device
        vocab_map = self.vocab_map

        def map_word(word):
            if False:
                while True:
                    i = 10
            idx = vocab_map.get(word, None)
            if idx is not None:
                return idx
            return vocab_map.get(word.lower(), UNK_ID)
        all_word_inputs = []
        all_word_labels = [[word.children[0].label for word in tagged_words] for tagged_words in tagged_word_lists]
        for (sentence_idx, tagged_words) in enumerate(tagged_word_lists):
            word_labels = all_word_labels[sentence_idx]
            word_idx = torch.stack([self.vocab_tensors[map_word(word.children[0].label)] for word in tagged_words])
            word_input = self.embedding(word_idx)
            if self.training:
                delta_labels = [None if word in self.rare_words and random.random() < self.args['rare_word_unknown_frequency'] else word for word in word_labels]
            else:
                delta_labels = word_labels
            delta_idx = torch.stack([self.delta_tensors[self.delta_word_map.get(word, UNK_ID)] for word in delta_labels])
            delta_input = self.delta_embedding(delta_idx)
            word_inputs = [word_input, delta_input]
            if self.tag_embedding_dim > 0:
                if self.training:
                    tag_labels = [None if random.random() < self.args['tag_unknown_frequency'] else word.label for word in tagged_words]
                else:
                    tag_labels = [word.label for word in tagged_words]
                tag_idx = torch.stack([self.tag_tensors[self.tag_map.get(tag, UNK_ID)] for tag in tag_labels])
                tag_input = self.tag_embedding(tag_idx)
                word_inputs.append(tag_input)
            all_word_inputs.append(word_inputs)
        if self.forward_charlm is not None:
            all_forward_chars = self.forward_charlm.build_char_representation(all_word_labels)
            for (word_inputs, forward_chars) in zip(all_word_inputs, all_forward_chars):
                word_inputs.append(forward_chars)
        if self.backward_charlm is not None:
            all_backward_chars = self.backward_charlm.build_char_representation(all_word_labels)
            for (word_inputs, backward_chars) in zip(all_word_inputs, all_backward_chars):
                word_inputs.append(backward_chars)
        all_word_inputs = [torch.cat(word_inputs, dim=1) for word_inputs in all_word_inputs]
        if self.sentence_boundary_vectors is not SentenceBoundary.NONE:
            word_start = self.word_start_embedding.unsqueeze(0)
            word_end = self.word_end_embedding.unsqueeze(0)
            all_word_inputs = [torch.cat([word_start, word_inputs, word_end], dim=0) for word_inputs in all_word_inputs]
        if self.bert_model is not None:
            bert_embeddings = extract_bert_embeddings(self.args['bert_model'], self.bert_tokenizer, self.bert_model, all_word_labels, device, keep_endpoints=self.sentence_boundary_vectors is not SentenceBoundary.NONE, num_layers=self.bert_layer_mix.in_features if self.bert_layer_mix is not None else None, detach=not self.args['bert_finetune'] and (not self.args['stage1_bert_finetune']))
            if self.bert_layer_mix is not None:
                bert_embeddings = [self.bert_layer_mix(feature).squeeze(2) + feature.sum(axis=2) / self.bert_layer_mix.in_features for feature in bert_embeddings]
            all_word_inputs = [torch.cat((x, y), axis=1) for (x, y) in zip(all_word_inputs, bert_embeddings)]
        if self.partitioned_transformer_module is not None:
            partitioned_embeddings = self.partitioned_transformer_module(None, all_word_inputs)
            all_word_inputs = [torch.cat((x, y[:x.shape[0], :]), axis=1) for (x, y) in zip(all_word_inputs, partitioned_embeddings)]
        if self.label_attention_module is not None:
            if self.args['lattn_combined_input']:
                labeled_representations = self.label_attention_module(all_word_inputs, tagged_word_lists)
            else:
                labeled_representations = self.label_attention_module(partitioned_embeddings, tagged_word_lists)
            all_word_inputs = [torch.cat((x, y[:x.shape[0], :]), axis=1) for (x, y) in zip(all_word_inputs, labeled_representations)]
        all_word_inputs = [self.word_dropout(word_inputs) for word_inputs in all_word_inputs]
        packed_word_input = torch.nn.utils.rnn.pack_sequence(all_word_inputs, enforce_sorted=False)
        (word_output, _) = self.word_lstm(packed_word_input)
        (word_output, word_output_lens) = torch.nn.utils.rnn.pad_packed_sequence(word_output)
        word_queues = []
        for (sentence_idx, tagged_words) in enumerate(tagged_word_lists):
            if self.sentence_boundary_vectors is not SentenceBoundary.NONE:
                sentence_output = word_output[:len(tagged_words) + 2, sentence_idx, :]
            else:
                sentence_output = word_output[:len(tagged_words), sentence_idx, :]
            sentence_output = self.word_to_constituent(sentence_output)
            sentence_output = self.nonlinearity(sentence_output)
            if self.sentence_boundary_vectors is not SentenceBoundary.NONE:
                word_queue = [WordNode(None, sentence_output[0, :])]
                word_queue += [WordNode(tag_node, sentence_output[idx + 1, :]) for (idx, tag_node) in enumerate(tagged_words)]
                word_queue.append(WordNode(None, sentence_output[len(tagged_words) + 1, :]))
            else:
                word_queue = [WordNode(None, self.word_zeros)]
                word_queue += [WordNode(tag_node, sentence_output[idx, :]) for (idx, tag_node) in enumerate(tagged_words)]
                word_queue.append(WordNode(None, self.word_zeros))
            if self.reverse_sentence():
                word_queue = list(reversed(word_queue))
            word_queues.append(word_queue)
        return word_queues

    def initial_transitions(self):
        if False:
            i = 10
            return i + 15
        '\n        Return an initial TreeStack with no transitions\n        '
        return self.transition_stack.initial_state()

    def initial_constituents(self):
        if False:
            print('Hello World!')
        '\n        Return an initial TreeStack with no constituents\n        '
        return self.constituent_stack.initial_state(Constituent(None, self.constituent_zeros, self.constituent_zeros))

    def get_word(self, word_node):
        if False:
            return 10
        return word_node.value

    def transform_word_to_constituent(self, state):
        if False:
            print('Hello World!')
        word_node = state.get_word(state.word_position)
        word = word_node.value
        if self.constituency_composition == ConstituencyComposition.TREE_LSTM:
            return Constituent(word, word_node.hx.view(self.num_tree_lstm_layers, self.hidden_size), self.word_zeros.view(self.num_tree_lstm_layers, self.hidden_size))
        elif self.constituency_composition == ConstituencyComposition.TREE_LSTM_CX:
            tag = word.label
            tree_hx = word_node.hx.view(self.num_tree_lstm_layers, self.hidden_size)
            tag_tensor = self.tag_tensors[self.tag_map.get(tag, UNK_ID)]
            tree_cx = self.constituent_reduce_embedding(tag_tensor)
            tree_cx = tree_cx.view(self.num_tree_lstm_layers, self.hidden_size)
            return Constituent(word, tree_hx, tree_cx * tree_hx)
        else:
            return Constituent(word, word_node.hx[:self.hidden_size].unsqueeze(0), None)

    def dummy_constituent(self, dummy):
        if False:
            for i in range(10):
                print('nop')
        label = dummy.label
        open_index = self.constituent_open_tensors[self.constituent_open_map[label]]
        hx = self.dummy_embedding(open_index)
        return Constituent(dummy, hx.unsqueeze(0), None)

    def build_constituents(self, labels, children_lists):
        if False:
            while True:
                i = 10
        '\n        Build new constituents with the given label from the list of children\n\n        labels is a list of labels for each of the new nodes to construct\n        children_lists is a list of children that go under each of the new nodes\n        lists of each are used so that we can stack operations\n        '
        if self.constituency_composition == ConstituencyComposition.BILSTM or self.constituency_composition == ConstituencyComposition.BILSTM_MAX:
            node_hx = [[child.value.tree_hx.squeeze(0) for child in children] for children in children_lists]
            label_hx = [self.constituent_open_embedding(self.constituent_open_tensors[self.constituent_open_map[label]]) for label in labels]
            max_length = max((len(children) for children in children_lists))
            zeros = torch.zeros(self.hidden_size, device=label_hx[0].device)
            unpacked_hx = [[lhx] + nhx + [lhx] + [zeros] * (max_length - len(nhx)) for (lhx, nhx) in zip(label_hx, node_hx)]
            unpacked_hx = [self.lstm_input_dropout(torch.stack(nhx)) for nhx in unpacked_hx]
            packed_hx = torch.stack(unpacked_hx, axis=1)
            packed_hx = torch.nn.utils.rnn.pack_padded_sequence(packed_hx, [len(x) + 2 for x in children_lists], enforce_sorted=False)
            lstm_output = self.constituent_reduce_lstm(packed_hx)
            if self.constituency_composition == ConstituencyComposition.BILSTM:
                lstm_output = lstm_output[1][0]
                forward_hx = lstm_output[-2, :, :]
                backward_hx = lstm_output[-1, :, :]
                hx = self.reduce_linear(torch.cat((forward_hx, backward_hx), axis=1))
            else:
                (lstm_output, lstm_lengths) = torch.nn.utils.rnn.pad_packed_sequence(lstm_output[0])
                lstm_output = [lstm_output[1:length - 1, x, :] for (x, length) in zip(range(len(lstm_lengths)), lstm_lengths)]
                lstm_output = torch.stack([torch.max(x, 0).values for x in lstm_output], axis=0)
                hx = self.reduce_forward(lstm_output[:, :self.hidden_size]) + self.reduce_backward(lstm_output[:, self.hidden_size:])
            lstm_hx = self.nonlinearity(hx).unsqueeze(0)
            lstm_cx = None
        elif self.constituency_composition == ConstituencyComposition.MAX:
            node_hx = [[child.value.tree_hx for child in children] for children in children_lists]
            unpacked_hx = [self.lstm_input_dropout(torch.max(torch.stack(nhx), 0).values) for nhx in node_hx]
            packed_hx = torch.stack(unpacked_hx, axis=1)
            hx = self.reduce_linear(packed_hx)
            lstm_hx = self.nonlinearity(hx)
            lstm_cx = None
        elif self.constituency_composition == ConstituencyComposition.UNTIED_MAX:
            node_hx = [[child.value.tree_hx for child in children] for children in children_lists]
            unpacked_hx = [self.lstm_input_dropout(torch.max(torch.stack(nhx), 0).values) for nhx in node_hx]
            label_indices = [self.constituent_open_map[label] for label in labels]
            hx = [torch.matmul(self.reduce_linear_weight[label_idx], hx_layer.squeeze(0)) + self.reduce_linear_bias[label_idx] for (label_idx, hx_layer) in zip(label_indices, unpacked_hx)]
            hx = torch.stack(hx, axis=0)
            hx = hx.unsqueeze(0)
            lstm_hx = self.nonlinearity(hx)
            lstm_cx = None
        elif self.constituency_composition == ConstituencyComposition.BIGRAM:
            node_hx = [[child.value.tree_hx for child in children] for children in children_lists]
            unpacked_hx = []
            for nhx in node_hx:
                stacked_nhx = self.lstm_input_dropout(torch.cat(nhx, axis=0))
                if stacked_nhx.shape[0] > 1:
                    bigram_hx = torch.cat((stacked_nhx[:-1, :], stacked_nhx[1:, :]), axis=1)
                    bigram_hx = self.reduce_bigram(bigram_hx) / 2
                    stacked_nhx = torch.cat((stacked_nhx, bigram_hx), axis=0)
                unpacked_hx.append(torch.max(stacked_nhx, 0).values)
            packed_hx = torch.stack(unpacked_hx, axis=0).unsqueeze(0)
            hx = self.reduce_linear(packed_hx)
            lstm_hx = self.nonlinearity(hx)
            lstm_cx = None
        elif self.constituency_composition == ConstituencyComposition.ATTN:
            node_hx = [[child.value.tree_hx for child in children] for children in children_lists]
            label_hx = [self.constituent_open_embedding(self.constituent_open_tensors[self.constituent_open_map[label]]) for label in labels]
            unpacked_hx = [torch.stack(nhx) for nhx in node_hx]
            unpacked_hx = [torch.cat((lhx.unsqueeze(0).unsqueeze(0), nhx), axis=0) for (lhx, nhx) in zip(label_hx, unpacked_hx)]
            unpacked_hx = [self.reduce_attn(nhx, nhx, nhx)[0].squeeze(1) for nhx in unpacked_hx]
            unpacked_hx = [self.lstm_input_dropout(torch.max(nhx, 0).values) for nhx in unpacked_hx]
            hx = torch.stack(unpacked_hx, axis=0)
            lstm_hx = self.nonlinearity(hx).unsqueeze(0)
            lstm_cx = None
        elif self.constituency_composition == ConstituencyComposition.KEY or self.constituency_composition == ConstituencyComposition.UNTIED_KEY:
            node_hx = [torch.stack([child.value.tree_hx for child in children]) for children in children_lists]
            node_hx = [self.reduce_position(x.reshape(x.shape[0], -1)) for x in node_hx]
            query_hx = [self.reduce_query(nhx) for nhx in node_hx]
            query_hx = [nhx.reshape(nhx.shape[0], self.reduce_heads, -1).transpose(0, 1) for nhx in query_hx]
            if self.constituency_composition == ConstituencyComposition.KEY:
                queries = [torch.matmul(nhx, self.reduce_key) for nhx in query_hx]
            else:
                label_indices = [self.constituent_open_map[label] for label in labels]
                queries = [torch.matmul(nhx, self.reduce_key[label_idx]) for (nhx, label_idx) in zip(query_hx, label_indices)]
            weights = [torch.nn.functional.softmax(nhx, dim=1).transpose(1, 2) for nhx in queries]
            value_hx = [self.reduce_value(nhx) for nhx in node_hx]
            value_hx = [nhx.reshape(nhx.shape[0], self.reduce_heads, -1).transpose(0, 1) for nhx in value_hx]
            unpacked_hx = [torch.matmul(weight, nhx).squeeze(1) for (weight, nhx) in zip(weights, value_hx)]
            unpacked_hx = [nhx.reshape(-1) for nhx in unpacked_hx]
            hx = torch.stack(unpacked_hx, axis=0).unsqueeze(0)
            lstm_hx = self.nonlinearity(hx)
            lstm_cx = None
        elif self.constituency_composition in (ConstituencyComposition.TREE_LSTM, ConstituencyComposition.TREE_LSTM_CX):
            label_hx = [self.lstm_input_dropout(self.constituent_open_embedding(self.constituent_open_tensors[self.constituent_open_map[label]])) for label in labels]
            label_hx = torch.stack(label_hx).unsqueeze(0)
            max_length = max((len(children) for children in children_lists))
            node_hx = [[child.value.tree_hx for child in children] for children in children_lists]
            unpacked_hx = [self.lstm_input_dropout(torch.stack(nhx)) for nhx in node_hx]
            unpacked_hx = [nhx.max(dim=0) for nhx in unpacked_hx]
            packed_hx = torch.stack([nhx.values for nhx in unpacked_hx], axis=1)
            node_cx = [torch.stack([child.value.tree_cx for child in children]) for children in children_lists]
            node_cx_indices = [uhx.indices.unsqueeze(0) for uhx in unpacked_hx]
            unpacked_cx = [ncx.gather(0, nci).squeeze(0) for (ncx, nci) in zip(node_cx, node_cx_indices)]
            packed_cx = torch.stack(unpacked_cx, axis=1)
            (_, (lstm_hx, lstm_cx)) = self.constituent_reduce_lstm(label_hx, (packed_hx, packed_cx))
        else:
            raise ValueError('Unhandled ConstituencyComposition: {}'.format(self.constituency_composition))
        constituents = []
        for (idx, (label, children)) in enumerate(zip(labels, children_lists)):
            children = [child.value.value for child in children]
            if isinstance(label, str):
                node = Tree(label=label, children=children)
            else:
                for value in reversed(label):
                    node = Tree(label=value, children=children)
                    children = node
            constituents.append(Constituent(node, lstm_hx[:, idx, :], lstm_cx[:, idx, :] if lstm_cx is not None else None))
        return constituents

    def push_constituents(self, constituent_stacks, constituents):
        if False:
            i = 10
            return i + 15
        current_nodes = [stack.value for stack in constituent_stacks]
        constituent_input = torch.stack([x.tree_hx[-1:] for x in constituents], axis=1)
        return self.constituent_stack.push_states(constituent_stacks, constituents, constituent_input)

    def get_top_constituent(self, constituents):
        if False:
            while True:
                i = 10
        "\n        Extract only the top constituent from a state's constituent\n        sequence, even though it has multiple addition pieces of\n        information\n        "
        constituent_node = constituents.value.value
        return constituent_node.value

    def push_transitions(self, transition_stacks, transitions):
        if False:
            return 10
        '\n        Push all of the given transitions on to the stack as a batch operations.\n\n        Significantly faster than doing one transition at a time.\n        '
        transition_idx = torch.stack([self.transition_tensors[self.transition_map[transition]] for transition in transitions])
        transition_input = self.transition_embedding(transition_idx).unsqueeze(0)
        return self.transition_stack.push_states(transition_stacks, transitions, transition_input)

    def get_top_transition(self, transitions):
        if False:
            return 10
        "\n        Extract only the top transition from a state's transition\n        sequence, even though it has multiple addition pieces of\n        information\n        "
        transition_node = transitions.value
        return transition_node.value

    def forward(self, states):
        if False:
            while True:
                i = 10
        "\n        Return logits for a prediction of what transition to make next\n\n        We've basically done all the work analyzing the state as\n        part of applying the transitions, so this method is very simple\n\n        return shape: (num_states, num_transitions)\n        "
        word_hx = torch.stack([state.get_word(state.word_position).hx for state in states])
        transition_hx = torch.stack([self.transition_stack.output(state.transitions) for state in states])
        constituent_hx = torch.stack([self.constituent_stack.output(state.constituents) for state in states])
        hx = torch.cat((word_hx, transition_hx, constituent_hx), axis=1)
        for (idx, output_layer) in enumerate(self.output_layers):
            hx = self.predict_dropout(hx)
            if not self.maxout_k and idx < len(self.output_layers) - 1:
                hx = self.nonlinearity(hx)
            hx = output_layer(hx)
        return hx

    def predict(self, states, is_legal=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate and return predictions, along with the transitions those predictions represent\n\n        If is_legal is set to True, will only return legal transitions.\n        This means returning None if there are no legal transitions.\n        Hopefully the constraints prevent that from happening\n        '
        predictions = self.forward(states)
        pred_max = torch.argmax(predictions, dim=1)
        scores = torch.take_along_dim(predictions, pred_max.unsqueeze(1), dim=1)
        pred_max = pred_max.detach().cpu()
        pred_trans = [self.transitions[pred_max[idx]] for idx in range(len(states))]
        if is_legal:
            for (idx, (state, trans)) in enumerate(zip(states, pred_trans)):
                if not trans.is_legal(state, self):
                    (_, indices) = predictions[idx, :].sort(descending=True)
                    for index in indices:
                        if self.transitions[index].is_legal(state, self):
                            pred_trans[idx] = self.transitions[index]
                            scores[idx] = predictions[idx, index]
                            break
                    else:
                        pred_trans[idx] = None
                        scores[idx] = None
        return (predictions, pred_trans, scores.squeeze(1))

    def weighted_choice(self, states):
        if False:
            i = 10
            return i + 15
        '\n        Generate and return predictions, and randomly choose a prediction weighted by the scores\n\n        TODO: pass in a temperature\n        '
        predictions = self.forward(states)
        pred_trans = []
        all_scores = []
        for (state, prediction) in zip(states, predictions):
            legal_idx = [idx for idx in range(prediction.shape[0]) if self.transitions[idx].is_legal(state, self)]
            if len(legal_idx) == 0:
                pred_trans.append(None)
                continue
            scores = prediction[legal_idx]
            scores = torch.softmax(scores, dim=0)
            idx = torch.multinomial(scores, 1)
            idx = legal_idx[idx]
            pred_trans.append(self.transitions[idx])
            all_scores.append(prediction[idx])
        all_scores = torch.stack(all_scores)
        return (predictions, pred_trans, all_scores)

    def predict_gold(self, states):
        if False:
            i = 10
            return i + 15
        '\n        For each State, return the next item in the gold_sequence\n        '
        predictions = self.forward(states)
        transitions = [y.gold_sequence[y.num_transitions()] for y in states]
        indices = torch.tensor([self.transition_map[t] for t in transitions], device=predictions.device)
        scores = torch.take_along_dim(predictions, indices.unsqueeze(1), dim=1)
        return (predictions, transitions, scores.squeeze(1))

    def get_params(self, skip_modules=True):
        if False:
            while True:
                i = 10
        '\n        Get a dictionary for saving the model\n        '
        model_state = self.state_dict()
        if skip_modules:
            skipped = [k for k in model_state.keys() if self.is_unsaved_module(k)]
            for k in skipped:
                del model_state[k]
        params = {'model': model_state, 'model_type': 'LSTM', 'config': self.args, 'transitions': self.transitions, 'constituents': self.constituents, 'tags': self.tags, 'words': self.delta_words, 'rare_words': self.rare_words, 'root_labels': self.root_labels, 'constituent_opens': self.constituent_opens, 'unary_limit': self.unary_limit()}
        return params