import copy
import math
import numpy as np
import torch
from configuration_bertabs import BertAbsConfig
from torch import nn
from torch.nn.init import xavier_uniform_
from transformers import BertConfig, BertModel, PreTrainedModel
MAX_SIZE = 5000
BERTABS_FINETUNED_MODEL_ARCHIVE_LIST = ['remi/bertabs-finetuned-cnndm-extractive-abstractive-summarization']

class BertAbsPreTrainedModel(PreTrainedModel):
    config_class = BertAbsConfig
    load_tf_weights = False
    base_model_prefix = 'bert'

class BertAbs(BertAbsPreTrainedModel):

    def __init__(self, args, checkpoint=None, bert_extractive_checkpoint=None):
        if False:
            return 10
        super().__init__(args)
        self.args = args
        self.bert = Bert()
        load_bert_pretrained_extractive = True if bert_extractive_checkpoint else False
        if load_bert_pretrained_extractive:
            self.bert.model.load_state_dict({n[11:]: p for (n, p) in bert_extractive_checkpoint.items() if n.startswith('bert.model')}, strict=True)
        self.vocab_size = self.bert.model.config.vocab_size
        if args.max_pos > 512:
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None, :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
        self.decoder = TransformerDecoder(self.args.dec_layers, self.args.dec_hidden_size, heads=self.args.dec_heads, d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings, vocab_size=self.vocab_size)
        gen_func = nn.LogSoftmax(dim=-1)
        self.generator = nn.Sequential(nn.Linear(args.dec_hidden_size, args.vocab_size), gen_func)
        self.generator[0].weight = self.decoder.embeddings.weight
        load_from_checkpoints = False if checkpoint is None else True
        if load_from_checkpoints:
            self.load_state_dict(checkpoint)

    def init_weights(self):
        if False:
            while True:
                i = 10
        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        for p in self.generator.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                p.data.zero_()

    def forward(self, encoder_input_ids, decoder_input_ids, token_type_ids, encoder_attention_mask, decoder_attention_mask):
        if False:
            while True:
                i = 10
        encoder_output = self.bert(input_ids=encoder_input_ids, token_type_ids=token_type_ids, attention_mask=encoder_attention_mask)
        encoder_hidden_states = encoder_output[0]
        dec_state = self.decoder.init_decoder_state(encoder_input_ids, encoder_hidden_states)
        (decoder_outputs, _) = self.decoder(decoder_input_ids[:, :-1], encoder_hidden_states, dec_state)
        return decoder_outputs

class Bert(nn.Module):
    """This class is not really necessary and should probably disappear."""

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        self.model = BertModel(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        if False:
            while True:
                i = 10
        self.eval()
        with torch.no_grad():
            (encoder_outputs, _) = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, **kwargs)
        return encoder_outputs

class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".

    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a separate copy attention
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, vocab_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)
        self.transformer_layers = nn.ModuleList([TransformerDecoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, input_ids, encoder_hidden_states=None, state=None, attention_mask=None, memory_lengths=None, step=None, cache=None, encoder_attention_mask=None):
        if False:
            i = 10
            return i + 15
        '\n        See :obj:`onmt.modules.RNNDecoderBase.forward()`\n        memory_bank = encoder_hidden_states\n        '
        tgt = input_ids
        memory_bank = encoder_hidden_states
        memory_mask = encoder_attention_mask
        src_words = state.src
        (src_batch, src_len) = src_words.size()
        padding_idx = self.embeddings.padding_idx
        tgt_words = tgt
        (tgt_batch, tgt_len) = tgt_words.size()
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1).expand(tgt_batch, tgt_len, tgt_len)
        if memory_mask is not None:
            src_len = memory_mask.size(-1)
            src_pad_mask = memory_mask.expand(src_batch, tgt_len, src_len)
        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1).expand(src_batch, tgt_len, src_len)
        emb = self.embeddings(input_ids)
        output = self.pos_emb(emb, step)
        assert emb.dim() == 3
        if state.cache is None:
            saved_inputs = []
        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            (output, all_input) = self.transformer_layers[i](output, memory_bank, src_pad_mask, tgt_pad_mask, previous_input=prev_layer_input, layer_cache=state.cache['layer_{}'.format(i)] if state.cache is not None else None, step=step)
            if state.cache is None:
                saved_inputs.append(all_input)
        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)
        output = self.layer_norm(output)
        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)
        return (output, state)

    def init_decoder_state(self, src, memory_bank, with_cache=False):
        if False:
            print('Hello World!')
        'Init decoder state'
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state

class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        if False:
            for i in range(10):
                print('nop')
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        if False:
            for i in range(10):
                print('nop')
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]
        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        if False:
            i = 10
            return i + 15
        return self.pe[:, :emb.size(1)]

class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        if False:
            return 10
        super().__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.context_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-06)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-06)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, previous_input=None, layer_cache=None, step=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`\n            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`\n            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`\n            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`\n\n        Returns:\n            (`FloatTensor`, `FloatTensor`, `FloatTensor`):\n\n            * output `[batch_size x 1 x model_dim]`\n            * attn `[batch_size x 1 x src_len]`\n            * all_input `[batch_size x current_step x model_dim]`\n\n        '
        dec_mask = torch.gt(tgt_pad_mask + self.mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None
        query = self.self_attn(all_input, all_input, input_norm, mask=dec_mask, layer_cache=layer_cache, type='self')
        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)
        mid = self.context_attn(memory_bank, memory_bank, query_norm, mask=src_pad_mask, layer_cache=layer_cache, type='context')
        output = self.feed_forward(self.drop(mid) + query)
        return (output, all_input)

    def _get_attn_subsequent_mask(self, size):
        if False:
            while True:
                i = 10
        '\n        Get an attention mask to avoid using the subsequent info.\n\n        Args:\n            size: int\n\n        Returns:\n            (`LongTensor`):\n\n            * subsequent_mask `[1 x size x size]`\n        '
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        if False:
            for i in range(10):
                print('nop')
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super().__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if self.use_final_linear:
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None, layer_cache=None, type=None, predefined_graph_1=None):
        if False:
            while True:
                i = 10
        '\n        Compute the context vector and the attention vectors.\n\n        Args:\n           key (`FloatTensor`): set of `key_len`\n                key vectors `[batch, key_len, dim]`\n           value (`FloatTensor`): set of `key_len`\n                value vectors `[batch, key_len, dim]`\n           query (`FloatTensor`): set of `query_len`\n                 query vectors  `[batch, query_len, dim]`\n           mask: binary mask indicating which keys have\n                 non-zero attention `[batch, query_len, key_len]`\n        Returns:\n           (`FloatTensor`, `FloatTensor`) :\n\n           * output context vectors `[batch, query_len, dim]`\n           * one of the attention vectors `[batch, query_len, key_len]`\n        '
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            if False:
                i = 10
                return i + 15
            'projection'
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            if False:
                for i in range(10):
                    print('nop')
            'compute context'
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)
        if layer_cache is not None:
            if type == 'self':
                (query, key, value) = (self.linear_query(query), self.linear_keys(query), self.linear_values(query))
                key = shape(key)
                value = shape(value)
                if layer_cache is not None:
                    device = key.device
                    if layer_cache['self_keys'] is not None:
                        key = torch.cat((layer_cache['self_keys'].to(device), key), dim=2)
                    if layer_cache['self_values'] is not None:
                        value = torch.cat((layer_cache['self_values'].to(device), value), dim=2)
                    layer_cache['self_keys'] = key
                    layer_cache['self_values'] = value
            elif type == 'context':
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache['memory_keys'] is None:
                        (key, value) = (self.linear_keys(key), self.linear_values(value))
                        key = shape(key)
                        value = shape(value)
                    else:
                        (key, value) = (layer_cache['memory_keys'], layer_cache['memory_values'])
                    layer_cache['memory_keys'] = key
                    layer_cache['memory_values'] = value
                else:
                    (key, value) = (self.linear_keys(key), self.linear_values(value))
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)
        query = shape(query)
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e+18)
        attn = self.softmax(scores)
        if predefined_graph_1 is not None:
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (torch.sum(attn_masked, 2).unsqueeze(2) + 1e-09)
            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)
        drop_attn = self.dropout(attn)
        if self.use_final_linear:
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context

class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """

    def detach(self):
        if False:
            i = 10
            return i + 15
        'Need to document this'
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        if False:
            print('Hello World!')
        'Need to document this'
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size, sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size, br // beam_size, sizes[2], sizes[3])[:, :, idx]
            sent_states.data.copy_(sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

class TransformerDecoderState(DecoderState):
    """Transformer Decoder state base class"""

    def __init__(self, src):
        if False:
            print('Hello World!')
        '\n        Args:\n            src (FloatTensor): a sequence of source words tensors\n                    with optional feature tensors, of size (len x batch).\n        '
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        if False:
            return 10
        '\n        Contains attributes that need to be updated in self.beam_update().\n        '
        if self.previous_input is not None and self.previous_layer_inputs is not None:
            return (self.previous_input, self.previous_layer_inputs, self.src)
        else:
            return (self.src,)

    def detach(self):
        if False:
            return 10
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        if False:
            print('Hello World!')
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        if False:
            print('Hello World!')
        self.cache = {}
        for l in range(num_layers):
            layer_cache = {'memory_keys': None, 'memory_values': None}
            layer_cache['self_keys'] = None
            layer_cache['self_values'] = None
            self.cache['layer_{}'.format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        if False:
            while True:
                i = 10
        'Repeat beam_size times along batch dimension.'
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        if False:
            for i in range(10):
                print('nop')

        def _recursive_map(struct, batch_dim=0):
            if False:
                print('Hello World!')
            for (k, v) in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)
        self.src = fn(self.src, 0)
        if self.cache is not None:
            _recursive_map(self.cache)

def gelu(x):
    if False:
        print('Hello World!')
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    """A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        if False:
            print('Hello World!')
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

def build_predictor(args, tokenizer, symbols, model, logger=None):
    if False:
        for i in range(10):
            print('nop')
    scorer = GNMTGlobalScorer(args.alpha, length_penalty='wu')
    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator

class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha, length_penalty):
        if False:
            while True:
                i = 10
        self.alpha = alpha
        penalty_builder = PenaltyBuilder(length_penalty)
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Rescores a prediction based on penalty functions\n        '
        normalized_probs = self.length_penalty(beam, logprobs, self.alpha)
        return normalized_probs

class PenaltyBuilder(object):
    """
    Returns the Length and Coverage Penalty function for Beam Search.

    Args:
        length_pen (str): option name of length pen
        cov_pen (str): option name of cov pen
    """

    def __init__(self, length_pen):
        if False:
            i = 10
            return i + 15
        self.length_pen = length_pen

    def length_penalty(self):
        if False:
            for i in range(10):
                print('nop')
        if self.length_pen == 'wu':
            return self.length_wu
        elif self.length_pen == 'avg':
            return self.length_average
        else:
            return self.length_none
    '\n    Below are all the different penalty terms implemented so far\n    '

    def length_wu(self, beam, logprobs, alpha=0.0):
        if False:
            i = 10
            return i + 15
        '\n        NMT length re-ranking score from\n        "Google\'s Neural Machine Translation System" :cite:`wu2016google`.\n        '
        modifier = (5 + len(beam.next_ys)) ** alpha / (5 + 1) ** alpha
        return logprobs / modifier

    def length_average(self, beam, logprobs, alpha=0.0):
        if False:
            return 10
        '\n        Returns the average probability of tokens in a sequence.\n        '
        return logprobs / len(beam.next_ys)

    def length_none(self, beam, logprobs, alpha=0.0, beta=0.0):
        if False:
            return 10
        '\n        Returns unmodified scores.\n        '
        return logprobs

class Translator(object):
    """
    Uses a model to translate a batch of sentences.

    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self, args, model, vocab, symbols, global_scorer=None, logger=None):
        if False:
            return 10
        self.logger = logger
        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']
        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

    def translate(self, batch, step, attn_debug=False):
        if False:
            print('Hello World!')
        'Generates summaries from one batch of data.'
        self.model.eval()
        with torch.no_grad():
            batch_data = self.translate_batch(batch)
            translations = self.from_batch(batch_data)
        return translations

    def translate_batch(self, batch, fast=False):
        if False:
            while True:
                i = 10
        '\n        Translate a batch of sentences.\n\n        Mostly a wrapper around :obj:`Beam`.\n\n        Args:\n           batch (:obj:`Batch`): a batch from a dataset object\n           fast (bool): enables fast beam search (may not support all features)\n        '
        with torch.no_grad():
            return self._fast_translate_batch(batch, self.max_length, min_length=self.min_length)

    def _fast_translate_batch(self, batch, max_length, min_length=0):
        if False:
            while True:
                i = 10
        'Beam Search using the encoder inputs contained in `batch`.'
        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src
        src_features = self.model.bert(src, segs, mask_src)
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features.device
        dec_states.map_batch_fn(lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features, beam_size, dim=0)
        batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device)
        alive_seq = torch.full([batch_size * beam_size, 1], self.start_token, dtype=torch.long, device=device)
        topk_log_probs = torch.tensor([0.0] + [float('-inf')] * (beam_size - 1), device=device).repeat(batch_size)
        hypotheses = [[] for _ in range(batch_size)]
        results = {}
        results['predictions'] = [[] for _ in range(batch_size)]
        results['scores'] = [[] for _ in range(batch_size)]
        results['gold_score'] = [0] * batch_size
        results['batch'] = batch
        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)
            decoder_input = decoder_input.transpose(0, 1)
            (dec_out, dec_states) = self.model.decoder(decoder_input, src_features, dec_states, step=step)
            log_probs = self.generator(dec_out.transpose(0, 1).squeeze(0))
            vocab_size = log_probs.size(-1)
            if step < min_length:
                log_probs[:, self.end_token] = -1e+20
            log_probs += topk_log_probs.view(-1).unsqueeze(1)
            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores = log_probs / length_penalty
            if self.args.block_trigram:
                cur_len = alive_seq.size(1)
                if cur_len > 3:
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = ' '.join(words).replace(' ##', '').split()
                        if len(words) <= 3:
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -1e+21
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            (topk_scores, topk_ids) = curr_scores.topk(beam_size, dim=-1)
            topk_log_probs = topk_scores * length_penalty
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)
            batch_index = topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1)
            select_indices = batch_index.view(-1)
            alive_seq = torch.cat([alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1)
            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            end_condition = is_finished[:, 0].eq(1)
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    for j in finished_hyp:
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))
                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        (score, pred) = best_hyp[0]
                        results['scores'][b].append(score)
                        results['predictions'][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                if len(non_finished) == 0:
                    break
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(lambda state, dim: state.index_select(dim, select_indices))
        return results

    def from_batch(self, translation_batch):
        if False:
            return 10
        batch = translation_batch['batch']
        assert len(translation_batch['gold_score']) == len(translation_batch['predictions'])
        batch_size = batch.batch_size
        (preds, _, _, tgt_str, src) = (translation_batch['predictions'], translation_batch['scores'], translation_batch['gold_score'], batch.tgt_str, batch.src)
        translations = []
        for b in range(batch_size):
            pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
            pred_sents = ' '.join(pred_sents).replace(' ##', '')
            gold_sent = ' '.join(tgt_str[b].split())
            raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = ' '.join(raw_src)
            translation = (pred_sents, gold_sent, raw_src)
            translations.append(translation)
        return translations

def tile(x, count, dim=0):
    if False:
        while True:
            i = 10
    '\n    Tiles x on dimension dim count times.\n    '
    perm = list(range(len(x.size())))
    if dim != 0:
        (perm[0], perm[dim]) = (perm[dim], perm[0])
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1).transpose(0, 1).repeat(count, 1).transpose(0, 1).contiguous().view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

class BertSumOptimizer(object):
    """Specific optimizer for BertSum.

    As described in [1], the authors fine-tune BertSum for abstractive
    summarization using two Adam Optimizers with different warm-up steps and
    learning rate. They also use a custom learning rate scheduler.

    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    """

    def __init__(self, model, lr, warmup_steps, beta_1=0.99, beta_2=0.999, eps=1e-08):
        if False:
            for i in range(10):
                print('nop')
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.optimizers = {'encoder': torch.optim.Adam(model.encoder.parameters(), lr=lr['encoder'], betas=(beta_1, beta_2), eps=eps), 'decoder': torch.optim.Adam(model.decoder.parameters(), lr=lr['decoder'], betas=(beta_1, beta_2), eps=eps)}
        self._step = 0
        self.current_learning_rates = {}

    def _update_rate(self, stack):
        if False:
            i = 10
            return i + 15
        return self.lr[stack] * min(self._step ** (-0.5), self._step * self.warmup_steps[stack] ** (-1.5))

    def zero_grad(self):
        if False:
            return 10
        self.optimizer_decoder.zero_grad()
        self.optimizer_encoder.zero_grad()

    def step(self):
        if False:
            return 10
        self._step += 1
        for (stack, optimizer) in self.optimizers.items():
            new_rate = self._update_rate(stack)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_rate
            optimizer.step()
            self.current_learning_rates[stack] = new_rate