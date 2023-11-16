import numpy as np
from legacy_test.test_dist_base import TestParallelDyGraphRunnerBase, runtime_main
import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base.dygraph import to_variable
from paddle.nn import Layer
from paddle.optimizer.lr import NoamDecay
'\nNote(chenweihang): To compare loss of single-card and multi-card\n    in our dist test framework, two parameters need to be adjusted:\n  1. set the dropout rate to 0.\n  2. set the weights for Transformer.forward to constant.\n  3. to test sparse optimize, set weight_sharing to be False\n'

class TrainTaskConfig:
    """
    TrainTaskConfig
    """
    pass_num = 20
    batch_size = 32
    learning_rate = 2.0
    beta1 = 0.9
    beta2 = 0.997
    eps = 1e-09
    warmup_steps = 8000
    label_smooth_eps = 0.1

class ModelHyperParams:
    src_vocab_size = 10000
    trg_vocab_size = 10000
    bos_idx = 0
    eos_idx = 1
    unk_idx = 2
    max_length = 4
    d_model = 512
    d_inner_hid = 2048
    d_key = 64
    d_value = 64
    n_head = 8
    n_layer = 6
    prepostprocess_dropout = 0
    attention_dropout = 0
    relu_dropout = 0
    preprocess_cmd = 'n'
    postprocess_cmd = 'da'
    dropout_seed = None
    weight_sharing = False
batch_size = -1
seq_len = ModelHyperParams.max_length
input_descs = {'src_word': [(batch_size, seq_len, 1), 'int64', 2], 'src_pos': [(batch_size, seq_len, 1), 'int64'], 'src_slf_attn_bias': [(batch_size, ModelHyperParams.n_head, seq_len, seq_len), 'float32'], 'trg_word': [(batch_size, seq_len, 1), 'int64', 2], 'trg_pos': [(batch_size, seq_len, 1), 'int64'], 'trg_slf_attn_bias': [(batch_size, ModelHyperParams.n_head, seq_len, seq_len), 'float32'], 'trg_src_attn_bias': [(batch_size, ModelHyperParams.n_head, seq_len, seq_len), 'float32'], 'enc_output': [(batch_size, seq_len, ModelHyperParams.d_model), 'float32'], 'lbl_word': [(batch_size * seq_len, 1), 'int64'], 'lbl_weight': [(batch_size * seq_len, 1), 'float32'], 'init_score': [(batch_size, 1), 'float32', 2], 'init_idx': [(batch_size,), 'int32']}
word_emb_param_names = ('src_word_emb_table', 'trg_word_emb_table')
pos_enc_param_names = ('src_pos_enc_table', 'trg_pos_enc_table')
encoder_data_input_fields = ('src_word', 'src_pos', 'src_slf_attn_bias')
decoder_data_input_fields = ('trg_word', 'trg_pos', 'trg_slf_attn_bias', 'trg_src_attn_bias', 'enc_output')
label_data_input_fields = ('lbl_word', 'lbl_weight')
fast_decoder_data_input_fields = ('trg_word', 'trg_src_attn_bias')

def position_encoding_init(n_position, d_pos_vec):
    if False:
        print('Hello World!')
    '\n    Generate the initial values for the sinusoid position encoding table.\n    '
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = np.log(10000.0 / float(1)) / (num_timescales - 1)
    inv_timescales = np.exp(np.arange(num_timescales)) * -log_timescale_increment
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype('float32')
pos_inp1 = position_encoding_init(ModelHyperParams.max_length, ModelHyperParams.d_model)
pos_inp2 = position_encoding_init(ModelHyperParams.max_length, ModelHyperParams.d_model)

class PrePostProcessLayer(Layer):

    def __init__(self, d_model, process_cmd, shape_len=None):
        if False:
            print('Hello World!')
        super().__init__()
        for cmd in process_cmd:
            if cmd == 'n':
                self._layer_norm = paddle.nn.LayerNorm(normalized_shape=d_model, weight_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0)), bias_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)))

    def forward(self, prev_out, out, process_cmd, dropout_rate=0.0):
        if False:
            while True:
                i = 10
        for cmd in process_cmd:
            if cmd == 'a':
                out = out + prev_out if prev_out is not None else out
            elif cmd == 'n':
                out = self._layer_norm(out)
            elif cmd == 'd':
                if dropout_rate:
                    out = paddle.nn.functional.dropout(out, p=dropout_rate)
        return out

class PositionwiseFeedForwardLayer(Layer):

    def __init__(self, d_inner_hid, d_hid, dropout_rate):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._i2h = paddle.nn.Linear(d_hid, d_inner_hid)
        self._h2o = paddle.nn.Linear(d_inner_hid, d_hid)
        self._dropout_rate = dropout_rate

    def forward(self, x):
        if False:
            print('Hello World!')
        hidden = self._i2h(x)
        if self._dropout_rate:
            hidden = paddle.nn.functional.dropout(hidden, p=self._dropout_rate)
        out = self._h2o(hidden)
        return out

class MultiHeadAttentionLayer(Layer):

    def __init__(self, d_key, d_value, d_model, n_head=1, dropout_rate=0.0, cache=None, gather_idx=None, static_kv=False):
        if False:
            return 10
        super().__init__()
        self._n_head = n_head
        self._d_key = d_key
        self._d_value = d_value
        self._d_model = d_model
        self._dropout_rate = dropout_rate
        self._q_fc = paddle.nn.Linear(self._d_model, d_key * n_head, bias_attr=False)
        self._k_fc = paddle.nn.Linear(self._d_model, d_key * n_head, bias_attr=False)
        self._v_fc = paddle.nn.Linear(self._d_model, d_value * n_head, bias_attr=False)
        self._proj_fc = paddle.nn.Linear(d_value * n_head, self._d_model, bias_attr=False)

    def forward(self, queries, keys, values, attn_bias):
        if False:
            return 10
        keys = queries if keys is None else keys
        values = keys if values is None else values
        q = self._q_fc(queries)
        k = self._k_fc(keys)
        v = self._v_fc(values)
        reshaped_q = paddle.reshape(x=q, shape=[0, 0, self._n_head, self._d_key])
        transpose_q = paddle.transpose(x=reshaped_q, perm=[0, 2, 1, 3])
        reshaped_k = paddle.reshape(x=k, shape=[0, 0, self._n_head, self._d_key])
        transpose_k = paddle.transpose(x=reshaped_k, perm=[0, 2, 1, 3])
        reshaped_v = paddle.reshape(x=v, shape=[0, 0, self._n_head, self._d_value])
        transpose_v = paddle.transpose(x=reshaped_v, perm=[0, 2, 1, 3])
        product = paddle.matmul(x=transpose_q, y=transpose_k, transpose_y=True)
        product = paddle.scale(product, scale=self._d_model ** (-0.5))
        if attn_bias is not None:
            product += attn_bias
        weights = paddle.nn.functional.softmax(product)
        if self._dropout_rate:
            weights_droped = paddle.nn.functional.dropout(weights, p=self._dropout_rate)
            out = paddle.matmul(weights_droped, transpose_v)
        else:
            out = paddle.matmul(weights, transpose_v)
        if len(out.shape) != 4:
            raise ValueError('Input(x) should be a 4-D Tensor.')
        trans_x = paddle.transpose(out, perm=[0, 2, 1, 3])
        final_out = paddle.reshape(x=trans_x, shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]])
        proj_out = self._proj_fc(final_out)
        return proj_out

class EncoderSubLayer(Layer):

    def __init__(self, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd='n', postprocess_cmd='da'):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._preprocess_cmd = preprocess_cmd
        self._postprocess_cmd = postprocess_cmd
        self._prepostprocess_dropout = prepostprocess_dropout
        self._preprocess_layer = PrePostProcessLayer(d_model, self._preprocess_cmd, 3)
        self._multihead_attention_layer = MultiHeadAttentionLayer(d_key, d_value, d_model, n_head, attention_dropout)
        self._postprocess_layer = PrePostProcessLayer(d_model, self._postprocess_cmd, None)
        self._preprocess_layer2 = PrePostProcessLayer(d_model, self._preprocess_cmd, 3)
        self._positionwise_feed_forward = PositionwiseFeedForwardLayer(d_inner_hid, d_model, relu_dropout)
        self._postprocess_layer2 = PrePostProcessLayer(d_model, self._postprocess_cmd, None)

    def forward(self, enc_input, attn_bias):
        if False:
            i = 10
            return i + 15
        pre_process_multihead = self._preprocess_layer(None, enc_input, self._preprocess_cmd, self._prepostprocess_dropout)
        attn_output = self._multihead_attention_layer(pre_process_multihead, None, None, attn_bias)
        attn_output = self._postprocess_layer(enc_input, attn_output, self._postprocess_cmd, self._prepostprocess_dropout)
        pre_process2_output = self._preprocess_layer2(None, attn_output, self._preprocess_cmd, self._prepostprocess_dropout)
        ffd_output = self._positionwise_feed_forward(pre_process2_output)
        return self._postprocess_layer2(attn_output, ffd_output, self._postprocess_cmd, self._prepostprocess_dropout)

class EncoderLayer(Layer):

    def __init__(self, n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd='n', postprocess_cmd='da'):
        if False:
            print('Hello World!')
        super().__init__()
        self._preprocess_cmd = preprocess_cmd
        self._encoder_sublayers = []
        self._prepostprocess_dropout = prepostprocess_dropout
        self._n_layer = n_layer
        self._preprocess_layer = PrePostProcessLayer(d_model, self._preprocess_cmd, 3)
        for i in range(n_layer):
            self._encoder_sublayers.append(self.add_sublayer('esl_%d' % i, EncoderSubLayer(n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd)))

    def forward(self, enc_input, attn_bias):
        if False:
            for i in range(10):
                print('nop')
        for i in range(self._n_layer):
            enc_output = self._encoder_sublayers[i](enc_input, attn_bias)
            enc_input = enc_output
        return self._preprocess_layer(None, enc_output, self._preprocess_cmd, self._prepostprocess_dropout)

class PrepareEncoderDecoderLayer(Layer):

    def __init__(self, src_vocab_size, src_emb_dim, src_max_len, dropout_rate, is_sparse=False, word_emb_param_name=None, pos_enc_param_name=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._src_max_len = src_max_len
        self._src_emb_dim = src_emb_dim
        self._src_vocab_size = src_vocab_size
        self._dropout_rate = dropout_rate
        self._input_emb = paddle.nn.Embedding(src_vocab_size, src_emb_dim, sparse=is_sparse, weight_attr=base.ParamAttr(name=word_emb_param_name, initializer=paddle.nn.initializer.Normal(0.0, src_emb_dim ** (-0.5))))
        if pos_enc_param_name is pos_enc_param_names[0]:
            pos_inp = pos_inp1
        else:
            pos_inp = pos_inp2
        self._pos_emb = paddle.nn.Embedding(self._src_max_len, src_emb_dim, sparse=is_sparse, weight_attr=base.ParamAttr(name=pos_enc_param_name, initializer=paddle.nn.initializer.Assign(pos_inp), trainable=False))

    def forward(self, src_word, src_pos):
        if False:
            for i in range(10):
                print('nop')
        src_word_emb = self._input_emb(src_word)
        src_word_emb = paddle.scale(x=src_word_emb, scale=self._src_emb_dim ** 0.5)
        src_pos_emb = self._pos_emb(src_pos)
        src_pos_emb.stop_gradient = True
        enc_input = src_word_emb + src_pos_emb
        return paddle.nn.functional.dropout(enc_input, p=self._dropout_rate) if self._dropout_rate else enc_input

class WrapEncoderLayer(Layer):

    def __init__(self, src_vocab_size, max_length, n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd, weight_sharing, is_sparse=False):
        if False:
            while True:
                i = 10
        '\n        The wrapper assembles together all needed layers for the encoder.\n        '
        super().__init__()
        self._prepare_encoder_layer = PrepareEncoderDecoderLayer(src_vocab_size, d_model, max_length, prepostprocess_dropout, is_sparse=is_sparse, word_emb_param_name=word_emb_param_names[0], pos_enc_param_name=pos_enc_param_names[0])
        self._encoder = EncoderLayer(n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd)

    def forward(self, enc_inputs):
        if False:
            i = 10
            return i + 15
        (src_word, src_pos, src_slf_attn_bias) = enc_inputs
        enc_input = self._prepare_encoder_layer(src_word, src_pos)
        enc_output = self._encoder(enc_input, src_slf_attn_bias)
        return enc_output

class DecoderSubLayer(Layer):

    def __init__(self, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd, cache=None, gather_idx=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._postprocess_cmd = postprocess_cmd
        self._preprocess_cmd = preprocess_cmd
        self._prepostprcess_dropout = prepostprocess_dropout
        self._pre_process_layer = PrePostProcessLayer(d_model, preprocess_cmd, 3)
        self._multihead_attention_layer = MultiHeadAttentionLayer(d_key, d_value, d_model, n_head, attention_dropout, cache=cache, gather_idx=gather_idx)
        self._post_process_layer = PrePostProcessLayer(d_model, postprocess_cmd, None)
        self._pre_process_layer2 = PrePostProcessLayer(d_model, preprocess_cmd, 3)
        self._multihead_attention_layer2 = MultiHeadAttentionLayer(d_key, d_value, d_model, n_head, attention_dropout, cache=cache, gather_idx=gather_idx, static_kv=True)
        self._post_process_layer2 = PrePostProcessLayer(d_model, postprocess_cmd, None)
        self._pre_process_layer3 = PrePostProcessLayer(d_model, preprocess_cmd, 3)
        self._positionwise_feed_forward_layer = PositionwiseFeedForwardLayer(d_inner_hid, d_model, relu_dropout)
        self._post_process_layer3 = PrePostProcessLayer(d_model, postprocess_cmd, None)

    def forward(self, dec_input, enc_output, slf_attn_bias, dec_enc_attn_bias):
        if False:
            print('Hello World!')
        pre_process_rlt = self._pre_process_layer(None, dec_input, self._preprocess_cmd, self._prepostprcess_dropout)
        slf_attn_output = self._multihead_attention_layer(pre_process_rlt, None, None, slf_attn_bias)
        slf_attn_output_pp = self._post_process_layer(dec_input, slf_attn_output, self._postprocess_cmd, self._prepostprcess_dropout)
        pre_process_rlt2 = self._pre_process_layer2(None, slf_attn_output_pp, self._preprocess_cmd, self._prepostprcess_dropout)
        enc_attn_output_pp = self._multihead_attention_layer2(pre_process_rlt2, enc_output, enc_output, dec_enc_attn_bias)
        enc_attn_output = self._post_process_layer2(slf_attn_output_pp, enc_attn_output_pp, self._postprocess_cmd, self._prepostprcess_dropout)
        pre_process_rlt3 = self._pre_process_layer3(None, enc_attn_output, self._preprocess_cmd, self._prepostprcess_dropout)
        ffd_output = self._positionwise_feed_forward_layer(pre_process_rlt3)
        dec_output = self._post_process_layer3(enc_attn_output, ffd_output, self._postprocess_cmd, self._prepostprcess_dropout)
        return dec_output

class DecoderLayer(Layer):

    def __init__(self, n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd, caches=None, gather_idx=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self._pre_process_layer = PrePostProcessLayer(d_model, preprocess_cmd, 3)
        self._decoder_sub_layers = []
        self._n_layer = n_layer
        self._preprocess_cmd = preprocess_cmd
        self._prepostprocess_dropout = prepostprocess_dropout
        for i in range(n_layer):
            self._decoder_sub_layers.append(self.add_sublayer('dsl_%d' % i, DecoderSubLayer(n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd, cache=None if caches is None else caches[i], gather_idx=gather_idx)))

    def forward(self, dec_input, enc_output, dec_slf_attn_bias, dec_enc_attn_bias):
        if False:
            print('Hello World!')
        for i in range(self._n_layer):
            tmp_dec_output = self._decoder_sub_layers[i](dec_input, enc_output, dec_slf_attn_bias, dec_enc_attn_bias)
            dec_input = tmp_dec_output
        dec_output = self._pre_process_layer(None, tmp_dec_output, self._preprocess_cmd, self._prepostprocess_dropout)
        return dec_output

class WrapDecoderLayer(Layer):

    def __init__(self, trg_vocab_size, max_length, n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd, weight_sharing, caches=None, gather_idx=None, is_sparse=False):
        if False:
            print('Hello World!')
        '\n        The wrapper assembles together all needed layers for the encoder.\n        '
        super().__init__()
        self._prepare_decoder_layer = PrepareEncoderDecoderLayer(trg_vocab_size, d_model, max_length, prepostprocess_dropout, is_sparse=is_sparse, word_emb_param_name=word_emb_param_names[1], pos_enc_param_name=pos_enc_param_names[1])
        self._decoder_layer = DecoderLayer(n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd, caches=caches, gather_idx=gather_idx)
        self._weight_sharing = weight_sharing
        if not weight_sharing:
            self._fc = paddle.nn.Linear(d_model, trg_vocab_size, bias_attr=False)

    def forward(self, dec_inputs=None, enc_output=None):
        if False:
            return 10
        (trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias) = dec_inputs
        dec_input = self._prepare_decoder_layer(trg_word, trg_pos)
        dec_output = self._decoder_layer(dec_input, enc_output, trg_slf_attn_bias, trg_src_attn_bias)
        dec_output_reshape = paddle.reshape(dec_output, shape=[-1, dec_output.shape[-1]])
        if self._weight_sharing:
            predict = paddle.matmul(x=dec_output_reshape, y=self._prepare_decoder_layer._input_emb.weight, transpose_y=True)
        else:
            predict = self._fc(dec_output_reshape)
        if dec_inputs is None:
            predict_out = paddle.nn.functional.softmax(predict)
            return predict_out
        return predict

class TransFormer(Layer):

    def __init__(self, src_vocab_size, trg_vocab_size, max_length, n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd, weight_sharing, label_smooth_eps, use_py_reader=False, is_test=False, is_sparse=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._label_smooth_eps = label_smooth_eps
        self._trg_vocab_size = trg_vocab_size
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, 'Vocabularies in source and target should be same for weight sharing.'
        self._wrap_encoder_layer = WrapEncoderLayer(src_vocab_size, max_length, n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd, weight_sharing, is_sparse=is_sparse)
        self._wrap_decoder_layer = WrapDecoderLayer(trg_vocab_size, max_length, n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd, weight_sharing, is_sparse=is_sparse)
        if weight_sharing:
            self._wrap_decoder_layer._prepare_decoder_layer._input_emb.weight = self._wrap_encoder_layer._prepare_encoder_layer._input_emb.weight

    def forward(self, enc_inputs, dec_inputs, label, weights):
        if False:
            return 10
        enc_output = self._wrap_encoder_layer(enc_inputs)
        predict = self._wrap_decoder_layer(dec_inputs, enc_output)
        if self._label_smooth_eps:
            label_out = F.label_smooth(label=paddle.squeeze(paddle.nn.functional.one_hot(label, self._trg_vocab_size)), epsilon=self._label_smooth_eps)
        cost = paddle.nn.functional.softmax_with_cross_entropy(logits=predict, label=label_out, soft_label=True if self._label_smooth_eps else False)
        weighted_cost = cost * weights
        sum_cost = paddle.sum(weighted_cost)
        token_num = paddle.sum(weights)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num
        return (sum_cost, avg_cost, predict, token_num)
batch_num = 5

def fake_data_reader():
    if False:
        return 10

    def __reader__():
        if False:
            return 10
        iteration = TrainTaskConfig.batch_size * batch_num
        for _ in range(iteration):
            np.random.seed = 90
            src_word_np = np.arange(1, seq_len + 1).reshape([seq_len]).astype('int64')
            src_pos_np = np.random.randint(1, seq_len, size=seq_len, dtype='int64')
            src_slf_attn_bias_np = np.random.randn(ModelHyperParams.n_head, seq_len, seq_len).astype('float32')
            trg_word_np = np.arange(1, seq_len + 1).reshape([seq_len]).astype('int64')
            trg_pos_np = np.random.randint(1, seq_len, size=seq_len, dtype='int64')
            trg_slf_attn_bias_np = np.random.randn(ModelHyperParams.n_head, seq_len, seq_len).astype('float32')
            trg_src_attn_bias_np = np.random.randn(ModelHyperParams.n_head, seq_len, seq_len).astype('float32')
            lbl_word_np = np.random.randint(1, ModelHyperParams.src_vocab_size - 1, size=(seq_len, 1), dtype='int64')
            lbl_weight_np = np.ones((seq_len, 1)).astype('int64')
            data_inputs = [src_word_np, src_pos_np, src_slf_attn_bias_np, trg_word_np, trg_pos_np, trg_slf_attn_bias_np, trg_src_attn_bias_np, lbl_word_np, lbl_weight_np]
            yield data_inputs
    return __reader__

def np_to_variable(data):
    if False:
        print('Hello World!')
    batch_size = len(data)
    src_word_np = np.array([x[0] for x in data]).astype('int64')
    src_pos_np = np.array([x[1] for x in data]).astype('int64')
    src_slf_attn_bias_np = np.array([x[2] for x in data]).astype('float32')
    trg_word_np = np.array([x[3] for x in data]).astype('int64')
    trg_pos_np = np.array([x[4] for x in data]).astype('int64')
    trg_slf_attn_bias_np = np.array([x[5] for x in data]).astype('float32')
    trg_src_attn_bias_np = np.array([x[6] for x in data]).astype('float32')
    lbl_word_np = np.array([x[7] for x in data]).astype('int64')
    lbl_weight_np = np.array([x[8] for x in data]).astype('float32')
    lbl_word_np = lbl_word_np.reshape(batch_size * seq_len, 1)
    lbl_weight_np = lbl_weight_np.reshape(batch_size * seq_len, 1)
    data_inputs = [src_word_np, src_pos_np, src_slf_attn_bias_np, trg_word_np, trg_pos_np, trg_slf_attn_bias_np, trg_src_attn_bias_np, lbl_word_np, lbl_weight_np]
    var_inputs = []
    for (i, field) in enumerate(encoder_data_input_fields + decoder_data_input_fields[:-1] + label_data_input_fields):
        var_inputs.append(to_variable(data_inputs[i], name=field))
    enc_inputs = var_inputs[0:len(encoder_data_input_fields)]
    dec_inputs = var_inputs[len(encoder_data_input_fields):len(encoder_data_input_fields) + len(decoder_data_input_fields[:-1])]
    label = var_inputs[-2]
    weights = var_inputs[-1]
    return (enc_inputs, dec_inputs, label, weights)
naive_optimize = True

class TestTransformer(TestParallelDyGraphRunnerBase):

    def get_model(self):
        if False:
            i = 10
            return i + 15
        model = TransFormer(ModelHyperParams.src_vocab_size, ModelHyperParams.trg_vocab_size, ModelHyperParams.max_length + 1, ModelHyperParams.n_layer, ModelHyperParams.n_head, ModelHyperParams.d_key, ModelHyperParams.d_value, ModelHyperParams.d_model, ModelHyperParams.d_inner_hid, ModelHyperParams.prepostprocess_dropout, ModelHyperParams.attention_dropout, ModelHyperParams.relu_dropout, ModelHyperParams.preprocess_cmd, ModelHyperParams.postprocess_cmd, ModelHyperParams.weight_sharing, TrainTaskConfig.label_smooth_eps, is_sparse=True)
        train_reader = paddle.batch(fake_data_reader(), TrainTaskConfig.batch_size)
        if naive_optimize:
            optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
        else:
            optimizer = paddle.optimizer.Adam(learning_rate=NoamDecay(ModelHyperParams.d_model, TrainTaskConfig.warmup_steps, TrainTaskConfig.learning_rate), beta1=TrainTaskConfig.beta1, beta2=TrainTaskConfig.beta2, epsilon=TrainTaskConfig.eps, parameters=model.parameters())
        return (model, train_reader, optimizer)

    def run_one_loop(self, model, optimizer, batch):
        if False:
            print('Hello World!')
        (enc_inputs, dec_inputs, label, weights) = np_to_variable(batch)
        (dy_sum_cost, dy_avg_cost, dy_predict, dy_token_num) = model(enc_inputs, dec_inputs, label, weights)
        return dy_avg_cost
if __name__ == '__main__':
    runtime_main(TestTransformer)