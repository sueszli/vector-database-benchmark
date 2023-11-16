"""
 A TF 2.0 Adaptive Softmax for Transformer XL model.
"""
import tensorflow as tf
from ...tf_utils import shape_list

class TFAdaptiveSoftmaxMask(tf.keras.layers.Layer):

    def __init__(self, vocab_size, d_embed, d_proj, cutoffs, div_val=1, keep_order=False, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.cutoffs = cutoffs + [vocab_size]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        self.keep_order = keep_order
        self.out_layers = []
        self.out_projs = []

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        if self.n_clusters > 0:
            self.cluster_weight = self.add_weight(shape=(self.n_clusters, self.d_embed), initializer='zeros', trainable=True, name='cluster_weight')
            self.cluster_bias = self.add_weight(shape=(self.n_clusters,), initializer='zeros', trainable=True, name='cluster_bias')
        if self.div_val == 1:
            for i in range(len(self.cutoffs)):
                if self.d_proj != self.d_embed:
                    weight = self.add_weight(shape=(self.d_embed, self.d_proj), initializer='zeros', trainable=True, name=f'out_projs_._{i}')
                    self.out_projs.append(weight)
                else:
                    self.out_projs.append(None)
                weight = self.add_weight(shape=(self.vocab_size, self.d_embed), initializer='zeros', trainable=True, name=f'out_layers_._{i}_._weight')
                bias = self.add_weight(shape=(self.vocab_size,), initializer='zeros', trainable=True, name=f'out_layers_._{i}_._bias')
                self.out_layers.append((weight, bias))
        else:
            for i in range(len(self.cutoffs)):
                (l_idx, r_idx) = (self.cutoff_ends[i], self.cutoff_ends[i + 1])
                d_emb_i = self.d_embed // self.div_val ** i
                weight = self.add_weight(shape=(d_emb_i, self.d_proj), initializer='zeros', trainable=True, name=f'out_projs_._{i}')
                self.out_projs.append(weight)
                weight = self.add_weight(shape=(r_idx - l_idx, d_emb_i), initializer='zeros', trainable=True, name=f'out_layers_._{i}_._weight')
                bias = self.add_weight(shape=(r_idx - l_idx,), initializer='zeros', trainable=True, name=f'out_layers_._{i}_._bias')
                self.out_layers.append((weight, bias))
        super().build(input_shape)

    @staticmethod
    def _logit(x, W, b, proj=None):
        if False:
            for i in range(10):
                print('nop')
        y = x
        if proj is not None:
            y = tf.einsum('ibd,ed->ibe', y, proj)
        return tf.einsum('ibd,nd->ibn', y, W) + b

    @staticmethod
    def _gather_logprob(logprob, target):
        if False:
            i = 10
            return i + 15
        lp_size = shape_list(logprob)
        r = tf.range(lp_size[0], dtype=target.dtype)
        idx = tf.stack([r, target], 1)
        return tf.gather_nd(logprob, idx)

    def call(self, hidden, target, return_mean=True, training=False):
        if False:
            i = 10
            return i + 15
        head_logprob = 0
        if self.n_clusters == 0:
            output = self._logit(hidden, self.out_layers[0][0], self.out_layers[0][1], self.out_projs[0])
            if target is not None:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output)
            out = tf.nn.log_softmax(output, axis=-1)
        else:
            hidden_sizes = shape_list(hidden)
            out = []
            loss = tf.zeros(hidden_sizes[:2])
            for i in range(len(self.cutoffs)):
                (l_idx, r_idx) = (self.cutoff_ends[i], self.cutoff_ends[i + 1])
                if target is not None:
                    mask = (target >= l_idx) & (target < r_idx)
                    mask_idx = tf.where(mask)
                    cur_target = tf.boolean_mask(target, mask) - l_idx
                if self.div_val == 1:
                    cur_W = self.out_layers[0][0][l_idx:r_idx]
                    cur_b = self.out_layers[0][1][l_idx:r_idx]
                else:
                    cur_W = self.out_layers[i][0]
                    cur_b = self.out_layers[i][1]
                if i == 0:
                    cur_W = tf.concat([cur_W, self.cluster_weight], 0)
                    cur_b = tf.concat([cur_b, self.cluster_bias], 0)
                    head_logit = self._logit(hidden, cur_W, cur_b, self.out_projs[0])
                    head_logprob = tf.nn.log_softmax(head_logit)
                    out.append(head_logprob[..., :self.cutoffs[0]])
                    if target is not None:
                        cur_head_logprob = tf.boolean_mask(head_logprob, mask)
                        cur_logprob = self._gather_logprob(cur_head_logprob, cur_target)
                else:
                    tail_logit = self._logit(hidden, cur_W, cur_b, self.out_projs[i])
                    tail_logprob = tf.nn.log_softmax(tail_logit)
                    cluster_prob_idx = self.cutoffs[0] + i - 1
                    logprob_i = head_logprob[..., cluster_prob_idx, None] + tail_logprob
                    out.append(logprob_i)
                    if target is not None:
                        cur_head_logprob = tf.boolean_mask(head_logprob, mask)
                        cur_tail_logprob = tf.boolean_mask(tail_logprob, mask)
                        cur_logprob = self._gather_logprob(cur_tail_logprob, cur_target)
                        cur_logprob += cur_head_logprob[:, self.cutoff_ends[1] + i - 1]
                if target is not None:
                    loss += tf.scatter_nd(mask_idx, -cur_logprob, shape_list(loss))
            out = tf.concat(out, axis=-1)
        if target is not None:
            if return_mean:
                loss = tf.reduce_mean(loss)
            self.add_loss(loss)
            self.add_metric(loss, name=self.name, aggregation='mean' if return_mean else '')
        return out