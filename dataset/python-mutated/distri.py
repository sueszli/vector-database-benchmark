"""
functions for sampling from hidden state
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from .util import fc

class Pd:
    """
    A particular probability distribution
    """

    def flatparam(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def mode(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def neglogp(self, x):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def kl(self, other):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def entropy(self):
        if False:
            return 10
        raise NotImplementedError

    def sample(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def logp(self, x):
        if False:
            while True:
                i = 10
        return -self.neglogp(x)

    def get_shape(self):
        if False:
            return 10
        return self.flatparam().shape

    @property
    def shape(self):
        if False:
            return 10
        return self.get_shape()

    def __getitem__(self, idx):
        if False:
            return 10
        return self.__class__(self.flatparam()[idx])

class PdType:
    """
    Parametrized family of probability distributions
    """

    def pdclass(self):
        if False:
            return 10
        raise NotImplementedError

    def pdfromflat(self, flat, mask, nsteps, size, is_act_model):
        if False:
            for i in range(10):
                print('nop')
        return self.pdclass()(flat, mask, nsteps, size, is_act_model)

    def pdfromlatent(self, latent_vector, init_scale, init_bias):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def param_shape(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def sample_shape(self):
        if False:
            return 10
        raise NotImplementedError

    def sample_dtype(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        if False:
            i = 10
            return i + 15
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        if False:
            return 10
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)

class CategoricalPd(Pd):
    """
    Categorical probability distribution
    """

    def __init__(self, logits, mask_npinf, nsteps, size, is_act_model):
        if False:
            return 10
        self.logits = logits
        self.mask_npinf = mask_npinf
        self.nsteps = nsteps
        self.size = size
        self.is_act_model = is_act_model

    def flatparam(self):
        if False:
            for i in range(10):
                print('nop')
        return self.logits

    def mode(self):
        if False:
            while True:
                i = 10
        return tf.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        if False:
            print('Hello World!')
        return tf.nn.softmax(self.logits)

    def neglogp(self, x):
        if False:
            print('Hello World!')
        "\n        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)\n        Note: we can't use sparse_softmax_cross_entropy_with_logits because\n              the implementation does not allow second-order derivatives...\n        "
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            x_shape_list = x.shape.as_list()
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            for (xs, ls) in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)
            x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        else:
            assert x.shape.as_list() == self.logits.shape.as_list()
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=x)

    def kl(self, other):
        if False:
            for i in range(10):
                print('nop')
        'kl'
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        if False:
            print('Hello World!')
        'compute entropy'
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        if False:
            while True:
                i = 10
        'sample from logits'
        if not self.is_act_model:
            re_res = tf.reshape(self.logits, [-1, self.nsteps, self.size])
            masked_res = tf.math.add(re_res, self.mask_npinf)
            re_masked_res = tf.reshape(masked_res, [-1, self.size])
            u = tf.random_uniform(tf.shape(re_masked_res), dtype=self.logits.dtype)
            return tf.argmax(re_masked_res - tf.log(-1 * tf.log(u)), axis=-1)
        else:
            u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
            return tf.argmax(self.logits - tf.log(-1 * tf.log(u)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        if False:
            while True:
                i = 10
        return cls(flat)

class CategoricalPdType(PdType):
    """
    To create CategoricalPd
    """

    def __init__(self, ncat, nsteps, np_mask, is_act_model):
        if False:
            for i in range(10):
                print('nop')
        self.ncat = ncat
        self.nsteps = nsteps
        self.np_mask = np_mask
        self.is_act_model = is_act_model

    def pdclass(self):
        if False:
            print('Hello World!')
        return CategoricalPd

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        if False:
            while True:
                i = 10
        'add fc and create CategoricalPd'
        (pdparam, mask, mask_npinf) = _matching_fc(latent_vector, 'pi', self.ncat, self.nsteps, init_scale=init_scale, init_bias=init_bias, np_mask=self.np_mask, is_act_model=self.is_act_model)
        return (self.pdfromflat(pdparam, mask_npinf, self.nsteps, self.ncat, self.is_act_model), pdparam, mask, mask_npinf)

    def param_shape(self):
        if False:
            print('Hello World!')
        return [self.ncat]

    def sample_shape(self):
        if False:
            for i in range(10):
                print('nop')
        return []

    def sample_dtype(self):
        if False:
            print('Hello World!')
        return tf.int32

def _matching_fc(tensor, name, size, nsteps, init_scale, init_bias, np_mask, is_act_model):
    if False:
        while True:
            i = 10
    '\n    Add fc op, and add mask op when not in action mode\n    '
    if tensor.shape[-1] == size:
        assert False
        return tensor
    else:
        mask = tf.get_variable('act_mask', dtype=tf.float32, initializer=np_mask[0], trainable=False)
        mask_npinf = tf.get_variable('act_mask_npinf', dtype=tf.float32, initializer=np_mask[1], trainable=False)
        res = fc(tensor, name, size, init_scale=init_scale, init_bias=init_bias)
        if not is_act_model:
            re_res = tf.reshape(res, [-1, nsteps, size])
            masked_res = tf.math.multiply(re_res, mask)
            re_masked_res = tf.reshape(masked_res, [-1, size])
            return (re_masked_res, mask, mask_npinf)
        else:
            return (res, mask, mask_npinf)