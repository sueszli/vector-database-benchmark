"""Adversarial losses for text models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('perturb_norm_length', 5.0, 'Norm length of adversarial perturbation to be optimized with validation. 5.0 is optimal on IMDB with virtual adversarial training. ')
flags.DEFINE_integer('num_power_iteration', 1, 'The number of power iteration')
flags.DEFINE_float('small_constant_for_finite_diff', 0.1, 'Small constant for finite difference method')
flags.DEFINE_string('adv_training_method', None, 'The flag which specifies training method. ""    : non-adversarial training (e.g. for running the         semi-supervised sequence learning model) "rp"  : random perturbation training "at"  : adversarial training "vat" : virtual adversarial training "atvat" : at + vat ')
flags.DEFINE_float('adv_reg_coeff', 1.0, 'Regularization coefficient of adversarial loss.')

def random_perturbation_loss(embedded, length, loss_fn):
    if False:
        return 10
    'Adds noise to embeddings and recomputes classification loss.'
    noise = tf.random_normal(shape=tf.shape(embedded))
    perturb = _scale_l2(_mask_by_length(noise, length), FLAGS.perturb_norm_length)
    return loss_fn(embedded + perturb)

def adversarial_loss(embedded, loss, loss_fn):
    if False:
        while True:
            i = 10
    'Adds gradient to embedding and recomputes classification loss.'
    (grad,) = tf.gradients(loss, embedded, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    grad = tf.stop_gradient(grad)
    perturb = _scale_l2(grad, FLAGS.perturb_norm_length)
    return loss_fn(embedded + perturb)

def virtual_adversarial_loss(logits, embedded, inputs, logits_from_embedding_fn):
    if False:
        print('Hello World!')
    'Virtual adversarial loss.\n\n  Computes virtual adversarial perturbation by finite difference method and\n  power iteration, adds it to the embedding, and computes the KL divergence\n  between the new logits and the original logits.\n\n  Args:\n    logits: 3-D float Tensor, [batch_size, num_timesteps, m], where m=1 if\n      num_classes=2, otherwise m=num_classes.\n    embedded: 3-D float Tensor, [batch_size, num_timesteps, embedding_dim].\n    inputs: VatxtInput.\n    logits_from_embedding_fn: callable that takes embeddings and returns\n      classifier logits.\n\n  Returns:\n    kl: float scalar.\n  '
    logits = tf.stop_gradient(logits)
    weights = inputs.eos_weights
    assert weights is not None
    if FLAGS.single_label:
        indices = tf.stack([tf.range(FLAGS.batch_size), inputs.length - 1], 1)
        weights = tf.expand_dims(tf.gather_nd(inputs.eos_weights, indices), 1)
    d = tf.random_normal(shape=tf.shape(embedded))
    for _ in xrange(FLAGS.num_power_iteration):
        d = _scale_l2(_mask_by_length(d, inputs.length), FLAGS.small_constant_for_finite_diff)
        d_logits = logits_from_embedding_fn(embedded + d)
        kl = _kl_divergence_with_logits(logits, d_logits, weights)
        (d,) = tf.gradients(kl, d, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        d = tf.stop_gradient(d)
    perturb = _scale_l2(d, FLAGS.perturb_norm_length)
    vadv_logits = logits_from_embedding_fn(embedded + perturb)
    return _kl_divergence_with_logits(logits, vadv_logits, weights)

def random_perturbation_loss_bidir(embedded, length, loss_fn):
    if False:
        return 10
    'Adds noise to embeddings and recomputes classification loss.'
    noise = [tf.random_normal(shape=tf.shape(emb)) for emb in embedded]
    masked = [_mask_by_length(n, length) for n in noise]
    scaled = [_scale_l2(m, FLAGS.perturb_norm_length) for m in masked]
    return loss_fn([e + s for (e, s) in zip(embedded, scaled)])

def adversarial_loss_bidir(embedded, loss, loss_fn):
    if False:
        i = 10
        return i + 15
    'Adds gradient to embeddings and recomputes classification loss.'
    grads = tf.gradients(loss, embedded, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    adv_exs = [emb + _scale_l2(tf.stop_gradient(g), FLAGS.perturb_norm_length) for (emb, g) in zip(embedded, grads)]
    return loss_fn(adv_exs)

def virtual_adversarial_loss_bidir(logits, embedded, inputs, logits_from_embedding_fn):
    if False:
        print('Hello World!')
    'Virtual adversarial loss for bidirectional models.'
    logits = tf.stop_gradient(logits)
    (f_inputs, _) = inputs
    weights = f_inputs.eos_weights
    if FLAGS.single_label:
        indices = tf.stack([tf.range(FLAGS.batch_size), f_inputs.length - 1], 1)
        weights = tf.expand_dims(tf.gather_nd(f_inputs.eos_weights, indices), 1)
    assert weights is not None
    perturbs = [_mask_by_length(tf.random_normal(shape=tf.shape(emb)), f_inputs.length) for emb in embedded]
    for _ in xrange(FLAGS.num_power_iteration):
        perturbs = [_scale_l2(d, FLAGS.small_constant_for_finite_diff) for d in perturbs]
        d_logits = logits_from_embedding_fn([emb + d for (emb, d) in zip(embedded, perturbs)])
        kl = _kl_divergence_with_logits(logits, d_logits, weights)
        perturbs = tf.gradients(kl, perturbs, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        perturbs = [tf.stop_gradient(d) for d in perturbs]
    perturbs = [_scale_l2(d, FLAGS.perturb_norm_length) for d in perturbs]
    vadv_logits = logits_from_embedding_fn([emb + d for (emb, d) in zip(embedded, perturbs)])
    return _kl_divergence_with_logits(logits, vadv_logits, weights)

def _mask_by_length(t, length):
    if False:
        while True:
            i = 10
    'Mask t, 3-D [batch, time, dim], by length, 1-D [batch,].'
    maxlen = t.get_shape().as_list()[1]
    mask = tf.sequence_mask(length - 1, maxlen=maxlen)
    mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
    return t * mask

def _scale_l2(x, norm_length):
    if False:
        return 10
    alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-06)
    x_unit = x / l2_norm
    return norm_length * x_unit

def _kl_divergence_with_logits(q_logits, p_logits, weights):
    if False:
        while True:
            i = 10
    'Returns weighted KL divergence between distributions q and p.\n\n  Args:\n    q_logits: logits for 1st argument of KL divergence shape\n              [batch_size, num_timesteps, num_classes] if num_classes > 2, and\n              [batch_size, num_timesteps] if num_classes == 2.\n    p_logits: logits for 2nd argument of KL divergence with same shape q_logits.\n    weights: 1-D float tensor with shape [batch_size, num_timesteps].\n             Elements should be 1.0 only on end of sequences\n\n  Returns:\n    KL: float scalar.\n  '
    if FLAGS.num_classes == 2:
        q = tf.nn.sigmoid(q_logits)
        kl = -tf.nn.sigmoid_cross_entropy_with_logits(logits=q_logits, labels=q) + tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logits, labels=q)
        kl = tf.squeeze(kl, 2)
    else:
        q = tf.nn.softmax(q_logits)
        kl = tf.reduce_sum(q * (tf.nn.log_softmax(q_logits) - tf.nn.log_softmax(p_logits)), -1)
    num_labels = tf.reduce_sum(weights)
    num_labels = tf.where(tf.equal(num_labels, 0.0), 1.0, num_labels)
    kl.get_shape().assert_has_rank(2)
    weights.get_shape().assert_has_rank(2)
    loss = tf.identity(tf.reduce_sum(weights * kl) / num_labels, name='kl')
    return loss