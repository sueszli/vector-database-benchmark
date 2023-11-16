"""Defines NeuMF model for NCF framework.

Some abbreviations used in the code base:
NeuMF: Neural Matrix Factorization
NCF: Neural Collaborative Filtering
GMF: Generalized Matrix Factorization
MLP: Multi-Layer Perceptron

GMF applies a linear kernel to model the latent feature interactions, and MLP
uses a nonlinear kernel to learn the interaction function from data. NeuMF model
is a fused model of GMF and MLP to better model the complex user-item
interactions, and unifies the strengths of linearity of MF and non-linearity of
MLP for modeling the user-item latent structures.

In NeuMF model, it allows GMF and MLP to learn separate embeddings, and combine
the two models by concatenating their last hidden layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from six.moves import xrange
import tensorflow as tf
from official.recommendation import constants as rconst
from official.recommendation import movielens
from official.recommendation import ncf_common
from official.recommendation import stat_utils
from official.utils.logs import mlperf_helper

def sparse_to_dense_grads(grads_and_vars):
    if False:
        while True:
            i = 10
    'Convert sparse gradients to dense gradients.\n\n  All sparse gradients, which are represented as instances of tf.IndexedSlices,\n  are converted to dense Tensors. Dense gradients, which are represents as\n  Tensors, are unchanged.\n\n  The purpose of this conversion is that for small embeddings, which are used by\n  this model, applying dense gradients with the AdamOptimizer is faster than\n  applying sparse gradients.\n\n  Args\n    grads_and_vars: A list of (gradient, variable) tuples. Each gradient can\n      be a Tensor or an IndexedSlices. Tensors are unchanged, and IndexedSlices\n      are converted to dense Tensors.\n  Returns:\n    The same list of (gradient, variable) as `grads_and_vars`, except each\n    IndexedSlices gradient is converted to a Tensor.\n  '
    return [(tf.convert_to_tensor(g), v) for (g, v) in grads_and_vars]

def neumf_model_fn(features, labels, mode, params):
    if False:
        print('Hello World!')
    'Model Function for NeuMF estimator.'
    if params.get('use_seed'):
        tf.set_random_seed(stat_utils.random_int32())
    users = features[movielens.USER_COLUMN]
    items = features[movielens.ITEM_COLUMN]
    user_input = tf.keras.layers.Input(tensor=users)
    item_input = tf.keras.layers.Input(tensor=items)
    logits = construct_model(user_input, item_input, params).output
    softmax_logits = ncf_common.convert_to_softmax_logits(logits)
    if mode == tf.estimator.ModeKeys.EVAL:
        duplicate_mask = tf.cast(features[rconst.DUPLICATE_MASK], tf.float32)
        return _get_estimator_spec_with_metrics(logits, softmax_logits, duplicate_mask, params['num_neg'], params['match_mlperf'], use_tpu_spec=params['use_xla_for_gpu'])
    elif mode == tf.estimator.ModeKeys.TRAIN:
        labels = tf.cast(labels, tf.int32)
        valid_pt_mask = features[rconst.VALID_POINT_MASK]
        mlperf_helper.ncf_print(key=mlperf_helper.TAGS.OPT_NAME, value='adam')
        mlperf_helper.ncf_print(key=mlperf_helper.TAGS.OPT_LR, value=params['learning_rate'])
        mlperf_helper.ncf_print(key=mlperf_helper.TAGS.OPT_HP_ADAM_BETA1, value=params['beta1'])
        mlperf_helper.ncf_print(key=mlperf_helper.TAGS.OPT_HP_ADAM_BETA2, value=params['beta2'])
        mlperf_helper.ncf_print(key=mlperf_helper.TAGS.OPT_HP_ADAM_EPSILON, value=params['epsilon'])
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['learning_rate'], beta1=params['beta1'], beta2=params['beta2'], epsilon=params['epsilon'])
        if params['use_tpu']:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        mlperf_helper.ncf_print(key=mlperf_helper.TAGS.MODEL_HP_LOSS_FN, value=mlperf_helper.TAGS.BCE)
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=softmax_logits, weights=tf.cast(valid_pt_mask, tf.float32))
        tf.identity(loss, name='cross_entropy')
        global_step = tf.compat.v1.train.get_global_step()
        tvars = tf.compat.v1.trainable_variables()
        gradients = optimizer.compute_gradients(loss, tvars, colocate_gradients_with_ops=True)
        gradients = sparse_to_dense_grads(gradients)
        minimize_op = optimizer.apply_gradients(gradients, global_step=global_step, name='train')
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    else:
        raise NotImplementedError

def _strip_first_and_last_dimension(x, batch_size):
    if False:
        return 10
    return tf.reshape(x[0, :], (batch_size,))

def construct_model(user_input, item_input, params):
    if False:
        i = 10
        return i + 15
    'Initialize NeuMF model.\n\n  Args:\n    user_input: keras input layer for users\n    item_input: keras input layer for items\n    params: Dict of hyperparameters.\n  Raises:\n    ValueError: if the first model layer is not even.\n  Returns:\n    model:  a keras Model for computing the logits\n  '
    num_users = params['num_users']
    num_items = params['num_items']
    model_layers = params['model_layers']
    mf_regularization = params['mf_regularization']
    mlp_reg_layers = params['mlp_reg_layers']
    mf_dim = params['mf_dim']
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.MODEL_HP_MF_DIM, value=mf_dim)
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.MODEL_HP_MLP_LAYER_SIZES, value=model_layers)
    if model_layers[0] % 2 != 0:
        raise ValueError('The first layer size should be multiple of 2!')
    embedding_initializer = 'glorot_uniform'

    def mf_slice_fn(x):
        if False:
            print('Hello World!')
        x = tf.squeeze(x, [1])
        return x[:, :mf_dim]

    def mlp_slice_fn(x):
        if False:
            for i in range(10):
                print('nop')
        x = tf.squeeze(x, [1])
        return x[:, mf_dim:]
    embedding_user = tf.keras.layers.Embedding(num_users, mf_dim + model_layers[0] // 2, embeddings_initializer=embedding_initializer, embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization), input_length=1, name='embedding_user')(user_input)
    embedding_item = tf.keras.layers.Embedding(num_items, mf_dim + model_layers[0] // 2, embeddings_initializer=embedding_initializer, embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization), input_length=1, name='embedding_item')(item_input)
    mf_user_latent = tf.keras.layers.Lambda(mf_slice_fn, name='embedding_user_mf')(embedding_user)
    mf_item_latent = tf.keras.layers.Lambda(mf_slice_fn, name='embedding_item_mf')(embedding_item)
    mlp_user_latent = tf.keras.layers.Lambda(mlp_slice_fn, name='embedding_user_mlp')(embedding_user)
    mlp_item_latent = tf.keras.layers.Lambda(mlp_slice_fn, name='embedding_item_mlp')(embedding_item)
    mf_vector = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])
    mlp_vector = tf.keras.layers.concatenate([mlp_user_latent, mlp_item_latent])
    num_layer = len(model_layers)
    for layer in xrange(1, num_layer):
        model_layer = tf.keras.layers.Dense(model_layers[layer], kernel_regularizer=tf.keras.regularizers.l2(mlp_reg_layers[layer]), activation='relu')
        mlp_vector = model_layer(mlp_vector)
    predict_vector = tf.keras.layers.concatenate([mf_vector, mlp_vector])
    logits = tf.keras.layers.Dense(1, activation=None, kernel_initializer='lecun_uniform', name=movielens.RATING_COLUMN)(predict_vector)
    model = tf.keras.models.Model([user_input, item_input], logits)
    model.summary()
    sys.stdout.flush()
    return model

def _get_estimator_spec_with_metrics(logits, softmax_logits, duplicate_mask, num_training_neg, match_mlperf=False, use_tpu_spec=False):
    if False:
        print('Hello World!')
    'Returns a EstimatorSpec that includes the metrics.'
    (cross_entropy, metric_fn, in_top_k, ndcg, metric_weights) = compute_eval_loss_and_metrics_helper(logits, softmax_logits, duplicate_mask, num_training_neg, match_mlperf, use_tpu_spec)
    if use_tpu_spec:
        return tf.contrib.tpu.TPUEstimatorSpec(mode=tf.estimator.ModeKeys.EVAL, loss=cross_entropy, eval_metrics=(metric_fn, [in_top_k, ndcg, metric_weights]))
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL, loss=cross_entropy, eval_metric_ops=metric_fn(in_top_k, ndcg, metric_weights))

def compute_eval_loss_and_metrics_helper(logits, softmax_logits, duplicate_mask, num_training_neg, match_mlperf=False, use_tpu_spec=False):
    if False:
        return 10
    "Model evaluation with HR and NDCG metrics.\n\n  The evaluation protocol is to rank the test interacted item (truth items)\n  among the randomly chosen 999 items that are not interacted by the user.\n  The performance of the ranked list is judged by Hit Ratio (HR) and Normalized\n  Discounted Cumulative Gain (NDCG).\n\n  For evaluation, the ranked list is truncated at 10 for both metrics. As such,\n  the HR intuitively measures whether the test item is present on the top-10\n  list, and the NDCG accounts for the position of the hit by assigning higher\n  scores to hits at top ranks. Both metrics are calculated for each test user,\n  and the average scores are reported.\n\n  If `match_mlperf` is True, then the HR and NDCG computations are done in a\n  slightly unusual way to match the MLPerf reference implementation.\n  Specifically, if the evaluation negatives contain duplicate items, it will be\n  treated as if the item only appeared once. Effectively, for duplicate items in\n  a row, the predicted score for all but one of the items will be set to\n  -infinity\n\n  For example, suppose we have that following inputs:\n  logits_by_user:     [[ 2,  3,  3],\n                       [ 5,  4,  4]]\n\n  items_by_user:     [[10, 20, 20],\n                      [30, 40, 40]]\n\n  # Note: items_by_user is not explicitly present. Instead the relevant           information is contained within `duplicate_mask`\n\n  top_k: 2\n\n  Then with match_mlperf=True, the HR would be 2/2 = 1.0. With\n  match_mlperf=False, the HR would be 1/2 = 0.5. This is because each user has\n  predicted scores for only 2 unique items: 10 and 20 for the first user, and 30\n  and 40 for the second. Therefore, with match_mlperf=True, it's guaranteed the\n  first item's score is in the top 2. With match_mlperf=False, this function\n  would compute the first user's first item is not in the top 2, because item 20\n  has a higher score, and item 20 occurs twice.\n\n  Args:\n    logits: A tensor containing the predicted logits for each user. The shape\n      of logits is (num_users_per_batch * (1 + NUM_EVAL_NEGATIVES),) Logits\n      for a user are grouped, and the last element of the group is the true\n      element.\n\n    softmax_logits: The same tensor, but with zeros left-appended.\n\n    duplicate_mask: A vector with the same shape as logits, with a value of 1\n      if the item corresponding to the logit at that position has already\n      appeared for that user.\n\n    num_training_neg: The number of negatives per positive during training.\n\n    match_mlperf: Use the MLPerf reference convention for computing rank.\n\n    use_tpu_spec: Should a TPUEstimatorSpec be returned instead of an\n      EstimatorSpec. Required for TPUs and if XLA is done on a GPU. Despite its\n      name, TPUEstimatorSpecs work with GPUs\n\n  Returns:\n    cross_entropy: the loss\n    metric_fn: the metrics function\n    in_top_k: hit rate metric\n    ndcg: ndcg metric\n    metric_weights: metric weights\n  "
    (in_top_k, ndcg, metric_weights, logits_by_user) = compute_top_k_and_ndcg(logits, duplicate_mask, match_mlperf)
    eval_labels = tf.reshape(shape=(-1,), tensor=tf.one_hot(tf.zeros(shape=(logits_by_user.shape[0],), dtype=tf.int32) + rconst.NUM_EVAL_NEGATIVES, logits_by_user.shape[1], dtype=tf.int32))
    eval_labels_float = tf.cast(eval_labels, tf.float32)
    negative_scale_factor = num_training_neg / rconst.NUM_EVAL_NEGATIVES
    example_weights = (eval_labels_float + (1 - eval_labels_float) * negative_scale_factor) * (1 + rconst.NUM_EVAL_NEGATIVES) / (1 + num_training_neg)
    expanded_metric_weights = tf.reshape(tf.tile(metric_weights[:, tf.newaxis], (1, rconst.NUM_EVAL_NEGATIVES + 1)), (-1,))
    example_weights *= tf.cast(expanded_metric_weights, tf.float32)
    cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(logits=softmax_logits, labels=eval_labels, weights=example_weights)

    def metric_fn(top_k_tensor, ndcg_tensor, weight_tensor):
        if False:
            for i in range(10):
                print('nop')
        return {rconst.HR_KEY: tf.compat.v1.metrics.mean(top_k_tensor, weights=weight_tensor, name=rconst.HR_METRIC_NAME), rconst.NDCG_KEY: tf.compat.v1.metrics.mean(ndcg_tensor, weights=weight_tensor, name=rconst.NDCG_METRIC_NAME)}
    return (cross_entropy, metric_fn, in_top_k, ndcg, metric_weights)

def compute_top_k_and_ndcg(logits, duplicate_mask, match_mlperf=False):
    if False:
        i = 10
        return i + 15
    'Compute inputs of metric calculation.\n\n  Args:\n    logits: A tensor containing the predicted logits for each user. The shape\n      of logits is (num_users_per_batch * (1 + NUM_EVAL_NEGATIVES),) Logits\n      for a user are grouped, and the first element of the group is the true\n      element.\n    duplicate_mask: A vector with the same shape as logits, with a value of 1\n      if the item corresponding to the logit at that position has already\n      appeared for that user.\n    match_mlperf: Use the MLPerf reference convention for computing rank.\n\n  Returns:\n    is_top_k, ndcg and weights, all of which has size (num_users_in_batch,), and\n    logits_by_user which has size\n    (num_users_in_batch, (rconst.NUM_EVAL_NEGATIVES + 1)).\n  '
    logits_by_user = tf.reshape(logits, (-1, rconst.NUM_EVAL_NEGATIVES + 1))
    duplicate_mask_by_user = tf.cast(tf.reshape(duplicate_mask, (-1, rconst.NUM_EVAL_NEGATIVES + 1)), logits_by_user.dtype)
    if match_mlperf:
        logits_by_user *= 1 - duplicate_mask_by_user
        logits_by_user += duplicate_mask_by_user * logits_by_user.dtype.min
    sort_indices = tf.argsort(logits_by_user, axis=1, direction='DESCENDING')
    one_hot_position = tf.cast(tf.equal(sort_indices, rconst.NUM_EVAL_NEGATIVES), tf.int32)
    sparse_positions = tf.multiply(one_hot_position, tf.range(logits_by_user.shape[1])[tf.newaxis, :])
    position_vector = tf.reduce_sum(sparse_positions, axis=1)
    in_top_k = tf.cast(tf.less(position_vector, rconst.TOP_K), tf.float32)
    ndcg = tf.math.log(2.0) / tf.math.log(tf.cast(position_vector, tf.float32) + 2)
    ndcg *= in_top_k
    metric_weights = tf.not_equal(tf.reduce_sum(duplicate_mask_by_user, axis=1), rconst.NUM_EVAL_NEGATIVES)
    return (in_top_k, ndcg, metric_weights, logits_by_user)