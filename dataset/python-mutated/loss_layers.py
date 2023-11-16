"""Loss functions for learning global objectives.

These functions have two return values: a Tensor with the value of
the loss, and a dictionary of internal quantities for customizability.
"""
import numpy
import tensorflow as tf
from global_objectives import util

def precision_recall_auc_loss(labels, logits, precision_range=(0.0, 1.0), num_anchors=20, weights=1.0, dual_rate_factor=0.1, label_priors=None, surrogate_type='xent', lambdas_initializer=tf.constant_initializer(1.0), reuse=None, variables_collections=None, trainable=True, scope=None):
    if False:
        return 10
    "Computes precision-recall AUC loss.\n\n  The loss is based on a sum of losses for recall at a range of\n  precision values (anchor points). This sum is a Riemann sum that\n  approximates the area under the precision-recall curve.\n\n  The per-example `weights` argument changes not only the coefficients of\n  individual training examples, but how the examples are counted toward the\n  constraint. If `label_priors` is given, it MUST take `weights` into account.\n  That is,\n      label_priors = P / (P + N)\n  where\n      P = sum_i (wt_i on positives)\n      N = sum_i (wt_i on negatives).\n\n  Args:\n    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].\n    logits: A `Tensor` with the same shape as `labels`.\n    precision_range: A length-two tuple, the range of precision values over\n      which to compute AUC. The entries must be nonnegative, increasing, and\n      less than or equal to 1.0.\n    num_anchors: The number of grid points used to approximate the Riemann sum.\n    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape\n      [batch_size] or [batch_size, num_labels].\n    dual_rate_factor: A floating point value which controls the step size for\n      the Lagrange multipliers.\n    label_priors: None, or a floating point `Tensor` of shape [num_labels]\n      containing the prior probability of each label (i.e. the fraction of the\n      training data consisting of positive examples). If None, the label\n      priors are computed from `labels` with a moving average. See the notes\n      above regarding the interaction with `weights` and do not set this unless\n      you have a good reason to do so.\n    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound\n      should be used for indicator functions.\n    lambdas_initializer: An initializer for the Lagrange multipliers.\n    reuse: Whether or not the layer and its variables should be reused. To be\n      able to reuse the layer scope must be given.\n    variables_collections: Optional list of collections for the variables.\n    trainable: If `True` also add variables to the graph collection\n      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).\n    scope: Optional scope for `variable_scope`.\n\n  Returns:\n    loss: A `Tensor` of the same shape as `logits` with the component-wise\n      loss.\n    other_outputs: A dictionary of useful internal quantities for debugging. For\n      more details, see http://arxiv.org/pdf/1608.04802.pdf.\n      lambdas: A Tensor of shape [1, num_labels, num_anchors] consisting of the\n        Lagrange multipliers.\n      biases: A Tensor of shape [1, num_labels, num_anchors] consisting of the\n        learned bias term for each.\n      label_priors: A Tensor of shape [1, num_labels, 1] consisting of the prior\n        probability of each label learned by the loss, if not provided.\n      true_positives_lower_bound: Lower bound on the number of true positives\n        given `labels` and `logits`. This is the same lower bound which is used\n        in the loss expression to be optimized.\n      false_positives_upper_bound: Upper bound on the number of false positives\n        given `labels` and `logits`. This is the same upper bound which is used\n        in the loss expression to be optimized.\n\n  Raises:\n    ValueError: If `surrogate_type` is not `xent` or `hinge`.\n  "
    with tf.variable_scope(scope, 'precision_recall_auc', [labels, logits, label_priors], reuse=reuse):
        (labels, logits, weights, original_shape) = _prepare_labels_logits_weights(labels, logits, weights)
        num_labels = util.get_num_labels(logits)
        dual_rate_factor = util.convert_and_cast(dual_rate_factor, 'dual_rate_factor', logits.dtype)
        (precision_values, delta) = _range_to_anchors_and_delta(precision_range, num_anchors, logits.dtype)
        (lambdas, lambdas_variable) = _create_dual_variable('lambdas', shape=[1, num_labels, num_anchors], dtype=logits.dtype, initializer=lambdas_initializer, collections=variables_collections, trainable=trainable, dual_rate_factor=dual_rate_factor)
        biases = tf.contrib.framework.model_variable(name='biases', shape=[1, num_labels, num_anchors], dtype=logits.dtype, initializer=tf.zeros_initializer(), collections=variables_collections, trainable=trainable)
        label_priors = maybe_create_label_priors(label_priors, labels, weights, variables_collections)
        label_priors = tf.reshape(label_priors, [1, num_labels, 1])
        logits = tf.expand_dims(logits, 2)
        labels = tf.expand_dims(labels, 2)
        weights = tf.expand_dims(weights, 2)
        loss = weights * util.weighted_surrogate_loss(labels, logits + biases, surrogate_type=surrogate_type, positive_weights=1.0 + lambdas * (1.0 - precision_values), negative_weights=lambdas * precision_values)
        maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
        maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
        lambda_term = lambdas * (1.0 - precision_values) * label_priors * maybe_log2
        per_anchor_loss = loss - lambda_term
        per_label_loss = delta * tf.reduce_sum(per_anchor_loss, 2)
        scaled_loss = tf.div(per_label_loss, precision_range[1] - precision_range[0] - delta, name='AUC_Normalize')
        scaled_loss = tf.reshape(scaled_loss, original_shape)
        other_outputs = {'lambdas': lambdas_variable, 'biases': biases, 'label_priors': label_priors, 'true_positives_lower_bound': true_positives_lower_bound(labels, logits, weights, surrogate_type), 'false_positives_upper_bound': false_positives_upper_bound(labels, logits, weights, surrogate_type)}
        return (scaled_loss, other_outputs)

def roc_auc_loss(labels, logits, weights=1.0, surrogate_type='xent', scope=None):
    if False:
        for i in range(10):
            print('nop')
    "Computes ROC AUC loss.\n\n  The area under the ROC curve is the probability p that a randomly chosen\n  positive example will be scored higher than a randomly chosen negative\n  example. This loss approximates 1-p by using a surrogate (either hinge loss or\n  cross entropy) for the indicator function. Specifically, the loss is:\n\n    sum_i sum_j w_i*w_j*loss(logit_i - logit_j)\n\n  where i ranges over the positive datapoints, j ranges over the negative\n  datapoints, logit_k denotes the logit (or score) of the k-th datapoint, and\n  loss is either the hinge or log loss given a positive label.\n\n  Args:\n    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].\n    logits: A `Tensor` with the same shape and dtype as `labels`.\n    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape\n      [batch_size] or [batch_size, num_labels].\n    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound\n      should be used for the indicator function.\n    scope: Optional scope for `name_scope`.\n\n  Returns:\n    loss: A `Tensor` of the same shape as `logits` with the component-wise loss.\n    other_outputs: An empty dictionary, for consistency.\n\n  Raises:\n    ValueError: If `surrogate_type` is not `xent` or `hinge`.\n  "
    with tf.name_scope(scope, 'roc_auc', [labels, logits, weights]):
        (labels, logits, weights, original_shape) = _prepare_labels_logits_weights(labels, logits, weights)
        logits_difference = tf.expand_dims(logits, 0) - tf.expand_dims(logits, 1)
        labels_difference = tf.expand_dims(labels, 0) - tf.expand_dims(labels, 1)
        weights_product = tf.expand_dims(weights, 0) * tf.expand_dims(weights, 1)
        signed_logits_difference = labels_difference * logits_difference
        raw_loss = util.weighted_surrogate_loss(labels=tf.ones_like(signed_logits_difference), logits=signed_logits_difference, surrogate_type=surrogate_type)
        weighted_loss = weights_product * raw_loss
        loss = tf.reduce_mean(tf.abs(labels_difference) * weighted_loss, 0) * 0.5
        loss = tf.reshape(loss, original_shape)
        return (loss, {})

def recall_at_precision_loss(labels, logits, target_precision, weights=1.0, dual_rate_factor=0.1, label_priors=None, surrogate_type='xent', lambdas_initializer=tf.constant_initializer(1.0), reuse=None, variables_collections=None, trainable=True, scope=None):
    if False:
        while True:
            i = 10
    "Computes recall at precision loss.\n\n  The loss is based on a surrogate of the form\n      wt * w(+) * loss(+) + wt * w(-) * loss(-) - c * pi,\n  where:\n  - w(+) =  1 + lambdas * (1 - target_precision)\n  - loss(+) is the cross-entropy loss on the positive examples\n  - w(-) = lambdas * target_precision\n  - loss(-) is the cross-entropy loss on the negative examples\n  - wt is a scalar or tensor of per-example weights\n  - c = lambdas * (1 - target_precision)\n  - pi is the label_priors.\n\n  The per-example weights change not only the coefficients of individual\n  training examples, but how the examples are counted toward the constraint.\n  If `label_priors` is given, it MUST take `weights` into account. That is,\n      label_priors = P / (P + N)\n  where\n      P = sum_i (wt_i on positives)\n      N = sum_i (wt_i on negatives).\n\n  Args:\n    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].\n    logits: A `Tensor` with the same shape as `labels`.\n    target_precision: The precision at which to compute the loss. Can be a\n      floating point value between 0 and 1 for a single precision value, or a\n      `Tensor` of shape [num_labels], holding each label's target precision\n      value.\n    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape\n      [batch_size] or [batch_size, num_labels].\n    dual_rate_factor: A floating point value which controls the step size for\n      the Lagrange multipliers.\n    label_priors: None, or a floating point `Tensor` of shape [num_labels]\n      containing the prior probability of each label (i.e. the fraction of the\n      training data consisting of positive examples). If None, the label\n      priors are computed from `labels` with a moving average. See the notes\n      above regarding the interaction with `weights` and do not set this unless\n      you have a good reason to do so.\n    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound\n      should be used for indicator functions.\n    lambdas_initializer: An initializer for the Lagrange multipliers.\n    reuse: Whether or not the layer and its variables should be reused. To be\n      able to reuse the layer scope must be given.\n    variables_collections: Optional list of collections for the variables.\n    trainable: If `True` also add variables to the graph collection\n      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).\n    scope: Optional scope for `variable_scope`.\n\n  Returns:\n    loss: A `Tensor` of the same shape as `logits` with the component-wise\n      loss.\n    other_outputs: A dictionary of useful internal quantities for debugging. For\n      more details, see http://arxiv.org/pdf/1608.04802.pdf.\n      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange\n        multipliers.\n      label_priors: A Tensor of shape [num_labels] consisting of the prior\n        probability of each label learned by the loss, if not provided.\n      true_positives_lower_bound: Lower bound on the number of true positives\n        given `labels` and `logits`. This is the same lower bound which is used\n        in the loss expression to be optimized.\n      false_positives_upper_bound: Upper bound on the number of false positives\n        given `labels` and `logits`. This is the same upper bound which is used\n        in the loss expression to be optimized.\n\n  Raises:\n    ValueError: If `logits` and `labels` do not have the same shape.\n  "
    with tf.variable_scope(scope, 'recall_at_precision', [logits, labels, label_priors], reuse=reuse):
        (labels, logits, weights, original_shape) = _prepare_labels_logits_weights(labels, logits, weights)
        num_labels = util.get_num_labels(logits)
        target_precision = util.convert_and_cast(target_precision, 'target_precision', logits.dtype)
        dual_rate_factor = util.convert_and_cast(dual_rate_factor, 'dual_rate_factor', logits.dtype)
        (lambdas, lambdas_variable) = _create_dual_variable('lambdas', shape=[num_labels], dtype=logits.dtype, initializer=lambdas_initializer, collections=variables_collections, trainable=trainable, dual_rate_factor=dual_rate_factor)
        label_priors = maybe_create_label_priors(label_priors, labels, weights, variables_collections)
        weighted_loss = weights * util.weighted_surrogate_loss(labels, logits, surrogate_type=surrogate_type, positive_weights=1.0 + lambdas * (1.0 - target_precision), negative_weights=lambdas * target_precision)
        maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
        maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
        lambda_term = lambdas * (1.0 - target_precision) * label_priors * maybe_log2
        loss = tf.reshape(weighted_loss - lambda_term, original_shape)
        other_outputs = {'lambdas': lambdas_variable, 'label_priors': label_priors, 'true_positives_lower_bound': true_positives_lower_bound(labels, logits, weights, surrogate_type), 'false_positives_upper_bound': false_positives_upper_bound(labels, logits, weights, surrogate_type)}
        return (loss, other_outputs)

def precision_at_recall_loss(labels, logits, target_recall, weights=1.0, dual_rate_factor=0.1, label_priors=None, surrogate_type='xent', lambdas_initializer=tf.constant_initializer(1.0), reuse=None, variables_collections=None, trainable=True, scope=None):
    if False:
        for i in range(10):
            print('nop')
    "Computes precision at recall loss.\n\n  The loss is based on a surrogate of the form\n     wt * loss(-) + lambdas * (pi * (b - 1) + wt * loss(+))\n  where:\n  - loss(-) is the cross-entropy loss on the negative examples\n  - loss(+) is the cross-entropy loss on the positive examples\n  - wt is a scalar or tensor of per-example weights\n  - b is the target recall\n  - pi is the label_priors.\n\n  The per-example weights change not only the coefficients of individual\n  training examples, but how the examples are counted toward the constraint.\n  If `label_priors` is given, it MUST take `weights` into account. That is,\n      label_priors = P / (P + N)\n  where\n      P = sum_i (wt_i on positives)\n      N = sum_i (wt_i on negatives).\n\n  Args:\n    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].\n    logits: A `Tensor` with the same shape as `labels`.\n    target_recall: The recall at which to compute the loss. Can be a floating\n      point value between 0 and 1 for a single target recall value, or a\n      `Tensor` of shape [num_labels] holding each label's target recall value.\n    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape\n      [batch_size] or [batch_size, num_labels].\n    dual_rate_factor: A floating point value which controls the step size for\n      the Lagrange multipliers.\n    label_priors: None, or a floating point `Tensor` of shape [num_labels]\n      containing the prior probability of each label (i.e. the fraction of the\n      training data consisting of positive examples). If None, the label\n      priors are computed from `labels` with a moving average. See the notes\n      above regarding the interaction with `weights` and do not set this unless\n      you have a good reason to do so.\n    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound\n      should be used for indicator functions.\n    lambdas_initializer: An initializer for the Lagrange multipliers.\n    reuse: Whether or not the layer and its variables should be reused. To be\n      able to reuse the layer scope must be given.\n    variables_collections: Optional list of collections for the variables.\n    trainable: If `True` also add variables to the graph collection\n      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).\n    scope: Optional scope for `variable_scope`.\n\n  Returns:\n    loss: A `Tensor` of the same shape as `logits` with the component-wise\n      loss.\n    other_outputs: A dictionary of useful internal quantities for debugging. For\n      more details, see http://arxiv.org/pdf/1608.04802.pdf.\n      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange\n        multipliers.\n      label_priors: A Tensor of shape [num_labels] consisting of the prior\n        probability of each label learned by the loss, if not provided.\n      true_positives_lower_bound: Lower bound on the number of true positives\n        given `labels` and `logits`. This is the same lower bound which is used\n        in the loss expression to be optimized.\n      false_positives_upper_bound: Upper bound on the number of false positives\n        given `labels` and `logits`. This is the same upper bound which is used\n        in the loss expression to be optimized.\n  "
    with tf.variable_scope(scope, 'precision_at_recall', [logits, labels, label_priors], reuse=reuse):
        (labels, logits, weights, original_shape) = _prepare_labels_logits_weights(labels, logits, weights)
        num_labels = util.get_num_labels(logits)
        target_recall = util.convert_and_cast(target_recall, 'target_recall', logits.dtype)
        dual_rate_factor = util.convert_and_cast(dual_rate_factor, 'dual_rate_factor', logits.dtype)
        (lambdas, lambdas_variable) = _create_dual_variable('lambdas', shape=[num_labels], dtype=logits.dtype, initializer=lambdas_initializer, collections=variables_collections, trainable=trainable, dual_rate_factor=dual_rate_factor)
        label_priors = maybe_create_label_priors(label_priors, labels, weights, variables_collections)
        weighted_loss = weights * util.weighted_surrogate_loss(labels, logits, surrogate_type, positive_weights=lambdas, negative_weights=1.0)
        maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
        maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
        lambda_term = lambdas * label_priors * (target_recall - 1.0) * maybe_log2
        loss = tf.reshape(weighted_loss + lambda_term, original_shape)
        other_outputs = {'lambdas': lambdas_variable, 'label_priors': label_priors, 'true_positives_lower_bound': true_positives_lower_bound(labels, logits, weights, surrogate_type), 'false_positives_upper_bound': false_positives_upper_bound(labels, logits, weights, surrogate_type)}
        return (loss, other_outputs)

def false_positive_rate_at_true_positive_rate_loss(labels, logits, target_rate, weights=1.0, dual_rate_factor=0.1, label_priors=None, surrogate_type='xent', lambdas_initializer=tf.constant_initializer(1.0), reuse=None, variables_collections=None, trainable=True, scope=None):
    if False:
        while True:
            i = 10
    "Computes false positive rate at true positive rate loss.\n\n  Note that `true positive rate` is a synonym for Recall, and that minimizing\n  the false positive rate and maximizing precision are equivalent for a fixed\n  Recall. Therefore, this function is identical to precision_at_recall_loss.\n\n  The per-example weights change not only the coefficients of individual\n  training examples, but how the examples are counted toward the constraint.\n  If `label_priors` is given, it MUST take `weights` into account. That is,\n      label_priors = P / (P + N)\n  where\n      P = sum_i (wt_i on positives)\n      N = sum_i (wt_i on negatives).\n\n  Args:\n    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].\n    logits: A `Tensor` with the same shape as `labels`.\n    target_rate: The true positive rate at which to compute the loss. Can be a\n      floating point value between 0 and 1 for a single true positive rate, or\n      a `Tensor` of shape [num_labels] holding each label's true positive rate.\n    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape\n      [batch_size] or [batch_size, num_labels].\n    dual_rate_factor: A floating point value which controls the step size for\n      the Lagrange multipliers.\n    label_priors: None, or a floating point `Tensor` of shape [num_labels]\n      containing the prior probability of each label (i.e. the fraction of the\n      training data consisting of positive examples). If None, the label\n      priors are computed from `labels` with a moving average. See the notes\n      above regarding the interaction with `weights` and do not set this unless\n      you have a good reason to do so.\n    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound\n      should be used for indicator functions. 'xent' will use the cross-entropy\n      loss surrogate, and 'hinge' will use the hinge loss.\n    lambdas_initializer: An initializer op for the Lagrange multipliers.\n    reuse: Whether or not the layer and its variables should be reused. To be\n      able to reuse the layer scope must be given.\n    variables_collections: Optional list of collections for the variables.\n    trainable: If `True` also add variables to the graph collection\n      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).\n    scope: Optional scope for `variable_scope`.\n\n  Returns:\n    loss: A `Tensor` of the same shape as `logits` with the component-wise\n      loss.\n    other_outputs: A dictionary of useful internal quantities for debugging. For\n      more details, see http://arxiv.org/pdf/1608.04802.pdf.\n      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange\n        multipliers.\n      label_priors: A Tensor of shape [num_labels] consisting of the prior\n        probability of each label learned by the loss, if not provided.\n      true_positives_lower_bound: Lower bound on the number of true positives\n        given `labels` and `logits`. This is the same lower bound which is used\n        in the loss expression to be optimized.\n      false_positives_upper_bound: Upper bound on the number of false positives\n        given `labels` and `logits`. This is the same upper bound which is used\n        in the loss expression to be optimized.\n\n  Raises:\n    ValueError: If `surrogate_type` is not `xent` or `hinge`.\n  "
    return precision_at_recall_loss(labels=labels, logits=logits, target_recall=target_rate, weights=weights, dual_rate_factor=dual_rate_factor, label_priors=label_priors, surrogate_type=surrogate_type, lambdas_initializer=lambdas_initializer, reuse=reuse, variables_collections=variables_collections, trainable=trainable, scope=scope)

def true_positive_rate_at_false_positive_rate_loss(labels, logits, target_rate, weights=1.0, dual_rate_factor=0.1, label_priors=None, surrogate_type='xent', lambdas_initializer=tf.constant_initializer(1.0), reuse=None, variables_collections=None, trainable=True, scope=None):
    if False:
        while True:
            i = 10
    "Computes true positive rate at false positive rate loss.\n\n  The loss is based on a surrogate of the form\n      wt * loss(+) + lambdas * (wt * loss(-) - r * (1 - pi))\n  where:\n  - loss(-) is the loss on the negative examples\n  - loss(+) is the loss on the positive examples\n  - wt is a scalar or tensor of per-example weights\n  - r is the target rate\n  - pi is the label_priors.\n\n  The per-example weights change not only the coefficients of individual\n  training examples, but how the examples are counted toward the constraint.\n  If `label_priors` is given, it MUST take `weights` into account. That is,\n      label_priors = P / (P + N)\n  where\n      P = sum_i (wt_i on positives)\n      N = sum_i (wt_i on negatives).\n\n  Args:\n    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].\n    logits: A `Tensor` with the same shape as `labels`.\n    target_rate: The false positive rate at which to compute the loss. Can be a\n      floating point value between 0 and 1 for a single false positive rate, or\n      a `Tensor` of shape [num_labels] holding each label's false positive rate.\n    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape\n      [batch_size] or [batch_size, num_labels].\n    dual_rate_factor: A floating point value which controls the step size for\n      the Lagrange multipliers.\n    label_priors: None, or a floating point `Tensor` of shape [num_labels]\n      containing the prior probability of each label (i.e. the fraction of the\n      training data consisting of positive examples). If None, the label\n      priors are computed from `labels` with a moving average. See the notes\n      above regarding the interaction with `weights` and do not set this unless\n      you have a good reason to do so.\n    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound\n      should be used for indicator functions. 'xent' will use the cross-entropy\n      loss surrogate, and 'hinge' will use the hinge loss.\n    lambdas_initializer: An initializer op for the Lagrange multipliers.\n    reuse: Whether or not the layer and its variables should be reused. To be\n      able to reuse the layer scope must be given.\n    variables_collections: Optional list of collections for the variables.\n    trainable: If `True` also add variables to the graph collection\n      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).\n    scope: Optional scope for `variable_scope`.\n\n  Returns:\n    loss: A `Tensor` of the same shape as `logits` with the component-wise\n      loss.\n    other_outputs: A dictionary of useful internal quantities for debugging. For\n      more details, see http://arxiv.org/pdf/1608.04802.pdf.\n      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange\n        multipliers.\n      label_priors: A Tensor of shape [num_labels] consisting of the prior\n        probability of each label learned by the loss, if not provided.\n      true_positives_lower_bound: Lower bound on the number of true positives\n        given `labels` and `logits`. This is the same lower bound which is used\n        in the loss expression to be optimized.\n      false_positives_upper_bound: Upper bound on the number of false positives\n        given `labels` and `logits`. This is the same upper bound which is used\n        in the loss expression to be optimized.\n\n  Raises:\n    ValueError: If `surrogate_type` is not `xent` or `hinge`.\n  "
    with tf.variable_scope(scope, 'tpr_at_fpr', [labels, logits, label_priors], reuse=reuse):
        (labels, logits, weights, original_shape) = _prepare_labels_logits_weights(labels, logits, weights)
        num_labels = util.get_num_labels(logits)
        target_rate = util.convert_and_cast(target_rate, 'target_rate', logits.dtype)
        dual_rate_factor = util.convert_and_cast(dual_rate_factor, 'dual_rate_factor', logits.dtype)
        (lambdas, lambdas_variable) = _create_dual_variable('lambdas', shape=[num_labels], dtype=logits.dtype, initializer=lambdas_initializer, collections=variables_collections, trainable=trainable, dual_rate_factor=dual_rate_factor)
        label_priors = maybe_create_label_priors(label_priors, labels, weights, variables_collections)
        weighted_loss = weights * util.weighted_surrogate_loss(labels, logits, surrogate_type=surrogate_type, positive_weights=1.0, negative_weights=lambdas)
        maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
        maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
        lambda_term = lambdas * target_rate * (1.0 - label_priors) * maybe_log2
        loss = tf.reshape(weighted_loss - lambda_term, original_shape)
        other_outputs = {'lambdas': lambdas_variable, 'label_priors': label_priors, 'true_positives_lower_bound': true_positives_lower_bound(labels, logits, weights, surrogate_type), 'false_positives_upper_bound': false_positives_upper_bound(labels, logits, weights, surrogate_type)}
    return (loss, other_outputs)

def _prepare_labels_logits_weights(labels, logits, weights):
    if False:
        for i in range(10):
            print('nop')
    'Validates labels, logits, and weights.\n\n  Converts inputs to tensors, checks shape compatibility, and casts dtype if\n  necessary.\n\n  Args:\n    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].\n    logits: A `Tensor` with the same shape as `labels`.\n    weights: Either `None` or a `Tensor` with shape broadcastable to `logits`.\n\n  Returns:\n    labels: Same as `labels` arg after possible conversion to tensor, cast, and\n      reshape.\n    logits: Same as `logits` arg after possible conversion to tensor and\n      reshape.\n    weights: Same as `weights` arg after possible conversion, cast, and reshape.\n    original_shape: Shape of `labels` and `logits` before reshape.\n\n  Raises:\n    ValueError: If `labels` and `logits` do not have the same shape.\n  '
    logits = tf.convert_to_tensor(logits, name='logits')
    labels = util.convert_and_cast(labels, 'labels', logits.dtype.base_dtype)
    weights = util.convert_and_cast(weights, 'weights', logits.dtype.base_dtype)
    try:
        labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
        raise ValueError('logits and labels must have the same shape (%s vs %s)' % (logits.get_shape(), labels.get_shape()))
    original_shape = labels.get_shape().as_list()
    if labels.get_shape().ndims > 0:
        original_shape[0] = -1
    if labels.get_shape().ndims <= 1:
        labels = tf.reshape(labels, [-1, 1])
        logits = tf.reshape(logits, [-1, 1])
    if weights.get_shape().ndims == 1:
        weights = tf.reshape(weights, [-1, 1])
    if weights.get_shape().ndims == 0:
        weights *= tf.ones_like(logits)
    return (labels, logits, weights, original_shape)

def _range_to_anchors_and_delta(precision_range, num_anchors, dtype):
    if False:
        print('Hello World!')
    'Calculates anchor points from precision range.\n\n  Args:\n    precision_range: As required in precision_recall_auc_loss.\n    num_anchors: int, number of equally spaced anchor points.\n    dtype: Data type of returned tensors.\n\n  Returns:\n    precision_values: A `Tensor` of data type dtype with equally spaced values\n      in the interval precision_range.\n    delta: The spacing between the values in precision_values.\n\n  Raises:\n    ValueError: If precision_range is invalid.\n  '
    if not 0 <= precision_range[0] <= precision_range[-1] <= 1:
        raise ValueError('precision values must obey 0 <= %f <= %f <= 1' % (precision_range[0], precision_range[-1]))
    if not 0 < len(precision_range) < 3:
        raise ValueError('length of precision_range (%d) must be 1 or 2' % len(precision_range))
    values = numpy.linspace(start=precision_range[0], stop=precision_range[1], num=num_anchors + 2)[1:-1]
    precision_values = util.convert_and_cast(values, 'precision_values', dtype)
    delta = util.convert_and_cast(values[0] - precision_range[0], 'delta', dtype)
    precision_values = util.expand_outer(precision_values, 3)
    return (precision_values, delta)

def _create_dual_variable(name, shape, dtype, initializer, collections, trainable, dual_rate_factor):
    if False:
        i = 10
        return i + 15
    'Creates a new dual variable.\n\n  Dual variables are required to be nonnegative. If trainable, their gradient\n  is reversed so that they are maximized (rather than minimized) by the\n  optimizer.\n\n  Args:\n    name: A string, the name for the new variable.\n    shape: Shape of the new variable.\n    dtype: Data type for the new variable.\n    initializer: Initializer for the new variable.\n    collections: List of graph collections keys. The new variable is added to\n      these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.\n    trainable: If `True`, the default, also adds the variable to the graph\n      collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as\n      the default list of variables to use by the `Optimizer` classes.\n    dual_rate_factor: A floating point value or `Tensor`. The learning rate for\n      the dual variable is scaled by this factor.\n\n  Returns:\n    dual_value: An op that computes the absolute value of the dual variable\n      and reverses its gradient.\n    dual_variable: The underlying variable itself.\n  '
    partitioner = tf.get_variable_scope().partitioner
    try:
        tf.get_variable_scope().set_partitioner(None)
        dual_variable = tf.contrib.framework.model_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, collections=collections, trainable=trainable)
    finally:
        tf.get_variable_scope().set_partitioner(partitioner)
    dual_value = tf.abs(dual_variable)
    if trainable:
        dual_value = tf.stop_gradient((1.0 + dual_rate_factor) * dual_value) - dual_rate_factor * dual_value
    return (dual_value, dual_variable)

def maybe_create_label_priors(label_priors, labels, weights, variables_collections):
    if False:
        print('Hello World!')
    'Creates moving average ops to track label priors, if necessary.\n\n  Args:\n    label_priors: As required in e.g. precision_recall_auc_loss.\n    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].\n    weights: As required in e.g. precision_recall_auc_loss.\n    variables_collections: Optional list of collections for the variables, if\n      any must be created.\n\n  Returns:\n    label_priors: A Tensor of shape [num_labels] consisting of the\n      weighted label priors, after updating with moving average ops if created.\n  '
    if label_priors is not None:
        label_priors = util.convert_and_cast(label_priors, name='label_priors', dtype=labels.dtype.base_dtype)
        return tf.squeeze(label_priors)
    label_priors = util.build_label_priors(labels, weights, variables_collections=variables_collections)
    return label_priors

def true_positives_lower_bound(labels, logits, weights, surrogate_type):
    if False:
        return 10
    "Calculate a lower bound on the number of true positives.\n\n  This lower bound on the number of true positives given `logits` and `labels`\n  is the same one used in the global objectives loss functions.\n\n  Args:\n    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].\n    logits: A `Tensor` of shape [batch_size, num_labels] or\n      [batch_size, num_labels, num_anchors]. If the third dimension is present,\n      the lower bound is computed on each slice [:, :, k] independently.\n    weights: Per-example loss coefficients, with shape broadcast-compatible with\n        that of `labels`.\n    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound\n      should be used for indicator functions.\n\n  Returns:\n    A `Tensor` of shape [num_labels] or [num_labels, num_anchors].\n  "
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    if logits.get_shape().ndims == 3 and labels.get_shape().ndims < 3:
        labels = tf.expand_dims(labels, 2)
    loss_on_positives = util.weighted_surrogate_loss(labels, logits, surrogate_type, negative_weights=0.0) / maybe_log2
    return tf.reduce_sum(weights * (labels - loss_on_positives), 0)

def false_positives_upper_bound(labels, logits, weights, surrogate_type):
    if False:
        while True:
            i = 10
    "Calculate an upper bound on the number of false positives.\n\n  This upper bound on the number of false positives given `logits` and `labels`\n  is the same one used in the global objectives loss functions.\n\n  Args:\n    labels: A `Tensor` of shape [batch_size, num_labels]\n    logits: A `Tensor` of shape [batch_size, num_labels]  or\n      [batch_size, num_labels, num_anchors]. If the third dimension is present,\n      the lower bound is computed on each slice [:, :, k] independently.\n    weights: Per-example loss coefficients, with shape broadcast-compatible with\n        that of `labels`.\n    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound\n      should be used for indicator functions.\n\n  Returns:\n    A `Tensor` of shape [num_labels] or [num_labels, num_anchors].\n  "
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    loss_on_negatives = util.weighted_surrogate_loss(labels, logits, surrogate_type, positive_weights=0.0) / maybe_log2
    return tf.reduce_sum(weights * loss_on_negatives, 0)