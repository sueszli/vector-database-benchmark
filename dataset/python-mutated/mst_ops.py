"""TensorFlow ops for maximum spanning tree problems."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import dragnn.python.load_mst_cc_impl
from dragnn.mst.ops import gen_mst_ops
from dragnn.python import digraph_ops
from syntaxnet.util import check
maximum_spanning_tree = gen_mst_ops.maximum_spanning_tree

@tf.RegisterGradient('MaximumSpanningTree')
def maximum_spanning_tree_gradient(mst_op, d_loss_d_max_scores, *_):
    if False:
        i = 10
        return i + 15
    'Returns a subgradient of the MaximumSpanningTree op.\n\n  Note that MaximumSpanningTree is only differentiable w.r.t. its |scores| input\n  and its |max_scores| output.\n\n  Args:\n    mst_op: The MaximumSpanningTree op being differentiated.\n    d_loss_d_max_scores: [B] vector where entry b is the gradient of the network\n                         loss w.r.t. entry b of the |max_scores| output of the\n                         |mst_op|.\n    *_: The gradients w.r.t. the other outputs; ignored.\n\n  Returns:\n    1. None, since the op is not differentiable w.r.t. its |num_nodes| input.\n    2. [B,M,M] tensor where entry b,t,s is a subgradient of the network loss\n       w.r.t. entry b,t,s of the |scores| input, with the same dtype as\n       |d_loss_d_max_scores|.\n  '
    dtype = d_loss_d_max_scores.dtype.base_dtype
    check.NotNone(dtype)
    argmax_sources_bxm = mst_op.outputs[1]
    input_dim = tf.shape(argmax_sources_bxm)[1]
    indicators_bxmxm = tf.one_hot(argmax_sources_bxm, input_dim, dtype=dtype)
    d_loss_d_max_scores_bx1 = tf.expand_dims(d_loss_d_max_scores, -1)
    d_loss_d_max_scores_bx1x1 = tf.expand_dims(d_loss_d_max_scores_bx1, -1)
    d_loss_d_scores_bxmxm = indicators_bxmxm * d_loss_d_max_scores_bx1x1
    return (None, d_loss_d_scores_bxmxm)

def log_partition_function(num_nodes, scores, forest=False, max_dynamic_range=None):
    if False:
        return 10
    'Returns the log of the sum-of-product of spanning trees or forests.\n\n  Computing the sum-of-product in the log domain reduces the chance of overflow\n  or underflow, and ML techniques (e.g., CRF loss functions) typically require\n  the log partition function anyways.  For similar reasons, the scores input is\n  assumed to be specified in the log domain.\n\n  The partition function is caluclated via application of the Matrix-Tree\n  theorem; see the following for details:\n    https://en.wikipedia.org/wiki/Kirchhoff%27s_theorem\n    http://www.aclweb.org/anthology/D/D07/D07-1015.pdf\n\n  Computing the gradient of the log partition function requires inverting the\n  Laplacian matrix.  Numerical issues may occur if the Laplacian is singular or\n  nearly-so.  (Intuitively, the Laplacian will be close to singular when the\n  input scores strongly favor invalid structures such as cycles).  In the EMNLP\n  paper, we alleviated the numerical issues by clipping the difference between\n  the minimum and maximum score for each node to 20 (in the log domain).  The\n  |max_dynamic_range| argument can be used for this purpose.\n\n  TODO(googleuser): Try improving the condition number of the Laplacian matrix\n  directly, instead of using the indirect approach above.  For example, one\n  could add c*I to the Laplacian (i.e., Tikhonov regularization).\n\n  Args:\n    num_nodes: [B] vector of graph sizes per batch item.\n    scores: [B,M,M] tensor of padded batched arc and root scores, in the format\n      used by the maximum_spanning_tree() op.  Padding values must be finite.\n    forest: If true, sum over spanning forests instead of trees.\n    max_dynamic_range: If specified, incoming scores for each node are clipped\n      to at most this far from the maximum such score (in the log domain).\n\n  Returns:\n    [B] vector Z of log partition function values, where\n      Z[b] = log(\n          \\sum_{tree spanning batch item b}\n              score(root_of(tree)) \\prod_{arc in tree} score(arc))\n  '
    orig_dtype = scores.dtype.base_dtype
    scores_bxmxm = tf.to_double(scores)
    shape_bxmxm = tf.shape(scores_bxmxm)
    batch_size = shape_bxmxm[0]
    max_nodes = shape_bxmxm[1]
    total_nodes = batch_size * max_nodes
    (_, valid_tokens_bxm) = digraph_ops.ValidArcAndTokenMasks(num_nodes, max_nodes, dtype=tf.int32)
    valid_tokens_bx1xm = tf.expand_dims(valid_tokens_bxm, 1)
    valid_sources_bxmxm = tf.tile(valid_tokens_bx1xm, [1, max_nodes, 1])
    sequence_bm = 1 + tf.range(total_nodes, dtype=tf.int32)
    sequence_bxmx1 = tf.reshape(sequence_bm, [batch_size, max_nodes, 1])
    target_ids_bxmxm = valid_sources_bxmxm * sequence_bxmx1
    max_scores_bm1 = tf.unsorted_segment_max(scores_bxmxm, target_ids_bxmxm, total_nodes + 1)
    max_scores_bm = max_scores_bm1[1:]
    sequence_b = 1 + tf.range(batch_size, dtype=tf.int32)
    sequence_bx1 = tf.expand_dims(sequence_b, 1)
    batch_ids_bxm = valid_tokens_bxm * sequence_bx1
    batch_ids_bm = tf.reshape(batch_ids_bxm, [-1])
    log_normalization_factor_b1 = tf.unsorted_segment_sum(max_scores_bm, batch_ids_bm, batch_size + 1)
    log_normalization_factor_b = log_normalization_factor_b1[1:]
    max_scores_bxmx1 = tf.reshape(max_scores_bm, [batch_size, max_nodes, 1])
    scores_bxmxm -= max_scores_bxmx1
    if max_dynamic_range is not None:
        scores_bxmxm = tf.maximum(scores_bxmxm, -max_dynamic_range)
    scores_bxmxm = tf.exp(scores_bxmxm)
    exp_normalized_laplacian_bxmxm = digraph_ops.LaplacianMatrix(num_nodes, scores_bxmxm, forest=forest)
    log_normalized_partition_function_b = tf.log(tf.matrix_determinant(exp_normalized_laplacian_bxmxm))
    log_partition_function_b = log_normalized_partition_function_b + log_normalization_factor_b
    return tf.cast(log_partition_function_b, orig_dtype)