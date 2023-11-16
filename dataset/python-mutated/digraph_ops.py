"""TensorFlow ops for directed graphs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from syntaxnet.util import check

def ArcPotentialsFromTokens(source_tokens, target_tokens, weights):
    if False:
        i = 10
        return i + 15
    'Returns arc potentials computed from token activations and weights.\n\n  For each batch of source and target token activations, computes a scalar\n  potential for each arc as the 3-way product between the activation vectors of\n  the source and target of the arc and the |weights|.  Specifically,\n\n    arc[b,s,t] =\n        \\sum_{i,j} source_tokens[b,s,i] * weights[i,j] * target_tokens[b,t,j]\n\n  Note that the token activations can be extended with bias terms to implement a\n  "biaffine" model (Dozat and Manning, 2017).\n\n  Args:\n    source_tokens: [B,N,S] tensor of batched activations for the source token in\n                   each arc.\n    target_tokens: [B,N,T] tensor of batched activations for the target token in\n                   each arc.\n    weights: [S,T] matrix of weights.\n\n    B,N may be statically-unknown, but S,T must be statically-known.  The dtype\n    of all arguments must be compatible.\n\n  Returns:\n    [B,N,N] tensor A of arc potentials where A_{b,s,t} is the potential of the\n    arc from s to t in batch element b.  The dtype of A is the same as that of\n    the arguments.  Note that the diagonal entries (i.e., where s==t) represent\n    self-loops and may not be meaningful.\n  '
    check.Eq(source_tokens.get_shape().ndims, 3, 'source_tokens must be rank 3')
    check.Eq(target_tokens.get_shape().ndims, 3, 'target_tokens must be rank 3')
    check.Eq(weights.get_shape().ndims, 2, 'weights must be a matrix')
    num_source_activations = weights.get_shape().as_list()[0]
    num_target_activations = weights.get_shape().as_list()[1]
    check.NotNone(num_source_activations, 'unknown source activation dimension')
    check.NotNone(num_target_activations, 'unknown target activation dimension')
    check.Eq(source_tokens.get_shape().as_list()[2], num_source_activations, 'dimension mismatch between weights and source_tokens')
    check.Eq(target_tokens.get_shape().as_list()[2], num_target_activations, 'dimension mismatch between weights and target_tokens')
    check.Same([weights.dtype.base_dtype, source_tokens.dtype.base_dtype, target_tokens.dtype.base_dtype], 'dtype mismatch')
    source_tokens_shape = tf.shape(source_tokens)
    target_tokens_shape = tf.shape(target_tokens)
    batch_size = source_tokens_shape[0]
    num_tokens = source_tokens_shape[1]
    with tf.control_dependencies([tf.assert_equal(batch_size, target_tokens_shape[0]), tf.assert_equal(num_tokens, target_tokens_shape[1])]):
        targets_bnxt = tf.reshape(target_tokens, [-1, num_target_activations])
        weights_targets_bnxs = tf.matmul(targets_bnxt, weights, transpose_b=True)
        weights_targets_bxnxs = tf.reshape(weights_targets_bnxs, [batch_size, num_tokens, num_source_activations])
        arcs_bxnxn = tf.matmul(source_tokens, weights_targets_bxnxs, transpose_b=True)
        return arcs_bxnxn

def ArcSourcePotentialsFromTokens(tokens, weights):
    if False:
        return 10
    'Returns arc source potentials computed from tokens and weights.\n\n  For each batch of token activations, computes a scalar potential for each arc\n  as the product between the activations of the source token and the |weights|.\n  Specifically,\n\n    arc[b,s,:] = \\sum_{i} weights[i] * tokens[b,s,i]\n\n  Args:\n    tokens: [B,N,S] tensor of batched activations for source tokens.\n    weights: [S] vector of weights.\n\n    B,N may be statically-unknown, but S must be statically-known.  The dtype of\n    all arguments must be compatible.\n\n  Returns:\n    [B,N,N] tensor A of arc potentials as defined above.  The dtype of A is the\n    same as that of the arguments.  Note that the diagonal entries (i.e., where\n    s==t) represent self-loops and may not be meaningful.\n  '
    check.Eq(tokens.get_shape().ndims, 3, 'tokens must be rank 3')
    check.Eq(weights.get_shape().ndims, 1, 'weights must be a vector')
    num_source_activations = weights.get_shape().as_list()[0]
    check.NotNone(num_source_activations, 'unknown source activation dimension')
    check.Eq(tokens.get_shape().as_list()[2], num_source_activations, 'dimension mismatch between weights and tokens')
    check.Same([weights.dtype.base_dtype, tokens.dtype.base_dtype], 'dtype mismatch')
    tokens_shape = tf.shape(tokens)
    batch_size = tokens_shape[0]
    num_tokens = tokens_shape[1]
    tokens_bnxs = tf.reshape(tokens, [-1, num_source_activations])
    weights_sx1 = tf.expand_dims(weights, 1)
    sources_bnx1 = tf.matmul(tokens_bnxs, weights_sx1)
    sources_bnxn = tf.tile(sources_bnx1, [1, num_tokens])
    sources_bxnxn = tf.reshape(sources_bnxn, [batch_size, num_tokens, num_tokens])
    return sources_bxnxn

def RootPotentialsFromTokens(root, tokens, weights_arc, weights_source):
    if False:
        while True:
            i = 10
    'Returns root selection potentials computed from tokens and weights.\n\n  For each batch of token activations, computes a scalar potential for each root\n  selection as the 3-way product between the activations of the artificial root\n  token, the token activations, and the |weights|.  Specifically,\n\n    roots[b,r] = \\sum_{i,j} root[i] * weights[i,j] * tokens[b,r,j]\n\n  Args:\n    root: [S] vector of activations for the artificial root token.\n    tokens: [B,N,T] tensor of batched activations for root tokens.\n    weights_arc: [S,T] matrix of weights.\n    weights_source: [S] vector of weights.\n\n    B,N may be statically-unknown, but S,T must be statically-known.  The dtype\n    of all arguments must be compatible.\n\n  Returns:\n    [B,N] matrix R of root-selection potentials as defined above.  The dtype of\n    R is the same as that of the arguments.\n  '
    check.Eq(root.get_shape().ndims, 1, 'root must be a vector')
    check.Eq(tokens.get_shape().ndims, 3, 'tokens must be rank 3')
    check.Eq(weights_arc.get_shape().ndims, 2, 'weights_arc must be a matrix')
    check.Eq(weights_source.get_shape().ndims, 1, 'weights_source must be a vector')
    num_source_activations = weights_arc.get_shape().as_list()[0]
    num_target_activations = weights_arc.get_shape().as_list()[1]
    check.NotNone(num_source_activations, 'unknown source activation dimension')
    check.NotNone(num_target_activations, 'unknown target activation dimension')
    check.Eq(root.get_shape().as_list()[0], num_source_activations, 'dimension mismatch between weights_arc and root')
    check.Eq(tokens.get_shape().as_list()[2], num_target_activations, 'dimension mismatch between weights_arc and tokens')
    check.Eq(weights_source.get_shape().as_list()[0], num_source_activations, 'dimension mismatch between weights_arc and weights_source')
    check.Same([weights_arc.dtype.base_dtype, weights_source.dtype.base_dtype, root.dtype.base_dtype, tokens.dtype.base_dtype], 'dtype mismatch')
    root_1xs = tf.expand_dims(root, 0)
    weights_source_sx1 = tf.expand_dims(weights_source, 1)
    tokens_shape = tf.shape(tokens)
    batch_size = tokens_shape[0]
    num_tokens = tokens_shape[1]
    tokens_bnxt = tf.reshape(tokens, [-1, num_target_activations])
    weights_targets_bnxs = tf.matmul(tokens_bnxt, weights_arc, transpose_b=True)
    roots_1xbn = tf.matmul(root_1xs, weights_targets_bnxs, transpose_b=True)
    roots_1xbn += tf.matmul(root_1xs, weights_source_sx1)
    roots_bxn = tf.reshape(roots_1xbn, [batch_size, num_tokens])
    return roots_bxn

def CombineArcAndRootPotentials(arcs, roots):
    if False:
        for i in range(10):
            print('nop')
    'Combines arc and root potentials into a single set of potentials.\n\n  Args:\n    arcs: [B,N,N] tensor of batched arc potentials.\n    roots: [B,N] matrix of batched root potentials.\n\n  Returns:\n    [B,N,N] tensor P of combined potentials where\n      P_{b,s,t} = s == t ? roots[b,t] : arcs[b,s,t]\n  '
    check.Eq(arcs.get_shape().ndims, 3, 'arcs must be rank 3')
    check.Eq(roots.get_shape().ndims, 2, 'roots must be a matrix')
    dtype = arcs.dtype.base_dtype
    check.Same([dtype, roots.dtype.base_dtype], 'dtype mismatch')
    roots_shape = tf.shape(roots)
    arcs_shape = tf.shape(arcs)
    batch_size = roots_shape[0]
    num_tokens = roots_shape[1]
    with tf.control_dependencies([tf.assert_equal(batch_size, arcs_shape[0]), tf.assert_equal(num_tokens, arcs_shape[1]), tf.assert_equal(num_tokens, arcs_shape[2])]):
        return tf.matrix_set_diag(arcs, roots)

def LabelPotentialsFromTokens(tokens, weights):
    if False:
        for i in range(10):
            print('nop')
    'Computes label potentials from tokens and weights.\n\n  For each batch of token activations, computes a scalar potential for each\n  label as the product between the activations of the source token and the\n  |weights|.  Specifically,\n\n    labels[b,t,l] = \\sum_{i} weights[l,i] * tokens[b,t,i]\n\n  Args:\n    tokens: [B,N,T] tensor of batched token activations.\n    weights: [L,T] matrix of weights.\n\n    B,N may be dynamic, but L,T must be static.  The dtype of all arguments must\n    be compatible.\n\n  Returns:\n    [B,N,L] tensor of label potentials as defined above, with the same dtype as\n    the arguments.\n  '
    check.Eq(tokens.get_shape().ndims, 3, 'tokens must be rank 3')
    check.Eq(weights.get_shape().ndims, 2, 'weights must be a matrix')
    num_labels = weights.get_shape().as_list()[0]
    num_activations = weights.get_shape().as_list()[1]
    check.NotNone(num_labels, 'unknown number of labels')
    check.NotNone(num_activations, 'unknown activation dimension')
    check.Eq(tokens.get_shape().as_list()[2], num_activations, 'activation mismatch between weights and tokens')
    tokens_shape = tf.shape(tokens)
    batch_size = tokens_shape[0]
    num_tokens = tokens_shape[1]
    check.Same([tokens.dtype.base_dtype, weights.dtype.base_dtype], 'dtype mismatch')
    tokens_bnxt = tf.reshape(tokens, [-1, num_activations])
    labels_bnxl = tf.matmul(tokens_bnxt, weights, transpose_b=True)
    labels_bxnxl = tf.reshape(labels_bnxl, [batch_size, num_tokens, num_labels])
    return labels_bxnxl

def LabelPotentialsFromTokenPairs(sources, targets, weights):
    if False:
        i = 10
        return i + 15
    'Computes label potentials from source and target tokens and weights.\n\n  For each aligned pair of source and target token activations, computes a\n  scalar potential for each label on the arc from the source to the target.\n  Specifically,\n\n    labels[b,t,l] = \\sum_{i,j} sources[b,t,i] * weights[l,i,j] * targets[b,t,j]\n\n  Args:\n    sources: [B,N,S] tensor of batched source token activations.\n    targets: [B,N,T] tensor of batched target token activations.\n    weights: [L,S,T] tensor of weights.\n\n    B,N may be dynamic, but L,S,T must be static.  The dtype of all arguments\n    must be compatible.\n\n  Returns:\n    [B,N,L] tensor of label potentials as defined above, with the same dtype as\n    the arguments.\n  '
    check.Eq(sources.get_shape().ndims, 3, 'sources must be rank 3')
    check.Eq(targets.get_shape().ndims, 3, 'targets must be rank 3')
    check.Eq(weights.get_shape().ndims, 3, 'weights must be rank 3')
    num_labels = weights.get_shape().as_list()[0]
    num_source_activations = weights.get_shape().as_list()[1]
    num_target_activations = weights.get_shape().as_list()[2]
    check.NotNone(num_labels, 'unknown number of labels')
    check.NotNone(num_source_activations, 'unknown source activation dimension')
    check.NotNone(num_target_activations, 'unknown target activation dimension')
    check.Eq(sources.get_shape().as_list()[2], num_source_activations, 'activation mismatch between weights and source tokens')
    check.Eq(targets.get_shape().as_list()[2], num_target_activations, 'activation mismatch between weights and target tokens')
    check.Same([sources.dtype.base_dtype, targets.dtype.base_dtype, weights.dtype.base_dtype], 'dtype mismatch')
    sources_shape = tf.shape(sources)
    targets_shape = tf.shape(targets)
    batch_size = sources_shape[0]
    num_tokens = sources_shape[1]
    with tf.control_dependencies([tf.assert_equal(batch_size, targets_shape[0]), tf.assert_equal(num_tokens, targets_shape[1])]):
        weights_lsxt = tf.reshape(weights, [num_labels * num_source_activations, num_target_activations])
        targets_bnxt = tf.reshape(targets, [-1, num_target_activations])
        weights_targets_bnxls = tf.matmul(targets_bnxt, weights_lsxt, transpose_b=True)
        weights_targets_bxnxlxs = tf.reshape(weights_targets_bnxls, [batch_size, num_tokens, num_labels, num_source_activations])
        sources_bxnx1xs = tf.expand_dims(sources, 2)
        labels_bxnxlx1 = tf.matmul(weights_targets_bxnxlxs, sources_bxnx1xs, transpose_b=True)
        labels_bxnxl = tf.squeeze(labels_bxnxlx1, [3])
        return labels_bxnxl

def ValidArcAndTokenMasks(lengths, max_length, dtype=tf.float32):
    if False:
        return 10
    'Returns 0/1 masks for valid arcs and tokens.\n\n  Args:\n    lengths: [B] vector of input sequence lengths.\n    max_length: Scalar maximum input sequence length, aka M.\n    dtype: Data type for output mask.\n\n  Returns:\n    [B,M,M] tensor A with 0/1 indicators of valid arcs.  Specifically,\n      A_{b,t,s} = t,s < lengths[b] ? 1 : 0\n    [B,M] matrix T with 0/1 indicators of valid tokens.  Specifically,\n      T_{b,t} = t < lengths[b] ? 1 : 0\n  '
    lengths_bx1 = tf.expand_dims(lengths, 1)
    sequence_m = tf.range(tf.cast(max_length, lengths.dtype.base_dtype))
    sequence_1xm = tf.expand_dims(sequence_m, 0)
    valid_token_bxm = tf.cast(sequence_1xm < lengths_bx1, dtype)
    valid_arc_bxmxm = tf.matmul(tf.expand_dims(valid_token_bxm, 2), tf.expand_dims(valid_token_bxm, 1))
    return (valid_arc_bxmxm, valid_token_bxm)

def LaplacianMatrix(lengths, arcs, forest=False):
    if False:
        return 10
    'Returns the (root-augmented) Laplacian matrix for a batch of digraphs.\n\n  Args:\n    lengths: [B] vector of input sequence lengths.\n    arcs: [B,M,M] tensor of arc potentials where entry b,t,s is the potential of\n      the arc from s to t in the b\'th digraph, while b,t,t is the potential of t\n      as a root.  Entries b,t,s where t or s >= lengths[b] are ignored.\n    forest: Whether to produce a Laplacian for trees or forests.\n\n  Returns:\n    [B,M,M] tensor L with the Laplacian of each digraph, padded with an identity\n    matrix.  More concretely, the padding entries (t or s >= lengths[b]) are:\n      L_{b,t,t} = 1.0\n      L_{b,t,s} = 0.0\n    Note that this "identity matrix padding" ensures that the determinant of\n    each padded matrix equals the determinant of the unpadded matrix.  The\n    non-padding entries (t,s < lengths[b]) depend on whether the Laplacian is\n    constructed for trees or forests.  For trees:\n      L_{b,t,0} = arcs[b,t,t]\n      L_{b,t,t} = \\sum_{s < lengths[b], t != s} arcs[b,t,s]\n      L_{b,t,s} = -arcs[b,t,s]\n    For forests:\n      L_{b,t,t} = \\sum_{s < lengths[b]} arcs[b,t,s]\n      L_{b,t,s} = -arcs[b,t,s]\n    See http://www.aclweb.org/anthology/D/D07/D07-1015.pdf for details, though\n    note that our matrices are transposed from their notation.\n  '
    check.Eq(arcs.get_shape().ndims, 3, 'arcs must be rank 3')
    dtype = arcs.dtype.base_dtype
    arcs_shape = tf.shape(arcs)
    batch_size = arcs_shape[0]
    max_length = arcs_shape[1]
    with tf.control_dependencies([tf.assert_equal(max_length, arcs_shape[2])]):
        (valid_arc_bxmxm, valid_token_bxm) = ValidArcAndTokenMasks(lengths, max_length, dtype=dtype)
    invalid_token_bxm = tf.constant(1, dtype=dtype) - valid_token_bxm
    arcs_bxmxm = arcs * valid_arc_bxmxm
    zeros_bxm = tf.zeros([batch_size, max_length], dtype)
    if not forest:
        roots_bxm = tf.matrix_diag_part(arcs_bxmxm)
        arcs_bxmxm = tf.matrix_set_diag(arcs_bxmxm, zeros_bxm)
    sums_bxm = tf.reduce_sum(arcs_bxmxm, 2)
    if forest:
        arcs_bxmxm = tf.matrix_set_diag(arcs_bxmxm, zeros_bxm)
    diagonal_bxm = sums_bxm + invalid_token_bxm
    laplacian_bxmxm = tf.matrix_diag(diagonal_bxm) - arcs_bxmxm
    if not forest:
        roots_bxmx1 = tf.expand_dims(roots_bxm, 2)
        laplacian_bxmxm = tf.concat([roots_bxmx1, laplacian_bxmxm[:, :, 1:]], 2)
    return laplacian_bxmxm