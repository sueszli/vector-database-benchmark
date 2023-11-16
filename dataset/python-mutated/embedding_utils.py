"""Utilities for the instance embedding for segmentation."""
import numpy as np
import tensorflow as tf
from deeplab import model
from deeplab.core import preprocess_utils
from feelvos.utils import mask_damaging
slim = tf.contrib.slim
resolve_shape = preprocess_utils.resolve_shape
WRONG_LABEL_PADDING_DISTANCE = 1e+20
USE_CORRELATION_COST = False
if USE_CORRELATION_COST:
    from correlation_cost.python.ops import correlation_cost_op

def pairwise_distances(x, y):
    if False:
        while True:
            i = 10
    'Computes pairwise squared l2 distances between tensors x and y.\n\n  Args:\n    x: Tensor of shape [n, feature_dim].\n    y: Tensor of shape [m, feature_dim].\n\n  Returns:\n    Float32 distances tensor of shape [n, m].\n  '
    xs = tf.reduce_sum(x * x, axis=1)[:, tf.newaxis]
    ys = tf.reduce_sum(y * y, axis=1)[tf.newaxis, :]
    d = xs + ys - 2 * tf.matmul(x, y, transpose_b=True)
    return d

def pairwise_distances2(x, y):
    if False:
        i = 10
        return i + 15
    'Computes pairwise squared l2 distances between tensors x and y.\n\n  Naive implementation, high memory use. Could be useful to test the more\n  efficient implementation.\n\n  Args:\n    x: Tensor of shape [n, feature_dim].\n    y: Tensor of shape [m, feature_dim].\n\n  Returns:\n    distances of shape [n, m].\n  '
    return tf.reduce_sum(tf.squared_difference(x[:, tf.newaxis], y[tf.newaxis, :]), axis=-1)

def cross_correlate(x, y, max_distance=9):
    if False:
        print('Hello World!')
    'Efficiently computes the cross correlation of x and y.\n\n  Optimized implementation using correlation_cost.\n  Note that we do not normalize by the feature dimension.\n\n  Args:\n    x: Float32 tensor of shape [height, width, feature_dim].\n    y: Float32 tensor of shape [height, width, feature_dim].\n    max_distance: Integer, the maximum distance in pixel coordinates\n      per dimension which is considered to be in the search window.\n\n  Returns:\n    Float32 tensor of shape [height, width, (2 * max_distance + 1) ** 2].\n  '
    with tf.name_scope('cross_correlation'):
        corr = correlation_cost_op.correlation_cost(x[tf.newaxis], y[tf.newaxis], kernel_size=1, max_displacement=max_distance, stride_1=1, stride_2=1, pad=max_distance)
        corr = tf.squeeze(corr, axis=0)
        feature_dim = resolve_shape(x)[-1]
        corr *= feature_dim
        return corr

def local_pairwise_distances(x, y, max_distance=9):
    if False:
        return 10
    'Computes pairwise squared l2 distances using a local search window.\n\n  Optimized implementation using correlation_cost.\n\n  Args:\n    x: Float32 tensor of shape [height, width, feature_dim].\n    y: Float32 tensor of shape [height, width, feature_dim].\n    max_distance: Integer, the maximum distance in pixel coordinates\n      per dimension which is considered to be in the search window.\n\n  Returns:\n    Float32 distances tensor of shape\n      [height, width, (2 * max_distance + 1) ** 2].\n  '
    with tf.name_scope('local_pairwise_distances'):
        corr = cross_correlate(x, y, max_distance=max_distance)
        xs = tf.reduce_sum(x * x, axis=2)[..., tf.newaxis]
        ys = tf.reduce_sum(y * y, axis=2)[..., tf.newaxis]
        ones_ys = tf.ones_like(ys)
        ys = cross_correlate(ones_ys, ys, max_distance=max_distance)
        d = xs + ys - 2 * corr
        boundary = tf.equal(cross_correlate(ones_ys, ones_ys, max_distance=max_distance), 0)
        d = tf.where(boundary, tf.fill(tf.shape(d), tf.constant(np.float('inf'))), d)
        return d

def local_pairwise_distances2(x, y, max_distance=9):
    if False:
        for i in range(10):
            print('nop')
    'Computes pairwise squared l2 distances using a local search window.\n\n  Naive implementation using map_fn.\n  Used as a slow fallback for when correlation_cost is not available.\n\n  Args:\n    x: Float32 tensor of shape [height, width, feature_dim].\n    y: Float32 tensor of shape [height, width, feature_dim].\n    max_distance: Integer, the maximum distance in pixel coordinates\n      per dimension which is considered to be in the search window.\n\n  Returns:\n    Float32 distances tensor of shape\n      [height, width, (2 * max_distance + 1) ** 2].\n  '
    with tf.name_scope('local_pairwise_distances2'):
        padding_val = 1e+20
        padded_y = tf.pad(y, [[max_distance, max_distance], [max_distance, max_distance], [0, 0]], constant_values=padding_val)
        (height, width, _) = resolve_shape(x)
        dists = []
        for y_start in range(2 * max_distance + 1):
            y_end = y_start + height
            y_slice = padded_y[y_start:y_end]
            for x_start in range(2 * max_distance + 1):
                x_end = x_start + width
                offset_y = y_slice[:, x_start:x_end]
                dist = tf.reduce_sum(tf.squared_difference(x, offset_y), axis=2)
                dists.append(dist)
        dists = tf.stack(dists, axis=2)
        return dists

def majority_vote(labels):
    if False:
        i = 10
        return i + 15
    'Performs a label majority vote along axis 1.\n\n  Second try, hopefully this time more efficient.\n  We assume that the labels are contiguous starting from 0.\n  It will also work for non-contiguous labels, but be inefficient.\n\n  Args:\n    labels: Int tensor of shape [n, k]\n\n  Returns:\n    The majority of labels along axis 1\n  '
    max_label = tf.reduce_max(labels)
    one_hot = tf.one_hot(labels, depth=max_label + 1)
    summed = tf.reduce_sum(one_hot, axis=1)
    majority = tf.argmax(summed, axis=1)
    return majority

def assign_labels_by_nearest_neighbors(reference_embeddings, query_embeddings, reference_labels, k=1):
    if False:
        while True:
            i = 10
    'Segments by nearest neighbor query wrt the reference frame.\n\n  Args:\n    reference_embeddings: Tensor of shape [height, width, embedding_dim],\n      the embedding vectors for the reference frame\n    query_embeddings: Tensor of shape [n_query_images, height, width,\n      embedding_dim], the embedding vectors for the query frames\n    reference_labels: Tensor of shape [height, width, 1], the class labels of\n      the reference frame\n    k: Integer, the number of nearest neighbors to use\n\n  Returns:\n    The labels of the nearest neighbors as [n_query_frames, height, width, 1]\n    tensor\n\n  Raises:\n    ValueError: If k < 1.\n  '
    if k < 1:
        raise ValueError('k must be at least 1')
    dists = flattened_pairwise_distances(reference_embeddings, query_embeddings)
    if k == 1:
        nn_indices = tf.argmin(dists, axis=1)[..., tf.newaxis]
    else:
        (_, nn_indices) = tf.nn.top_k(-dists, k, sorted=False)
    reference_labels = tf.reshape(reference_labels, [-1])
    nn_labels = tf.gather(reference_labels, nn_indices)
    if k == 1:
        nn_labels = tf.squeeze(nn_labels, axis=1)
    else:
        nn_labels = majority_vote(nn_labels)
    height = tf.shape(reference_embeddings)[0]
    width = tf.shape(reference_embeddings)[1]
    n_query_frames = query_embeddings.shape[0]
    nn_labels = tf.reshape(nn_labels, [n_query_frames, height, width, 1])
    return nn_labels

def flattened_pairwise_distances(reference_embeddings, query_embeddings):
    if False:
        for i in range(10):
            print('nop')
    'Calculates flattened tensor of pairwise distances between ref and query.\n\n  Args:\n    reference_embeddings: Tensor of shape [..., embedding_dim],\n      the embedding vectors for the reference frame\n    query_embeddings: Tensor of shape [n_query_images, height, width,\n      embedding_dim], the embedding vectors for the query frames.\n\n  Returns:\n    A distance tensor of shape [reference_embeddings.size / embedding_dim,\n    query_embeddings.size / embedding_dim]\n  '
    embedding_dim = resolve_shape(query_embeddings)[-1]
    reference_embeddings = tf.reshape(reference_embeddings, [-1, embedding_dim])
    first_dim = -1
    query_embeddings = tf.reshape(query_embeddings, [first_dim, embedding_dim])
    dists = pairwise_distances(query_embeddings, reference_embeddings)
    return dists

def nearest_neighbor_features_per_object(reference_embeddings, query_embeddings, reference_labels, max_neighbors_per_object, k_nearest_neighbors, gt_ids=None, n_chunks=100):
    if False:
        for i in range(10):
            print('nop')
    'Calculates the distance to the nearest neighbor per object.\n\n  For every pixel of query_embeddings calculate the distance to the\n  nearest neighbor in the (possibly subsampled) reference_embeddings per object.\n\n  Args:\n    reference_embeddings: Tensor of shape [height, width, embedding_dim],\n      the embedding vectors for the reference frame.\n    query_embeddings: Tensor of shape [n_query_images, height, width,\n      embedding_dim], the embedding vectors for the query frames.\n    reference_labels: Tensor of shape [height, width, 1], the class labels of\n      the reference frame.\n    max_neighbors_per_object: Integer, the maximum number of candidates\n      for the nearest neighbor query per object after subsampling,\n      or 0 for no subsampling.\n    k_nearest_neighbors: Integer, the number of nearest neighbors to use.\n    gt_ids: Int tensor of shape [n_objs] of the sorted unique ground truth\n      ids in the first frame. If None, it will be derived from\n      reference_labels.\n    n_chunks: Integer, the number of chunks to use to save memory\n      (set to 1 for no chunking).\n\n  Returns:\n    nn_features: A float32 tensor of nearest neighbor features of shape\n      [n_query_images, height, width, n_objects, feature_dim].\n    gt_ids: An int32 tensor of the unique sorted object ids present\n      in the reference labels.\n  '
    with tf.name_scope('nn_features_per_object'):
        reference_labels_flat = tf.reshape(reference_labels, [-1])
        if gt_ids is None:
            (ref_obj_ids, _) = tf.unique(reference_labels_flat)
            ref_obj_ids = tf.contrib.framework.sort(ref_obj_ids)
            gt_ids = ref_obj_ids
        embedding_dim = resolve_shape(reference_embeddings)[-1]
        reference_embeddings_flat = tf.reshape(reference_embeddings, [-1, embedding_dim])
        (reference_embeddings_flat, reference_labels_flat) = subsample_reference_embeddings_and_labels(reference_embeddings_flat, reference_labels_flat, gt_ids, max_neighbors_per_object)
        shape = resolve_shape(query_embeddings)
        query_embeddings_flat = tf.reshape(query_embeddings, [-1, embedding_dim])
        nn_features = _nearest_neighbor_features_per_object_in_chunks(reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, gt_ids, k_nearest_neighbors, n_chunks)
        nn_features_dim = resolve_shape(nn_features)[-1]
        nn_features_reshaped = tf.reshape(nn_features, tf.stack(shape[:3] + [tf.size(gt_ids), nn_features_dim]))
        return (nn_features_reshaped, gt_ids)

def _nearest_neighbor_features_per_object_in_chunks(reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, ref_obj_ids, k_nearest_neighbors, n_chunks):
    if False:
        print('Hello World!')
    'Calculates the nearest neighbor features per object in chunks to save mem.\n\n  Uses chunking to bound the memory use.\n\n  Args:\n    reference_embeddings_flat: Tensor of shape [n, embedding_dim],\n      the embedding vectors for the reference frame.\n    query_embeddings_flat: Tensor of shape [m, embedding_dim], the embedding\n      vectors for the query frames.\n    reference_labels_flat: Tensor of shape [n], the class labels of the\n      reference frame.\n    ref_obj_ids: int tensor of unique object ids in the reference labels.\n    k_nearest_neighbors: Integer, the number of nearest neighbors to use.\n    n_chunks: Integer, the number of chunks to use to save memory\n      (set to 1 for no chunking).\n\n  Returns:\n    nn_features: A float32 tensor of nearest neighbor features of shape\n      [m, n_objects, feature_dim].\n  '
    chunk_size = tf.cast(tf.ceil(tf.cast(tf.shape(query_embeddings_flat)[0], tf.float32) / n_chunks), tf.int32)
    wrong_label_mask = tf.not_equal(reference_labels_flat, ref_obj_ids[:, tf.newaxis])
    all_features = []
    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        with tf.control_dependencies(all_features):
            features = _nn_features_per_object_for_chunk(reference_embeddings_flat, query_embeddings_flat_chunk, wrong_label_mask, k_nearest_neighbors)
        all_features.append(features)
    if n_chunks == 1:
        nn_features = all_features[0]
    else:
        nn_features = tf.concat(all_features, axis=0)
    return nn_features

def _nn_features_per_object_for_chunk(reference_embeddings, query_embeddings, wrong_label_mask, k_nearest_neighbors):
    if False:
        while True:
            i = 10
    'Extracts features for each object using nearest neighbor attention.\n\n  Args:\n    reference_embeddings: Tensor of shape [n_chunk, embedding_dim],\n      the embedding vectors for the reference frame.\n    query_embeddings: Tensor of shape [m_chunk, embedding_dim], the embedding\n      vectors for the query frames.\n    wrong_label_mask:\n    k_nearest_neighbors: Integer, the number of nearest neighbors to use.\n\n  Returns:\n    nn_features: A float32 tensor of nearest neighbor features of shape\n      [m_chunk, n_objects, feature_dim].\n  '
    reference_embeddings_key = reference_embeddings
    query_embeddings_key = query_embeddings
    dists = flattened_pairwise_distances(reference_embeddings_key, query_embeddings_key)
    dists = dists[:, tf.newaxis, :] + tf.cast(wrong_label_mask[tf.newaxis, :, :], tf.float32) * WRONG_LABEL_PADDING_DISTANCE
    if k_nearest_neighbors == 1:
        features = tf.reduce_min(dists, axis=2, keepdims=True)
    else:
        (dists, _) = tf.nn.top_k(-dists, k=k_nearest_neighbors)
        dists = -dists
        valid_mask = tf.less(dists, WRONG_LABEL_PADDING_DISTANCE)
        masked_dists = dists * tf.cast(valid_mask, tf.float32)
        pad_dist = tf.tile(tf.reduce_max(masked_dists, axis=2)[..., tf.newaxis], multiples=[1, 1, k_nearest_neighbors])
        dists = tf.where(valid_mask, dists, pad_dist)
        features = tf.reduce_mean(dists, axis=2, keepdims=True)
    return features

def create_embedding_segmentation_features(features, feature_dimension, n_layers, kernel_size, reuse, atrous_rates=None):
    if False:
        i = 10
        return i + 15
    'Extracts features which can be used to estimate the final segmentation.\n\n  Args:\n    features: input features of shape [batch, height, width, features]\n    feature_dimension: Integer, the dimensionality used in the segmentation\n      head layers.\n    n_layers: Integer, the number of layers in the segmentation head.\n    kernel_size: Integer, the kernel size used in the segmentation head.\n    reuse: reuse mode for the variable_scope.\n    atrous_rates: List of integers of length n_layers, the atrous rates to use.\n\n  Returns:\n    Features to be used to estimate the segmentation labels of shape\n      [batch, height, width, embedding_seg_feat_dim].\n  '
    if atrous_rates is None or not atrous_rates:
        atrous_rates = [1 for _ in range(n_layers)]
    assert len(atrous_rates) == n_layers
    with tf.variable_scope('embedding_seg', reuse=reuse):
        for n in range(n_layers):
            features = model.split_separable_conv2d(features, feature_dimension, kernel_size=kernel_size, rate=atrous_rates[n], scope='split_separable_conv2d_{}'.format(n))
        return features

def add_image_summaries(images, nn_features, logits, batch_size, prev_frame_nn_features=None):
    if False:
        return 10
    'Adds image summaries of input images, attention features and logits.\n\n  Args:\n    images: Image tensor of shape [batch, height, width, channels].\n    nn_features: Nearest neighbor attention features of shape\n      [batch_size, height, width, n_objects, 1].\n    logits: Float32 tensor of logits.\n    batch_size: Integer, the number of videos per clone per mini-batch.\n    prev_frame_nn_features: Nearest neighbor attention features wrt. the\n      last frame of shape [batch_size, height, width, n_objects, 1].\n      Can be None.\n  '
    reshaped_images = tf.reshape(images, tf.stack([batch_size, -1] + resolve_shape(images)[1:]))
    reference_images = reshaped_images[:, 0]
    query_images = reshaped_images[:, 1:]
    query_images_reshaped = tf.reshape(query_images, tf.stack([-1] + resolve_shape(images)[1:]))
    tf.summary.image('ref_images', reference_images, max_outputs=batch_size)
    tf.summary.image('query_images', query_images_reshaped, max_outputs=10)
    predictions = tf.cast(tf.argmax(logits, axis=-1), tf.uint8)[..., tf.newaxis]
    tf.summary.image('predictions', predictions * 32, max_outputs=10)
    tf.summary.image('nn_fg_features', nn_features[..., 0:1, 0], max_outputs=batch_size)
    if prev_frame_nn_features is not None:
        tf.summary.image('nn_fg_features_prev', prev_frame_nn_features[..., 0:1, 0], max_outputs=batch_size)
    tf.summary.image('nn_bg_features', nn_features[..., 1:2, 0], max_outputs=batch_size)
    if prev_frame_nn_features is not None:
        tf.summary.image('nn_bg_features_prev', prev_frame_nn_features[..., 1:2, 0], max_outputs=batch_size)

def get_embeddings(images, model_options, embedding_dimension):
    if False:
        i = 10
        return i + 15
    'Extracts embedding vectors for images. Should only be used for inference.\n\n  Args:\n    images: A tensor of shape [batch, height, width, channels].\n    model_options: A ModelOptions instance to configure models.\n    embedding_dimension: Integer, the dimension of the embedding.\n\n  Returns:\n    embeddings: A tensor of shape [batch, height, width, embedding_dimension].\n  '
    (features, end_points) = model.extract_features(images, model_options, is_training=False)
    if model_options.decoder_output_stride is not None:
        decoder_output_stride = min(model_options.decoder_output_stride)
        if model_options.crop_size is None:
            height = tf.shape(images)[1]
            width = tf.shape(images)[2]
        else:
            (height, width) = model_options.crop_size
        features = model.refine_by_decoder(features, end_points, crop_size=[height, width], decoder_output_stride=[decoder_output_stride], decoder_use_separable_conv=model_options.decoder_use_separable_conv, model_variant=model_options.model_variant, is_training=False)
    with tf.variable_scope('embedding'):
        embeddings = split_separable_conv2d_with_identity_initializer(features, embedding_dimension, scope='split_separable_conv2d')
    return embeddings

def get_logits_with_matching(images, model_options, weight_decay=0.0001, reuse=None, is_training=False, fine_tune_batch_norm=False, reference_labels=None, batch_size=None, num_frames_per_video=None, embedding_dimension=None, max_neighbors_per_object=0, k_nearest_neighbors=1, use_softmax_feedback=True, initial_softmax_feedback=None, embedding_seg_feature_dimension=256, embedding_seg_n_layers=4, embedding_seg_kernel_size=7, embedding_seg_atrous_rates=None, normalize_nearest_neighbor_distances=True, also_attend_to_previous_frame=True, damage_initial_previous_frame_mask=False, use_local_previous_frame_attention=True, previous_frame_attention_window_size=15, use_first_frame_matching=True, also_return_embeddings=False, ref_embeddings=None):
    if False:
        return 10
    'Gets the logits by atrous/image spatial pyramid pooling using attention.\n\n  Args:\n    images: A tensor of size [batch, height, width, channels].\n    model_options: A ModelOptions instance to configure models.\n    weight_decay: The weight decay for model variables.\n    reuse: Reuse the model variables or not.\n    is_training: Is training or not.\n    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.\n    reference_labels: The segmentation labels of the reference frame on which\n      attention is applied.\n    batch_size: Integer, the number of videos on a batch\n    num_frames_per_video: Integer, the number of frames per video\n    embedding_dimension: Integer, the dimension of the embedding\n    max_neighbors_per_object: Integer, the maximum number of candidates\n      for the nearest neighbor query per object after subsampling.\n      Can be 0 for no subsampling.\n    k_nearest_neighbors: Integer, the number of nearest neighbors to use.\n    use_softmax_feedback: Boolean, whether to give the softmax predictions of\n      the last frame as additional input to the segmentation head.\n    initial_softmax_feedback: List of Float32 tensors, or None. Can be used to\n      initialize the softmax predictions used for the feedback loop.\n      Only has an effect if use_softmax_feedback is True.\n    embedding_seg_feature_dimension: Integer, the dimensionality used in the\n      segmentation head layers.\n    embedding_seg_n_layers: Integer, the number of layers in the segmentation\n      head.\n    embedding_seg_kernel_size: Integer, the kernel size used in the\n      segmentation head.\n    embedding_seg_atrous_rates: List of integers of length\n      embedding_seg_n_layers, the atrous rates to use for the segmentation head.\n    normalize_nearest_neighbor_distances: Boolean, whether to normalize the\n      nearest neighbor distances to [0,1] using sigmoid, scale and shift.\n    also_attend_to_previous_frame: Boolean, whether to also use nearest\n      neighbor attention with respect to the previous frame.\n    damage_initial_previous_frame_mask: Boolean, whether to artificially damage\n      the initial previous frame mask. Only has an effect if\n      also_attend_to_previous_frame is True.\n    use_local_previous_frame_attention: Boolean, whether to restrict the\n      previous frame attention to a local search window.\n      Only has an effect, if also_attend_to_previous_frame is True.\n    previous_frame_attention_window_size: Integer, the window size used for\n      local previous frame attention, if use_local_previous_frame_attention\n      is True.\n    use_first_frame_matching: Boolean, whether to extract features by matching\n      to the reference frame. This should always be true except for ablation\n      experiments.\n    also_return_embeddings: Boolean, whether to return the embeddings as well.\n    ref_embeddings: Tuple of\n      (first_frame_embeddings, previous_frame_embeddings),\n      each of shape [batch, height, width, embedding_dimension], or None.\n  Returns:\n    outputs_to_logits: A map from output_type to logits.\n    If also_return_embeddings is True, it will also return an embeddings\n      tensor of shape [batch, height, width, embedding_dimension].\n  '
    (features, end_points) = model.extract_features(images, model_options, weight_decay=weight_decay, reuse=reuse, is_training=is_training, fine_tune_batch_norm=fine_tune_batch_norm)
    if model_options.decoder_output_stride:
        decoder_output_stride = min(model_options.decoder_output_stride)
        if model_options.crop_size is None:
            height = tf.shape(images)[1]
            width = tf.shape(images)[2]
        else:
            (height, width) = model_options.crop_size
        decoder_height = model.scale_dimension(height, 1.0 / decoder_output_stride)
        decoder_width = model.scale_dimension(width, 1.0 / decoder_output_stride)
        features = model.refine_by_decoder(features, end_points, crop_size=[height, width], decoder_output_stride=[decoder_output_stride], decoder_use_separable_conv=model_options.decoder_use_separable_conv, model_variant=model_options.model_variant, weight_decay=weight_decay, reuse=reuse, is_training=is_training, fine_tune_batch_norm=fine_tune_batch_norm)
    with tf.variable_scope('embedding', reuse=reuse):
        embeddings = split_separable_conv2d_with_identity_initializer(features, embedding_dimension, scope='split_separable_conv2d')
        embeddings = tf.identity(embeddings, name='embeddings')
    scaled_reference_labels = tf.image.resize_nearest_neighbor(reference_labels, resolve_shape(embeddings, 4)[1:3], align_corners=True)
    (h, w) = (decoder_height, decoder_width)
    if num_frames_per_video is None:
        num_frames_per_video = tf.size(embeddings) // (batch_size * h * w * embedding_dimension)
    new_labels_shape = tf.stack([batch_size, -1, h, w, 1])
    reshaped_reference_labels = tf.reshape(scaled_reference_labels, new_labels_shape)
    new_embeddings_shape = tf.stack([batch_size, num_frames_per_video, h, w, embedding_dimension])
    reshaped_embeddings = tf.reshape(embeddings, new_embeddings_shape)
    all_nn_features = []
    all_ref_obj_ids = []
    for n in range(batch_size):
        embedding = reshaped_embeddings[n]
        if ref_embeddings is None:
            n_chunks = 100
            reference_embedding = embedding[0]
            if also_attend_to_previous_frame or use_softmax_feedback:
                queries_embedding = embedding[2:]
            else:
                queries_embedding = embedding[1:]
        else:
            if USE_CORRELATION_COST:
                n_chunks = 20
            else:
                n_chunks = 500
            reference_embedding = ref_embeddings[0][n]
            queries_embedding = embedding
        reference_labels = reshaped_reference_labels[n][0]
        (nn_features_n, ref_obj_ids) = nearest_neighbor_features_per_object(reference_embedding, queries_embedding, reference_labels, max_neighbors_per_object, k_nearest_neighbors, n_chunks=n_chunks)
        if normalize_nearest_neighbor_distances:
            nn_features_n = (tf.nn.sigmoid(nn_features_n) - 0.5) * 2
        all_nn_features.append(nn_features_n)
        all_ref_obj_ids.append(ref_obj_ids)
    feat_dim = resolve_shape(features)[-1]
    features = tf.reshape(features, tf.stack([batch_size, num_frames_per_video, h, w, feat_dim]))
    if ref_embeddings is None:
        if also_attend_to_previous_frame or use_softmax_feedback:
            features = features[:, 2:]
        else:
            features = features[:, 1:]
    outputs_to_logits = {output: [] for output in model_options.outputs_to_num_classes}
    for n in range(batch_size):
        features_n = features[n]
        nn_features_n = all_nn_features[n]
        nn_features_n_tr = tf.transpose(nn_features_n, [3, 0, 1, 2, 4])
        n_objs = tf.shape(nn_features_n_tr)[0]
        features_n_tiled = tf.tile(features_n[tf.newaxis], multiples=[n_objs, 1, 1, 1, 1])
        prev_frame_labels = None
        if also_attend_to_previous_frame:
            prev_frame_labels = reshaped_reference_labels[n, 1]
            if is_training and damage_initial_previous_frame_mask:
                prev_frame_labels = mask_damaging.damage_masks(prev_frame_labels, dilate=False)
            tf.summary.image('prev_frame_labels', tf.cast(prev_frame_labels[tf.newaxis], tf.uint8) * 32)
            initial_softmax_feedback_n = create_initial_softmax_from_labels(prev_frame_labels, reshaped_reference_labels[n][0], decoder_output_stride=None, reduce_labels=True)
        elif initial_softmax_feedback is not None:
            initial_softmax_feedback_n = initial_softmax_feedback[n]
        else:
            initial_softmax_feedback_n = None
        if initial_softmax_feedback_n is None:
            last_softmax = tf.zeros((n_objs, h, w, 1), dtype=tf.float32)
        else:
            last_softmax = tf.transpose(initial_softmax_feedback_n, [2, 0, 1])[..., tf.newaxis]
        assert len(model_options.outputs_to_num_classes) == 1
        output = model_options.outputs_to_num_classes.keys()[0]
        logits = []
        n_ref_frames = 1
        prev_frame_nn_features_n = None
        if also_attend_to_previous_frame or use_softmax_feedback:
            n_ref_frames += 1
        if ref_embeddings is not None:
            n_ref_frames = 0
        for t in range(num_frames_per_video - n_ref_frames):
            to_concat = [features_n_tiled[:, t]]
            if use_first_frame_matching:
                to_concat.append(nn_features_n_tr[:, t])
            if use_softmax_feedback:
                to_concat.append(last_softmax)
            if also_attend_to_previous_frame:
                assert normalize_nearest_neighbor_distances, 'previous frame attention currently only works when normalized distances are used'
                embedding = reshaped_embeddings[n]
                if ref_embeddings is None:
                    last_frame_embedding = embedding[t + 1]
                    query_embeddings = embedding[t + 2, tf.newaxis]
                else:
                    last_frame_embedding = ref_embeddings[1][0]
                    query_embeddings = embedding
                if use_local_previous_frame_attention:
                    assert query_embeddings.shape[0] == 1
                    prev_frame_nn_features_n = local_previous_frame_nearest_neighbor_features_per_object(last_frame_embedding, query_embeddings[0], prev_frame_labels, all_ref_obj_ids[n], max_distance=previous_frame_attention_window_size)
                else:
                    (prev_frame_nn_features_n, _) = nearest_neighbor_features_per_object(last_frame_embedding, query_embeddings, prev_frame_labels, max_neighbors_per_object, k_nearest_neighbors, gt_ids=all_ref_obj_ids[n])
                    prev_frame_nn_features_n = (tf.nn.sigmoid(prev_frame_nn_features_n) - 0.5) * 2
                prev_frame_nn_features_n_sq = tf.squeeze(prev_frame_nn_features_n, axis=0)
                prev_frame_nn_features_n_tr = tf.transpose(prev_frame_nn_features_n_sq, [2, 0, 1, 3])
                to_concat.append(prev_frame_nn_features_n_tr)
            features_n_concat_t = tf.concat(to_concat, axis=-1)
            embedding_seg_features_n_t = create_embedding_segmentation_features(features_n_concat_t, embedding_seg_feature_dimension, embedding_seg_n_layers, embedding_seg_kernel_size, reuse or n > 0, atrous_rates=embedding_seg_atrous_rates)
            logits_t = model.get_branch_logits(embedding_seg_features_n_t, 1, model_options.atrous_rates, aspp_with_batch_norm=model_options.aspp_with_batch_norm, kernel_size=model_options.logits_kernel_size, weight_decay=weight_decay, reuse=reuse or n > 0 or t > 0, scope_suffix=output)
            logits.append(logits_t)
            prev_frame_labels = tf.transpose(tf.argmax(logits_t, axis=0), [2, 0, 1])
            last_softmax = tf.nn.softmax(logits_t, axis=0)
        logits = tf.stack(logits, axis=1)
        logits_shape = tf.stack([n_objs, num_frames_per_video - n_ref_frames] + resolve_shape(logits)[2:-1])
        logits_reshaped = tf.reshape(logits, logits_shape)
        logits_transposed = tf.transpose(logits_reshaped, [1, 2, 3, 0])
        outputs_to_logits[output].append(logits_transposed)
        add_image_summaries(images[n * num_frames_per_video:(n + 1) * num_frames_per_video], nn_features_n, logits_transposed, batch_size=1, prev_frame_nn_features=prev_frame_nn_features_n)
    if also_return_embeddings:
        return (outputs_to_logits, embeddings)
    else:
        return outputs_to_logits

def subsample_reference_embeddings_and_labels(reference_embeddings_flat, reference_labels_flat, ref_obj_ids, max_neighbors_per_object):
    if False:
        while True:
            i = 10
    'Subsamples the reference embedding vectors and labels.\n\n  After subsampling, at most max_neighbors_per_object items will remain per\n    class.\n\n  Args:\n    reference_embeddings_flat: Tensor of shape [n, embedding_dim],\n      the embedding vectors for the reference frame.\n    reference_labels_flat: Tensor of shape [n, 1],\n      the class labels of the reference frame.\n    ref_obj_ids: An int32 tensor of the unique object ids present\n      in the reference labels.\n    max_neighbors_per_object: Integer, the maximum number of candidates\n      for the nearest neighbor query per object after subsampling,\n      or 0 for no subsampling.\n\n  Returns:\n    reference_embeddings_flat: Tensor of shape [n_sub, embedding_dim],\n      the subsampled embedding vectors for the reference frame.\n    reference_labels_flat: Tensor of shape [n_sub, 1],\n      the class labels of the reference frame.\n  '
    if max_neighbors_per_object == 0:
        return (reference_embeddings_flat, reference_labels_flat)
    same_label_mask = tf.equal(reference_labels_flat[tf.newaxis, :], ref_obj_ids[:, tf.newaxis])
    max_neighbors_per_object_repeated = tf.tile(tf.constant(max_neighbors_per_object)[tf.newaxis], multiples=[tf.size(ref_obj_ids)])
    with tf.device('cpu:0'):
        subsampled_indices = tf.map_fn(_create_subsampling_mask, (same_label_mask, max_neighbors_per_object_repeated), dtype=tf.int64, name='subsample_labels_map_fn', parallel_iterations=1)
    mask = tf.not_equal(subsampled_indices, tf.constant(-1, dtype=tf.int64))
    masked_indices = tf.boolean_mask(subsampled_indices, mask)
    reference_embeddings_flat = tf.gather(reference_embeddings_flat, masked_indices)
    reference_labels_flat = tf.gather(reference_labels_flat, masked_indices)
    return (reference_embeddings_flat, reference_labels_flat)

def _create_subsampling_mask(args):
    if False:
        while True:
            i = 10
    'Creates boolean mask which can be used to subsample the labels.\n\n  Args:\n    args: tuple of (label_mask, max_neighbors_per_object), where label_mask\n      is the mask to be subsampled and max_neighbors_per_object is a int scalar,\n      the maximum number of neighbors to be retained after subsampling.\n\n  Returns:\n    The boolean mask for subsampling the labels.\n  '
    (label_mask, max_neighbors_per_object) = args
    indices = tf.squeeze(tf.where(label_mask), axis=1)
    shuffled_indices = tf.random_shuffle(indices)
    subsampled_indices = shuffled_indices[:max_neighbors_per_object]
    n_pad = max_neighbors_per_object - tf.size(subsampled_indices)
    padded_label = -1
    padding = tf.fill((n_pad,), tf.constant(padded_label, dtype=tf.int64))
    padded = tf.concat([subsampled_indices, padding], axis=0)
    return padded

def conv2d_identity_initializer(scale=1.0, mean=0, stddev=0.03):
    if False:
        for i in range(10):
            print('nop')
    'Creates an identity initializer for TensorFlow conv2d.\n\n  We add a small amount of normal noise to the initialization matrix.\n  Code copied from lcchen@.\n\n  Args:\n    scale: The scale coefficient for the identity weight matrix.\n    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the\n      truncated normal distribution.\n    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation\n      of the truncated normal distribution.\n\n  Returns:\n    An identity initializer function for TensorFlow conv2d.\n  '

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if False:
            i = 10
            return i + 15
        'Returns the identity matrix scaled by `scale`.\n\n    Args:\n      shape: A tuple of int32 numbers indicating the shape of the initializing\n         matrix.\n      dtype: The data type of the initializing matrix.\n      partition_info: (Optional) variable_scope._PartitionInfo object holding\n      additional information about how the variable is partitioned. This input\n        is not used in our case, but is required by TensorFlow.\n\n    Returns:\n      A identity matrix.\n\n    Raises:\n      ValueError: If len(shape) != 4, or shape[0] != shape[1], or shape[0] is\n        not odd, or shape[1] is not odd..\n    '
        del partition_info
        if len(shape) != 4:
            raise ValueError('Expect shape length to be 4.')
        if shape[0] != shape[1]:
            raise ValueError('Expect shape[0] = shape[1].')
        if shape[0] % 2 != 1:
            raise ValueError('Expect shape[0] to be odd value.')
        if shape[1] % 2 != 1:
            raise ValueError('Expect shape[1] to be odd value.')
        weights = np.zeros(shape, dtype=np.float32)
        center_y = shape[0] / 2
        center_x = shape[1] / 2
        min_channel = min(shape[2], shape[3])
        for i in range(min_channel):
            weights[center_y, center_x, i, i] = scale
        return tf.constant(weights, dtype=dtype) + tf.truncated_normal(shape, mean=mean, stddev=stddev, dtype=dtype)
    return _initializer

def split_separable_conv2d_with_identity_initializer(inputs, filters, kernel_size=3, rate=1, weight_decay=4e-05, scope=None):
    if False:
        print('Hello World!')
    'Splits a separable conv2d into depthwise and pointwise conv2d.\n\n  This operation differs from `tf.layers.separable_conv2d` as this operation\n  applies activation function between depthwise and pointwise conv2d.\n\n  Args:\n    inputs: Input tensor with shape [batch, height, width, channels].\n    filters: Number of filters in the 1x1 pointwise convolution.\n    kernel_size: A list of length 2: [kernel_height, kernel_width] of\n      of the filters. Can be an int if both values are the same.\n    rate: Atrous convolution rate for the depthwise convolution.\n    weight_decay: The weight decay to use for regularizing the model.\n    scope: Optional scope for the operation.\n\n  Returns:\n    Computed features after split separable conv2d.\n  '
    initializer = conv2d_identity_initializer()
    outputs = slim.separable_conv2d(inputs, None, kernel_size=kernel_size, depth_multiplier=1, rate=rate, weights_initializer=initializer, weights_regularizer=None, scope=scope + '_depthwise')
    return slim.conv2d(outputs, filters, 1, weights_initializer=initializer, weights_regularizer=slim.l2_regularizer(weight_decay), scope=scope + '_pointwise')

def create_initial_softmax_from_labels(last_frame_labels, reference_labels, decoder_output_stride, reduce_labels):
    if False:
        for i in range(10):
            print('nop')
    "Creates initial softmax predictions from last frame labels.\n\n  Args:\n    last_frame_labels: last frame labels of shape [1, height, width, 1].\n    reference_labels: reference frame labels of shape [1, height, width, 1].\n    decoder_output_stride: Integer, the stride of the decoder. Can be None, in\n      this case it's assumed that the last_frame_labels and reference_labels\n      are already scaled to the decoder output resolution.\n    reduce_labels: Boolean, whether to reduce the depth of the softmax one_hot\n      encoding to the actual number of labels present in the reference frame\n      (otherwise the depth will be the highest label index + 1).\n\n  Returns:\n    init_softmax: the initial softmax predictions.\n  "
    if decoder_output_stride is None:
        labels_output_size = last_frame_labels
        reference_labels_output_size = reference_labels
    else:
        h = tf.shape(last_frame_labels)[1]
        w = tf.shape(last_frame_labels)[2]
        h_sub = model.scale_dimension(h, 1.0 / decoder_output_stride)
        w_sub = model.scale_dimension(w, 1.0 / decoder_output_stride)
        labels_output_size = tf.image.resize_nearest_neighbor(last_frame_labels, [h_sub, w_sub], align_corners=True)
        reference_labels_output_size = tf.image.resize_nearest_neighbor(reference_labels, [h_sub, w_sub], align_corners=True)
    if reduce_labels:
        (unique_labels, _) = tf.unique(tf.reshape(reference_labels_output_size, [-1]))
        depth = tf.size(unique_labels)
    else:
        depth = tf.reduce_max(reference_labels_output_size) + 1
    one_hot_assertion = tf.assert_less(tf.reduce_max(labels_output_size), depth)
    with tf.control_dependencies([one_hot_assertion]):
        init_softmax = tf.one_hot(tf.squeeze(labels_output_size, axis=-1), depth=depth, dtype=tf.float32)
    return init_softmax

def local_previous_frame_nearest_neighbor_features_per_object(prev_frame_embedding, query_embedding, prev_frame_labels, gt_ids, max_distance=9):
    if False:
        return 10
    'Computes nearest neighbor features while only allowing local matches.\n\n  Args:\n    prev_frame_embedding: Tensor of shape [height, width, embedding_dim],\n      the embedding vectors for the last frame.\n    query_embedding: Tensor of shape [height, width, embedding_dim],\n      the embedding vectors for the query frames.\n    prev_frame_labels: Tensor of shape [height, width, 1], the class labels of\n      the previous frame.\n    gt_ids: Int Tensor of shape [n_objs] of the sorted unique ground truth\n      ids in the first frame.\n    max_distance: Integer, the maximum distance allowed for local matching.\n\n  Returns:\n    nn_features: A float32 np.array of nearest neighbor features of shape\n      [1, height, width, n_objects, 1].\n  '
    with tf.name_scope('local_previous_frame_nearest_neighbor_features_per_object'):
        if USE_CORRELATION_COST:
            tf.logging.info('Using correlation_cost.')
            d = local_pairwise_distances(query_embedding, prev_frame_embedding, max_distance=max_distance)
        else:
            tf.logging.warn('correlation cost is not available, using slow fallback implementation.')
            d = local_pairwise_distances2(query_embedding, prev_frame_embedding, max_distance=max_distance)
        d = (tf.nn.sigmoid(d) - 0.5) * 2
        height = tf.shape(prev_frame_embedding)[0]
        width = tf.shape(prev_frame_embedding)[1]
        if USE_CORRELATION_COST:
            offset_labels = correlation_cost_op.correlation_cost(tf.ones((1, height, width, 1)), tf.cast(prev_frame_labels + 1, tf.float32)[tf.newaxis], kernel_size=1, max_displacement=max_distance, stride_1=1, stride_2=1, pad=max_distance)
            offset_labels = tf.squeeze(offset_labels, axis=0)[..., tf.newaxis]
            offset_labels = tf.round(offset_labels - 1)
            offset_masks = tf.equal(offset_labels, tf.cast(gt_ids, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis, :])
        else:
            masks = tf.equal(prev_frame_labels, gt_ids[tf.newaxis, tf.newaxis])
            padded_masks = tf.pad(masks, [[max_distance, max_distance], [max_distance, max_distance], [0, 0]])
            offset_masks = []
            for y_start in range(2 * max_distance + 1):
                y_end = y_start + height
                masks_slice = padded_masks[y_start:y_end]
                for x_start in range(2 * max_distance + 1):
                    x_end = x_start + width
                    offset_mask = masks_slice[:, x_start:x_end]
                    offset_masks.append(offset_mask)
            offset_masks = tf.stack(offset_masks, axis=2)
        pad = tf.ones((height, width, (2 * max_distance + 1) ** 2, tf.size(gt_ids)))
        d_tiled = tf.tile(d[..., tf.newaxis], multiples=(1, 1, 1, tf.size(gt_ids)))
        d_masked = tf.where(offset_masks, d_tiled, pad)
        dists = tf.reduce_min(d_masked, axis=2)
        dists = tf.reshape(dists, (1, height, width, tf.size(gt_ids), 1))
        return dists