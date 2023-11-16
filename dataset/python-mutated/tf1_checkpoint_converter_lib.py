"""Convert checkpoints created by Estimator (tf1) to be Keras compatible."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf
BERT_NAME_REPLACEMENTS = (('bert', 'bert_model'), ('embeddings/word_embeddings', 'word_embeddings/embeddings'), ('embeddings/token_type_embeddings', 'embedding_postprocessor/type_embeddings'), ('embeddings/position_embeddings', 'embedding_postprocessor/position_embeddings'), ('embeddings/LayerNorm', 'embedding_postprocessor/layer_norm'), ('attention/self', 'self_attention'), ('attention/output/dense', 'self_attention_output'), ('attention/output/LayerNorm', 'self_attention_layer_norm'), ('intermediate/dense', 'intermediate'), ('output/dense', 'output'), ('output/LayerNorm', 'output_layer_norm'), ('pooler/dense', 'pooler_transform'))
BERT_V2_NAME_REPLACEMENTS = (('bert/', ''), ('encoder', 'transformer'), ('embeddings/word_embeddings', 'word_embeddings/embeddings'), ('embeddings/token_type_embeddings', 'type_embeddings/embeddings'), ('embeddings/position_embeddings', 'position_embedding/embeddings'), ('embeddings/LayerNorm', 'embeddings/layer_norm'), ('attention/self', 'self_attention'), ('attention/output/dense', 'self_attention_output'), ('attention/output/LayerNorm', 'self_attention_layer_norm'), ('intermediate/dense', 'intermediate'), ('output/dense', 'output'), ('output/LayerNorm', 'output_layer_norm'), ('pooler/dense', 'pooler_transform'), ('cls/predictions/output_bias', 'cls/predictions/output_bias/bias'), ('cls/seq_relationship/output_bias', 'predictions/transform/logits/bias'), ('cls/seq_relationship/output_weights', 'predictions/transform/logits/kernel'))
BERT_PERMUTATIONS = ()
BERT_V2_PERMUTATIONS = (('cls/seq_relationship/output_weights', (1, 0)),)

def _bert_name_replacement(var_name, name_replacements):
    if False:
        for i in range(10):
            print('nop')
    'Gets the variable name replacement.'
    for (src_pattern, tgt_pattern) in name_replacements:
        if src_pattern in var_name:
            old_var_name = var_name
            var_name = var_name.replace(src_pattern, tgt_pattern)
            tf.logging.info('Converted: %s --> %s', old_var_name, var_name)
    return var_name

def _has_exclude_patterns(name, exclude_patterns):
    if False:
        while True:
            i = 10
    'Checks if a string contains substrings that match patterns to exclude.'
    for p in exclude_patterns:
        if p in name:
            return True
    return False

def _get_permutation(name, permutations):
    if False:
        for i in range(10):
            print('nop')
    'Checks whether a variable requires transposition by pattern matching.'
    for (src_pattern, permutation) in permutations:
        if src_pattern in name:
            tf.logging.info('Permuted: %s --> %s', name, permutation)
            return permutation
    return None

def _get_new_shape(name, shape, num_heads):
    if False:
        i = 10
        return i + 15
    'Checks whether a variable requires reshape by pattern matching.'
    if 'attention/output/dense/kernel' in name:
        return tuple([num_heads, shape[0] // num_heads, shape[1]])
    if 'attention/output/dense/bias' in name:
        return shape
    patterns = ['attention/self/query', 'attention/self/value', 'attention/self/key']
    for pattern in patterns:
        if pattern in name:
            if 'kernel' in name:
                return tuple([shape[0], num_heads, shape[1] // num_heads])
            if 'bias' in name:
                return tuple([num_heads, shape[0] // num_heads])
    return None

def create_v2_checkpoint(model, src_checkpoint, output_path):
    if False:
        i = 10
        return i + 15
    'Converts a name-based matched TF V1 checkpoint to TF V2 checkpoint.'
    model.load_weights(src_checkpoint).assert_existing_objects_matched()
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.save(output_path)

def convert(checkpoint_from_path, checkpoint_to_path, num_heads, name_replacements, permutations, exclude_patterns=None):
    if False:
        for i in range(10):
            print('nop')
    'Migrates the names of variables within a checkpoint.\n\n  Args:\n    checkpoint_from_path: Path to source checkpoint to be read in.\n    checkpoint_to_path: Path to checkpoint to be written out.\n    num_heads: The number of heads of the model.\n    name_replacements: A list of tuples of the form (match_str, replace_str)\n      describing variable names to adjust.\n    permutations: A list of tuples of the form (match_str, permutation)\n      describing permutations to apply to given variables. Note that match_str\n      should match the original variable name, not the replaced one.\n    exclude_patterns: A list of string patterns to exclude variables from\n      checkpoint conversion.\n\n  Returns:\n    A dictionary that maps the new variable names to the Variable objects.\n    A dictionary that maps the old variable names to the new variable names.\n  '
    with tf.Graph().as_default():
        tf.logging.info('Reading checkpoint_from_path %s', checkpoint_from_path)
        reader = tf.train.NewCheckpointReader(checkpoint_from_path)
        name_shape_map = reader.get_variable_to_shape_map()
        new_variable_map = {}
        conversion_map = {}
        for var_name in name_shape_map:
            if exclude_patterns and _has_exclude_patterns(var_name, exclude_patterns):
                continue
            tensor = reader.get_tensor(var_name)
            new_var_name = _bert_name_replacement(var_name, name_replacements)
            new_shape = None
            if num_heads > 0:
                new_shape = _get_new_shape(var_name, tensor.shape, num_heads)
            if new_shape:
                tf.logging.info('Veriable %s has a shape change from %s to %s', var_name, tensor.shape, new_shape)
                tensor = np.reshape(tensor, new_shape)
            permutation = _get_permutation(var_name, permutations)
            if permutation:
                tensor = np.transpose(tensor, permutation)
            var = tf.Variable(tensor, name=var_name)
            new_variable_map[new_var_name] = var
            if new_var_name != var_name:
                conversion_map[var_name] = new_var_name
        saver = tf.train.Saver(new_variable_map)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.logging.info('Writing checkpoint_to_path %s', checkpoint_to_path)
            saver.save(sess, checkpoint_to_path)
    tf.logging.info('Summary:')
    tf.logging.info('  Converted %d variable name(s).', len(new_variable_map))
    tf.logging.info('  Converted: %s', str(conversion_map))