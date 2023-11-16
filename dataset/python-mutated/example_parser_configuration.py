"""Extract parse_example op configuration to a proto."""
from tensorflow.core.example import example_parser_configuration_pb2
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util

def extract_example_parser_configuration(parse_example_op, sess):
    if False:
        while True:
            i = 10
    'Returns an ExampleParserConfig proto.\n\n  Args:\n    parse_example_op: A ParseExample or ParseExampleV2 `Operation`\n    sess: A tf.compat.v1.Session needed to obtain some configuration values.\n  Returns:\n    A ExampleParserConfig proto.\n\n  Raises:\n    ValueError: If attributes are inconsistent.\n  '
    if parse_example_op.type == 'ParseExample':
        return _extract_from_parse_example(parse_example_op, sess)
    elif parse_example_op.type == 'ParseExampleV2':
        return _extract_from_parse_example_v2(parse_example_op, sess)
    else:
        raise ValueError(f'Found unexpected type when parsing example. Expected `ParseExample` object. Received type: {parse_example_op.type}')

def _extract_from_parse_example(parse_example_op, sess):
    if False:
        while True:
            i = 10
    'Extract ExampleParserConfig from ParseExample op.'
    config = example_parser_configuration_pb2.ExampleParserConfiguration()
    num_sparse = parse_example_op.get_attr('Nsparse')
    num_dense = parse_example_op.get_attr('Ndense')
    total_features = num_dense + num_sparse
    sparse_types = parse_example_op.get_attr('sparse_types')
    dense_types = parse_example_op.get_attr('Tdense')
    dense_shapes = parse_example_op.get_attr('dense_shapes')
    if len(sparse_types) != num_sparse:
        raise ValueError('len(sparse_types) attribute does not match Nsparse attribute (%d vs %d)' % (len(sparse_types), num_sparse))
    if len(dense_types) != num_dense:
        raise ValueError('len(dense_types) attribute does not match Ndense attribute (%d vs %d)' % (len(dense_types), num_dense))
    if len(dense_shapes) != num_dense:
        raise ValueError('len(dense_shapes) attribute does not match Ndense attribute (%d vs %d)' % (len(dense_shapes), num_dense))
    fetch_list = parse_example_op.inputs[2:]
    if len(fetch_list) != total_features + num_dense:
        raise ValueError('len(fetch_list) does not match total features + num_dense (%d vs %d)' % (len(fetch_list), total_features + num_dense))
    fetched = sess.run(fetch_list)
    if len(fetched) != len(fetch_list):
        raise ValueError('len(fetched) does not match len(fetch_list) (%d vs %d)' % (len(fetched), len(fetch_list)))
    sparse_keys_start = 0
    dense_keys_start = sparse_keys_start + num_sparse
    dense_def_start = dense_keys_start + num_dense
    sparse_indices_start = 0
    sparse_values_start = num_sparse
    sparse_shapes_start = sparse_values_start + num_sparse
    dense_values_start = sparse_shapes_start + num_sparse
    for i in range(num_dense):
        key = fetched[dense_keys_start + i]
        feature_config = config.feature_map[key]
        fixed_config = feature_config.fixed_len_feature
        fixed_config.default_value.CopyFrom(tensor_util.make_tensor_proto(fetched[dense_def_start + i]))
        fixed_config.shape.CopyFrom(tensor_shape.TensorShape(dense_shapes[i]).as_proto())
        fixed_config.dtype = dense_types[i].as_datatype_enum
        fixed_config.values_output_tensor_name = parse_example_op.outputs[dense_values_start + i].name
    for i in range(num_sparse):
        key = fetched[sparse_keys_start + i]
        feature_config = config.feature_map[key]
        var_len_feature = feature_config.var_len_feature
        var_len_feature.dtype = sparse_types[i].as_datatype_enum
        var_len_feature.indices_output_tensor_name = parse_example_op.outputs[sparse_indices_start + i].name
        var_len_feature.values_output_tensor_name = parse_example_op.outputs[sparse_values_start + i].name
        var_len_feature.shapes_output_tensor_name = parse_example_op.outputs[sparse_shapes_start + i].name
    return config

def _extract_from_parse_example_v2(parse_example_op, sess):
    if False:
        while True:
            i = 10
    'Extract ExampleParserConfig from ParseExampleV2 op.'
    config = example_parser_configuration_pb2.ExampleParserConfiguration()
    dense_types = parse_example_op.get_attr('Tdense')
    num_sparse = parse_example_op.get_attr('num_sparse')
    sparse_types = parse_example_op.get_attr('sparse_types')
    ragged_value_types = parse_example_op.get_attr('ragged_value_types')
    ragged_split_types = parse_example_op.get_attr('ragged_split_types')
    dense_shapes = parse_example_op.get_attr('dense_shapes')
    num_dense = len(dense_types)
    num_ragged = len(ragged_value_types)
    assert len(ragged_value_types) == len(ragged_split_types)
    assert len(parse_example_op.inputs) == 5 + num_dense
    fetched = sess.run(parse_example_op.inputs[2:])
    sparse_keys = fetched[0].tolist()
    dense_keys = fetched[1].tolist()
    ragged_keys = fetched[2].tolist()
    dense_defaults = fetched[3:]
    assert len(sparse_keys) == num_sparse
    assert len(dense_keys) == num_dense
    assert len(ragged_keys) == num_ragged
    sparse_indices_start = 0
    sparse_values_start = num_sparse
    sparse_shapes_start = sparse_values_start + num_sparse
    dense_values_start = sparse_shapes_start + num_sparse
    ragged_values_start = dense_values_start + num_dense
    ragged_row_splits_start = ragged_values_start + num_ragged
    for i in range(num_dense):
        key = dense_keys[i]
        feature_config = config.feature_map[key]
        fixed_config = feature_config.fixed_len_feature
        fixed_config.default_value.CopyFrom(tensor_util.make_tensor_proto(dense_defaults[i]))
        fixed_config.shape.CopyFrom(tensor_shape.TensorShape(dense_shapes[i]).as_proto())
        fixed_config.dtype = dense_types[i].as_datatype_enum
        fixed_config.values_output_tensor_name = parse_example_op.outputs[dense_values_start + i].name
    for i in range(num_sparse):
        key = sparse_keys[i]
        feature_config = config.feature_map[key]
        var_len_feature = feature_config.var_len_feature
        var_len_feature.dtype = sparse_types[i].as_datatype_enum
        var_len_feature.indices_output_tensor_name = parse_example_op.outputs[sparse_indices_start + i].name
        var_len_feature.values_output_tensor_name = parse_example_op.outputs[sparse_values_start + i].name
        var_len_feature.shapes_output_tensor_name = parse_example_op.outputs[sparse_shapes_start + i].name
    if num_ragged != 0:
        del ragged_values_start
        del ragged_row_splits_start
        raise ValueError('Ragged features are not yet supported by example_parser_configuration.proto')
    return config