"""Functions for reading and updating configuration files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.lib.io import file_io
from object_detection.protos import eval_pb2
from object_detection.protos import graph_rewriter_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import train_pb2

def get_image_resizer_config(model_config):
    if False:
        for i in range(10):
            print('nop')
    'Returns the image resizer config from a model config.\n\n  Args:\n    model_config: A model_pb2.DetectionModel.\n\n  Returns:\n    An image_resizer_pb2.ImageResizer.\n\n  Raises:\n    ValueError: If the model type is not recognized.\n  '
    meta_architecture = model_config.WhichOneof('model')
    if meta_architecture == 'faster_rcnn':
        return model_config.faster_rcnn.image_resizer
    if meta_architecture == 'ssd':
        return model_config.ssd.image_resizer
    raise ValueError('Unknown model type: {}'.format(meta_architecture))

def get_spatial_image_size(image_resizer_config):
    if False:
        i = 10
        return i + 15
    'Returns expected spatial size of the output image from a given config.\n\n  Args:\n    image_resizer_config: An image_resizer_pb2.ImageResizer.\n\n  Returns:\n    A list of two integers of the form [height, width]. `height` and `width` are\n    set  -1 if they cannot be determined during graph construction.\n\n  Raises:\n    ValueError: If the model type is not recognized.\n  '
    if image_resizer_config.HasField('fixed_shape_resizer'):
        return [image_resizer_config.fixed_shape_resizer.height, image_resizer_config.fixed_shape_resizer.width]
    if image_resizer_config.HasField('keep_aspect_ratio_resizer'):
        if image_resizer_config.keep_aspect_ratio_resizer.pad_to_max_dimension:
            return [image_resizer_config.keep_aspect_ratio_resizer.max_dimension] * 2
        else:
            return [-1, -1]
    if image_resizer_config.HasField('identity_resizer') or image_resizer_config.HasField('conditional_shape_resizer'):
        return [-1, -1]
    raise ValueError('Unknown image resizer type.')

def get_configs_from_pipeline_file(pipeline_config_path, config_override=None):
    if False:
        for i in range(10):
            print('nop')
    'Reads config from a file containing pipeline_pb2.TrainEvalPipelineConfig.\n\n  Args:\n    pipeline_config_path: Path to pipeline_pb2.TrainEvalPipelineConfig text\n      proto.\n    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to\n      override pipeline_config_path.\n\n  Returns:\n    Dictionary of configuration objects. Keys are `model`, `train_config`,\n      `train_input_config`, `eval_config`, `eval_input_config`. Value are the\n      corresponding config objects.\n  '
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    if config_override:
        text_format.Merge(config_override, pipeline_config)
    return create_configs_from_pipeline_proto(pipeline_config)

def create_configs_from_pipeline_proto(pipeline_config):
    if False:
        print('Hello World!')
    'Creates a configs dictionary from pipeline_pb2.TrainEvalPipelineConfig.\n\n  Args:\n    pipeline_config: pipeline_pb2.TrainEvalPipelineConfig proto object.\n\n  Returns:\n    Dictionary of configuration objects. Keys are `model`, `train_config`,\n      `train_input_config`, `eval_config`, `eval_input_configs`. Value are\n      the corresponding config objects or list of config objects (only for\n      eval_input_configs).\n  '
    configs = {}
    configs['model'] = pipeline_config.model
    configs['train_config'] = pipeline_config.train_config
    configs['train_input_config'] = pipeline_config.train_input_reader
    configs['eval_config'] = pipeline_config.eval_config
    configs['eval_input_configs'] = pipeline_config.eval_input_reader
    if configs['eval_input_configs']:
        configs['eval_input_config'] = configs['eval_input_configs'][0]
    if pipeline_config.HasField('graph_rewriter'):
        configs['graph_rewriter_config'] = pipeline_config.graph_rewriter
    return configs

def get_graph_rewriter_config_from_file(graph_rewriter_config_file):
    if False:
        i = 10
        return i + 15
    'Parses config for graph rewriter.\n\n  Args:\n    graph_rewriter_config_file: file path to the graph rewriter config.\n\n  Returns:\n    graph_rewriter_pb2.GraphRewriter proto\n  '
    graph_rewriter_config = graph_rewriter_pb2.GraphRewriter()
    with tf.gfile.GFile(graph_rewriter_config_file, 'r') as f:
        text_format.Merge(f.read(), graph_rewriter_config)
    return graph_rewriter_config

def create_pipeline_proto_from_configs(configs):
    if False:
        return 10
    'Creates a pipeline_pb2.TrainEvalPipelineConfig from configs dictionary.\n\n  This function performs the inverse operation of\n  create_configs_from_pipeline_proto().\n\n  Args:\n    configs: Dictionary of configs. See get_configs_from_pipeline_file().\n\n  Returns:\n    A fully populated pipeline_pb2.TrainEvalPipelineConfig.\n  '
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.CopyFrom(configs['model'])
    pipeline_config.train_config.CopyFrom(configs['train_config'])
    pipeline_config.train_input_reader.CopyFrom(configs['train_input_config'])
    pipeline_config.eval_config.CopyFrom(configs['eval_config'])
    pipeline_config.eval_input_reader.extend(configs['eval_input_configs'])
    if 'graph_rewriter_config' in configs:
        pipeline_config.graph_rewriter.CopyFrom(configs['graph_rewriter_config'])
    return pipeline_config

def save_pipeline_config(pipeline_config, directory):
    if False:
        return 10
    'Saves a pipeline config text file to disk.\n\n  Args:\n    pipeline_config: A pipeline_pb2.TrainEvalPipelineConfig.\n    directory: The model directory into which the pipeline config file will be\n      saved.\n  '
    if not file_io.file_exists(directory):
        file_io.recursive_create_dir(directory)
    pipeline_config_path = os.path.join(directory, 'pipeline.config')
    config_text = text_format.MessageToString(pipeline_config)
    with tf.gfile.Open(pipeline_config_path, 'wb') as f:
        tf.logging.info('Writing pipeline config file to %s', pipeline_config_path)
        f.write(config_text)

def get_configs_from_multiple_files(model_config_path='', train_config_path='', train_input_config_path='', eval_config_path='', eval_input_config_path='', graph_rewriter_config_path=''):
    if False:
        while True:
            i = 10
    'Reads training configuration from multiple config files.\n\n  Args:\n    model_config_path: Path to model_pb2.DetectionModel.\n    train_config_path: Path to train_pb2.TrainConfig.\n    train_input_config_path: Path to input_reader_pb2.InputReader.\n    eval_config_path: Path to eval_pb2.EvalConfig.\n    eval_input_config_path: Path to input_reader_pb2.InputReader.\n    graph_rewriter_config_path: Path to graph_rewriter_pb2.GraphRewriter.\n\n  Returns:\n    Dictionary of configuration objects. Keys are `model`, `train_config`,\n      `train_input_config`, `eval_config`, `eval_input_config`. Key/Values are\n        returned only for valid (non-empty) strings.\n  '
    configs = {}
    if model_config_path:
        model_config = model_pb2.DetectionModel()
        with tf.gfile.GFile(model_config_path, 'r') as f:
            text_format.Merge(f.read(), model_config)
            configs['model'] = model_config
    if train_config_path:
        train_config = train_pb2.TrainConfig()
        with tf.gfile.GFile(train_config_path, 'r') as f:
            text_format.Merge(f.read(), train_config)
            configs['train_config'] = train_config
    if train_input_config_path:
        train_input_config = input_reader_pb2.InputReader()
        with tf.gfile.GFile(train_input_config_path, 'r') as f:
            text_format.Merge(f.read(), train_input_config)
            configs['train_input_config'] = train_input_config
    if eval_config_path:
        eval_config = eval_pb2.EvalConfig()
        with tf.gfile.GFile(eval_config_path, 'r') as f:
            text_format.Merge(f.read(), eval_config)
            configs['eval_config'] = eval_config
    if eval_input_config_path:
        eval_input_config = input_reader_pb2.InputReader()
        with tf.gfile.GFile(eval_input_config_path, 'r') as f:
            text_format.Merge(f.read(), eval_input_config)
            configs['eval_input_configs'] = [eval_input_config]
    if graph_rewriter_config_path:
        configs['graph_rewriter_config'] = get_graph_rewriter_config_from_file(graph_rewriter_config_path)
    return configs

def get_number_of_classes(model_config):
    if False:
        for i in range(10):
            print('nop')
    'Returns the number of classes for a detection model.\n\n  Args:\n    model_config: A model_pb2.DetectionModel.\n\n  Returns:\n    Number of classes.\n\n  Raises:\n    ValueError: If the model type is not recognized.\n  '
    meta_architecture = model_config.WhichOneof('model')
    if meta_architecture == 'faster_rcnn':
        return model_config.faster_rcnn.num_classes
    if meta_architecture == 'ssd':
        return model_config.ssd.num_classes
    raise ValueError("Expected the model to be one of 'faster_rcnn' or 'ssd'.")

def get_optimizer_type(train_config):
    if False:
        for i in range(10):
            print('nop')
    'Returns the optimizer type for training.\n\n  Args:\n    train_config: A train_pb2.TrainConfig.\n\n  Returns:\n    The type of the optimizer\n  '
    return train_config.optimizer.WhichOneof('optimizer')

def get_learning_rate_type(optimizer_config):
    if False:
        i = 10
        return i + 15
    'Returns the learning rate type for training.\n\n  Args:\n    optimizer_config: An optimizer_pb2.Optimizer.\n\n  Returns:\n    The type of the learning rate.\n  '
    return optimizer_config.learning_rate.WhichOneof('learning_rate')

def _is_generic_key(key):
    if False:
        i = 10
        return i + 15
    'Determines whether the key starts with a generic config dictionary key.'
    for prefix in ['graph_rewriter_config', 'model', 'train_input_config', 'train_config', 'eval_config']:
        if key.startswith(prefix + '.'):
            return True
    return False

def _check_and_convert_legacy_input_config_key(key):
    if False:
        for i in range(10):
            print('nop')
    "Checks key and converts legacy input config update to specific update.\n\n  Args:\n    key: string indicates the target of update operation.\n\n  Returns:\n    is_valid_input_config_key: A boolean indicating whether the input key is to\n      update input config(s).\n    key_name: 'eval_input_configs' or 'train_input_config' string if\n      is_valid_input_config_key is true. None if is_valid_input_config_key is\n      false.\n    input_name: always returns None since legacy input config key never\n      specifies the target input config. Keeping this output only to match the\n      output form defined for input config update.\n    field_name: the field name in input config. `key` itself if\n      is_valid_input_config_key is false.\n  "
    key_name = None
    input_name = None
    field_name = key
    is_valid_input_config_key = True
    if field_name == 'train_shuffle':
        key_name = 'train_input_config'
        field_name = 'shuffle'
    elif field_name == 'eval_shuffle':
        key_name = 'eval_input_configs'
        field_name = 'shuffle'
    elif field_name == 'train_input_path':
        key_name = 'train_input_config'
        field_name = 'input_path'
    elif field_name == 'eval_input_path':
        key_name = 'eval_input_configs'
        field_name = 'input_path'
    elif field_name == 'append_train_input_path':
        key_name = 'train_input_config'
        field_name = 'input_path'
    elif field_name == 'append_eval_input_path':
        key_name = 'eval_input_configs'
        field_name = 'input_path'
    else:
        is_valid_input_config_key = False
    return (is_valid_input_config_key, key_name, input_name, field_name)

def check_and_parse_input_config_key(configs, key):
    if False:
        while True:
            i = 10
    "Checks key and returns specific fields if key is valid input config update.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    key: string indicates the target of update operation.\n\n  Returns:\n    is_valid_input_config_key: A boolean indicate whether the input key is to\n      update input config(s).\n    key_name: 'eval_input_configs' or 'train_input_config' string if\n      is_valid_input_config_key is true. None if is_valid_input_config_key is\n      false.\n    input_name: the name of the input config to be updated. None if\n      is_valid_input_config_key is false.\n    field_name: the field name in input config. `key` itself if\n      is_valid_input_config_key is false.\n\n  Raises:\n    ValueError: when the input key format doesn't match any known formats.\n    ValueError: if key_name doesn't match 'eval_input_configs' or\n      'train_input_config'.\n    ValueError: if input_name doesn't match any name in train or eval input\n      configs.\n    ValueError: if field_name doesn't match any supported fields.\n  "
    key_name = None
    input_name = None
    field_name = None
    fields = key.split(':')
    if len(fields) == 1:
        field_name = key
        return _check_and_convert_legacy_input_config_key(key)
    elif len(fields) == 3:
        key_name = fields[0]
        input_name = fields[1]
        field_name = fields[2]
    else:
        raise ValueError('Invalid key format when overriding configs.')
    if key_name not in ['eval_input_configs', 'train_input_config']:
        raise ValueError('Invalid key_name when overriding input config.')
    if isinstance(configs[key_name], input_reader_pb2.InputReader):
        is_valid_input_name = configs[key_name].name == input_name
    else:
        is_valid_input_name = input_name in [eval_input_config.name for eval_input_config in configs[key_name]]
    if not is_valid_input_name:
        raise ValueError('Invalid input_name when overriding input config.')
    if field_name not in ['input_path', 'label_map_path', 'shuffle', 'mask_type', 'sample_1_of_n_examples']:
        raise ValueError('Invalid field_name when overriding input config.')
    return (True, key_name, input_name, field_name)

def merge_external_params_with_configs(configs, hparams=None, kwargs_dict=None):
    if False:
        while True:
            i = 10
    'Updates `configs` dictionary based on supplied parameters.\n\n  This utility is for modifying specific fields in the object detection configs.\n  Say that one would like to experiment with different learning rates, momentum\n  values, or batch sizes. Rather than creating a new config text file for each\n  experiment, one can use a single base config file, and update particular\n  values.\n\n  There are two types of field overrides:\n  1. Strategy-based overrides, which update multiple relevant configuration\n  options. For example, updating `learning_rate` will update both the warmup and\n  final learning rates.\n  In this case key can be one of the following formats:\n      1. legacy update: single string that indicates the attribute to be\n        updated. E.g. \'label_map_path\', \'eval_input_path\', \'shuffle\'.\n        Note that when updating fields (e.g. eval_input_path, eval_shuffle) in\n        eval_input_configs, the override will only be applied when\n        eval_input_configs has exactly 1 element.\n      2. specific update: colon separated string that indicates which field in\n        which input_config to update. It should have 3 fields:\n        - key_name: Name of the input config we should update, either\n          \'train_input_config\' or \'eval_input_configs\'\n        - input_name: a \'name\' that can be used to identify elements, especially\n          when configs[key_name] is a repeated field.\n        - field_name: name of the field that you want to override.\n        For example, given configs dict as below:\n          configs = {\n            \'model\': {...}\n            \'train_config\': {...}\n            \'train_input_config\': {...}\n            \'eval_config\': {...}\n            \'eval_input_configs\': [{ name:"eval_coco", ...},\n                                   { name:"eval_voc", ... }]\n          }\n        Assume we want to update the input_path of the eval_input_config\n        whose name is \'eval_coco\'. The `key` would then be:\n        \'eval_input_configs:eval_coco:input_path\'\n  2. Generic key/value, which update a specific parameter based on namespaced\n  configuration keys. For example,\n  `model.ssd.loss.hard_example_miner.max_negatives_per_positive` will update the\n  hard example miner configuration for an SSD model config. Generic overrides\n  are automatically detected based on the namespaced keys.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    hparams: A `HParams`.\n    kwargs_dict: Extra keyword arguments that are treated the same way as\n      attribute/value pairs in `hparams`. Note that hyperparameters with the\n      same names will override keyword arguments.\n\n  Returns:\n    `configs` dictionary.\n\n  Raises:\n    ValueError: when the key string doesn\'t match any of its allowed formats.\n  '
    if kwargs_dict is None:
        kwargs_dict = {}
    if hparams:
        kwargs_dict.update(hparams.values())
    for (key, value) in kwargs_dict.items():
        tf.logging.info('Maybe overwriting %s: %s', key, value)
        if value == '' or value is None:
            continue
        elif _maybe_update_config_with_key_value(configs, key, value):
            continue
        elif _is_generic_key(key):
            _update_generic(configs, key, value)
        else:
            tf.logging.info('Ignoring config override key: %s', key)
    return configs

def _maybe_update_config_with_key_value(configs, key, value):
    if False:
        return 10
    "Checks key type and updates `configs` with the key value pair accordingly.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    key: String indicates the field(s) to be updated.\n    value: Value used to override existing field value.\n\n  Returns:\n    A boolean value that indicates whether the override succeeds.\n\n  Raises:\n    ValueError: when the key string doesn't match any of the formats above.\n  "
    (is_valid_input_config_key, key_name, input_name, field_name) = check_and_parse_input_config_key(configs, key)
    if is_valid_input_config_key:
        update_input_reader_config(configs, key_name=key_name, input_name=input_name, field_name=field_name, value=value)
    elif field_name == 'learning_rate':
        _update_initial_learning_rate(configs, value)
    elif field_name == 'batch_size':
        _update_batch_size(configs, value)
    elif field_name == 'momentum_optimizer_value':
        _update_momentum_optimizer_value(configs, value)
    elif field_name == 'classification_localization_weight_ratio':
        _update_classification_localization_weight_ratio(configs, value)
    elif field_name == 'focal_loss_gamma':
        _update_focal_loss_gamma(configs, value)
    elif field_name == 'focal_loss_alpha':
        _update_focal_loss_alpha(configs, value)
    elif field_name == 'train_steps':
        _update_train_steps(configs, value)
    elif field_name == 'label_map_path':
        _update_label_map_path(configs, value)
    elif field_name == 'mask_type':
        _update_mask_type(configs, value)
    elif field_name == 'sample_1_of_n_eval_examples':
        _update_all_eval_input_configs(configs, 'sample_1_of_n_examples', value)
    elif field_name == 'eval_num_epochs':
        _update_all_eval_input_configs(configs, 'num_epochs', value)
    elif field_name == 'eval_with_moving_averages':
        _update_use_moving_averages(configs, value)
    elif field_name == 'retain_original_images_in_eval':
        _update_retain_original_images(configs['eval_config'], value)
    elif field_name == 'use_bfloat16':
        _update_use_bfloat16(configs, value)
    elif field_name == 'retain_original_image_additional_channels_in_eval':
        _update_retain_original_image_additional_channels(configs['eval_config'], value)
    else:
        return False
    return True

def _update_tf_record_input_path(input_config, input_path):
    if False:
        print('Hello World!')
    'Updates input configuration to reflect a new input path.\n\n  The input_config object is updated in place, and hence not returned.\n\n  Args:\n    input_config: A input_reader_pb2.InputReader.\n    input_path: A path to data or list of paths.\n\n  Raises:\n    TypeError: if input reader type is not `tf_record_input_reader`.\n  '
    input_reader_type = input_config.WhichOneof('input_reader')
    if input_reader_type == 'tf_record_input_reader':
        input_config.tf_record_input_reader.ClearField('input_path')
        if isinstance(input_path, list):
            input_config.tf_record_input_reader.input_path.extend(input_path)
        else:
            input_config.tf_record_input_reader.input_path.append(input_path)
    else:
        raise TypeError('Input reader type must be `tf_record_input_reader`.')

def update_input_reader_config(configs, key_name=None, input_name=None, field_name=None, value=None, path_updater=_update_tf_record_input_path):
    if False:
        return 10
    'Updates specified input reader config field.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    key_name: Name of the input config we should update, either\n      \'train_input_config\' or \'eval_input_configs\'\n    input_name: String name used to identify input config to update with. Should\n      be either None or value of the \'name\' field in one of the input reader\n      configs.\n    field_name: Field name in input_reader_pb2.InputReader.\n    value: Value used to override existing field value.\n    path_updater: helper function used to update the input path. Only used when\n      field_name is "input_path".\n\n  Raises:\n    ValueError: when input field_name is None.\n    ValueError: when input_name is None and number of eval_input_readers does\n      not equal to 1.\n  '
    if isinstance(configs[key_name], input_reader_pb2.InputReader):
        target_input_config = configs[key_name]
        if field_name == 'input_path':
            path_updater(input_config=target_input_config, input_path=value)
        else:
            setattr(target_input_config, field_name, value)
    elif input_name is None and len(configs[key_name]) == 1:
        target_input_config = configs[key_name][0]
        if field_name == 'input_path':
            path_updater(input_config=target_input_config, input_path=value)
        else:
            setattr(target_input_config, field_name, value)
    elif input_name is not None and len(configs[key_name]):
        update_count = 0
        for input_config in configs[key_name]:
            if input_config.name == input_name:
                setattr(input_config, field_name, value)
                update_count = update_count + 1
        if not update_count:
            raise ValueError('Input name {} not found when overriding.'.format(input_name))
        elif update_count > 1:
            raise ValueError('Duplicate input name found when overriding.')
    else:
        key_name = 'None' if key_name is None else key_name
        input_name = 'None' if input_name is None else input_name
        field_name = 'None' if field_name is None else field_name
        raise ValueError('Unknown input config overriding: key_name:{}, input_name:{}, field_name:{}.'.format(key_name, input_name, field_name))

def _update_initial_learning_rate(configs, learning_rate):
    if False:
        for i in range(10):
            print('nop')
    'Updates `configs` to reflect the new initial learning rate.\n\n  This function updates the initial learning rate. For learning rate schedules,\n  all other defined learning rates in the pipeline config are scaled to maintain\n  their same ratio with the initial learning rate.\n  The configs dictionary is updated in place, and hence not returned.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    learning_rate: Initial learning rate for optimizer.\n\n  Raises:\n    TypeError: if optimizer type is not supported, or if learning rate type is\n      not supported.\n  '
    optimizer_type = get_optimizer_type(configs['train_config'])
    if optimizer_type == 'rms_prop_optimizer':
        optimizer_config = configs['train_config'].optimizer.rms_prop_optimizer
    elif optimizer_type == 'momentum_optimizer':
        optimizer_config = configs['train_config'].optimizer.momentum_optimizer
    elif optimizer_type == 'adam_optimizer':
        optimizer_config = configs['train_config'].optimizer.adam_optimizer
    else:
        raise TypeError('Optimizer %s is not supported.' % optimizer_type)
    learning_rate_type = get_learning_rate_type(optimizer_config)
    if learning_rate_type == 'constant_learning_rate':
        constant_lr = optimizer_config.learning_rate.constant_learning_rate
        constant_lr.learning_rate = learning_rate
    elif learning_rate_type == 'exponential_decay_learning_rate':
        exponential_lr = optimizer_config.learning_rate.exponential_decay_learning_rate
        exponential_lr.initial_learning_rate = learning_rate
    elif learning_rate_type == 'manual_step_learning_rate':
        manual_lr = optimizer_config.learning_rate.manual_step_learning_rate
        original_learning_rate = manual_lr.initial_learning_rate
        learning_rate_scaling = float(learning_rate) / original_learning_rate
        manual_lr.initial_learning_rate = learning_rate
        for schedule in manual_lr.schedule:
            schedule.learning_rate *= learning_rate_scaling
    elif learning_rate_type == 'cosine_decay_learning_rate':
        cosine_lr = optimizer_config.learning_rate.cosine_decay_learning_rate
        learning_rate_base = cosine_lr.learning_rate_base
        warmup_learning_rate = cosine_lr.warmup_learning_rate
        warmup_scale_factor = warmup_learning_rate / learning_rate_base
        cosine_lr.learning_rate_base = learning_rate
        cosine_lr.warmup_learning_rate = warmup_scale_factor * learning_rate
    else:
        raise TypeError('Learning rate %s is not supported.' % learning_rate_type)

def _update_batch_size(configs, batch_size):
    if False:
        for i in range(10):
            print('nop')
    'Updates `configs` to reflect the new training batch size.\n\n  The configs dictionary is updated in place, and hence not returned.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    batch_size: Batch size to use for training (Ideally a power of 2). Inputs\n      are rounded, and capped to be 1 or greater.\n  '
    configs['train_config'].batch_size = max(1, int(round(batch_size)))

def _validate_message_has_field(message, field):
    if False:
        for i in range(10):
            print('nop')
    if not message.HasField(field):
        raise ValueError('Expecting message to have field %s' % field)

def _update_generic(configs, key, value):
    if False:
        return 10
    'Update a pipeline configuration parameter based on a generic key/value.\n\n  Args:\n    configs: Dictionary of pipeline configuration protos.\n    key: A string key, dot-delimited to represent the argument key.\n      e.g. "model.ssd.train_config.batch_size"\n    value: A value to set the argument to. The type of the value must match the\n      type for the protocol buffer. Note that setting the wrong type will\n      result in a TypeError.\n      e.g. 42\n\n  Raises:\n    ValueError if the message key does not match the existing proto fields.\n    TypeError the value type doesn\'t match the protobuf field type.\n  '
    fields = key.split('.')
    first_field = fields.pop(0)
    last_field = fields.pop()
    message = configs[first_field]
    for field in fields:
        _validate_message_has_field(message, field)
        message = getattr(message, field)
    _validate_message_has_field(message, last_field)
    setattr(message, last_field, value)

def _update_momentum_optimizer_value(configs, momentum):
    if False:
        while True:
            i = 10
    'Updates `configs` to reflect the new momentum value.\n\n  Momentum is only supported for RMSPropOptimizer and MomentumOptimizer. For any\n  other optimizer, no changes take place. The configs dictionary is updated in\n  place, and hence not returned.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    momentum: New momentum value. Values are clipped at 0.0 and 1.0.\n\n  Raises:\n    TypeError: If the optimizer type is not `rms_prop_optimizer` or\n    `momentum_optimizer`.\n  '
    optimizer_type = get_optimizer_type(configs['train_config'])
    if optimizer_type == 'rms_prop_optimizer':
        optimizer_config = configs['train_config'].optimizer.rms_prop_optimizer
    elif optimizer_type == 'momentum_optimizer':
        optimizer_config = configs['train_config'].optimizer.momentum_optimizer
    else:
        raise TypeError('Optimizer type must be one of `rms_prop_optimizer` or `momentum_optimizer`.')
    optimizer_config.momentum_optimizer_value = min(max(0.0, momentum), 1.0)

def _update_classification_localization_weight_ratio(configs, ratio):
    if False:
        print('Hello World!')
    'Updates the classification/localization weight loss ratio.\n\n  Detection models usually define a loss weight for both classification and\n  objectness. This function updates the weights such that the ratio between\n  classification weight to localization weight is the ratio provided.\n  Arbitrarily, localization weight is set to 1.0.\n\n  Note that in the case of Faster R-CNN, this same ratio is applied to the first\n  stage objectness loss weight relative to localization loss weight.\n\n  The configs dictionary is updated in place, and hence not returned.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    ratio: Desired ratio of classification (and/or objectness) loss weight to\n      localization loss weight.\n  '
    meta_architecture = configs['model'].WhichOneof('model')
    if meta_architecture == 'faster_rcnn':
        model = configs['model'].faster_rcnn
        model.first_stage_localization_loss_weight = 1.0
        model.first_stage_objectness_loss_weight = ratio
        model.second_stage_localization_loss_weight = 1.0
        model.second_stage_classification_loss_weight = ratio
    if meta_architecture == 'ssd':
        model = configs['model'].ssd
        model.loss.localization_weight = 1.0
        model.loss.classification_weight = ratio

def _get_classification_loss(model_config):
    if False:
        while True:
            i = 10
    'Returns the classification loss for a model.'
    meta_architecture = model_config.WhichOneof('model')
    if meta_architecture == 'faster_rcnn':
        model = model_config.faster_rcnn
        classification_loss = model.second_stage_classification_loss
    elif meta_architecture == 'ssd':
        model = model_config.ssd
        classification_loss = model.loss.classification_loss
    else:
        raise TypeError('Did not recognize the model architecture.')
    return classification_loss

def _update_focal_loss_gamma(configs, gamma):
    if False:
        i = 10
        return i + 15
    'Updates the gamma value for a sigmoid focal loss.\n\n  The configs dictionary is updated in place, and hence not returned.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    gamma: Exponent term in focal loss.\n\n  Raises:\n    TypeError: If the classification loss is not `weighted_sigmoid_focal`.\n  '
    classification_loss = _get_classification_loss(configs['model'])
    classification_loss_type = classification_loss.WhichOneof('classification_loss')
    if classification_loss_type != 'weighted_sigmoid_focal':
        raise TypeError('Classification loss must be `weighted_sigmoid_focal`.')
    classification_loss.weighted_sigmoid_focal.gamma = gamma

def _update_focal_loss_alpha(configs, alpha):
    if False:
        return 10
    'Updates the alpha value for a sigmoid focal loss.\n\n  The configs dictionary is updated in place, and hence not returned.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    alpha: Class weight multiplier for sigmoid loss.\n\n  Raises:\n    TypeError: If the classification loss is not `weighted_sigmoid_focal`.\n  '
    classification_loss = _get_classification_loss(configs['model'])
    classification_loss_type = classification_loss.WhichOneof('classification_loss')
    if classification_loss_type != 'weighted_sigmoid_focal':
        raise TypeError('Classification loss must be `weighted_sigmoid_focal`.')
    classification_loss.weighted_sigmoid_focal.alpha = alpha

def _update_train_steps(configs, train_steps):
    if False:
        i = 10
        return i + 15
    'Updates `configs` to reflect new number of training steps.'
    configs['train_config'].num_steps = int(train_steps)

def _update_all_eval_input_configs(configs, field, value):
    if False:
        print('Hello World!')
    'Updates the content of `field` with `value` for all eval input configs.'
    for eval_input_config in configs['eval_input_configs']:
        setattr(eval_input_config, field, value)

def _update_label_map_path(configs, label_map_path):
    if False:
        print('Hello World!')
    'Updates the label map path for both train and eval input readers.\n\n  The configs dictionary is updated in place, and hence not returned.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    label_map_path: New path to `StringIntLabelMap` pbtxt file.\n  '
    configs['train_input_config'].label_map_path = label_map_path
    _update_all_eval_input_configs(configs, 'label_map_path', label_map_path)

def _update_mask_type(configs, mask_type):
    if False:
        while True:
            i = 10
    'Updates the mask type for both train and eval input readers.\n\n  The configs dictionary is updated in place, and hence not returned.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    mask_type: A string name representing a value of\n      input_reader_pb2.InstanceMaskType\n  '
    configs['train_input_config'].mask_type = mask_type
    _update_all_eval_input_configs(configs, 'mask_type', mask_type)

def _update_use_moving_averages(configs, use_moving_averages):
    if False:
        return 10
    'Updates the eval config option to use or not use moving averages.\n\n  The configs dictionary is updated in place, and hence not returned.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    use_moving_averages: Boolean indicating whether moving average variables\n      should be loaded during evaluation.\n  '
    configs['eval_config'].use_moving_averages = use_moving_averages

def _update_retain_original_images(eval_config, retain_original_images):
    if False:
        print('Hello World!')
    'Updates eval config with option to retain original images.\n\n  The eval_config object is updated in place, and hence not returned.\n\n  Args:\n    eval_config: A eval_pb2.EvalConfig.\n    retain_original_images: Boolean indicating whether to retain original images\n      in eval mode.\n  '
    eval_config.retain_original_images = retain_original_images

def _update_use_bfloat16(configs, use_bfloat16):
    if False:
        print('Hello World!')
    'Updates `configs` to reflect the new setup on whether to use bfloat16.\n\n  The configs dictionary is updated in place, and hence not returned.\n\n  Args:\n    configs: Dictionary of configuration objects. See outputs from\n      get_configs_from_pipeline_file() or get_configs_from_multiple_files().\n    use_bfloat16: A bool, indicating whether to use bfloat16 for training.\n  '
    configs['train_config'].use_bfloat16 = use_bfloat16

def _update_retain_original_image_additional_channels(eval_config, retain_original_image_additional_channels):
    if False:
        print('Hello World!')
    'Updates eval config to retain original image additional channels or not.\n\n  The eval_config object is updated in place, and hence not returned.\n\n  Args:\n    eval_config: A eval_pb2.EvalConfig.\n    retain_original_image_additional_channels: Boolean indicating whether to\n      retain original image additional channels in eval mode.\n  '
    eval_config.retain_original_image_additional_channels = retain_original_image_additional_channels

def remove_unecessary_ema(variables_to_restore, no_ema_collection=None):
    if False:
        for i in range(10):
            print('nop')
    "Remap and Remove EMA variable that are not created during training.\n\n  ExponentialMovingAverage.variables_to_restore() returns a map of EMA names\n  to tf variables to restore. E.g.:\n  {\n      conv/batchnorm/gamma/ExponentialMovingAverage: conv/batchnorm/gamma,\n      conv_4/conv2d_params/ExponentialMovingAverage: conv_4/conv2d_params,\n      global_step: global_step\n  }\n  This function takes care of the extra ExponentialMovingAverage variables\n  that get created during eval but aren't available in the checkpoint, by\n  remapping the key to the shallow copy of the variable itself, and remove\n  the entry of its EMA from the variables to restore. An example resulting\n  dictionary would look like:\n  {\n      conv/batchnorm/gamma: conv/batchnorm/gamma,\n      conv_4/conv2d_params: conv_4/conv2d_params,\n      global_step: global_step\n  }\n  Args:\n    variables_to_restore: A dictionary created by ExponentialMovingAverage.\n      variables_to_restore().\n    no_ema_collection: A list of namescope substrings to match the variables\n      to eliminate EMA.\n\n  Returns:\n    A variables_to_restore dictionary excluding the collection of unwanted\n    EMA mapping.\n  "
    if no_ema_collection is None:
        return variables_to_restore
    for key in variables_to_restore:
        if 'ExponentialMovingAverage' in key:
            for name in no_ema_collection:
                if name in key:
                    variables_to_restore[key.replace('/ExponentialMovingAverage', '')] = variables_to_restore[key]
                    del variables_to_restore[key]
    return variables_to_restore