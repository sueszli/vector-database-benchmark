""" PyTorch - TF 2.0 general utilities."""
import os
import re
import numpy
from .utils import ExplicitEnum, expand_dims, is_numpy_array, is_torch_tensor, logging, reshape, squeeze, tensor_size
from .utils import transpose as transpose_func
logger = logging.get_logger(__name__)

class TransposeType(ExplicitEnum):
    """
    Possible ...
    """
    NO = 'no'
    SIMPLE = 'simple'
    CONV1D = 'conv1d'
    CONV2D = 'conv2d'

def convert_tf_weight_name_to_pt_weight_name(tf_name, start_prefix_to_remove='', tf_weight_shape=None, name_scope=None):
    if False:
        print('Hello World!')
    "\n    Convert a TF 2.0 model variable name in a pytorch model weight name.\n\n    Conventions for TF2.0 scopes -> PyTorch attribute names conversions:\n\n        - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)\n        - '_._' is replaced by a new level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)\n\n    return tuple with:\n\n        - pytorch model weight name\n        - transpose: `TransposeType` member indicating whether and how TF2.0 and PyTorch weights matrices should be\n          transposed with regards to each other\n    "
    if name_scope is not None:
        if not tf_name.startswith(name_scope):
            raise ValueError(f'Weight name {tf_name} does not start with name_scope {name_scope}. This is an internal error in Transformers, so (unless you were doing something really evil) please open an issue to report it!')
        tf_name = tf_name[len(name_scope):]
        tf_name = tf_name.lstrip('/')
    tf_name = tf_name.replace(':0', '')
    tf_name = re.sub('/[^/]*___([^/]*)/', '/\\1/', tf_name)
    tf_name = tf_name.replace('_._', '/')
    tf_name = re.sub('//+', '/', tf_name)
    tf_name = tf_name.split('/')
    if len(tf_name) > 1:
        tf_name = tf_name[1:]
    tf_weight_shape = list(tf_weight_shape)
    if tf_name[-1] == 'kernel' and tf_weight_shape is not None and (len(tf_weight_shape) == 4):
        transpose = TransposeType.CONV2D
    elif tf_name[-1] == 'kernel' and tf_weight_shape is not None and (len(tf_weight_shape) == 3):
        transpose = TransposeType.CONV1D
    elif bool(tf_name[-1] in ['kernel', 'pointwise_kernel', 'depthwise_kernel'] or 'emb_projs' in tf_name or 'out_projs' in tf_name):
        transpose = TransposeType.SIMPLE
    else:
        transpose = TransposeType.NO
    if tf_name[-1] == 'kernel' or tf_name[-1] == 'embeddings' or tf_name[-1] == 'gamma':
        tf_name[-1] = 'weight'
    if tf_name[-1] == 'beta':
        tf_name[-1] = 'bias'
    if tf_name[-1] == 'pointwise_kernel' or tf_name[-1] == 'depthwise_kernel':
        tf_name[-1] = tf_name[-1].replace('_kernel', '.weight')
    tf_name = '.'.join(tf_name)
    if start_prefix_to_remove:
        tf_name = tf_name.replace(start_prefix_to_remove, '', 1)
    return (tf_name, transpose)

def apply_transpose(transpose: TransposeType, weight, match_shape=None, pt_to_tf=True):
    if False:
        print('Hello World!')
    '\n    Apply a transpose to some weight then tries to reshape the weight to the same shape as a given shape, all in a\n    framework agnostic way.\n    '
    if transpose is TransposeType.CONV2D:
        axes = (2, 3, 1, 0) if pt_to_tf else (3, 2, 0, 1)
        weight = transpose_func(weight, axes=axes)
    elif transpose is TransposeType.CONV1D:
        weight = transpose_func(weight, axes=(2, 1, 0))
    elif transpose is TransposeType.SIMPLE:
        weight = transpose_func(weight)
    if match_shape is None:
        return weight
    if len(match_shape) < len(weight.shape):
        weight = squeeze(weight)
    elif len(match_shape) > len(weight.shape):
        weight = expand_dims(weight, axis=0)
    if list(match_shape) != list(weight.shape):
        try:
            weight = reshape(weight, match_shape)
        except AssertionError as e:
            e.args += (match_shape, match_shape)
            raise e
    return weight

def load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path, tf_inputs=None, allow_missing_keys=False, output_loading_info=False, _prefix=None, tf_to_pt_weight_rename=None):
    if False:
        return 10
    'Load pytorch checkpoints in a TF 2.0 model'
    try:
        import tensorflow as tf
        import torch
        from safetensors.torch import load_file as safe_load_file
    except ImportError:
        logger.error('Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.')
        raise
    if isinstance(pytorch_checkpoint_path, str):
        pytorch_checkpoint_path = [pytorch_checkpoint_path]
    pt_state_dict = {}
    for path in pytorch_checkpoint_path:
        pt_path = os.path.abspath(path)
        logger.info(f'Loading PyTorch weights from {pt_path}')
        if pt_path.endswith('.safetensors'):
            state_dict = safe_load_file(pt_path)
        else:
            state_dict = torch.load(pt_path, map_location='cpu')
        pt_state_dict.update(state_dict)
    logger.info(f'PyTorch checkpoint contains {sum((t.numel() for t in pt_state_dict.values())):,} parameters')
    return load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info, _prefix=_prefix, tf_to_pt_weight_rename=tf_to_pt_weight_rename)

def load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=None, allow_missing_keys=False):
    if False:
        i = 10
        return i + 15
    'Load pytorch checkpoints in a TF 2.0 model'
    pt_state_dict = pt_model.state_dict()
    return load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys)

def load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=None, allow_missing_keys=False, output_loading_info=False, _prefix=None, tf_to_pt_weight_rename=None):
    if False:
        i = 10
        return i + 15
    'Load pytorch state_dict in a TF 2.0 model.'
    try:
        import tensorflow as tf
        import torch
    except ImportError:
        logger.error('Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.')
        raise
    pt_state_dict = {k: v.numpy() for (k, v) in pt_state_dict.items()}
    return load_pytorch_state_dict_in_tf2_model(tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info, _prefix=_prefix, tf_to_pt_weight_rename=tf_to_pt_weight_rename)

def load_pytorch_state_dict_in_tf2_model(tf_model, pt_state_dict, tf_inputs=None, allow_missing_keys=False, output_loading_info=False, _prefix=None, tf_to_pt_weight_rename=None, ignore_mismatched_sizes=False):
    if False:
        i = 10
        return i + 15
    'Load a pytorch state_dict in a TF 2.0 model. pt_state_dict can be either an actual dict or a lazy-loading\n    safetensors archive created with the safe_open() function.'
    import tensorflow as tf
    from keras import backend as K
    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs
    if _prefix is None:
        _prefix = ''
    if tf_inputs:
        with tf.name_scope(_prefix):
            tf_model(tf_inputs, training=False)
    tf_keys_to_pt_keys = {}
    for key in pt_state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if 'running_var' in key:
            new_key = key.replace('running_var', 'moving_variance')
        if 'running_mean' in key:
            new_key = key.replace('running_mean', 'moving_mean')
        key_components = key.split('.')
        name = None
        if key_components[-3::2] == ['parametrizations', 'original0']:
            name = key_components[-2] + '_g'
        elif key_components[-3::2] == ['parametrizations', 'original1']:
            name = key_components[-2] + '_v'
        if name is not None:
            key_components = key_components[:-3] + [name]
            new_key = '.'.join(key_components)
        if new_key is None:
            new_key = key
        tf_keys_to_pt_keys[new_key] = key
    start_prefix_to_remove = ''
    if not any((s.startswith(tf_model.base_model_prefix) for s in tf_keys_to_pt_keys.keys())):
        start_prefix_to_remove = tf_model.base_model_prefix + '.'
    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    tf_loaded_numel = 0
    all_pytorch_weights = set(tf_keys_to_pt_keys.keys())
    missing_keys = []
    mismatched_keys = []
    is_safetensor_archive = hasattr(pt_state_dict, 'get_tensor')
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        (name, transpose) = convert_tf_weight_name_to_pt_weight_name(sw_name, start_prefix_to_remove=start_prefix_to_remove, tf_weight_shape=symbolic_weight.shape, name_scope=_prefix)
        if tf_to_pt_weight_rename is not None:
            name = tf_to_pt_weight_rename(name)
        if name not in tf_keys_to_pt_keys:
            if allow_missing_keys:
                missing_keys.append(name)
                continue
            elif tf_model._keys_to_ignore_on_load_missing is not None:
                if any((re.search(pat, name) is not None for pat in tf_model._keys_to_ignore_on_load_missing)):
                    continue
            raise AttributeError(f'{name} not found in PyTorch model')
        state_dict_name = tf_keys_to_pt_keys[name]
        if is_safetensor_archive:
            array = pt_state_dict.get_tensor(state_dict_name)
        else:
            array = pt_state_dict[state_dict_name]
        try:
            array = apply_transpose(transpose, array, symbolic_weight.shape)
        except tf.errors.InvalidArgumentError as e:
            if not ignore_mismatched_sizes:
                error_msg = str(e)
                error_msg += '\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method.'
                raise tf.errors.InvalidArgumentError(error_msg)
            else:
                mismatched_keys.append((name, array.shape, symbolic_weight.shape))
                continue
        tf_loaded_numel += tensor_size(array)
        K.set_value(symbolic_weight, array)
        del array
        all_pytorch_weights.discard(name)
    logger.info(f'Loaded {tf_loaded_numel:,} parameters in the TF 2.0 model.')
    unexpected_keys = list(all_pytorch_weights)
    if tf_model._keys_to_ignore_on_load_missing is not None:
        for pat in tf_model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
    if tf_model._keys_to_ignore_on_load_unexpected is not None:
        for pat in tf_model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
    if len(unexpected_keys) > 0:
        logger.warning(f'Some weights of the PyTorch model were not used when initializing the TF 2.0 model {tf_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing {tf_model.__class__.__name__} from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).\n- This IS NOT expected if you are initializing {tf_model.__class__.__name__} from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).')
    else:
        logger.warning(f'All PyTorch model weights were used when initializing {tf_model.__class__.__name__}.\n')
    if len(missing_keys) > 0:
        logger.warning(f'Some weights or buffers of the TF 2.0 model {tf_model.__class__.__name__} were not initialized from the PyTorch model and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
    else:
        logger.warning(f'All the weights of {tf_model.__class__.__name__} were initialized from the PyTorch model.\nIf your task is similar to the task the model of the checkpoint was trained on, you can already use {tf_model.__class__.__name__} for predictions without further training.')
    if len(mismatched_keys) > 0:
        mismatched_warning = '\n'.join([f'- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated' for (key, shape1, shape2) in mismatched_keys])
        logger.warning(f'Some weights of {tf_model.__class__.__name__} were not initialized from the model checkpoint are newly initialized because the shapes did not match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
    if output_loading_info:
        loading_info = {'missing_keys': missing_keys, 'unexpected_keys': unexpected_keys, 'mismatched_keys': mismatched_keys}
        return (tf_model, loading_info)
    return tf_model

def load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path, tf_inputs=None, allow_missing_keys=False, output_loading_info=False):
    if False:
        i = 10
        return i + 15
    '\n    Load TF 2.0 HDF5 checkpoint in a PyTorch model We use HDF5 to easily do transfer learning (see\n    https://github.com/tensorflow/tensorflow/blob/ee16fcac960ae660e0e4496658a366e2f745e1f0/tensorflow/python/keras/engine/network.py#L1352-L1357).\n    '
    try:
        import tensorflow as tf
        import torch
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.')
        raise
    import transformers
    from .modeling_tf_utils import load_tf_weights
    logger.info(f'Loading TensorFlow weights from {tf_checkpoint_path}')
    tf_model_class_name = 'TF' + pt_model.__class__.__name__
    tf_model_class = getattr(transformers, tf_model_class_name)
    tf_model = tf_model_class(pt_model.config)
    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs
    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)
    load_tf_weights(tf_model, tf_checkpoint_path)
    return load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info)

def load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=False, output_loading_info=False):
    if False:
        i = 10
        return i + 15
    'Load TF 2.0 model in a pytorch model'
    weights = tf_model.weights
    return load_tf2_weights_in_pytorch_model(pt_model, weights, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info)

def load_tf2_weights_in_pytorch_model(pt_model, tf_weights, allow_missing_keys=False, output_loading_info=False):
    if False:
        return 10
    'Load TF2.0 symbolic weights in a PyTorch model'
    try:
        import tensorflow as tf
        import torch
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_state_dict = {tf_weight.name: tf_weight.numpy() for tf_weight in tf_weights}
    return load_tf2_state_dict_in_pytorch_model(pt_model, tf_state_dict, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info)

def load_tf2_state_dict_in_pytorch_model(pt_model, tf_state_dict, allow_missing_keys=False, output_loading_info=False):
    if False:
        while True:
            i = 10
    import torch
    new_pt_params_dict = {}
    current_pt_params_dict = dict(pt_model.named_parameters())
    start_prefix_to_remove = ''
    if not any((s.startswith(pt_model.base_model_prefix) for s in current_pt_params_dict.keys())):
        start_prefix_to_remove = pt_model.base_model_prefix + '.'
    tf_weights_map = {}
    for (name, tf_weight) in tf_state_dict.items():
        (pt_name, transpose) = convert_tf_weight_name_to_pt_weight_name(name, start_prefix_to_remove=start_prefix_to_remove, tf_weight_shape=tf_weight.shape)
        tf_weights_map[pt_name] = (tf_weight, transpose)
    all_tf_weights = set(tf_weights_map.keys())
    loaded_pt_weights_data_ptr = {}
    missing_keys_pt = []
    for (pt_weight_name, pt_weight) in current_pt_params_dict.items():
        if pt_weight.data_ptr() in loaded_pt_weights_data_ptr:
            new_pt_params_dict[pt_weight_name] = loaded_pt_weights_data_ptr[pt_weight.data_ptr()]
            continue
        pt_weight_name_to_check = pt_weight_name
        key_components = pt_weight_name.split('.')
        name = None
        if key_components[-3::2] == ['parametrizations', 'original0']:
            name = key_components[-2] + '_g'
        elif key_components[-3::2] == ['parametrizations', 'original1']:
            name = key_components[-2] + '_v'
        if name is not None:
            key_components = key_components[:-3] + [name]
            pt_weight_name_to_check = '.'.join(key_components)
        if pt_weight_name_to_check not in tf_weights_map:
            if allow_missing_keys:
                missing_keys_pt.append(pt_weight_name)
                continue
            raise AttributeError(f'{pt_weight_name} not found in TF 2.0 model')
        (array, transpose) = tf_weights_map[pt_weight_name_to_check]
        array = apply_transpose(transpose, array, pt_weight.shape, pt_to_tf=False)
        if numpy.isscalar(array):
            array = numpy.array(array)
        if not is_torch_tensor(array) and (not is_numpy_array(array)):
            array = array.numpy()
        if is_numpy_array(array):
            array = torch.from_numpy(array)
        new_pt_params_dict[pt_weight_name] = array
        loaded_pt_weights_data_ptr[pt_weight.data_ptr()] = array
        all_tf_weights.discard(pt_weight_name)
    (missing_keys, unexpected_keys) = pt_model.load_state_dict(new_pt_params_dict, strict=False)
    missing_keys += missing_keys_pt
    if pt_model._keys_to_ignore_on_load_missing is not None:
        for pat in pt_model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
    if pt_model._keys_to_ignore_on_load_unexpected is not None:
        for pat in pt_model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
    if len(unexpected_keys) > 0:
        logger.warning(f'Some weights of the TF 2.0 model were not used when initializing the PyTorch model {pt_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing {pt_model.__class__.__name__} from a TF 2.0 model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a TFBertForPreTraining model).\n- This IS NOT expected if you are initializing {pt_model.__class__.__name__} from a TF 2.0 model that you expect to be exactly identical (e.g. initializing a BertForSequenceClassification model from a TFBertForSequenceClassification model).')
    else:
        logger.warning(f'All TF 2.0 model weights were used when initializing {pt_model.__class__.__name__}.\n')
    if len(missing_keys) > 0:
        logger.warning(f'Some weights of {pt_model.__class__.__name__} were not initialized from the TF 2.0 model and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
    else:
        logger.warning(f'All the weights of {pt_model.__class__.__name__} were initialized from the TF 2.0 model.\nIf your task is similar to the task the model of the checkpoint was trained on, you can already use {pt_model.__class__.__name__} for predictions without further training.')
    logger.info(f'Weights or buffers not loaded from TF 2.0 model: {all_tf_weights}')
    if output_loading_info:
        loading_info = {'missing_keys': missing_keys, 'unexpected_keys': unexpected_keys}
        return (pt_model, loading_info)
    return pt_model