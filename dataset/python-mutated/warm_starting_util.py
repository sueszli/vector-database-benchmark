"""Utilities to warm-start TF.Learn Estimators."""
import collections
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_ops
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['train.VocabInfo'])
class VocabInfo(collections.namedtuple('VocabInfo', ['new_vocab', 'new_vocab_size', 'num_oov_buckets', 'old_vocab', 'old_vocab_size', 'backup_initializer', 'axis'])):
    """Vocabulary information for warm-starting.

  See `tf.estimator.WarmStartSettings` for examples of using
  VocabInfo to warm-start.

  Args:
    new_vocab: [Required] A path to the new vocabulary file (used with the model
      to be trained).
    new_vocab_size: [Required] An integer indicating how many entries of the new
      vocabulary will used in training.
    num_oov_buckets: [Required] An integer indicating how many OOV buckets are
      associated with the vocabulary.
    old_vocab: [Required] A path to the old vocabulary file (used with the
      checkpoint to be warm-started from).
    old_vocab_size: [Optional] An integer indicating how many entries of the old
      vocabulary were used in the creation of the checkpoint. If not provided,
      the entire old vocabulary will be used.
    backup_initializer: [Optional] A variable initializer used for variables
      corresponding to new vocabulary entries and OOV. If not provided, these
      entries will be zero-initialized.
    axis: [Optional] Denotes what axis the vocabulary corresponds to.  The
      default, 0, corresponds to the most common use case (embeddings or
      linear weights for binary classification / regression).  An axis of 1
      could be used for warm-starting output layers with class vocabularies.

  Returns:
    A `VocabInfo` which represents the vocabulary information for warm-starting.

  Raises:
    ValueError: `axis` is neither 0 or 1.

      Example Usage:
```python
      embeddings_vocab_info = tf.VocabInfo(
          new_vocab='embeddings_vocab',
          new_vocab_size=100,
          num_oov_buckets=1,
          old_vocab='pretrained_embeddings_vocab',
          old_vocab_size=10000,
          backup_initializer=tf.compat.v1.truncated_normal_initializer(
              mean=0.0, stddev=(1 / math.sqrt(embedding_dim))),
          axis=0)

      softmax_output_layer_kernel_vocab_info = tf.VocabInfo(
          new_vocab='class_vocab',
          new_vocab_size=5,
          num_oov_buckets=0,  # No OOV for classes.
          old_vocab='old_class_vocab',
          old_vocab_size=8,
          backup_initializer=tf.compat.v1.glorot_uniform_initializer(),
          axis=1)

      softmax_output_layer_bias_vocab_info = tf.VocabInfo(
          new_vocab='class_vocab',
          new_vocab_size=5,
          num_oov_buckets=0,  # No OOV for classes.
          old_vocab='old_class_vocab',
          old_vocab_size=8,
          backup_initializer=tf.compat.v1.zeros_initializer(),
          axis=0)

      #Currently, only axis=0 and axis=1 are supported.
  ```
  """

    def __new__(cls, new_vocab, new_vocab_size, num_oov_buckets, old_vocab, old_vocab_size=-1, backup_initializer=None, axis=0):
        if False:
            i = 10
            return i + 15
        if axis != 0 and axis != 1:
            raise ValueError('The only supported values for the axis argument are 0 and 1.  Provided axis: {}'.format(axis))
        return super(VocabInfo, cls).__new__(cls, new_vocab, new_vocab_size, num_oov_buckets, old_vocab, old_vocab_size, backup_initializer, axis)

def _infer_var_name(var):
    if False:
        i = 10
        return i + 15
    'Returns name of the `var`.\n\n  Args:\n    var: A list. The list can contain either of the following:\n      (i) A single `Variable`\n      (ii) A single `ResourceVariable`\n      (iii) Multiple `Variable` objects which must be slices of the same larger\n        variable.\n      (iv) A single `PartitionedVariable`\n\n  Returns:\n    Name of the `var`\n  '
    name_to_var_dict = saveable_object_util.op_list_to_dict(var)
    if len(name_to_var_dict) > 1:
        raise TypeError('`var` = %s passed as arg violates the constraints.  name_to_var_dict = %s' % (var, name_to_var_dict))
    return list(name_to_var_dict.keys())[0]

def _get_var_info(var, prev_tensor_name=None):
    if False:
        i = 10
        return i + 15
    "Helper method for standarizing Variable and naming.\n\n  Args:\n    var: Current graph's variable that needs to be warm-started (initialized).\n      Can be either of the following: (i) `Variable` (ii) `ResourceVariable`\n      (iii) list of `Variable`: The list must contain slices of the same larger\n        variable. (iv) `PartitionedVariable`\n    prev_tensor_name: Name of the tensor to lookup in provided `prev_ckpt`. If\n      None, we lookup tensor with same name as given `var`.\n\n  Returns:\n    A tuple of the Tensor name and var.\n  "
    if checkpoint_utils._is_variable(var):
        current_var_name = _infer_var_name([var])
    elif isinstance(var, list) and all((checkpoint_utils._is_variable(v) for v in var)):
        current_var_name = _infer_var_name(var)
    elif isinstance(var, variables_lib.PartitionedVariable):
        current_var_name = _infer_var_name([var])
        var = var._get_variable_list()
    else:
        raise TypeError('var MUST be one of the following: a Variable, list of Variable or PartitionedVariable, but is {}'.format(type(var)))
    if not prev_tensor_name:
        prev_tensor_name = current_var_name
    return (prev_tensor_name, var)

def _warm_start_var_with_vocab(var, current_vocab_path, current_vocab_size, prev_ckpt, prev_vocab_path, previous_vocab_size=-1, current_oov_buckets=0, prev_tensor_name=None, initializer=None, axis=0):
    if False:
        print('Hello World!')
    "Warm-starts given variable from `prev_tensor_name` tensor in `prev_ckpt`.\n\n  Use this method when the `var` is backed by vocabulary. This method stitches\n  the given `var` such that values corresponding to individual features in the\n  vocabulary remain consistent irrespective of changing order of the features\n  between old and new vocabularies.\n\n  Args:\n    var: Current graph's variable that needs to be warm-started (initialized).\n      Can be either of the following:\n      (i) `Variable`\n      (ii) `ResourceVariable`\n      (iii) list of `Variable`: The list must contain slices of the same larger\n        variable.\n      (iv) `PartitionedVariable`\n    current_vocab_path: Path to the vocab file used for the given `var`.\n    current_vocab_size: An `int` specifying the number of entries in the current\n      vocab.\n    prev_ckpt: A string specifying the directory with checkpoint file(s) or path\n      to checkpoint. The given checkpoint must have tensor with name\n      `prev_tensor_name` (if not None) or tensor with name same as given `var`.\n    prev_vocab_path: Path to the vocab file used for the tensor in `prev_ckpt`.\n    previous_vocab_size: If provided, will constrain previous vocab to the first\n      `previous_vocab_size` entries.  -1 means use the entire previous vocab.\n    current_oov_buckets: An `int` specifying the number of out-of-vocabulary\n      buckets used for given `var`.\n    prev_tensor_name: Name of the tensor to lookup in provided `prev_ckpt`. If\n      None, we lookup tensor with same name as given `var`.\n    initializer: Variable initializer to be used for missing entries.  If None,\n      missing entries will be zero-initialized.\n    axis: Axis of the variable that the provided vocabulary corresponds to.\n\n  Raises:\n    ValueError: If required args are not provided.\n  "
    if not (current_vocab_path and current_vocab_size and prev_ckpt and prev_vocab_path):
        raise ValueError('Invalid args: Must provide all of [current_vocab_path, current_vocab_size, prev_ckpt, prev_vocab_path}.')
    if checkpoint_utils._is_variable(var):
        var = [var]
    elif isinstance(var, list) and all((checkpoint_utils._is_variable(v) for v in var)):
        var = var
    elif isinstance(var, variables_lib.PartitionedVariable):
        var = var._get_variable_list()
    else:
        raise TypeError('var MUST be one of the following: a Variable, list of Variable or PartitionedVariable, but is {}'.format(type(var)))
    if not prev_tensor_name:
        prev_tensor_name = _infer_var_name(var)
    total_v_first_axis = sum((v.get_shape().as_list()[0] for v in var))
    for v in var:
        v_shape = v.get_shape().as_list()
        slice_info = v._get_save_slice_info()
        partition_info = None
        if slice_info:
            partition_info = variable_scope._PartitionInfo(full_shape=slice_info.full_shape, var_offset=slice_info.var_offset)
        if axis == 0:
            new_row_vocab_size = current_vocab_size
            new_col_vocab_size = v_shape[1]
            old_row_vocab_size = previous_vocab_size
            old_row_vocab_file = prev_vocab_path
            new_row_vocab_file = current_vocab_path
            old_col_vocab_file = None
            new_col_vocab_file = None
            num_row_oov_buckets = current_oov_buckets
            num_col_oov_buckets = 0
        elif axis == 1:
            new_row_vocab_size = total_v_first_axis
            new_col_vocab_size = current_vocab_size
            old_row_vocab_size = -1
            old_row_vocab_file = None
            new_row_vocab_file = None
            old_col_vocab_file = prev_vocab_path
            new_col_vocab_file = current_vocab_path
            num_row_oov_buckets = 0
            num_col_oov_buckets = current_oov_buckets
        else:
            raise ValueError('The only supported values for the axis argument are 0 and 1.  Provided axis: {}'.format(axis))
        init = checkpoint_ops._load_and_remap_matrix_initializer(ckpt_path=checkpoint_utils._get_checkpoint_filename(prev_ckpt), old_tensor_name=prev_tensor_name, new_row_vocab_size=new_row_vocab_size, new_col_vocab_size=new_col_vocab_size, old_row_vocab_size=old_row_vocab_size, old_row_vocab_file=old_row_vocab_file, new_row_vocab_file=new_row_vocab_file, old_col_vocab_file=old_col_vocab_file, new_col_vocab_file=new_col_vocab_file, num_row_oov_buckets=num_row_oov_buckets, num_col_oov_buckets=num_col_oov_buckets, initializer=initializer)
        new_init_val = ops.convert_to_tensor(init(shape=v_shape, partition_info=partition_info))
        v._initializer_op = state_ops.assign(v, new_init_val)

def _get_grouped_variables(vars_to_warm_start):
    if False:
        for i in range(10):
            print('nop')
    'Collects and groups (possibly partitioned) variables into a dictionary.\n\n  The variables can be provided explicitly through vars_to_warm_start, or they\n  are retrieved from collections (see below).\n\n  Args:\n    vars_to_warm_start: One of the following:\n\n      - A regular expression (string) that captures which variables to\n        warm-start (see tf.compat.v1.get_collection).  This expression will\n        only consider variables in the TRAINABLE_VARIABLES collection.\n      - A list of strings, each representing a full variable name to warm-start.\n        These will consider variables in GLOBAL_VARIABLES collection.\n      - A list of Variables to warm-start.\n      - `None`, in which case all variables in TRAINABLE_VARIABLES will be used.\n  Returns:\n    A dictionary mapping variable names (strings) to lists of Variables.\n  Raises:\n    ValueError: If vars_to_warm_start is not a string, `None`, a list of\n      `Variables`, or a list of strings.\n  '
    if isinstance(vars_to_warm_start, str) or vars_to_warm_start is None:
        logging.info('Warm-starting variables only in TRAINABLE_VARIABLES.')
        list_of_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope=vars_to_warm_start)
    elif isinstance(vars_to_warm_start, list):
        if all((isinstance(v, str) for v in vars_to_warm_start)):
            list_of_vars = []
            for v in vars_to_warm_start:
                list_of_vars += ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope=v)
        elif all((checkpoint_utils._is_variable(v) for v in vars_to_warm_start)):
            list_of_vars = vars_to_warm_start
        else:
            raise ValueError('If `vars_to_warm_start` is a list, it must be all `Variable` or all `str`.  Given types are {}'.format([type(v) for v in vars_to_warm_start]))
    else:
        raise ValueError('`vars_to_warm_start must be a `list` or `str`.  Given type is {}'.format(type(vars_to_warm_start)))
    grouped_variables = {}
    for v in list_of_vars:
        t = [v] if not isinstance(v, list) else v
        var_name = _infer_var_name(t)
        grouped_variables.setdefault(var_name, []).append(v)
    return grouped_variables

def _get_object_checkpoint_renames(path, variable_names):
    if False:
        i = 10
        return i + 15
    'Returns a dictionary mapping variable names to checkpoint keys.\n\n  The warm-starting utility expects variable names to match with the variable\n  names in the checkpoint. For object-based checkpoints, the variable names\n  and names in the checkpoint are different. Thus, for object-based checkpoints,\n  this function is used to obtain the map from variable names to checkpoint\n  keys.\n\n  Args:\n    path: path to checkpoint directory or file.\n    variable_names: list of variable names to load from the checkpoint.\n\n  Returns:\n    If the checkpoint is object-based, this function returns a map from variable\n    names to their corresponding checkpoint keys.\n    If the checkpoint is name-based, this returns an empty dict.\n\n  Raises:\n    ValueError: If the object-based checkpoint is missing variables.\n  '
    fname = checkpoint_utils._get_checkpoint_filename(path)
    try:
        names_to_keys = saver_lib.object_graph_key_mapping(fname)
    except errors.NotFoundError:
        return {}
    missing_names = set(variable_names) - set(names_to_keys.keys())
    if missing_names:
        raise ValueError('Attempting to warm-start from an object-based checkpoint, but found that the checkpoint did not contain values for all variables. The following variables were missing: {}'.format(missing_names))
    return {name: names_to_keys[name] for name in variable_names}

@tf_export(v1=['train.warm_start'])
def warm_start(ckpt_to_initialize_from, vars_to_warm_start='.*', var_name_to_vocab_info=None, var_name_to_prev_var_name=None):
    if False:
        while True:
            i = 10
    'Warm-starts a model using the given settings.\n\n  If you are using a tf.estimator.Estimator, this will automatically be called\n  during training.\n\n  Args:\n    ckpt_to_initialize_from: [Required] A string specifying the directory with\n      checkpoint file(s) or path to checkpoint from which to warm-start the\n      model parameters.\n    vars_to_warm_start: [Optional] One of the following:\n\n      - A regular expression (string) that captures which variables to\n        warm-start (see tf.compat.v1.get_collection).  This expression will only\n        consider variables in the TRAINABLE_VARIABLES collection -- if you need\n        to warm-start non_TRAINABLE vars (such as optimizer accumulators or\n        batch norm statistics), please use the below option.\n      - A list of strings, each a regex scope provided to\n        tf.compat.v1.get_collection with GLOBAL_VARIABLES (please see\n        tf.compat.v1.get_collection).  For backwards compatibility reasons,\n        this is separate from the single-string argument type.\n      - A list of Variables to warm-start.  If you do not have access to the\n        `Variable` objects at the call site, please use the above option.\n      - `None`, in which case only TRAINABLE variables specified in\n        `var_name_to_vocab_info` will be warm-started.\n\n      Defaults to `\'.*\'`, which warm-starts all variables in the\n      TRAINABLE_VARIABLES collection.  Note that this excludes variables such\n      as accumulators and moving statistics from batch norm.\n    var_name_to_vocab_info: [Optional] Dict of variable names (strings) to\n      `tf.estimator.VocabInfo`. The variable names should be "full" variables,\n      not the names of the partitions.  If not explicitly provided, the variable\n      is assumed to have no (changes to) vocabulary.\n    var_name_to_prev_var_name: [Optional] Dict of variable names (strings) to\n      name of the previously-trained variable in `ckpt_to_initialize_from`. If\n      not explicitly provided, the name of the variable is assumed to be same\n      between previous checkpoint and current model.  Note that this has no\n      effect on the set of variables that is warm-started, and only controls\n      name mapping (use `vars_to_warm_start` for controlling what variables to\n      warm-start).\n\n  Raises:\n    ValueError: If the WarmStartSettings contains prev_var_name or VocabInfo\n      configuration for variable names that are not used.  This is to ensure\n      a stronger check for variable configuration than relying on users to\n      examine the logs.\n  '
    logging.info('Warm-starting from: {}'.format(ckpt_to_initialize_from))
    grouped_variables = _get_grouped_variables(vars_to_warm_start)
    if var_name_to_vocab_info is None:
        var_name_to_vocab_info = {}
    if not var_name_to_prev_var_name:
        var_name_to_prev_var_name = _get_object_checkpoint_renames(ckpt_to_initialize_from, grouped_variables.keys())
    warmstarted_count = 0
    prev_var_name_used = set()
    vocab_info_used = set()
    vocabless_vars = {}
    for (var_name, variable) in grouped_variables.items():
        prev_var_name = var_name_to_prev_var_name.get(var_name)
        if prev_var_name:
            prev_var_name_used.add(var_name)
        vocab_info = var_name_to_vocab_info.get(var_name)
        if vocab_info:
            vocab_info_used.add(var_name)
            warmstarted_count += 1
            logging.debug('Warm-starting variable: {}; current_vocab: {} current_vocab_size: {} prev_vocab: {} prev_vocab_size: {} current_oov: {} prev_tensor: {} initializer: {}'.format(var_name, vocab_info.new_vocab, vocab_info.new_vocab_size, vocab_info.old_vocab, vocab_info.old_vocab_size if vocab_info.old_vocab_size > 0 else 'All', vocab_info.num_oov_buckets, prev_var_name or 'Unchanged', vocab_info.backup_initializer or 'zero-initialized'))
            _warm_start_var_with_vocab(variable, current_vocab_path=vocab_info.new_vocab, current_vocab_size=vocab_info.new_vocab_size, prev_ckpt=ckpt_to_initialize_from, prev_vocab_path=vocab_info.old_vocab, previous_vocab_size=vocab_info.old_vocab_size, current_oov_buckets=vocab_info.num_oov_buckets, prev_tensor_name=prev_var_name, initializer=vocab_info.backup_initializer, axis=vocab_info.axis)
        elif vars_to_warm_start:
            warmstarted_count += 1
            logging.debug('Warm-starting variable: {}; prev_var_name: {}'.format(var_name, prev_var_name or 'Unchanged'))
            if len(variable) == 1:
                variable = variable[0]
            (prev_tensor_name, var) = _get_var_info(variable, prev_var_name)
            if prev_tensor_name in vocabless_vars:
                logging.debug('Requested prev_var_name {} initialize both {} and {}; calling init_from_checkpoint.'.format(prev_tensor_name, vocabless_vars[prev_tensor_name], var))
                checkpoint_utils.init_from_checkpoint(ckpt_to_initialize_from, vocabless_vars)
                vocabless_vars.clear()
            vocabless_vars[prev_tensor_name] = var
    if vocabless_vars:
        checkpoint_utils.init_from_checkpoint(ckpt_to_initialize_from, vocabless_vars)
    prev_var_name_not_used = set(var_name_to_prev_var_name.keys()) - prev_var_name_used
    vocab_info_not_used = set(var_name_to_vocab_info.keys()) - vocab_info_used
    logging.info('Warm-started %d variables.', warmstarted_count)
    if prev_var_name_not_used:
        raise ValueError('You provided the following variables in var_name_to_prev_var_name that were not used: {0}.  Perhaps you misspelled them?  Here is the list of viable variable names: {1}'.format(prev_var_name_not_used, grouped_variables.keys()))
    if vocab_info_not_used:
        raise ValueError('You provided the following variables in var_name_to_vocab_info that were not used: {0}.  Perhaps you misspelled them?  Here is the list of viable variable names: {1}'.format(vocab_info_not_used, grouped_variables.keys()))