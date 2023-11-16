"""Part of the Keras training engine related to plain array data."""
import functools
import numpy as np
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils.generic_utils import make_batches
from tensorflow.python.keras.utils.generic_utils import slice_arrays
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
try:
    from scipy.sparse import issparse
except ImportError:
    issparse = None

def model_iteration(model, inputs, targets=None, sample_weights=None, batch_size=None, epochs=1, verbose=1, callbacks=None, val_inputs=None, val_targets=None, val_sample_weights=None, shuffle=True, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, mode=ModeKeys.TRAIN, validation_in_fit=False, prepared_feed_values_from_dataset=False, steps_name='steps', **kwargs):
    if False:
        print('Hello World!')
    'Loop function for arrays of data with modes TRAIN/TEST/PREDICT.\n\n  Args:\n      model: Keras Model instance.\n      inputs: Either a list or dictionary of arrays, or a dataset instance.\n      targets: List/dictionary of input arrays.\n      sample_weights: Optional list of sample weight arrays.\n      batch_size: Integer batch size or None if unknown.\n      epochs: Number of times to iterate over the data\n      verbose: 0, 1, or 2. Verbosity mode.\n        0 = silent, 1 = progress bar, 2 = one line per epoch.\n        Note that the progress bar is not particularly useful when\n        logged to a file, so verbose=2 is recommended when not running\n        interactively (eg, in a production environment).\n      callbacks: List of callbacks to be called during training\n      val_inputs: Either a list or dictionary of arrays, or a dataset instance.\n      val_targets: List/dictionary of target arrays.\n      val_sample_weights: Optional list of sample weight arrays.\n      shuffle: Whether to shuffle the data at the beginning of each epoch\n        concatenation of list the display names of the outputs of `f` and the\n        list of display names of the outputs of `f_val`.\n      initial_epoch: Epoch at which to start training (useful for resuming a\n        previous training run)\n      steps_per_epoch: Total number of steps (batches of samples) before\n        declaring one epoch finished and starting the next epoch. Ignored with\n        the default value of `None`.\n      validation_steps: Number of steps to run validation for (only if doing\n        validation from data tensors). Ignored with the default value of\n        `None`.\n      validation_freq: Only relevant if validation data is provided. Integer or\n        `collections.abc.Container` instance (e.g. list, tuple, etc.). If an\n        integer, specifies how many training epochs to run before a new\n        validation run is performed, e.g. `validation_freq=2` runs\n        validation every 2 epochs. If a Container, specifies the epochs on\n        which to run validation, e.g. `validation_freq=[1, 2, 10]` runs\n        validation at the end of the 1st, 2nd, and 10th epochs.\n      mode: One of ModeKeys.TRAIN/ModeKeys.TEST/ModeKeys.PREDICT.\n      validation_in_fit: if true, then this method is invoked from within\n        training iteration (for validation). In the case where `val_inputs` is\n        a dataset, this flag indicates that its iterator and feed values are\n        already created so should properly reuse resources.\n      prepared_feed_values_from_dataset: if True, `inputs` is a list of feed\n        tensors returned from `_prepare_feed_values` call on the validation\n        dataset, so do not call it again on `inputs`. Should only be used for\n        inline validation (i.e., only if `validation_in_fit` is also True).\n      steps_name: The string name of the steps argument, either `steps`,\n        `validation_steps`, or `steps_per_epoch`. Only used for error message\n        formatting.\n      **kwargs: Additional arguments for backwards compatibility.\n\n  Returns:\n      - In TRAIN mode: `History` object.\n      - In TEST mode: Evaluation metrics.\n      - In PREDICT mode: Outputs of the Model called on inputs.\n\n  Raises:\n      ValueError: in case of invalid arguments.\n  '
    if 'steps' in kwargs:
        steps_per_epoch = kwargs.pop('steps')
    if kwargs:
        raise TypeError('Unknown arguments: %s' % (kwargs,))
    reset_dataset_after_each_epoch = False
    input_iterator = None
    is_dataset = isinstance(inputs, (data_types.DatasetV1, data_types.DatasetV2))
    if is_dataset:
        if steps_per_epoch is None:
            reset_dataset_after_each_epoch = True
            steps_per_epoch = training_utils_v1.infer_steps_for_dataset(model, inputs, steps_per_epoch, epochs=epochs, steps_name=steps_name)
        input_iterator = _get_iterator(inputs, model._distribution_strategy)
    if model._distribution_strategy:
        scope = distributed_training_utils_v1.distributed_scope(strategy=model._distribution_strategy, learning_phase=1 if mode == ModeKeys.TRAIN else 0)
        scope.__enter__()
    use_steps = is_dataset or steps_per_epoch is not None
    do_validation = val_inputs is not None
    inputs = input_iterator or inputs
    if validation_in_fit and prepared_feed_values_from_dataset:
        ins = inputs
    else:
        ins = _prepare_feed_values(model, inputs, targets, sample_weights, mode)
    if not is_dataset:
        num_samples_or_steps = _get_num_samples_or_steps(ins, batch_size, steps_per_epoch)
    else:
        num_samples_or_steps = steps_per_epoch
    _update_sample_weight_mode(model, mode, ins)
    f = _make_execution_function(model, mode)
    val_iterator = None
    if isinstance(val_inputs, (data_types.DatasetV1, data_types.DatasetV2)):
        if validation_steps is None:
            validation_steps = training_utils_v1.infer_steps_for_dataset(model, val_inputs, validation_steps, epochs=epochs, steps_name='validation_steps')
        val_iterator = _get_iterator(val_inputs, model._distribution_strategy)
        val_inputs = _prepare_feed_values(model, val_iterator, val_targets, val_sample_weights, ModeKeys.TEST)
        val_samples_or_steps = validation_steps
    else:
        val_samples_or_steps = val_inputs and nest.flatten(val_inputs)[0].shape[0] or None
    if mode == ModeKeys.TRAIN and verbose:
        _print_train_info(num_samples_or_steps, val_samples_or_steps, is_dataset)
    count_mode = 'steps' if use_steps else 'samples'
    callbacks = cbks.configure_callbacks(callbacks, model, do_validation=do_validation, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, samples=num_samples_or_steps, count_mode=count_mode, verbose=verbose, mode=mode)
    if issparse is not None and (not use_steps):
        indices_for_conversion_to_dense = []
        feed = _get_model_feed(model, mode)
        for (i, (input_data, feed_tensor)) in enumerate(zip(ins, feed)):
            if issparse(input_data) and (not backend.is_sparse(feed_tensor)):
                indices_for_conversion_to_dense.append(i)
    if mode == ModeKeys.PREDICT:
        aggregator = training_utils_v1.OutputsAggregator(use_steps, num_samples=None if steps_per_epoch else num_samples_or_steps, steps=steps_per_epoch)
    else:
        aggregator = training_utils_v1.MetricsAggregator(use_steps, num_samples=None if steps_per_epoch else num_samples_or_steps, steps=steps_per_epoch)
    if model._compile_distribution:
        distributed_training_utils_v1._copy_weights_to_distributed_model(model, mode)
    callbacks.model.stop_training = False
    callbacks._call_begin_hook(mode)
    initial_epoch = model._maybe_load_initial_epoch_from_ckpt(initial_epoch, mode)
    for epoch in range(initial_epoch, epochs):
        if callbacks.model.stop_training:
            break
        epoch_logs = {}
        if mode != ModeKeys.PREDICT:
            model.reset_metrics()
        if mode == ModeKeys.TRAIN:
            callbacks.on_epoch_begin(epoch, epoch_logs)
        if use_steps:
            if steps_per_epoch is None:
                target_steps = np.inf
            else:
                target_steps = steps_per_epoch
            step = 0
            while step < target_steps:
                batch_logs = {'batch': step, 'size': 1}
                callbacks._call_batch_hook(mode, 'begin', step, batch_logs)
                try:
                    if not callable(ins) or (model._distribution_strategy and (not distributed_training_utils_v1.is_distributing_by_cloning(model))):
                        actual_inputs = ins
                    else:
                        actual_inputs = ins()
                    batch_outs = f(actual_inputs)
                except errors.OutOfRangeError:
                    if is_dataset:
                        if steps_per_epoch:
                            callbacks.model.stop_training = True
                            logging.warning('Your dataset ran out of data; interrupting training. Make sure that your dataset can generate at least `%s * epochs` batches (in this case, %d batches). You may need to use the repeat() function when building your dataset.' % (steps_name, steps_per_epoch * epochs))
                        elif step > 0:
                            steps_per_epoch = step
                            aggregator.steps = steps_per_epoch
                    else:
                        callbacks.model.stop_training = True
                        logging.warning('Your dataset iterator ran out of data; interrupting training. Make sure that your iterator can generate at least `%s * epochs` batches (in this case, %d batches). You may need touse the repeat() function when building your dataset.' % (steps_name, steps_per_epoch * epochs))
                    break
                if not isinstance(batch_outs, list):
                    batch_outs = [batch_outs]
                if model._distribution_strategy:
                    batch_outs = distributed_training_utils_v1._per_replica_aggregate_batch(model._distribution_strategy, batch_outs, model, mode)
                if step == 0:
                    aggregator.create(batch_outs)
                aggregator.aggregate(batch_outs)
                batch_logs = cbks.make_logs(model, batch_logs, batch_outs, mode)
                callbacks._call_batch_hook(mode, 'end', step, batch_logs)
                step += 1
                if callbacks.model.stop_training:
                    break
        else:
            index_array = np.arange(num_samples_or_steps)
            if shuffle == 'batch':
                index_array = training_utils_v1.batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)
            batches = make_batches(num_samples_or_steps, batch_size)
            for (batch_index, (batch_start, batch_end)) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                if len(batches) == 1:
                    ins_batch = ins
                else:
                    try:
                        if ins and isinstance(ins[-1], int):
                            ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
                        else:
                            ins_batch = slice_arrays(ins, batch_ids)
                    except TypeError:
                        raise TypeError('TypeError while preparing batch. If using HDF5 input data, pass shuffle="batch".')
                if issparse is not None:
                    for i in indices_for_conversion_to_dense:
                        ins_batch[i] = ins_batch[i].toarray()
                batch_logs = {'batch': batch_index, 'size': len(batch_ids)}
                callbacks._call_batch_hook(mode, 'begin', batch_index, batch_logs)
                batch_outs = f(ins_batch)
                if not isinstance(batch_outs, list):
                    batch_outs = [batch_outs]
                if batch_index == 0:
                    aggregator.create(batch_outs)
                aggregator.aggregate(batch_outs, batch_start, batch_end)
                batch_logs = cbks.make_logs(model, batch_logs, batch_outs, mode)
                callbacks._call_batch_hook(mode, 'end', batch_index, batch_logs)
                if callbacks.model.stop_training:
                    break
        aggregator.finalize()
        results = aggregator.results
        epoch_logs = cbks.make_logs(model, epoch_logs, results, mode)
        if len(results) == 1:
            results = results[0]
        if do_validation and training_utils_v1.should_run_validation(validation_freq, epoch) and (not callbacks.model.stop_training):
            if model._compile_distribution:
                distributed_training_utils_v1._copy_weights_to_original_model(model, ModeKeys.TRAIN)
            val_results = model_iteration(model, val_inputs, targets=val_targets, sample_weights=val_sample_weights, batch_size=batch_size, steps_per_epoch=validation_steps, callbacks=callbacks, verbose=0, mode=ModeKeys.TEST, validation_in_fit=True, prepared_feed_values_from_dataset=val_iterator is not None, steps_name='validation_steps')
            if not isinstance(val_results, list):
                val_results = [val_results]
            epoch_logs = cbks.make_logs(model, epoch_logs, val_results, mode, prefix='val_')
            if val_iterator and epoch < epochs - 1:
                _reinitialize_iterator(val_iterator, model._distribution_strategy)
        if mode == ModeKeys.TRAIN:
            callbacks.on_epoch_end(epoch, epoch_logs)
        if reset_dataset_after_each_epoch and epoch < epochs - 1:
            _reinitialize_iterator(input_iterator, model._distribution_strategy)
    model._successful_loop_finish = True
    callbacks._call_end_hook(mode)
    if model._distribution_strategy:
        if model._compile_distribution:
            distributed_training_utils_v1._copy_weights_to_original_model(model, mode)
        scope.__exit__(None, None, None)
    if mode == ModeKeys.TRAIN:
        return model.history
    return results

def _get_model_feed(model, mode):
    if False:
        return 10
    if mode == ModeKeys.PREDICT:
        feed = model._feed_inputs
    else:
        feed = model._feed_inputs + model._feed_targets + model._feed_sample_weights
    return feed

def _print_train_info(num_samples_or_steps, val_samples_or_steps, is_dataset):
    if False:
        i = 10
        return i + 15
    increment = 'steps' if is_dataset else 'samples'
    msg = 'Train on {0} {increment}'.format(num_samples_or_steps, increment=increment)
    if val_samples_or_steps:
        msg += ', validate on {0} {increment}'.format(val_samples_or_steps, increment=increment)
    print(msg)

def _get_num_samples_or_steps(ins, batch_size, steps_per_epoch):
    if False:
        print('Hello World!')
    'Returns total number of samples (when training in batch mode) or steps.'
    if steps_per_epoch:
        return steps_per_epoch
    return training_utils_v1.check_num_samples(ins, batch_size, steps_per_epoch, 'steps_per_epoch')

def _prepare_feed_values(model, inputs, targets, sample_weights, mode):
    if False:
        print('Hello World!')
    'Prepare feed values to the model execution function.\n\n  Args:\n    model: Model to prepare feed values for.\n    inputs: List or dict of model inputs.\n    targets: Optional list of model targets.\n    sample_weights: Optional list of sample weight arrays.\n    mode: One of ModeKeys.TRAIN/ModeKeys.TEST/ModeKeys.PREDICT.\n\n  Returns:\n    Feed values for the model in the given mode.\n  '
    if model._distribution_strategy:
        if isinstance(inputs, (data_types.DatasetV1, data_types.DatasetV2)):
            inputs = distributed_training_utils_v1.get_iterator(inputs, model._distribution_strategy)

        def get_distributed_inputs():
            if False:
                return 10
            return distributed_training_utils_v1._prepare_feed_values(model, inputs, targets, sample_weights, mode)
        if context.executing_eagerly():
            return get_distributed_inputs
        else:
            return get_distributed_inputs()
    if isinstance(inputs, (data_types.DatasetV1, data_types.DatasetV2, iterator_ops.Iterator)):
        (inputs, targets, sample_weights) = model._standardize_user_data(inputs, extract_tensors_from_dataset=True)
    inputs = training_utils_v1.ModelInputs(inputs).as_list()
    targets = list(targets or [])
    sample_weights = list(sample_weights or [])
    ins = inputs + targets + sample_weights
    if mode == ModeKeys.TRAIN and (not isinstance(backend.symbolic_learning_phase(), int)):
        ins += [True]
    return ins

def _get_iterator(inputs, distribution_strategy=None):
    if False:
        for i in range(10):
            print('nop')
    if distribution_strategy:
        return distributed_training_utils_v1.get_iterator(inputs, distribution_strategy)
    return training_utils_v1.get_iterator(inputs)

def _reinitialize_iterator(iterator, distribution_strategy=None):
    if False:
        while True:
            i = 10
    if distribution_strategy:
        distributed_training_utils_v1.initialize_iterator(iterator, distribution_strategy)
    else:
        training_utils_v1.initialize_iterator(iterator)

def _make_execution_function(model, mode):
    if False:
        print('Hello World!')
    'Makes function to run one step of model execution.'
    if model._distribution_strategy:
        return distributed_training_utils_v1._make_execution_function(model, mode)
    return model._make_execution_function(mode)

def _update_sample_weight_mode(model, mode, inputs):
    if False:
        i = 10
        return i + 15
    'Updates the sample_weight_mode of a given model.'
    if mode == ModeKeys.PREDICT:
        return
    sample_weights = None
    if not callable(inputs):
        sample_weights = inputs[len(model._feed_inputs) + len(model._feed_targets):]
        has_learning_phase_pl = mode == ModeKeys.TRAIN and (not isinstance(backend.symbolic_learning_phase(), int))
        if has_learning_phase_pl:
            sample_weights = sample_weights[:-1]
        model._update_sample_weight_modes(sample_weights=sample_weights)
    if model._distribution_strategy:
        distributed_training_utils_v1._update_sample_weight_modes(model, mode, sample_weights)
fit_loop = functools.partial(model_iteration, mode=ModeKeys.TRAIN)
test_loop = functools.partial(model_iteration, mode=ModeKeys.TEST, shuffle=False)
predict_loop = functools.partial(model_iteration, mode=ModeKeys.PREDICT, shuffle=False)

class ArrayLikeTrainingLoop(training_utils_v1.TrainingLoop):
    """TrainingLoop that handle inputs like array.

  This is the default handler for most of the input data types, includes
  symbolic tensors or Numpy array-like, Datasets and iterators in graph mode
  (since they generate symbolic tensors). This Function is used to handle model
  with `run_eagerly` = False.
  """

    def fit(self, model, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        batch_size = model._validate_or_infer_batch_size(batch_size, steps_per_epoch, x)
        (x, y, sample_weights) = model._standardize_user_data(x, y, sample_weight=sample_weight, class_weight=class_weight, batch_size=batch_size, check_steps=True, steps_name='steps_per_epoch', steps=steps_per_epoch, validation_split=validation_split, shuffle=shuffle)
        if validation_data:
            (val_x, val_y, val_sample_weights) = model._prepare_validation_data(validation_data, batch_size, validation_steps)
        elif validation_split and 0.0 < validation_split < 1.0:
            (x, y, sample_weights, val_x, val_y, val_sample_weights) = training_utils_v1.split_training_and_validation_data(x, y, sample_weights, validation_split)
        else:
            if validation_steps:
                raise ValueError('`validation_steps` should not be specified if `validation_data` is None.')
            (val_x, val_y, val_sample_weights) = (None, None, None)
        return fit_loop(model, inputs=x, targets=y, sample_weights=sample_weights, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, val_inputs=val_x, val_targets=val_y, val_sample_weights=val_sample_weights, shuffle=shuffle, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_freq=validation_freq, steps_name='steps_per_epoch')

    def evaluate(self, model, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, **kwargs):
        if False:
            i = 10
            return i + 15
        batch_size = model._validate_or_infer_batch_size(batch_size, steps, x)
        (x, y, sample_weights) = model._standardize_user_data(x, y, sample_weight=sample_weight, batch_size=batch_size, check_steps=True, steps_name='steps', steps=steps)
        return test_loop(model, inputs=x, targets=y, sample_weights=sample_weights, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks)

    def predict(self, model, x, batch_size=None, verbose=0, steps=None, callbacks=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        batch_size = model._validate_or_infer_batch_size(batch_size, steps, x)
        (x, _, _) = model._standardize_user_data(x, check_steps=True, steps_name='steps', steps=steps)
        return predict_loop(model, x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks)