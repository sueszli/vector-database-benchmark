"""Keras training and evaluation routines for eager execution."""
import numpy as np
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

def _eager_loss_fn(outputs, targets, loss_fn, output_name):
    if False:
        return 10
    with backend.name_scope(output_name + '_loss'):
        loss = loss_fn(targets, outputs)
    return loss

def _eager_metrics_fn(model, outputs, targets, sample_weights=None, masks=None):
    if False:
        while True:
            i = 10
    'Calculates the metrics for each output of the given model.\n\n  Args:\n      model: The model on which metrics are being calculated.\n      outputs: The outputs of the given model.\n      targets: The predictions or targets of the given model.\n      sample_weights: Optional list of sample weights for each output.\n      masks: Optional list of masks for each output.\n\n  Returns:\n      Returns the metric results for each output of the model.\n  '
    outputs = nest.flatten(outputs)
    targets = nest.flatten(targets)
    metric_results = []
    if targets:
        if len(model._targets) != len(targets):
            new_targets = [None if t is None else targets.pop(0) for t in model._targets]
            targets = new_targets
        metric_results = model._handle_metrics(outputs, targets=targets, sample_weights=sample_weights, masks=masks, return_weighted_and_unweighted_metrics=True, skip_target_masks=model._prepare_skip_target_masks())
    metric_results.extend([m.result() for m in model.metrics if m not in model._compile_metric_functions])
    return metric_results

def _model_loss(model, inputs, targets, output_loss_metrics=None, sample_weights=None, training=False):
    if False:
        for i in range(10):
            print('nop')
    'Calculates the loss for a given model.\n\n  Args:\n      model: The model on which metrics are being calculated.\n      inputs: Either a dictionary of inputs to the model or a list of input\n        arrays.\n      targets: List of target arrays.\n      output_loss_metrics: List of metrics that are used to aggregated output\n        loss values.\n      sample_weights: Optional list of sample weight arrays.\n      training: Whether the model should be run in inference or training mode.\n\n  Returns:\n     Returns the model output, total loss, loss value calculated using the\n     specified loss function and masks for each output. The total loss includes\n     regularization losses and applies masking and sample weighting\n     to the loss value.\n  '
    total_loss = 0
    kwargs = {}
    if model._expects_training_arg:
        kwargs['training'] = training
    if len(inputs) == 1 and (not isinstance(inputs, dict)):
        inputs = inputs[0]
    if any((isinstance(input_t, (np.ndarray, float, int)) for input_t in nest.flatten(inputs))):
        inputs = nest.map_structure(tensor_conversion.convert_to_tensor_v2_with_dispatch, inputs)
    outs = model(inputs, **kwargs)
    outs = nest.flatten(outs)
    if targets:
        targets = training_utils_v1.cast_if_floating_dtype_and_mismatch(targets, outs)
    if sample_weights:
        new_sample_weights = []
        for val in sample_weights:
            if val is not None:
                new_sample_weights.append(training_utils_v1.cast_if_floating_dtype(tensor_conversion.convert_to_tensor_v2_with_dispatch(val)))
            else:
                new_sample_weights.append(None)
        sample_weights = new_sample_weights
    masks = [getattr(t, '_keras_mask', None) for t in outs]
    targets = nest.flatten(targets)
    output_losses = []
    with backend.name_scope('loss'):
        loss_fns = [loss_fn for loss_fn in model.loss_functions if loss_fn is not None]
        custom_losses = model.losses
        if not loss_fns and (not custom_losses):
            if training:
                raise ValueError('The model cannot be trained because it has no loss to optimize.')
            else:
                raise ValueError('The model cannot be evaluated because it has no loss to compute.')
        for (i, loss_fn) in enumerate(loss_fns):
            weights = sample_weights[i] if sample_weights else None
            mask = masks[i]
            with backend.name_scope(model.output_names[i] + '_loss'):
                if mask is not None:
                    mask = math_ops.cast(mask, outs[i].dtype)
                    if weights is None:
                        weights = mask
                    else:
                        weights = math_ops.cast(weights, outs[i].dtype)
                        (mask, _, weights) = losses_utils.squeeze_or_expand_dimensions(mask, sample_weight=weights)
                        weights *= mask
                if hasattr(loss_fn, 'reduction'):
                    per_sample_losses = loss_fn.call(targets[i], outs[i])
                    weighted_losses = losses_utils.compute_weighted_loss(per_sample_losses, sample_weight=weights, reduction=losses_utils.ReductionV2.NONE)
                    loss_reduction = loss_fn.reduction
                    if loss_reduction == losses_utils.ReductionV2.AUTO:
                        loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
                    output_loss = losses_utils.reduce_weighted_loss(weighted_losses, reduction=loss_reduction)
                else:
                    output_loss = loss_fn(targets[i], outs[i], sample_weight=weights)
                    loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
            if len(model.outputs) > 1:
                output_losses.append(output_loss_metrics[i](output_loss))
            if loss_reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE:
                output_loss = losses_utils.scale_loss_for_distribution(output_loss)
            total_loss += model._loss_weights_list[i] * output_loss
        if custom_losses:
            total_loss += losses_utils.scale_loss_for_distribution(math_ops.add_n(custom_losses))
    return (outs, total_loss, output_losses, masks)

def _process_single_batch(model, inputs, targets, output_loss_metrics=None, sample_weights=None, training=False):
    if False:
        for i in range(10):
            print('nop')
    "Calculate the loss and gradient for one input batch.\n\n     The model weights are updated if training is set to True.\n\n  Args:\n      model: Model whose loss has to be calculated.\n      inputs: List of input arrays.\n      targets: List of target arrays.\n      output_loss_metrics: List of metrics that are used to aggregated output\n        loss values.\n      sample_weights: Optional list of sample weight arrays.\n      training: The boolean represents if the weights of the model are updated.\n              'fit' methods will set this to True while 'evaluate' methods will\n              set this to False.\n\n  Returns:\n      output of the model, total loss, the loss and the mask\n      associated with each output.\n\n  Raises:\n      ValueError: If the model has no loss to optimize.\n  "
    with backend.eager_learning_phase_scope(1 if training else 0), training_utils.RespectCompiledTrainableState(model):
        with GradientTape() as tape:
            (outs, total_loss, output_losses, masks) = _model_loss(model, inputs, targets, output_loss_metrics=output_loss_metrics, sample_weights=sample_weights, training=training)
            if isinstance(model.optimizer, loss_scale_optimizer.LossScaleOptimizer):
                scaled_total_loss = model.optimizer.get_scaled_loss(total_loss)
            else:
                scaled_total_loss = total_loss
        if training:
            trainable_weights = model.trainable_weights
            if trainable_weights:
                if hasattr(model, '_backwards'):
                    model._backwards(tape, scaled_total_loss)
                else:
                    grads = tape.gradient(scaled_total_loss, trainable_weights)
                    if isinstance(model.optimizer, loss_scale_optimizer.LossScaleOptimizer):
                        grads = model.optimizer.get_unscaled_gradients(grads)
                    model.optimizer.apply_gradients(zip(grads, trainable_weights))
            else:
                logging.warning('The list of trainable weights is empty. Make sure that you are not setting model.trainable to False before compiling the model.')
        return (outs, total_loss, output_losses, masks)

def train_on_batch(model, inputs, targets, sample_weights=None, output_loss_metrics=None):
    if False:
        print('Hello World!')
    "Calculates the loss and gradient updates for one input batch.\n\n  Args:\n      model: Model whose loss has to be calculated.\n      inputs: Input batch data.\n      targets: Target batch data.\n      sample_weights: Sample weight batch data.\n      output_loss_metrics: List of metrics that are used to aggregated output\n        loss values.\n\n  Returns:\n      Dict with three items:\n        'total_loss': list with a single tensor for overall loss,\n        'output_losses': list of tensors for loss corresponding to each of the\n          model output. Could be a empty list when model has only one output.\n        'metrics': list of tensors for metric specified.\n  "
    inputs = training_utils_v1.cast_to_model_input_dtypes(inputs, model)
    (outs, total_loss, output_losses, masks) = _process_single_batch(model, inputs, targets, sample_weights=sample_weights, training=True, output_loss_metrics=output_loss_metrics)
    if not isinstance(outs, list):
        outs = [outs]
    metrics_results = _eager_metrics_fn(model, outs, targets, sample_weights=sample_weights, masks=masks)
    total_loss = nest.flatten(total_loss)
    return {'total_loss': total_loss, 'output_losses': output_losses, 'metrics': metrics_results}

def test_on_batch(model, inputs, targets, sample_weights=None, output_loss_metrics=None):
    if False:
        return 10
    "Calculates the loss for one input batch.\n\n  Args:\n      model: Model whose loss has to be calculated.\n      inputs: Input batch data.\n      targets: Target batch data.\n      sample_weights: Sample weight batch data.\n      output_loss_metrics: List of metrics that are used to aggregated output\n        loss values.\n\n  Returns:\n      Dict with three items:\n        'total_loss': single tensor for overall loss,\n        'output_losses': list of tensors for loss corresponding to each of the\n          model output. Could be a empty list when model has only one output.\n        'metrics': list of tensors for metric specified.\n  "
    inputs = training_utils_v1.cast_to_model_input_dtypes(inputs, model)
    with backend.eager_learning_phase_scope(0):
        (outs, total_loss, output_losses, masks) = _model_loss(model, inputs, targets, sample_weights=sample_weights, training=False, output_loss_metrics=output_loss_metrics)
    if not isinstance(outs, list):
        outs = [outs]
    metrics_results = _eager_metrics_fn(model, outs, targets, sample_weights=sample_weights, masks=masks)
    total_loss = nest.flatten(total_loss)
    return {'total_loss': total_loss, 'output_losses': output_losses, 'metrics': metrics_results}