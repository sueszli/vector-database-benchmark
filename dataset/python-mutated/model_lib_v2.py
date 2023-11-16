"""Constructs model, inputs, and training environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import time
import tensorflow as tf
from object_detection import eval_util
from object_detection import inputs
from object_detection import model_lib
from object_detection.builders import model_builder
from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import ops
from object_detection.utils import variables_helper
MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP

def _compute_losses_and_predictions_dicts(model, features, labels, add_regularization_loss=True):
    if False:
        print('Hello World!')
    "Computes the losses dict and predictions dict for a model on inputs.\n\n  Args:\n    model: a DetectionModel (based on Keras).\n    features: Dictionary of feature tensors from the input dataset.\n      Should be in the format output by `inputs.train_input` and\n      `inputs.eval_input`.\n        features[fields.InputDataFields.image] is a [batch_size, H, W, C]\n          float32 tensor with preprocessed images.\n        features[HASH_KEY] is a [batch_size] int32 tensor representing unique\n          identifiers for the images.\n        features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]\n          int32 tensor representing the true image shapes, as preprocessed\n          images could be padded.\n        features[fields.InputDataFields.original_image] (optional) is a\n          [batch_size, H, W, C] float32 tensor with original images.\n    labels: A dictionary of groundtruth tensors post-unstacking. The original\n      labels are of the form returned by `inputs.train_input` and\n      `inputs.eval_input`. The shapes may have been modified by unstacking with\n      `model_lib.unstack_batch`. However, the dictionary includes the following\n      fields.\n        labels[fields.InputDataFields.num_groundtruth_boxes] is a\n          int32 tensor indicating the number of valid groundtruth boxes\n          per image.\n        labels[fields.InputDataFields.groundtruth_boxes] is a float32 tensor\n          containing the corners of the groundtruth boxes.\n        labels[fields.InputDataFields.groundtruth_classes] is a float32\n          one-hot tensor of classes.\n        labels[fields.InputDataFields.groundtruth_weights] is a float32 tensor\n          containing groundtruth weights for the boxes.\n        -- Optional --\n        labels[fields.InputDataFields.groundtruth_instance_masks] is a\n          float32 tensor containing only binary values, which represent\n          instance masks for objects.\n        labels[fields.InputDataFields.groundtruth_keypoints] is a\n          float32 tensor containing keypoints for each box.\n    add_regularization_loss: Whether or not to include the model's\n      regularization loss in the losses dictionary.\n\n  Returns:\n    A tuple containing the losses dictionary (with the total loss under\n    the key 'Loss/total_loss'), and the predictions dictionary produced by\n    `model.predict`.\n\n  "
    model_lib.provide_groundtruth(model, labels)
    preprocessed_images = features[fields.InputDataFields.image]
    prediction_dict = model.predict(preprocessed_images, features[fields.InputDataFields.true_image_shape])
    prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)
    losses_dict = model.loss(prediction_dict, features[fields.InputDataFields.true_image_shape])
    losses = [loss_tensor for loss_tensor in losses_dict.values()]
    if add_regularization_loss:
        regularization_losses = model.regularization_losses()
        if regularization_losses:
            regularization_losses = ops.bfloat16_to_float32_nested(regularization_losses)
            regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
            losses.append(regularization_loss)
            losses_dict['Loss/regularization_loss'] = regularization_loss
    total_loss = tf.add_n(losses, name='total_loss')
    losses_dict['Loss/total_loss'] = total_loss
    return (losses_dict, prediction_dict)

def eager_train_step(detection_model, features, labels, unpad_groundtruth_tensors, optimizer, learning_rate, add_regularization_loss=True, clip_gradients_value=None, global_step=None, num_replicas=1.0):
    if False:
        while True:
            i = 10
    "Process a single training batch.\n\n  This method computes the loss for the model on a single training batch,\n  while tracking the gradients with a gradient tape. It then updates the\n  model variables with the optimizer, clipping the gradients if\n  clip_gradients_value is present.\n\n  This method can run eagerly or inside a tf.function.\n\n  Args:\n    detection_model: A DetectionModel (based on Keras) to train.\n    features: Dictionary of feature tensors from the input dataset.\n      Should be in the format output by `inputs.train_input.\n        features[fields.InputDataFields.image] is a [batch_size, H, W, C]\n          float32 tensor with preprocessed images.\n        features[HASH_KEY] is a [batch_size] int32 tensor representing unique\n          identifiers for the images.\n        features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]\n          int32 tensor representing the true image shapes, as preprocessed\n          images could be padded.\n        features[fields.InputDataFields.original_image] (optional, not used\n          during training) is a\n          [batch_size, H, W, C] float32 tensor with original images.\n    labels: A dictionary of groundtruth tensors. This method unstacks\n      these labels using model_lib.unstack_batch. The stacked labels are of\n      the form returned by `inputs.train_input` and `inputs.eval_input`.\n        labels[fields.InputDataFields.num_groundtruth_boxes] is a [batch_size]\n          int32 tensor indicating the number of valid groundtruth boxes\n          per image.\n        labels[fields.InputDataFields.groundtruth_boxes] is a\n          [batch_size, num_boxes, 4] float32 tensor containing the corners of\n          the groundtruth boxes.\n        labels[fields.InputDataFields.groundtruth_classes] is a\n          [batch_size, num_boxes, num_classes] float32 one-hot tensor of\n          classes. num_classes includes the background class.\n        labels[fields.InputDataFields.groundtruth_weights] is a\n          [batch_size, num_boxes] float32 tensor containing groundtruth weights\n          for the boxes.\n        -- Optional --\n        labels[fields.InputDataFields.groundtruth_instance_masks] is a\n          [batch_size, num_boxes, H, W] float32 tensor containing only binary\n          values, which represent instance masks for objects.\n        labels[fields.InputDataFields.groundtruth_keypoints] is a\n          [batch_size, num_boxes, num_keypoints, 2] float32 tensor containing\n          keypoints for each box.\n    unpad_groundtruth_tensors: A parameter passed to unstack_batch.\n    optimizer: The training optimizer that will update the variables.\n    learning_rate: The learning rate tensor for the current training step.\n      This is used only for TensorBoard logging purposes, it does not affect\n       model training.\n    add_regularization_loss: Whether or not to include the model's\n      regularization loss in the losses dictionary.\n    clip_gradients_value: If this is present, clip the gradients global norm\n      at this value using `tf.clip_by_global_norm`.\n    global_step: The current training step. Used for TensorBoard logging\n      purposes. This step is not updated by this function and must be\n      incremented separately.\n    num_replicas: The number of replicas in the current distribution strategy.\n      This is used to scale the total loss so that training in a distribution\n      strategy works correctly.\n\n  Returns:\n    The total loss observed at this training step\n  "
    is_training = True
    detection_model._is_training = is_training
    tf.keras.backend.set_learning_phase(is_training)
    labels = model_lib.unstack_batch(labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)
    with tf.GradientTape() as tape:
        (losses_dict, _) = _compute_losses_and_predictions_dicts(detection_model, features, labels, add_regularization_loss)
        total_loss = losses_dict['Loss/total_loss']
        total_loss = tf.math.divide(total_loss, tf.constant(num_replicas, dtype=tf.float32))
        losses_dict['Loss/normalized_total_loss'] = total_loss
    for loss_type in losses_dict:
        tf.compat.v2.summary.scalar(loss_type, losses_dict[loss_type], step=global_step)
    trainable_variables = detection_model.trainable_variables
    gradients = tape.gradient(total_loss, trainable_variables)
    if clip_gradients_value:
        (gradients, _) = tf.clip_by_global_norm(gradients, clip_gradients_value)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    tf.compat.v2.summary.scalar('learning_rate', learning_rate, step=global_step)
    return total_loss

def load_fine_tune_checkpoint(model, checkpoint_path, checkpoint_type, load_all_detection_checkpoint_vars, input_dataset, unpad_groundtruth_tensors):
    if False:
        while True:
            i = 10
    'Load a fine tuning classification or detection checkpoint.\n\n  To make sure the model variables are all built, this method first executes\n  the model by computing a dummy loss. (Models might not have built their\n  variables before their first execution)\n\n  It then loads a variable-name based classification or detection checkpoint\n  that comes from converted TF 1.x slim model checkpoints.\n\n  This method updates the model in-place and does not return a value.\n\n  Args:\n    model: A DetectionModel (based on Keras) to load a fine-tuning\n      checkpoint for.\n    checkpoint_path: Directory with checkpoints file or path to checkpoint.\n    checkpoint_type: Whether to restore from a full detection\n      checkpoint (with compatible variable names) or to restore from a\n      classification checkpoint for initialization prior to training.\n      Valid values: `detection`, `classification`.\n    load_all_detection_checkpoint_vars: whether to load all variables (when\n      `fine_tune_checkpoint_type` is `detection`). If False, only variables\n      within the feature extractor scopes are included. Default False.\n    input_dataset: The tf.data Dataset the model is being trained on. Needed\n      to get the shapes for the dummy loss computation.\n    unpad_groundtruth_tensors: A parameter passed to unstack_batch.\n  '
    (features, labels) = iter(input_dataset).next()

    def _dummy_computation_fn(features, labels):
        if False:
            print('Hello World!')
        model._is_training = False
        tf.keras.backend.set_learning_phase(False)
        labels = model_lib.unstack_batch(labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)
        return _compute_losses_and_predictions_dicts(model, features, labels)
    strategy = tf.compat.v2.distribute.get_strategy()
    strategy.experimental_run_v2(_dummy_computation_fn, args=(features, labels))
    var_map = model.restore_map(fine_tune_checkpoint_type=checkpoint_type, load_all_detection_checkpoint_vars=load_all_detection_checkpoint_vars)
    available_var_map = variables_helper.get_variables_available_in_checkpoint(var_map, checkpoint_path, include_global_step=False)
    tf.train.init_from_checkpoint(checkpoint_path, available_var_map)

def train_loop(hparams, pipeline_config_path, model_dir, config_override=None, train_steps=None, use_tpu=False, save_final_config=False, export_to_tpu=None, checkpoint_every_n=1000, **kwargs):
    if False:
        i = 10
        return i + 15
    'Trains a model using eager + functions.\n\n  This method:\n    1. Processes the pipeline configs\n    2. (Optionally) saves the as-run config\n    3. Builds the model & optimizer\n    4. Gets the training input data\n    5. Loads a fine-tuning detection or classification checkpoint if requested\n    6. Loops over the train data, executing distributed training steps inside\n       tf.functions.\n    7. Checkpoints the model every `checkpoint_every_n` training steps.\n    8. Logs the training metrics as TensorBoard summaries.\n\n  Args:\n    hparams: A `HParams`.\n    pipeline_config_path: A path to a pipeline config file.\n    model_dir:\n      The directory to save checkpoints and summaries to.\n    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to\n      override the config from `pipeline_config_path`.\n    train_steps: Number of training steps. If None, the number of training steps\n      is set from the `TrainConfig` proto.\n    use_tpu: Boolean, whether training and evaluation should run on TPU.\n    save_final_config: Whether to save final config (obtained after applying\n      overrides) to `model_dir`.\n    export_to_tpu: When use_tpu and export_to_tpu are true,\n      `export_savedmodel()` exports a metagraph for serving on TPU besides the\n      one on CPU. If export_to_tpu is not provided, we will look for it in\n      hparams too.\n    checkpoint_every_n:\n      Checkpoint every n training steps.\n    **kwargs: Additional keyword arguments for configuration override.\n  '
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP['get_configs_from_pipeline_file']
    merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP['merge_external_params_with_configs']
    create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP['create_pipeline_proto_from_configs']
    configs = get_configs_from_pipeline_file(pipeline_config_path, config_override=config_override)
    kwargs.update({'train_steps': train_steps, 'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu})
    configs = merge_external_params_with_configs(configs, hparams, kwargs_dict=kwargs)
    model_config = configs['model']
    train_config = configs['train_config']
    train_input_config = configs['train_input_config']
    unpad_groundtruth_tensors = train_config.unpad_groundtruth_tensors
    add_regularization_loss = train_config.add_regularization_loss
    clip_gradients_value = None
    if train_config.gradient_clipping_by_norm > 0:
        clip_gradients_value = train_config.gradient_clipping_by_norm
    if train_steps is None and train_config.num_steps != 0:
        train_steps = train_config.num_steps
    if export_to_tpu is None:
        export_to_tpu = hparams.get('export_to_tpu', False)
    tf.logging.info('train_loop: use_tpu %s, export_to_tpu %s', use_tpu, export_to_tpu)
    if kwargs['use_bfloat16']:
        tf.compat.v2.keras.mixed_precision.experimental.set_policy('mixed_bfloat16')
    if hparams.load_pretrained:
        fine_tune_checkpoint_path = train_config.fine_tune_checkpoint
    else:
        fine_tune_checkpoint_path = None
    load_all_detection_checkpoint_vars = train_config.load_all_detection_checkpoint_vars
    if not train_config.fine_tune_checkpoint_type:
        if train_config.from_detection_checkpoint:
            train_config.fine_tune_checkpoint_type = 'detection'
        else:
            train_config.fine_tune_checkpoint_type = 'classification'
    fine_tune_checkpoint_type = train_config.fine_tune_checkpoint_type
    if save_final_config:
        pipeline_config_final = create_pipeline_proto_from_configs(configs)
        config_util.save_pipeline_config(pipeline_config_final, model_dir)
    strategy = tf.compat.v2.distribute.get_strategy()
    with strategy.scope():
        detection_model = model_builder.build(model_config=model_config, is_training=True)
        train_input = inputs.train_input(train_config=train_config, train_input_config=train_input_config, model_config=model_config, model=detection_model)
        train_input = strategy.experimental_distribute_dataset(train_input.repeat())
        global_step = tf.compat.v2.Variable(0, trainable=False, dtype=tf.compat.v2.dtypes.int64, name='global_step')
        (optimizer, (learning_rate,)) = optimizer_builder.build(train_config.optimizer, global_step=global_step)
        if callable(learning_rate):
            learning_rate_fn = learning_rate
        else:
            learning_rate_fn = lambda : learning_rate
    summary_writer = tf.compat.v2.summary.create_file_writer(model_dir + '/train')
    with summary_writer.as_default():
        with strategy.scope():
            if fine_tune_checkpoint_path:
                load_fine_tune_checkpoint(detection_model, fine_tune_checkpoint_path, fine_tune_checkpoint_type, load_all_detection_checkpoint_vars, train_input, unpad_groundtruth_tensors)
            ckpt = tf.compat.v2.train.Checkpoint(step=global_step, model=detection_model, optimizer=optimizer)
            manager = tf.compat.v2.train.CheckpointManager(ckpt, model_dir, max_to_keep=7)
            ckpt.restore(manager.latest_checkpoint)

            def train_step_fn(features, labels):
                if False:
                    i = 10
                    return i + 15
                return eager_train_step(detection_model, features, labels, unpad_groundtruth_tensors, optimizer, learning_rate=learning_rate_fn(), add_regularization_loss=add_regularization_loss, clip_gradients_value=clip_gradients_value, global_step=global_step, num_replicas=strategy.num_replicas_in_sync)

            @tf.function
            def _dist_train_step(data_iterator):
                if False:
                    i = 10
                    return i + 15
                'A distributed train step.'
                (features, labels) = data_iterator.next()
                per_replica_losses = strategy.experimental_run_v2(train_step_fn, args=(features, labels))
                mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                return mean_loss
            train_input_iter = iter(train_input)
            for _ in range(train_steps - global_step.value()):
                start_time = time.time()
                loss = _dist_train_step(train_input_iter)
                global_step.assign_add(1)
                end_time = time.time()
                tf.compat.v2.summary.scalar('steps_per_sec', 1.0 / (end_time - start_time), step=global_step)
                if int(global_step.value()) % 100 == 0:
                    tf.logging.info('Step {} time taken {:.3f}s loss={:.3f}'.format(global_step.value(), end_time - start_time, loss))
                if int(global_step.value()) % checkpoint_every_n == 0:
                    manager.save()

def eager_eval_loop(detection_model, configs, eval_dataset, use_tpu=False, postprocess_on_cpu=False, global_step=None):
    if False:
        i = 10
        return i + 15
    'Evaluate the model eagerly on the evaluation dataset.\n\n  This method will compute the evaluation metrics specified in the configs on\n  the entire evaluation dataset, then return the metrics. It will also log\n  the metrics to TensorBoard\n\n  Args:\n    detection_model: A DetectionModel (based on Keras) to evaluate.\n    configs: Object detection configs that specify the evaluators that should\n      be used, as well as whether regularization loss should be included and\n      if bfloat16 should be used on TPUs.\n    eval_dataset: Dataset containing evaluation data.\n    use_tpu: Whether a TPU is being used to execute the model for evaluation.\n    postprocess_on_cpu: Whether model postprocessing should happen on\n      the CPU when using a TPU to execute the model.\n    global_step: A variable containing the training step this model was trained\n      to. Used for logging purposes.\n\n  Returns:\n    A dict of evaluation metrics representing the results of this evaluation.\n  '
    train_config = configs['train_config']
    eval_input_config = configs['eval_input_config']
    eval_config = configs['eval_config']
    add_regularization_loss = train_config.add_regularization_loss
    is_training = False
    detection_model._is_training = is_training
    tf.keras.backend.set_learning_phase(is_training)
    evaluator_options = eval_util.evaluator_options_from_eval_config(eval_config)
    class_agnostic_category_index = label_map_util.create_class_agnostic_category_index()
    class_agnostic_evaluators = eval_util.get_evaluators(eval_config, list(class_agnostic_category_index.values()), evaluator_options)
    class_aware_evaluators = None
    if eval_input_config.label_map_path:
        class_aware_category_index = label_map_util.create_category_index_from_labelmap(eval_input_config.label_map_path)
        class_aware_evaluators = eval_util.get_evaluators(eval_config, list(class_aware_category_index.values()), evaluator_options)
    evaluators = None
    loss_metrics = {}

    @tf.function
    def compute_eval_dict(features, labels):
        if False:
            while True:
                i = 10
        'Compute the evaluation result on an image.'
        boxes_shape = labels[fields.InputDataFields.groundtruth_boxes].get_shape().as_list()
        unpad_groundtruth_tensors = boxes_shape[1] is not None and (not use_tpu)
        labels = model_lib.unstack_batch(labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)
        (losses_dict, prediction_dict) = _compute_losses_and_predictions_dicts(detection_model, features, labels, add_regularization_loss)

        def postprocess_wrapper(args):
            if False:
                for i in range(10):
                    print('nop')
            return detection_model.postprocess(args[0], args[1])
        if use_tpu and postprocess_on_cpu:
            detections = tf.contrib.tpu.outside_compilation(postprocess_wrapper, (prediction_dict, features[fields.InputDataFields.true_image_shape]))
        else:
            detections = postprocess_wrapper((prediction_dict, features[fields.InputDataFields.true_image_shape]))
        class_agnostic = fields.DetectionResultFields.detection_classes not in detections
        groundtruth = model_lib._prepare_groundtruth_for_eval(detection_model, class_agnostic, eval_input_config.max_number_of_boxes)
        use_original_images = fields.InputDataFields.original_image in features
        if use_original_images:
            eval_images = features[fields.InputDataFields.original_image]
            true_image_shapes = tf.slice(features[fields.InputDataFields.true_image_shape], [0, 0], [-1, 3])
            original_image_spatial_shapes = features[fields.InputDataFields.original_image_spatial_shape]
        else:
            eval_images = features[fields.InputDataFields.image]
            true_image_shapes = None
            original_image_spatial_shapes = None
        eval_dict = eval_util.result_dict_for_batched_example(eval_images, features[inputs.HASH_KEY], detections, groundtruth, class_agnostic=class_agnostic, scale_to_absolute=True, original_image_spatial_shapes=original_image_spatial_shapes, true_image_shapes=true_image_shapes)
        return (eval_dict, losses_dict, class_agnostic)
    for (i, (features, labels)) in enumerate(eval_dataset):
        (eval_dict, losses_dict, class_agnostic) = compute_eval_dict(features, labels)
        if i % 100 == 0:
            tf.logging.info('Finished eval step %d', i)
        if evaluators is None:
            if class_agnostic:
                evaluators = class_agnostic_evaluators
            else:
                evaluators = class_aware_evaluators
        for evaluator in evaluators:
            evaluator.add_eval_dict(eval_dict)
        for (loss_key, loss_tensor) in iter(losses_dict.items()):
            if loss_key not in loss_metrics:
                loss_metrics[loss_key] = tf.keras.metrics.Mean()
            loss_metrics[loss_key].update_state(loss_tensor)
    eval_metrics = {}
    for evaluator in evaluators:
        eval_metrics.update(evaluator.evaluate())
    for loss_key in loss_metrics:
        eval_metrics[loss_key] = loss_metrics[loss_key].result()
    eval_metrics = {str(k): v for (k, v) in eval_metrics.items()}
    for k in eval_metrics:
        tf.compat.v2.summary.scalar(k, eval_metrics[k], step=global_step)
    return eval_metrics

def eval_continuously(hparams, pipeline_config_path, config_override=None, train_steps=None, sample_1_of_n_eval_examples=1, sample_1_of_n_eval_on_train_examples=1, use_tpu=False, override_eval_num_epochs=True, postprocess_on_cpu=False, export_to_tpu=None, model_dir=None, checkpoint_dir=None, wait_interval=180, **kwargs):
    if False:
        while True:
            i = 10
    'Run continuous evaluation of a detection model eagerly.\n\n  This method builds the model, and continously restores it from the most\n  recent training checkpoint in the checkpoint directory & evaluates it\n  on the evaluation data.\n\n  Args:\n    hparams: A `HParams`.\n    pipeline_config_path: A path to a pipeline config file.\n    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to\n      override the config from `pipeline_config_path`.\n    train_steps: Number of training steps. If None, the number of training steps\n      is set from the `TrainConfig` proto.\n    sample_1_of_n_eval_examples: Integer representing how often an eval example\n      should be sampled. If 1, will sample all examples.\n    sample_1_of_n_eval_on_train_examples: Similar to\n      `sample_1_of_n_eval_examples`, except controls the sampling of training\n      data for evaluation.\n    use_tpu: Boolean, whether training and evaluation should run on TPU.\n    override_eval_num_epochs: Whether to overwrite the number of epochs to 1 for\n      eval_input.\n    postprocess_on_cpu: When use_tpu and postprocess_on_cpu are true,\n      postprocess is scheduled on the host cpu.\n    export_to_tpu: When use_tpu and export_to_tpu are true,\n      `export_savedmodel()` exports a metagraph for serving on TPU besides the\n      one on CPU. If export_to_tpu is not provided, we will look for it in\n      hparams too.\n    model_dir:\n      Directory to output resulting evaluation summaries to.\n    checkpoint_dir:\n      Directory that contains the training checkpoints.\n    wait_interval:\n      Terminate evaluation in no new checkpoints arrive within this wait\n      interval (in seconds).\n    **kwargs: Additional keyword arguments for configuration override.\n  '
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP['get_configs_from_pipeline_file']
    merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP['merge_external_params_with_configs']
    configs = get_configs_from_pipeline_file(pipeline_config_path, config_override=config_override)
    kwargs.update({'sample_1_of_n_eval_examples': sample_1_of_n_eval_examples, 'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu})
    if train_steps is not None:
        kwargs['train_steps'] = train_steps
    if override_eval_num_epochs:
        kwargs.update({'eval_num_epochs': 1})
        tf.logging.warning('Forced number of epochs for all eval validations to be 1.')
    configs = merge_external_params_with_configs(configs, hparams, kwargs_dict=kwargs)
    model_config = configs['model']
    train_input_config = configs['train_input_config']
    eval_config = configs['eval_config']
    eval_input_configs = configs['eval_input_configs']
    eval_on_train_input_config = copy.deepcopy(train_input_config)
    eval_on_train_input_config.sample_1_of_n_examples = sample_1_of_n_eval_on_train_examples
    if override_eval_num_epochs and eval_on_train_input_config.num_epochs != 1:
        tf.logging.warning('Expected number of evaluation epochs is 1, but instead encountered `eval_on_train_input_config.num_epochs` = {}. Overwriting `num_epochs` to 1.'.format(eval_on_train_input_config.num_epochs))
        eval_on_train_input_config.num_epochs = 1
    if kwargs['use_bfloat16']:
        tf.compat.v2.keras.mixed_precision.experimental.set_policy('mixed_bfloat16')
    detection_model = model_builder.build(model_config=model_config, is_training=True)
    eval_inputs = []
    for eval_input_config in eval_input_configs:
        next_eval_input = inputs.eval_input(eval_config=eval_config, eval_input_config=eval_input_config, model_config=model_config, model=detection_model)
        eval_inputs.append((eval_input_config.name, next_eval_input))
    if export_to_tpu is None:
        export_to_tpu = hparams.get('export_to_tpu', False)
    tf.logging.info('eval_continuously: use_tpu %s, export_to_tpu %s', use_tpu, export_to_tpu)
    global_step = tf.compat.v2.Variable(0, trainable=False, dtype=tf.compat.v2.dtypes.int64)
    prev_checkpoint = None
    waiting = False
    while True:
        ckpt = tf.compat.v2.train.Checkpoint(step=global_step, model=detection_model)
        manager = tf.compat.v2.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
        latest_checkpoint = manager.latest_checkpoint
        if prev_checkpoint == latest_checkpoint:
            if prev_checkpoint is None:
                tf.logging.info('No checkpoints found yet. Trying again in %s seconds.' % wait_interval)
                time.sleep(wait_interval)
            elif waiting:
                tf.logging.info('Terminating eval after %s seconds of no new checkpoints.' % wait_interval)
                break
            else:
                tf.logging.info('No new checkpoint found. Will try again in %s seconds and terminate if no checkpoint appears.' % wait_interval)
                waiting = True
                time.sleep(wait_interval)
        else:
            tf.logging.info('New checkpoint found. Starting evaluation.')
            waiting = False
            prev_checkpoint = latest_checkpoint
            ckpt.restore(latest_checkpoint)
            for (eval_name, eval_input) in eval_inputs:
                summary_writer = tf.compat.v2.summary.create_file_writer(model_dir + '/eval' + eval_name)
                with summary_writer.as_default():
                    eager_eval_loop(detection_model, configs, eval_input, use_tpu=use_tpu, postprocess_on_cpu=postprocess_on_cpu, global_step=global_step)