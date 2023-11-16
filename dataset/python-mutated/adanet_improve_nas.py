"""Defines adanet estimator builder.

Copyright 2019 The AdaNet Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import adanet
import tensorflow.compat.v1 as tf
try:
    from adanet.research.improve_nas.trainer import improve_nas
    from adanet.research.improve_nas.trainer import optimizer
except ImportError as e:
    from trainer import improve_nas
    from trainer import optimizer

class GeneratorType(object):
    """Controls what generator is used."""
    DYNAMIC = 'dynamic'
    SIMPLE = 'simple'

class Builder(object):
    """An AdaNet estimator builder."""

    def estimator(self, data_provider, run_config, hparams, train_steps=None, seed=None):
        if False:
            return 10
        'Returns an AdaNet `Estimator` for train and evaluation.\n\n    Args:\n      data_provider: Data `Provider` for dataset to model.\n      run_config: `RunConfig` object to configure the runtime settings.\n      hparams: `HParams` instance defining custom hyperparameters.\n      train_steps: number of train steps.\n      seed: An integer seed if determinism is required.\n\n    Returns:\n      Returns an `Estimator`.\n    '
        max_iteration_steps = int(train_steps / hparams.boosting_iterations)
        optimizer_fn = optimizer.fn_with_name(hparams.optimizer, learning_rate_schedule=hparams.learning_rate_schedule, cosine_decay_steps=max_iteration_steps)
        hparams.add_hparam('total_training_steps', max_iteration_steps)
        if hparams.generator == GeneratorType.SIMPLE:
            subnetwork_generator = improve_nas.Generator(feature_columns=data_provider.get_feature_columns(), optimizer_fn=optimizer_fn, iteration_steps=max_iteration_steps, checkpoint_dir=run_config.model_dir, hparams=hparams, seed=seed)
        elif hparams.generator == GeneratorType.DYNAMIC:
            subnetwork_generator = improve_nas.DynamicGenerator(feature_columns=data_provider.get_feature_columns(), optimizer_fn=optimizer_fn, iteration_steps=max_iteration_steps, checkpoint_dir=run_config.model_dir, hparams=hparams, seed=seed)
        else:
            raise ValueError('Invalid generator: `%s`' % hparams.generator)
        evaluator = None
        if hparams.use_evaluator:
            evaluator = adanet.Evaluator(input_fn=data_provider.get_input_fn(partition='train', mode=tf.estimator.ModeKeys.EVAL, batch_size=hparams.evaluator_batch_size), steps=hparams.evaluator_steps)
        return adanet.Estimator(head=data_provider.get_head(), subnetwork_generator=subnetwork_generator, max_iteration_steps=max_iteration_steps, adanet_lambda=hparams.adanet_lambda, adanet_beta=hparams.adanet_beta, mixture_weight_type=hparams.mixture_weight_type, force_grow=hparams.force_grow, evaluator=evaluator, config=run_config, model_dir=run_config.model_dir)

    def hparams(self, default_batch_size, hparams_string):
        if False:
            print('Hello World!')
        "Returns hyperparameters, including any flag value overrides.\n\n    In order to allow for automated hyperparameter tuning, model hyperparameters\n    are aggregated within a tf.HParams object.  In this case, here are the\n    hyperparameters and their descriptions:\n    - optimizer: Name of the optimizer to use. See `optimizers.fn_with_name`.\n    - learning_rate_schedule: Learning rate schedule string.\n    - initial_learning_rate: The initial learning rate to use during training.\n    - num_cells: Number of cells in the model. Must be divisible by 3.\n    - num_conv_filters: The initial number of convolutional filters. The final\n        layer will have 24*num_conv_filters channels.\n    - weight_decay: Float amount of weight decay to apply to train loss.\n    - use_aux_head: Whether to create an auxiliary head for training. This adds\n        some non-determinism to training.\n    - knowledge_distillation: Whether subnetworks should learn from the\n        logits of the 'previous ensemble'/'previous subnetwork' in addition to\n        the labels to distill/transfer/compress the knowledge in a manner\n        inspired by Born Again Networks [Furlanello et al., 2018]\n        (https://arxiv.org/abs/1805.04770) and Distilling the Knowledge in\n        a Neural Network [Hinton at al., 2015]\n        (https://arxiv.org/abs/1503.02531).\n    - model_version: See `improve_nas.ModelVersion`.\n    - adanet_lambda: See `adanet.Estimator`.\n    - adanet_beta: See `adanet.Estimator`.\n    - generator: Type of generator. `simple` generator is just ensembling,\n        `dynamic` generator gradually grows the network.\n    - boosting_iterations: The number of boosting iterations to perform. The\n      final ensemble will have at most this many subnetworks comprising it.\n    - evaluator_batch_size: Batch size for the evaluator to use when comparing\n        candidates.\n    - evaluator_steps: Number of batches for the evaluator to use when\n        comparing candidates.\n    - learn_mixture_weights: Whether to learn adanet mixture weights.\n    - mixture_weight_type: Type of mxture weights.\n    - batch_size: Batch size for training.\n    - force_grow: Force AdaNet to add a candidate in each itteration, even if it\n        would decreases the performance of the ensemble.\n    - label_smoothing: Strength of label smoothing that will be applied (even\n        non true labels will have a non zero representation in one hot encoding\n        when computing loss).\n    - clip_gradients: Clip gradient to this value.\n    - aux_head_weight: NASNet cell parameter. Weight of auxiliary loss.\n    - stem_multiplier: NASNet cell parameter.\n    - drop_path_keep_prob: NASNet cell parameter. Propability for drop_path\n        regularization.\n    - dense_dropout_keep_prob: NASNet cell parameter. Dropout keep probability.\n    - filter_scaling_rate: NASNet cell parameter. Controls growth of number of\n        filters.\n    - num_reduction_layers: NASNet cell parameter. Number of reduction layers\n        that will be added to the architecture.\n    - data_format: NASNet cell parameter. Controls whether data is in channels\n        last or channels first format.\n    - skip_reduction_layer_input: NASNet cell parameter. Whether to skip\n        reduction layer.\n    - use_bounded_activation: NASNet cell parameter. Whether to use bounded\n        activations.\n    - use_evaluator: Boolean whether to use the adanet.Evaluator to choose the\n        best ensemble at each round.\n\n    Args:\n      default_batch_size: The default batch_size specified for training.\n      hparams_string: If the hparams_string is given, then it will use any\n        values specified in hparams to override any individually-set\n        hyperparameter. This logic allows tuners to override hyperparameter\n        settings to find optimal values.\n\n    Returns:\n      The hyperparameters as a tf.HParams object.\n    "
        hparams = tf.contrib.training.HParams(num_cells=3, num_conv_filters=10, aux_head_weight=0.4, stem_multiplier=3.0, drop_path_keep_prob=0.6, use_aux_head=True, dense_dropout_keep_prob=1.0, filter_scaling_rate=2.0, num_reduction_layers=2, data_format='NHWC', skip_reduction_layer_input=0, use_bounded_activation=False, clip_gradients=5, optimizer='momentum', learning_rate_schedule='cosine', initial_learning_rate=0.025, weight_decay=0.0005, label_smoothing=0.1, knowledge_distillation=improve_nas.KnowledgeDistillation.ADAPTIVE, model_version='cifar', adanet_lambda=0.0, adanet_beta=0.0, generator=GeneratorType.SIMPLE, boosting_iterations=3, force_grow=True, evaluator_batch_size=-1, evaluator_steps=-1, batch_size=default_batch_size, learn_mixture_weights=False, mixture_weight_type=adanet.MixtureWeightType.SCALAR, use_evaluator=True)
        if hparams_string:
            hparams = hparams.parse(hparams_string)
        if hparams.evaluator_batch_size < 0:
            hparams.evaluator_batch_size = default_batch_size
        if hparams.evaluator_steps < 0:
            hparams.evaluator_steps = None
        return hparams