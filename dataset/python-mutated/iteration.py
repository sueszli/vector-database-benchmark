"""An AdaNet iteration implementation in Tensorflow using a single graph.

Copyright 2018 The AdaNet Authors. All Rights Reserved.

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
import collections
import contextlib
import copy
import json
import os
from absl import logging
from adanet import distributed
from adanet import subnetwork
from adanet import tf_compat
from adanet.core.ensemble_builder import _EnsembleSpec
from adanet.core.eval_metrics import _IterationMetrics
import numpy as np
import tensorflow.compat.v2 as tf
from typing import Any

class _TrainManager(object):
    """Manages the training of SubnetworkSpecs and EnsembleSpecs.

  This object maintains a dictionary of states for each SubnetworkSpec and
  EnsembleSpec to coordinate and manage training. Users can check the
  training status of a spec, or request that it stops training.

  It also persists metadata about specs to disk in order to be consistent across
  runs and robust to preemptions.
  """

    def __init__(self, subnetwork_specs, ensemble_specs, train_manager_dir, is_chief):
        if False:
            print('Hello World!')
        'Initializes a _TrainManager instance.\n\n    Args:\n      subnetwork_specs: List of `_SubnetworkSpec` instances to monitor.\n      ensemble_specs: List of `EstimatorSpec` instances to monitor.\n      train_manager_dir: Directory for storing metadata about training. When a\n        spec should no longer be trained, a JSON file with its name and metadata\n        is written to this directory, to persist across runs and preemptions.\n      is_chief: Boolean whether the current worker is a chief.\n    '
        if not tf.io.gfile.exists(train_manager_dir):
            tf.io.gfile.makedirs(train_manager_dir)
        self._train_manager_dir = train_manager_dir
        self._is_training = {spec.name: not self._is_done_training(spec) for spec in subnetwork_specs + ensemble_specs}
        self._ensemble_specs = set([e.name for e in ensemble_specs])
        self._is_chief = is_chief

    def should_train(self, spec):
        if False:
            i = 10
            return i + 15
        'Whether the given spec should keep training.'
        return self._is_training[spec.name]

    def _is_done_training(self, spec):
        if False:
            for i in range(10):
                print('nop')
        'If the file exists, then the candidate is done training.'
        return tf.io.gfile.exists(self._filename_for(spec))

    def _filename_for(self, spec):
        if False:
            for i in range(10):
                print('nop')
        'Returns the filename to identify the spec.'
        return os.path.join(self._train_manager_dir, '{}.json'.format(spec.name))

    def request_stop(self, spec, message):
        if False:
            while True:
                i = 10
        'Registers that given spec should no longer train.'
        self._is_training[spec.name] = False
        if self._is_chief and (not self._is_done_training(spec)):
            with tf.io.gfile.GFile(self._filename_for(spec), 'w') as record_file:
                message = {'message': message}
                record_file.write(json.dumps(message))

    def is_over(self):
        if False:
            return 10
        'Whether all specs are done training and the iteration is over.'
        for k in sorted(self._is_training):
            if k in self._ensemble_specs:
                continue
            if self._is_training[k]:
                return False
        return True

class _NanLossHook(tf_compat.SessionRunHook):
    """Monitors a spec's loss tensor and stops its training if loss is NaN."""

    def __init__(self, train_manager, spec):
        if False:
            for i in range(10):
                print('nop')
        "Initializes a `NanTensorHook`.\n\n    Args:\n      train_manager: The current iteration's `_TrainManager`.\n      spec: Either a `SubnetworkSpec` or `EnsembleSpec` to monitor.\n    "
        self._train_manager = train_manager
        self._spec = spec

    def before_run(self, run_context):
        if False:
            return 10
        del run_context
        if self._train_manager.should_train(self._spec):
            return tf_compat.SessionRunArgs(self._spec.loss)

    def after_run(self, run_context, run_values):
        if False:
            print('Hello World!')
        loss = run_values.results
        if loss is None or not np.isnan(loss):
            return
        logging.warning("'%s' diverged with loss = NaN.", self._spec.name)

class _TrainingLimitHook(tf_compat.SessionRunHook):
    """Limits a given spec's training to a maximum number of steps.

  Is also responsible for incrementing the spec's step.
  """

    def __init__(self, train_manager, spec, max_steps, increment_step_op):
        if False:
            return 10
        "Initializes a _TrainingLimitHook instance.\n\n    Args:\n      train_manager: The current iteration's `_TrainManager`.\n      spec: Either a `SubnetworkSpec` or `EnsembleSpec` to monitor.\n      max_steps: Maximum number steps to train the given spec.\n      increment_step_op: That increments the current step and executes one train\n        op run.\n    "
        self._train_manager = train_manager
        self._spec = spec
        self._max_steps = max_steps
        self._increment_step_op = increment_step_op

    def after_create_session(self, session, coord):
        if False:
            i = 10
            return i + 15
        if not self._train_manager.should_train(self._spec):
            return
        if self._spec.step is None:
            self._train_manager.request_stop(self._spec, 'Dummy candidate to ignore.')
            return
        step_value = session.run(self._spec.step)
        if self._should_stop(step_value):
            logging.info("Skipping '%s' training which already trained %d steps", self._spec.name, step_value)
            self._train_manager.request_stop(self._spec, 'Training already complete.')

    def before_run(self, run_context):
        if False:
            while True:
                i = 10
        del run_context
        if not self._train_manager.should_train(self._spec):
            return None
        if self._increment_step_op is None:
            return tf_compat.SessionRunArgs(self._spec.step)
        return tf_compat.SessionRunArgs(self._increment_step_op)

    def after_run(self, run_context, run_values):
        if False:
            i = 10
            return i + 15
        step_value = run_values.results
        if step_value is None:
            return
        if self._should_stop(step_value):
            logging.info("Now stopping '%s' training after %d steps", self._spec.name, step_value)
            self._train_manager.request_stop(self._spec, 'Training complete after {} steps.'.format(step_value))

    def _should_stop(self, step):
        if False:
            print('Hello World!')
        return self._max_steps is not None and step >= self._max_steps

class _GlobalStepSetterHook(tf_compat.SessionRunHook):
    """A hook for setting the global step variable.

  Should only be run on CPU and GPU, but not TPU. TPUs run many training steps
  per hook run, so the global step should be incremented in an op along with the
  candidates' train ops.
  """

    def __init__(self, train_manager, subnetwork_specs, base_global_step, global_step_combiner_fn):
        if False:
            print('Hello World!')
        "Initializes a _GlobalStepSetterHook instance.\n\n    Args:\n      train_manager: The current iteration's `_TrainManager`.\n      subnetwork_specs: List of `_SubnetworkSpec` instances for this iteration.\n      base_global_step: Integer global step at the beginning of this iteration.\n      global_step_combiner_fn: Function for combining each subnetwork's\n        iteration step into the global step.\n    "
        self._train_manager = train_manager
        self._subnetwork_specs = subnetwork_specs
        self._base_global_step = base_global_step
        self._global_step_combiner_fn = global_step_combiner_fn

    def begin(self):
        if False:
            for i in range(10):
                print('nop')
        logging.info('Starting iteration at global step %s', self._base_global_step)
        steps = [self._base_global_step + s.step.read_value() for s in self._subnetwork_specs]
        updated_global_step = self._global_step_combiner_fn(steps)
        global_step = tf_compat.v1.train.get_global_step()
        self._assign_global_step_op = global_step.assign(updated_global_step)

    def after_run(self, run_context, run_values):
        if False:
            print('Hello World!')
        run_context.session.run(self._assign_global_step_op)

class _TrainingHookRunnerHook(tf_compat.SessionRunHook):
    """Hook wrapper for executing a spec's training hook.

  Will only run the hook according to the current TrainManager.
  """

    def __init__(self, train_manager, spec, hook):
        if False:
            for i in range(10):
                print('nop')
        "Initializes a _TrainingHookRunnerHook instance.\n\n    Only accepts a single hook, since merging hooks is complex and should be\n    handled by the MonitoredTrainingSession instead.\n\n    Args:\n      train_manager: The current iteration's `_TrainManager`.\n      spec: Either a `SubnetworkSpec` or `EnsembleSpec` to train.\n      hook: The spec's training hook to execute.\n    "
        self._train_manager = train_manager
        self._spec = spec
        self._hook = hook

    def begin(self):
        if False:
            for i in range(10):
                print('nop')
        self._hook.begin()

    @contextlib.contextmanager
    def _session_run_context(self):
        if False:
            i = 10
            return i + 15
        'Intercepts input out of range errors to gracefully stop spec training.'
        try:
            yield
        except (tf.errors.OutOfRangeError, StopIteration) as e:
            logging.info("Now stopping '%s' training after hitting end of input", self._spec.name)
            self._train_manager.request_stop(self._spec, 'OutOfRangeError: {}'.format(e))

    def after_create_session(self, session, coord):
        if False:
            i = 10
            return i + 15
        with self._session_run_context():
            self._hook.after_create_session(session, coord)

    def before_run(self, run_context):
        if False:
            for i in range(10):
                print('nop')
        if self._train_manager.should_train(self._spec):
            tmp_run_context = tf_compat.v1.train.SessionRunContext(run_context.original_args, run_context.session)
            with self._session_run_context():
                return self._hook.before_run(tmp_run_context)
            if tmp_run_context.stop_requested:
                self._train_manager.request_stop(self._spec, 'Stop requested.')

    def after_run(self, run_context, run_values):
        if False:
            i = 10
            return i + 15
        if self._train_manager.should_train(self._spec):
            tmp_run_context = tf_compat.v1.train.SessionRunContext(run_context.original_args, run_context.session)
            with self._session_run_context():
                self._hook.after_run(tmp_run_context, run_values)
            if tmp_run_context.stop_requested:
                self._train_manager.request_stop(self._spec, 'Stop requested.')

    def end(self, session):
        if False:
            i = 10
            return i + 15
        with self._session_run_context():
            self._hook.end(session)

class _Iteration(collections.namedtuple('_Iteration', ['number', 'candidates', 'subnetwork_specs', 'estimator_spec', 'best_candidate_index', 'summaries', 'train_manager', 'subnetwork_reports', 'checkpoint', 'previous_iteration'])):
    """An AdaNet iteration.

  An AdaNet iteration represents the simultaneous training of multiple
  candidates for one iteration of the AdaNet loop, and tracks the best
  candidate's loss, predictions, and evaluation metrics.

  There must be maximum one _Iteration per graph.
  """

    def __new__(cls, number, candidates, subnetwork_specs, estimator_spec, best_candidate_index, summaries, train_manager, subnetwork_reports, checkpoint, previous_iteration):
        if False:
            print('Hello World!')
        "Creates a validated `_Iteration` instance.\n\n    Args:\n      number: The iteration number.\n      candidates: List of `_Candidate` instances to track.\n      subnetwork_specs: List of `_SubnetworkSpec` instances.\n      estimator_spec: `EstimatorSpec` instance.\n      best_candidate_index: Int `Tensor` indicating the best candidate's index.\n      summaries: List of `adanet.Summary` instances for each candidate.\n      train_manager: The current `_TrainManager` for monitoring candidate per\n        training.\n      subnetwork_reports: Dict mapping string names to `subnetwork.Report`s, one\n        per candidate.\n      checkpoint: The `tf.train.Checkpoint` object associated with this\n        iteration.\n      previous_iteration: The iteration occuring before this one or None if this\n        is the first iteration.\n\n    Returns:\n      A validated `_Iteration` object.\n\n    Raises:\n      ValueError: If validation fails.\n    "
        if not isinstance(number, (int, np.integer)):
            raise ValueError('number must be an integer')
        if number < 0:
            raise ValueError('number must be greater than 0 got %d' % number)
        if not isinstance(candidates, list) or not candidates:
            raise ValueError('candidates must be a non-empty list')
        if estimator_spec is None:
            raise ValueError('estimator_spec is required')
        if best_candidate_index is None:
            raise ValueError('best_candidate_index is required')
        if not isinstance(subnetwork_reports, dict):
            raise ValueError('subnetwork_reports must be a dict')
        return super(_Iteration, cls).__new__(cls, number=number, candidates=candidates, subnetwork_specs=subnetwork_specs, estimator_spec=estimator_spec, best_candidate_index=best_candidate_index, summaries=summaries, train_manager=train_manager, subnetwork_reports=subnetwork_reports, checkpoint=checkpoint, previous_iteration=previous_iteration)

def _is_numeric(tensor):
    if False:
        return 10
    'Determines if given tensor is a float numeric.'
    if not isinstance(tensor, tf.Tensor):
        return False
    return tensor.dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]

class _IterationBuilder(object):
    """Builds AdaNet iterations."""

    def __init__(self, candidate_builder, subnetwork_manager, ensemble_builder, ensemblers, max_steps, summary_maker, global_step_combiner_fn=tf.math.reduce_mean, placement_strategy=distributed.ReplicationStrategy(), replicate_ensemble_in_training=False, use_tpu=False, debug=False, enable_ensemble_summaries=True, enable_subnetwork_summaries=True, enable_subnetwork_reports=True):
        if False:
            i = 10
            return i + 15
        "Creates an `_IterationBuilder` instance.\n\n    Args:\n      candidate_builder: A `_CandidateBuilder` instance.\n      subnetwork_manager: A `_SubnetworkManager` instance.\n      ensemble_builder: An `_EnsembleBuilder` instance.\n      ensemblers: An iterable of :class:`adanet.ensemble.Ensembler` objects that\n        define how to ensemble a group of subnetworks.\n      max_steps: Maximum number of steps to train candidate subnetworks.\n      summary_maker: A function that constructs an `adanet.Summary` instance\n        from (namespace, scope, and skip_summary).\n      global_step_combiner_fn: Function for combining each subnetwork's\n        iteration step into the global step.\n      placement_strategy: A `PlacementStrategy` for assigning subnetworks and\n        ensembles to specific workers.\n      replicate_ensemble_in_training: Whether to build the frozen subnetworks in\n        `training` mode during training.\n      use_tpu: Whether AdaNet is running on TPU.\n      debug: Boolean to enable debug mode which will check features and labels\n        for Infs and NaNs.\n      enable_ensemble_summaries: Whether to record summaries to display in\n        TensorBoard for each ensemble candidate. Disable to reduce memory and\n        disk usage per run.\n      enable_subnetwork_summaries: Whether to record summaries to display in\n        TensorBoard for each subnetwork. Disable to reduce memory and disk usage\n        per run.\n      enable_subnetwork_reports: Whether to enable generating subnetwork\n        reports.\n\n    Returns:\n      An `_IterationBuilder` object.\n    "
        if max_steps is not None and max_steps <= 0:
            raise ValueError('max_steps must be > 0 or None')
        self._candidate_builder = candidate_builder
        self._subnetwork_manager = subnetwork_manager
        self._ensemble_builder = ensemble_builder
        self._ensemblers = ensemblers
        self._max_steps = max_steps
        self._summary_maker = summary_maker
        self._global_step_combiner_fn = global_step_combiner_fn
        self._placement_strategy = placement_strategy
        self._replicate_ensemble_in_training = replicate_ensemble_in_training
        self._use_tpu = use_tpu
        self._debug = debug
        self._enable_ensemble_summaries = enable_ensemble_summaries
        self._enable_subnetwork_summaries = enable_subnetwork_summaries
        self._enable_subnetwork_reports = enable_subnetwork_reports
        super(_IterationBuilder, self).__init__()

    @property
    def placement_strategy(self):
        if False:
            return 10
        return self._placement_strategy

    @placement_strategy.setter
    def placement_strategy(self, new_placement_strategy):
        if False:
            for i in range(10):
                print('nop')
        self._placement_strategy = new_placement_strategy

    def _check_numerics(self, features, labels):
        if False:
            return 10
        'Checks for NaNs and Infs in input features and labels.\n\n    Args:\n      features: Dictionary of `Tensor` objects keyed by feature name.\n      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`\n        (for multi-head). Can be `None`.\n\n    Returns:\n      A features and labels tuple with same types and respective inputs, but\n      with numeric check ops wrapping them.\n    '
        if not self._debug:
            return (features, labels)
        (checked_features, checked_labels) = ({}, {})
        logging.info('DEBUG: Checking numerics of float features.')
        for name in sorted(features):
            if not _is_numeric(features[name]):
                continue
            logging.info("DEBUG: Checking numerics of float feature '%s'.", name)
            checked_features[name] = tf.debugging.check_numerics(features[name], "features '{}'".format(name))
        if isinstance(labels, dict):
            for name in sorted(labels):
                if not _is_numeric(labels[name]):
                    continue
                logging.info("DEBUG: Checking numerics of float label '%s'.", name)
                checked_labels[name] = tf.debugging.check_numerics(labels[name], "labels '{}'".format(name))
        elif labels is not None and _is_numeric(labels):
            logging.info('DEBUG: Checking numerics of labels.')
            checked_labels = tf.debugging.check_numerics(labels, "'labels'")
        return (checked_features, checked_labels)

    def build_iteration(self, base_global_step, iteration_number, ensemble_candidates, subnetwork_builders, features, mode, config, labels=None, previous_ensemble_summary=None, rebuilding=False, rebuilding_ensembler_name=None, best_ensemble_index_override=None, previous_iteration=None):
        if False:
            return 10
        "Builds and returns AdaNet iteration t.\n\n    This method uses the generated the candidate subnetworks given the ensemble\n    at iteration t-1 and creates graph operations to train them. The returned\n    `_Iteration` tracks the training of all candidates to know when the\n    iteration is over, and tracks the best candidate's predictions and loss, as\n    defined by lowest complexity-regularized loss on the train set.\n\n    Args:\n      base_global_step: Integer global step at the beginning of this iteration.\n      iteration_number: Integer iteration number.\n      ensemble_candidates: Iterable of `adanet.ensemble.Candidate` instances.\n      subnetwork_builders: A list of `Builders` for adding ` Subnetworks` to the\n        graph. Each subnetwork is then wrapped in a `_Candidate` to train.\n      features: Dictionary of `Tensor` objects keyed by feature name.\n      mode: Defines whether this is training, evaluation or prediction. See\n        `ModeKeys`.\n      config: The `tf.estimator.RunConfig` to use this iteration.\n      labels: `Tensor` of labels. Can be `None`.\n      previous_ensemble_summary: The `adanet.Summary` for the previous ensemble.\n      rebuilding: Boolean whether the iteration is being rebuilt only to restore\n        the previous best subnetworks and ensembles.\n      rebuilding_ensembler_name: Optional ensembler to restrict to, only\n        relevant when rebuilding is set as True.\n      best_ensemble_index_override: Integer index to identify the best ensemble\n        candidate instead of computing the best ensemble index dynamically\n        conditional on the ensemble AdaNet losses.\n      previous_iteration: The iteration occuring before this one or None if this\n        is the first iteration.\n\n    Returns:\n      An _Iteration instance.\n\n    Raises:\n      ValueError: If subnetwork_builders is empty.\n      ValueError: If two subnetworks share the same name.\n      ValueError: If two ensembles share the same name.\n    "
        self._placement_strategy.config = config
        logging.info('%s iteration %s', 'Rebuilding' if rebuilding else 'Building', iteration_number)
        if not subnetwork_builders:
            raise ValueError('Each iteration must have at least one Builder.')
        builder_mode = mode
        if rebuilding:
            builder_mode = tf.estimator.ModeKeys.EVAL
            if mode == tf.estimator.ModeKeys.PREDICT:
                builder_mode = mode
            if self._replicate_ensemble_in_training and mode == tf.estimator.ModeKeys.TRAIN:
                builder_mode = mode
        (features, labels) = self._check_numerics(features, labels)
        replay_indices_for_all = {}
        training = mode == tf.estimator.ModeKeys.TRAIN
        skip_summaries = mode == tf.estimator.ModeKeys.PREDICT or rebuilding
        with tf_compat.v1.variable_scope('iteration_{}'.format(iteration_number)):
            seen_builder_names = {}
            candidates = []
            summaries = []
            subnetwork_reports = {}
            previous_ensemble = None
            previous_ensemble_spec = None
            previous_iteration_checkpoint = None
            if previous_iteration:
                previous_iteration_checkpoint = previous_iteration.checkpoint
                previous_best_candidate = previous_iteration.candidates[-1]
                previous_ensemble_spec = previous_best_candidate.ensemble_spec
                previous_ensemble = previous_ensemble_spec.ensemble
                replay_indices_for_all[len(candidates)] = copy.copy(previous_ensemble_spec.architecture.replay_indices)
                seen_builder_names = {previous_ensemble_spec.name: True}
                candidates.append(previous_best_candidate)
                if self._enable_ensemble_summaries:
                    summaries.append(previous_ensemble_summary)
                if self._enable_subnetwork_reports and mode == tf.estimator.ModeKeys.EVAL:
                    metrics = previous_ensemble_spec.eval_metrics.eval_metrics_ops()
                    subnetwork_report = subnetwork.Report(hparams={}, attributes={}, metrics=metrics)
                    subnetwork_report.metrics['adanet_loss'] = tf_compat.v1.metrics.mean(previous_ensemble_spec.adanet_loss)
                    subnetwork_reports['previous_ensemble'] = subnetwork_report
            for subnetwork_builder in subnetwork_builders:
                if subnetwork_builder.name in seen_builder_names:
                    raise ValueError("Two subnetworks have the same name '{}'".format(subnetwork_builder.name))
                seen_builder_names[subnetwork_builder.name] = True
            subnetwork_specs = []
            num_subnetworks = len(subnetwork_builders)
            skip_summary = skip_summaries or not self._enable_subnetwork_summaries
            for (i, subnetwork_builder) in enumerate(subnetwork_builders):
                if not self._placement_strategy.should_build_subnetwork(num_subnetworks, i) and (not rebuilding):
                    continue
                with self._placement_strategy.subnetwork_devices(num_subnetworks, i):
                    subnetwork_name = 't{}_{}'.format(iteration_number, subnetwork_builder.name)
                    subnetwork_summary = self._summary_maker(namespace='subnetwork', scope=subnetwork_name, skip_summary=skip_summary)
                    if not skip_summary:
                        summaries.append(subnetwork_summary)
                    logging.info("%s subnetwork '%s'", 'Rebuilding' if rebuilding else 'Building', subnetwork_builder.name)
                    subnetwork_spec = self._subnetwork_manager.build_subnetwork_spec(name=subnetwork_name, subnetwork_builder=subnetwork_builder, summary=subnetwork_summary, features=features, mode=builder_mode, labels=labels, previous_ensemble=previous_ensemble, config=config)
                    subnetwork_specs.append(subnetwork_spec)
                    if not self._placement_strategy.should_build_ensemble(num_subnetworks) and (not rebuilding):
                        candidates.append(self._create_dummy_candidate(subnetwork_spec, subnetwork_builders, subnetwork_summary, training))
                if self._enable_subnetwork_reports and mode != tf.estimator.ModeKeys.PREDICT:
                    subnetwork_report = subnetwork_builder.build_subnetwork_report()
                    if not subnetwork_report:
                        subnetwork_report = subnetwork.Report(hparams={}, attributes={}, metrics={})
                    metrics = subnetwork_spec.eval_metrics.eval_metrics_ops()
                    for metric_name in sorted(metrics):
                        metric = metrics[metric_name]
                        subnetwork_report.metrics[metric_name] = metric
                    subnetwork_reports[subnetwork_builder.name] = subnetwork_report
            skip_summary = skip_summaries or not self._enable_ensemble_summaries
            seen_ensemble_names = {}
            for ensembler in self._ensemblers:
                if rebuilding and rebuilding_ensembler_name and (ensembler.name != rebuilding_ensembler_name):
                    continue
                for ensemble_candidate in ensemble_candidates:
                    if not self._placement_strategy.should_build_ensemble(num_subnetworks) and (not rebuilding):
                        continue
                    ensemble_name = 't{}_{}_{}'.format(iteration_number, ensemble_candidate.name, ensembler.name)
                    if ensemble_name in seen_ensemble_names:
                        raise ValueError("Two ensembles have the same name '{}'".format(ensemble_name))
                    seen_ensemble_names[ensemble_name] = True
                    summary = self._summary_maker(namespace='ensemble', scope=ensemble_name, skip_summary=skip_summary)
                    if not skip_summary:
                        summaries.append(summary)
                    ensemble_spec = self._ensemble_builder.build_ensemble_spec(name=ensemble_name, candidate=ensemble_candidate, ensembler=ensembler, subnetwork_specs=subnetwork_specs, summary=summary, features=features, mode=builder_mode, iteration_number=iteration_number, labels=labels, my_ensemble_index=len(candidates), previous_ensemble_spec=previous_ensemble_spec, previous_iteration_checkpoint=previous_iteration_checkpoint)
                    candidate = self._candidate_builder.build_candidate(ensemble_spec=ensemble_spec, training=training, summary=summary, rebuilding=rebuilding)
                    replay_indices_for_all[len(candidates)] = copy.copy(ensemble_spec.architecture.replay_indices)
                    candidates.append(candidate)
                    if len(ensemble_candidates) != len(subnetwork_builders):
                        continue
                    if len(ensemble_candidate.subnetwork_builders) > 1:
                        continue
                    if mode == tf.estimator.ModeKeys.PREDICT:
                        continue
                    builder_name = ensemble_candidate.subnetwork_builders[0].name
                    if self._enable_subnetwork_reports:
                        subnetwork_reports[builder_name].metrics['adanet_loss'] = tf_compat.v1.metrics.mean(ensemble_spec.adanet_loss)
            best_candidate_index = self._best_candidate_index(candidates, best_ensemble_index_override)
            best_predictions = self._best_predictions(candidates, best_candidate_index)
            best_loss = self._best_loss(candidates, best_candidate_index, mode)
            best_export_outputs = self._best_export_outputs(candidates, best_candidate_index, mode, best_predictions)
            train_manager_dir = os.path.join(config.model_dir, 'train_manager', 't{}'.format(iteration_number))
            (train_manager, training_chief_hooks, training_hooks) = self._create_hooks(base_global_step, subnetwork_specs, candidates, num_subnetworks, rebuilding, train_manager_dir, config.is_chief)
            local_init_ops = []
            if previous_ensemble_spec:
                for s in previous_ensemble_spec.ensemble.subnetworks:
                    if s.local_init_ops:
                        local_init_ops.extend(s.local_init_ops)
            for subnetwork_spec in subnetwork_specs:
                if subnetwork_spec and subnetwork_spec.subnetwork and subnetwork_spec.subnetwork.local_init_ops:
                    local_init_ops.extend(subnetwork_spec.subnetwork.local_init_ops)
            summary = self._summary_maker(namespace=None, scope=None, skip_summary=skip_summaries)
            summaries.append(summary)
            with summary.current_scope():
                summary.scalar('iteration/adanet/iteration', iteration_number)
                if best_loss is not None:
                    summary.scalar('loss', best_loss)
            iteration_metrics = _IterationMetrics(iteration_number, candidates, subnetwork_specs, self._use_tpu, replay_indices_for_all)
            checkpoint = self._make_checkpoint(candidates, subnetwork_specs, iteration_number, previous_iteration)
            if self._use_tpu:
                estimator_spec = tf_compat.v1.estimator.tpu.TPUEstimatorSpec(mode=mode, predictions=best_predictions, loss=best_loss, train_op=self._create_tpu_train_op(base_global_step, subnetwork_specs, candidates, mode, num_subnetworks, config), eval_metrics=iteration_metrics.best_eval_metrics_tuple(best_candidate_index, mode), export_outputs=best_export_outputs, training_hooks=training_hooks, scaffold_fn=self._get_scaffold_fn(local_init_ops))
            else:
                estimator_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=best_predictions, loss=best_loss, train_op=tf.no_op() if training else None, eval_metric_ops=iteration_metrics.best_eval_metric_ops(best_candidate_index, mode), export_outputs=best_export_outputs, training_chief_hooks=training_chief_hooks, training_hooks=training_hooks, scaffold=self._get_scaffold_fn(local_init_ops)())
            return _Iteration(number=iteration_number, candidates=candidates, subnetwork_specs=subnetwork_specs, estimator_spec=estimator_spec, best_candidate_index=best_candidate_index, summaries=summaries, train_manager=train_manager, subnetwork_reports=subnetwork_reports, checkpoint=checkpoint, previous_iteration=previous_iteration)

    def _get_scaffold_fn(self, local_init_ops):
        if False:
            for i in range(10):
                print('nop')
        'Creates a method generating a scaffold.\n\n    TODO: Make this code compatible with TPU estimators.\n\n    Args:\n      local_init_ops: List of tf.Operations to call during initialization.\n\n    Returns:\n      Method returning a `tf.train.Scaffold`.\n    '

        def get_scaffold():
            if False:
                i = 10
                return i + 15
            return tf_compat.v1.train.Scaffold(local_init_op=tf.group(local_init_ops + [tf_compat.v1.train.Scaffold.default_local_init_op()]))
        return get_scaffold

    def _create_dummy_candidate(self, subnetwork_spec, subnetwork_builders, subnetwork_summary, training):
        if False:
            print('Hello World!')
        'Returns a dummy candidate for the given SubnetworkSpec.\n\n    AdaNet only considers ensembles as candidate models, and ensembles\n    are represented as `_Candidates`. When training only subnetworks, such as\n    on a subnetwork-worker in the RoundRobinStrategy, then we still need a\n    candidate to manage the training of the subnetwork, even if it gets\n    discarded, hence the dummy candidate.\n\n    Args:\n      subnetwork_spec: The subnetwork spec for the dummy candidate to wrap.\n      subnetwork_builders: List of all subnetwork builders generated this\n        iteration.\n      subnetwork_summary: `_Summary` object to use for TensorBoard.\n      training: Whether or not we are currently training.\n    '
        dummy_ensemble_spec = _EnsembleSpec(name='dummy_{}'.format(subnetwork_spec.name), ensemble=None, architecture=None, subnetwork_builders=subnetwork_builders, predictions=subnetwork_spec.predictions, loss=subnetwork_spec.loss, step=None, adanet_loss=0.0, variables=[])
        return self._candidate_builder.build_candidate(ensemble_spec=dummy_ensemble_spec, training=training, summary=subnetwork_summary, track_moving_average=False)

    def _create_tpu_train_op(self, base_global_step, subnetwork_specs, candidates, mode, num_subnetworks, config):
        if False:
            for i in range(10):
                print('nop')
        'Returns the train op for this set of candidates.\n\n    This train op combines the train ops from all the candidates into a single\n    train op. Additionally, it is responsible for incrementing the global step.\n\n    The train op is only non-None during the `TRAIN` mode.\n\n    Args:\n      base_global_step: Integer global step at the beginning of this iteration.\n      subnetwork_specs: List of `_SubnetworkSpec` instances for this iteration.\n      candidates: List of `_Candidate` instances to train.\n      mode: Defines whether this is training, evaluation or inference. The train\n        op is only non-None during `TRAIN`. See `ModeKeys`.\n      num_subnetworks: Integer number of subnetwork builders generated for the\n        current iteration.\n      config: The `tf.estimator.RunConfig` to use this iteration.\n\n    Returns:\n      A `Tensor` train op.\n    '
        if mode != tf.estimator.ModeKeys.TRAIN:
            return None
        ensemble_specs = [c.ensemble_spec for c in candidates]
        with tf_compat.v1.variable_scope('train_op'):
            train_ops = []
            if self._placement_strategy.should_train_subnetworks(num_subnetworks):
                for subnetwork_spec in subnetwork_specs:
                    if subnetwork_spec.train_op is not None:
                        train_ops.append(subnetwork_spec.train_op.train_op)
            for ensemble_spec in ensemble_specs:
                if ensemble_spec.train_op is not None:
                    train_ops.append(ensemble_spec.train_op.train_op)
            with tf.control_dependencies(train_ops):
                increment_ops = [s.step.assign_add(1) for s in subnetwork_specs]
                increment_ops += [e.step.assign_add(1) for e in ensemble_specs]
                if not config.is_chief:
                    return tf.group(*increment_ops)
                with tf.control_dependencies(increment_ops):
                    steps = [s.step.read_value() for s in subnetwork_specs]
                    global_step = tf_compat.v1.train.get_global_step()
                    return global_step.assign(tf.cast(base_global_step + self._global_step_combiner_fn(steps), dtype=tf.int64))

    def _create_hooks(self, base_global_step, subnetwork_specs, candidates, num_subnetworks, rebuilding, train_manager_dir, is_chief):
        if False:
            while True:
                i = 10
        'Returns the hooks to monitor and train this iteration.\n\n    Args:\n      base_global_step: Integer global step at the beginning of this iteration.\n      subnetwork_specs: List of `_SubnetworkSpec` instances.\n      candidates: List of `_Candidate` instances to compare.\n      num_subnetworks: Integer number of subnetwork builders generated for the\n        current iteration.\n      rebuilding: Boolean whether the iteration is being rebuilt only to restore\n        the previous best subnetworks and ensembles.\n      train_manager_dir: Directory for the TrainManager to store spec metadata.\n      is_chief: Whether the current worker is chief.\n\n    Returns:\n      A 3-tuple of a _TrainManager for monitoring training, a list of\n      `SessionRunHooks` to run on chief, and a list of `SessionRunHooks` to run\n      on all workers.\n    '
        (training_chief_hooks, training_hooks) = ([], [])
        ensemble_specs = [c.ensemble_spec for c in candidates]
        train_manager = _TrainManager(subnetwork_specs, ensemble_specs, train_manager_dir, is_chief)
        if not self._use_tpu:
            training_chief_hooks.append(_GlobalStepSetterHook(train_manager, subnetwork_specs, base_global_step, self._global_step_combiner_fn))
        should_train_subnetworks = self._placement_strategy.should_train_subnetworks(num_subnetworks)
        for spec in subnetwork_specs:
            if not self._use_tpu:
                training_hooks.append(_NanLossHook(train_manager, spec))
            if self._use_tpu or not should_train_subnetworks or spec.train_op is None:
                increment_step_op = None
            else:
                with tf.control_dependencies([spec.train_op.train_op]):
                    increment_step_op = spec.step.assign_add(1)
            training_hooks.append(_TrainingLimitHook(train_manager, spec, self._max_steps, increment_step_op=increment_step_op))
            if not should_train_subnetworks and (not rebuilding):
                continue
            self._add_hooks(spec, train_manager, training_chief_hooks, training_hooks)
        for spec in ensemble_specs:
            if not self._use_tpu:
                training_hooks.append(_NanLossHook(train_manager, spec))
            if self._use_tpu or spec.train_op is None:
                increment_step_op = None
            else:
                with tf.control_dependencies([spec.train_op.train_op]):
                    increment_step_op = spec.step.assign_add(1)
            training_hooks.append(_TrainingLimitHook(train_manager, spec, self._max_steps, increment_step_op=increment_step_op))
            self._add_hooks(spec, train_manager, training_chief_hooks, training_hooks)
        return (train_manager, training_chief_hooks, training_hooks)

    def _add_hooks(self, spec, train_manager, training_chief_hooks, training_hooks):
        if False:
            print('Hello World!')
        'Appends spec train hooks to the given hook lists.'
        if not spec.train_op:
            return
        for hook in spec.train_op.chief_hooks:
            training_chief_hooks.append(_TrainingHookRunnerHook(train_manager, spec, hook))
        for hook in spec.train_op.hooks:
            training_hooks.append(_TrainingHookRunnerHook(train_manager, spec, hook))

    def _best_candidate_index(self, candidates, best_ensemble_index_override):
        if False:
            print('Hello World!')
        'Returns the index of the best candidate in the list.\n\n    The best candidate is the one with the smallest AdaNet loss, unless\n    `best_ensemble_index_override` is given.\n\n    TODO: Best ensemble index should always be static during EVAL\n    and PREDICT modes.\n\n    In case a candidate has a NaN loss, their loss is immediately set to\n    infinite, so that they are not selected. As long as one candidate ensemble\n    has a non-NaN loss during training, the dreaded `NanLossDuringTrainingError`\n    should not be raised.\n\n    Args:\n      candidates: List of `_Candidate` instances to choose from.\n      best_ensemble_index_override: Integer index to return instead of computing\n        the best ensemble index dynamically.\n\n    Returns:\n      An integer `Tensor` representing the index of the best candidate.\n    '
        with tf_compat.v1.variable_scope('best_candidate_index'):
            if best_ensemble_index_override is not None:
                return tf.constant(best_ensemble_index_override)
            if len(candidates) == 1:
                return tf.constant(0)
            adanet_losses = [candidate.adanet_loss for candidate in candidates]
            adanet_losses = tf.where(tf_compat.v1.is_nan(adanet_losses), tf.ones_like(adanet_losses) * -np.inf, adanet_losses)
            return tf.argmin(input=adanet_losses, axis=0)

    def _best_predictions(self, candidates, best_candidate_index):
        if False:
            i = 10
            return i + 15
        "Returns the best predictions from a set of candidates.\n\n    Args:\n      candidates: List of `_Candidate` instances to compare.\n      best_candidate_index: `Tensor` index of the best candidate in the list.\n\n    Returns:\n      A `Tensor` or dictionary of `Tensor`s representing the best candidate's\n      predictions (depending on what the subnetworks return).\n    "
        if len(candidates) == 1:
            return candidates[0].ensemble_spec.predictions
        with tf_compat.v1.variable_scope('best_predictions'):
            if isinstance(candidates[0].ensemble_spec.predictions, dict):
                predictions = {}
                for candidate in candidates:
                    ensemble_spec = candidate.ensemble_spec
                    for key in sorted(ensemble_spec.predictions):
                        tensor = ensemble_spec.predictions[key]
                        if key in predictions:
                            predictions[key].append(tensor)
                        else:
                            predictions[key] = [tensor]
            else:
                predictions = []
                for candidate in candidates:
                    ensemble_spec = candidate.ensemble_spec
                    predictions.append(ensemble_spec.predictions)
            if isinstance(predictions, dict):
                best_predictions = {}
                for key in sorted(predictions):
                    tensor_list = predictions[key]
                    best_predictions[key] = tf.stack(tensor_list)[best_candidate_index]
            else:
                best_predictions = tf.stack(predictions)[best_candidate_index]
            return best_predictions

    def _best_loss(self, candidates, best_candidate_index, mode):
        if False:
            i = 10
            return i + 15
        "Returns the best loss from a set of candidates.\n\n    Args:\n      candidates: List of `_Candidate` instances to compare.\n      best_candidate_index: `Tensor` index of the best candidate in the list.\n      mode: Defines whether this is training, evaluation or inference. Loss is\n        always None during inference. See `ModeKeys`.\n\n    Returns:\n      Float `Tensor` of the best candidate's loss.\n    "
        if mode == tf.estimator.ModeKeys.PREDICT:
            return None
        if len(candidates) == 1:
            return candidates[0].ensemble_spec.loss
        with tf_compat.v1.variable_scope('best_loss'):
            losses = [c.ensemble_spec.loss for c in candidates]
            loss = tf.slice(tf.stack(losses), [best_candidate_index], [1])
            return tf.reshape(loss, [])

    def _best_export_outputs(self, candidates, best_candidate_index, mode, best_predictions):
        if False:
            for i in range(10):
                print('nop')
        "Returns the best `SavedModel` export outputs from a set of candidates.\n\n    Assumes that all candidate ensembles have identical export output keys and\n    `ExportOutput` types.\n\n    Args:\n      candidates: List of `_Candidate` instances to compare.\n      best_candidate_index: `Tensor` index of the best candidate in the list.\n      mode: Defines whether this is training, evaluation or inference. Export\n        outputs are always None during training and evaluation. See `ModeKeys`.\n      best_predictions: A `Tensor` or dictionary of `Tensor`s representing the\n        best candidate's predictions (depending on what the subnetworks return).\n\n    Returns:\n      A `Tensor` dictionary representing the best candidate's export outputs.\n\n    Raises:\n      TypeError: If the `ExportOutput` type is not supported.\n    "
        if mode != tf.estimator.ModeKeys.PREDICT:
            return None
        if len(candidates) == 1:
            return candidates[0].ensemble_spec.export_outputs
        with tf_compat.v1.variable_scope('best_export_outputs'):
            export_outputs = {}
            for candidate in candidates:
                ensemble_spec = candidate.ensemble_spec
                for key in sorted(ensemble_spec.export_outputs):
                    export_output = ensemble_spec.export_outputs[key]
                    if isinstance(export_output, tf.estimator.export.ClassificationOutput):
                        if key not in export_outputs:
                            export_outputs[key] = ([], [])
                        if export_output.scores is not None:
                            export_outputs[key][0].append(export_output.scores)
                        if export_output.classes is not None:
                            export_outputs[key][1].append(export_output.classes)
                    elif isinstance(export_output, tf.estimator.export.RegressionOutput):
                        if key not in export_outputs:
                            export_outputs[key] = []
                        export_outputs[key].append(export_output.value)
                    elif isinstance(export_output, tf.estimator.export.PredictOutput):
                        continue
                    else:
                        raise TypeError('Values in export_outputs must be ClassificationOutput, RegressionOutput, or PredictOutput objects. Given: {}'.format(export_output))
            best_export_outputs = {}
            for key in sorted(candidates[0].ensemble_spec.export_outputs):
                export_output = candidates[0].ensemble_spec.export_outputs[key]
                if isinstance(export_output, tf.estimator.export.ClassificationOutput):
                    (scores, classes) = (None, None)
                    if export_outputs[key][0]:
                        scores = tf.stack(export_outputs[key][0])[best_candidate_index]
                    if export_outputs[key][1]:
                        classes = tf.stack(export_outputs[key][1])[best_candidate_index]
                    output = tf.estimator.export.ClassificationOutput(scores=scores, classes=classes)
                elif isinstance(export_output, tf.estimator.export.RegressionOutput):
                    value = tf.stack(export_outputs[key])[best_candidate_index]
                    output = tf.estimator.export.RegressionOutput(value)
                else:
                    predictions = copy.copy(export_output.outputs)
                    predictions.update(best_predictions)
                    output = tf.estimator.export.PredictOutput(predictions)
                best_export_outputs[key] = output
            return best_export_outputs

    def _make_checkpoint(self, candidates, subnetwork_specs, iteration_number, previous_iteration):
        if False:
            for i in range(10):
                print('nop')
        'Returns a `tf.train.Checkpoint` for the iteration.'
        trackable = {}
        for candidate in candidates:
            for ensemble_var in candidate.ensemble_spec.variables:
                trackable['{}_{}'.format(candidate.ensemble_spec.name, ensemble_var.name)] = ensemble_var
            for candidate_var in candidate.variables:
                trackable['candidate_{}_{}'.format(candidate.ensemble_spec.name, candidate_var.name)] = candidate_var
        for subnetwork_spec in subnetwork_specs:
            for subnetwork_var in subnetwork_spec.variables:
                trackable['{}_{}'.format(subnetwork_spec.name, subnetwork_var.name)] = subnetwork_var
        global_step = tf_compat.v1.train.get_global_step()
        if global_step is not None:
            trackable[tf_compat.v1.GraphKeys.GLOBAL_STEP] = global_step
        trackable['iteration_number'] = tf_compat.v1.get_variable('iteration_number', dtype=tf.int64, initializer=lambda : tf.constant(iteration_number, dtype=tf.int64), trainable=False)
        if previous_iteration:
            trackable['previous_iteration'] = previous_iteration.checkpoint
        logging.info('TRACKABLE: %s', trackable)
        checkpoint = tf_compat.v2.train.Checkpoint(**trackable)
        checkpoint.save_counter
        return checkpoint