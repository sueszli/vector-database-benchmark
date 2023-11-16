"""AdaNet metrics objects and functions.

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
import collections
import inspect
from absl import logging
from adanet import tf_compat
import six
import tensorflow.compat.v2 as tf

def _call_eval_metrics(eval_metrics):
    if False:
        for i in range(10):
            print('nop')
    if not eval_metrics:
        return {}
    (fn, args) = eval_metrics
    if isinstance(args, dict):
        return fn(**args)
    else:
        return fn(*args)

class _EvalMetricsStore(object):
    """Stores and manipulate eval_metric tuples."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._metric_fns = []
        self._args = []

    def add_eval_metrics(self, metric_fn, args):
        if False:
            print('Hello World!')
        'Adds an eval_metrics tuple to the internal store.'
        self._metric_fns.append(metric_fn)
        self._args.append(args)

    @property
    def metric_fns(self):
        if False:
            print('Hello World!')
        return self._metric_fns

    def flatten_args(self):
        if False:
            return 10
        'Flattens the eval_metrics arguments to a list.'
        from tensorflow.python.util import nest
        return nest.flatten(self._args)

    def pack_args(self, args):
        if False:
            print('Hello World!')
        'Packs the given list of arguments into the internal args structure.'
        from tensorflow.python.util import nest
        return nest.pack_sequence_as(self._args, args)

class _SubnetworkMetrics(object):
    """A object which creates evaluation metrics for Subnetworks."""

    def __init__(self, use_tpu=False):
        if False:
            while True:
                i = 10
        'Creates a _SubnetworkMetrics.\n\n    Args:\n      use_tpu: Whether to use TPU-specific variable sharing logic. This ensures\n        that eval metrics created on TPU can be written to disk on the host CPU.\n\n    Returns:\n      A `_SubnetworkMetrics` instance.\n    '
        self._use_tpu = use_tpu
        self._eval_metrics_store = _EvalMetricsStore()

    def create_eval_metrics(self, features, labels, estimator_spec, metric_fn):
        if False:
            for i in range(10):
                print('nop')
        'Creates evaluation metrics from the given arguments.\n\n    Args:\n      features: Input `dict` of `Tensor` objects.\n      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`\n        (for multi-head).\n      estimator_spec: The `EstimatorSpec` created by a `Head` instance.\n      metric_fn: A function which should obey the following signature:\n      - Args: can only have following three arguments in any order:\n        * predictions: Predictions `Tensor` or dict of `Tensor` created by given\n          `Head`.\n        * features: Input `dict` of `Tensor` objects created by `input_fn` which\n          is given to `estimator.evaluate` as an argument.\n        * labels:  Labels `Tensor` or dict of `Tensor` (for multi-head) created\n          by `input_fn` which is given to `estimator.evaluate` as an argument.\n      - Returns: Dict of metric results keyed by name. Final metrics are a union\n        of this and `estimator`s existing metrics. If there is a name conflict\n        between this and `estimator`s existing metrics, this will override the\n        existing one. The values of the dict are the results of calling a metric\n        function, namely a `(metric_tensor, update_op)` tuple.\n    '
        if isinstance(estimator_spec, tf.estimator.EstimatorSpec):
            (spec_fn, spec_args) = (lambda : estimator_spec.eval_metric_ops, [])
        else:
            (spec_fn, spec_args) = estimator_spec.eval_metrics
        self._eval_metrics_store.add_eval_metrics(self._templatize_metric_fn(spec_fn), spec_args)
        loss_fn = lambda loss: {'loss': tf_compat.v1.metrics.mean(loss)}
        loss_fn_args = [tf.reshape(estimator_spec.loss, [1])]
        if not self._use_tpu:
            loss_ops = _call_eval_metrics((loss_fn, loss_fn_args))
            (loss_fn, loss_fn_args) = (lambda : loss_ops, [])
        self._eval_metrics_store.add_eval_metrics(self._templatize_metric_fn(loss_fn), loss_fn_args)
        if metric_fn:
            metric_fn_args = {}
            argspec = inspect.getargs(metric_fn.__code__).args
            if 'features' in argspec:
                metric_fn_args['features'] = features
            if 'labels' in argspec:
                metric_fn_args['labels'] = labels
            if 'predictions' in argspec:
                metric_fn_args['predictions'] = estimator_spec.predictions
            if not self._use_tpu:
                metric_fn_ops = _call_eval_metrics((metric_fn, metric_fn_args))
                (metric_fn, metric_fn_args) = (lambda : metric_fn_ops, [])
            self._eval_metrics_store.add_eval_metrics(self._templatize_metric_fn(metric_fn), metric_fn_args)

    def _templatize_metric_fn(self, metric_fn):
        if False:
            while True:
                i = 10
        "Wraps the given metric_fn with a template so it's Variables are shared.\n\n    Hooks on TPU cannot depend on any graph Tensors. Instead the eval metrics\n    returned by metric_fn are stored in Variables. These variables are later\n    read from the evaluation hooks which run on the host CPU.\n\n    Args:\n      metric_fn: The function to wrap with a template.\n\n    Returns:\n      The original metric_fn wrapped with a template function.\n    "

        def _metric_fn(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            'The wrapping function to be returned.'
            args = args if args else kwargs
            metrics = _call_eval_metrics((metric_fn, args))
            if not self._use_tpu:
                return metrics
            logging.log_first_n(logging.INFO, 'Writing eval metrics to variables for TPU', 1)
            wrapped_metrics = {}
            for (i, key) in enumerate(sorted(metrics)):
                (tensor, op) = tf_compat.metric_op(metrics[key])
                var = tf_compat.v1.get_variable('metric_{}'.format(i), shape=tensor.shape, dtype=tensor.dtype, trainable=False, initializer=tf_compat.v1.zeros_initializer(), collections=[tf_compat.v1.GraphKeys.LOCAL_VARIABLES])
                if isinstance(op, tf.Operation) or op.shape != tensor.shape:
                    with tf.control_dependencies([op]):
                        op = var.assign(tensor)
                metric = (var, var.assign(op))
                wrapped_metrics[key] = metric
            return wrapped_metrics
        return tf_compat.v1.make_template('metric_fn_template', _metric_fn)

    def eval_metrics_tuple(self):
        if False:
            print('Hello World!')
        'Returns tuple of (metric_fn, tensors) which can be executed on TPU.'
        if not self._eval_metrics_store.metric_fns:
            return None

        def _metric_fn(*args):
            if False:
                while True:
                    i = 10
            metric_fns = self._eval_metrics_store.metric_fns
            metric_fn_args = self._eval_metrics_store.pack_args(args)
            eval_metric_ops = {}
            for (metric_fn, args) in zip(metric_fns, metric_fn_args):
                eval_metric_ops.update(_call_eval_metrics((metric_fn, args)))
            return eval_metric_ops
        return (_metric_fn, self._eval_metrics_store.flatten_args())

    def eval_metrics_ops(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the eval_metrics_ops.'
        return _call_eval_metrics(self.eval_metrics_tuple())

class _EnsembleMetrics(_SubnetworkMetrics):
    """A object which creates evaluation metrics for Ensembles."""

    def create_eval_metrics(self, features, labels, estimator_spec, metric_fn, architecture):
        if False:
            i = 10
            return i + 15
        "Overrides parent's method to also add the ensemble's architecture."
        super(_EnsembleMetrics, self).create_eval_metrics(features, labels, estimator_spec, metric_fn)
        self._eval_metrics_store.add_eval_metrics(self._architecture_as_metric(architecture), [])

    def _architecture_as_metric(self, architecture):
        if False:
            print('Hello World!')
        "Returns a representation of an ensemble's architecture as a tf.metric."

        def _architecture_metric_fn():
            if False:
                print('Hello World!')
            'Manually creates the tf.metric with a serialized tf.Summary proto.'
            architecture_ = ' | '.join([name for (_, name) in architecture.subnetworks])
            architecture_ = '| {} |'.format(architecture_)
            summary_metadata = tf_compat.v1.SummaryMetadata(plugin_data=tf_compat.v1.SummaryMetadata.PluginData(plugin_name='text'))
            summary_proto = tf_compat.v1.summary.Summary()
            summary_proto.value.add(metadata=summary_metadata, tag='architecture/adanet', tensor=tf_compat.v1.make_tensor_proto(architecture_, dtype=tf.string))
            architecture_summary = tf.convert_to_tensor(value=summary_proto.SerializeToString(), name='architecture')
            return {'architecture/adanet/ensembles': (architecture_summary, tf.no_op())}
        if not self._use_tpu:
            ops = _architecture_metric_fn()
            return lambda : ops
        else:
            return _architecture_metric_fn

class _IterationMetrics(object):
    """A object which creates evaluation metrics for an Iteration."""

    def __init__(self, iteration_number, candidates, subnetwork_specs, use_tpu=False, replay_indices_for_all=None):
        if False:
            i = 10
            return i + 15
        self._iteration_number = iteration_number
        self._candidates = candidates
        self._subnetwork_specs = subnetwork_specs
        self._use_tpu = use_tpu
        self._replay_indices_for_all = replay_indices_for_all
        self._candidates_eval_metrics_store = self._build_eval_metrics_store([candidate.ensemble_spec for candidate in self._candidates])
        self._subnetworks_eval_metrics_store = self._build_eval_metrics_store(self._subnetwork_specs)
        self._best_eval_metrics_tuple = None

    def _build_eval_metrics_store(self, specs):
        if False:
            print('Hello World!')
        'Creates an _EvalMetricsStore from Subnetwork or Ensemble specs.'
        store = _EvalMetricsStore()
        for spec in specs:
            if not spec.eval_metrics or not spec.eval_metrics.eval_metrics_tuple():
                continue
            (metric_fn, args) = spec.eval_metrics.eval_metrics_tuple()
            store.add_eval_metrics(metric_fn, args)
        return store

    def best_eval_metric_ops(self, best_candidate_index, mode):
        if False:
            while True:
                i = 10
        "Returns best ensemble's metrics."
        return _call_eval_metrics(self.best_eval_metrics_tuple(best_candidate_index, mode))

    def best_eval_metrics_tuple(self, best_candidate_index, mode):
        if False:
            i = 10
            return i + 15
        "Returns (metric_fn, tensors) which computes the best ensemble's metrics.\n\n    Specifically, when metric_fn(tensors) is called, it separates the metric ops\n    by metric name. All candidates are not required to have the same metrics.\n    When they all share a given metric, an additional metric is added which\n    represents that of the best candidate.\n\n    Args:\n      best_candidate_index: `Tensor` index of the best candidate in the list.\n      mode: Defines whether this is training, evaluation or inference. Eval\n        metrics are only defined during evaluation. See `ModeKeys`.\n\n    Returns:\n      Dict of metric results keyed by name. The values of the dict are the\n      results of calling a metric function.\n    "
        if mode != tf.estimator.ModeKeys.EVAL:
            return None
        candidate_args = self._candidates_eval_metrics_store.flatten_args()
        subnetwork_args = self._subnetworks_eval_metrics_store.flatten_args()
        args = candidate_args + subnetwork_args
        args.append(tf.reshape(best_candidate_index, [1]))

        def _replay_eval_metrics(best_candidate_idx, eval_metric_ops):
            if False:
                print('Hello World!')
            'Saves replay indices as eval metrics.'
            pad_value = max([len(v) for (_, v) in self._replay_indices_for_all.items()])
            replay_indices_as_tensor = tf.constant([value + [-1] * (pad_value - len(value)) for (_, value) in self._replay_indices_for_all.items()])
            for iteration in range(replay_indices_as_tensor.get_shape().as_list()[1]):
                index_t = replay_indices_as_tensor[best_candidate_idx, iteration]
                eval_metric_ops['best_ensemble_index_{}'.format(iteration)] = (index_t, index_t)

        def _best_eval_metrics_fn(*args):
            if False:
                for i in range(10):
                    print('nop')
            'Returns the best eval metrics.'
            with tf_compat.v1.variable_scope('best_eval_metrics'):
                args = list(args)
                (idx, idx_update_op) = tf_compat.v1.metrics.mean(args.pop())
                idx = tf.cast(idx, tf.int32)
                metric_fns = self._candidates_eval_metrics_store.metric_fns
                metric_fn_args = self._candidates_eval_metrics_store.pack_args(args[:len(candidate_args)])
                candidate_grouped_metrics = self._group_metric_ops(metric_fns, metric_fn_args)
                metric_fns = self._subnetworks_eval_metrics_store.metric_fns
                metric_fn_args = self._subnetworks_eval_metrics_store.pack_args(args[len(args) - len(subnetwork_args):])
                subnetwork_grouped_metrics = self._group_metric_ops(metric_fns, metric_fn_args)
                eval_metric_ops = {}
                for metric_name in sorted(candidate_grouped_metrics):
                    metric_ops = candidate_grouped_metrics[metric_name]
                    if len(metric_ops) != len(self._candidates):
                        continue
                    if metric_name == 'loss':
                        continue
                    (values, ops) = list(six.moves.zip(*metric_ops))
                    best_value = tf.stack(values)[idx]
                    ops = list(ops)
                    ops.append(idx_update_op)
                    ensemble_loss_ops = candidate_grouped_metrics.get('loss', tf.no_op())
                    all_ops = tf.group(ops, ensemble_loss_ops, subnetwork_grouped_metrics)
                    eval_metric_ops[metric_name] = (best_value, all_ops)
                iteration_number = tf.constant(self._iteration_number)
                eval_metric_ops['iteration'] = (iteration_number, iteration_number)
                if self._replay_indices_for_all:
                    _replay_eval_metrics(idx, eval_metric_ops)
                assert 'loss' not in eval_metric_ops
                return eval_metric_ops
        if not self._use_tpu:
            if not self._best_eval_metrics_tuple:
                best_ops = _call_eval_metrics((_best_eval_metrics_fn, args))
                self._best_eval_metrics_tuple = (lambda : best_ops, [])
            return self._best_eval_metrics_tuple
        return (_best_eval_metrics_fn, args)

    def _group_metric_ops(self, metric_fns, metric_fn_args):
        if False:
            i = 10
            return i + 15
        'Runs the metric_fns and groups the returned metric ops by name.\n\n    Args:\n      metric_fns: The eval_metrics functions to run.\n      metric_fn_args: The eval_metrics function arguments.\n\n    Returns:\n      The metric ops grouped by name.\n    '
        grouped_metrics = collections.defaultdict(list)
        for (metric_fn, args) in zip(metric_fns, metric_fn_args):
            eval_metric_ops = _call_eval_metrics((metric_fn, args))
            for metric_name in sorted(eval_metric_ops):
                metric_op = tf_compat.metric_op(eval_metric_ops[metric_name])
                grouped_metrics[metric_name].append(metric_op)
        return grouped_metrics