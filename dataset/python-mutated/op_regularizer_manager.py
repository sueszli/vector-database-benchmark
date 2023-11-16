"""A class for managing OpRegularizers.

OpRegularizerManager creates the required regulrizers and manages the
association between ops and their regularizers. OpRegularizerManager handles the
logic associated with the graph topology:
- Concatenating tensors is reflected in concatenating their regularizers.
- Skip-connections (aka residual connections), RNNs and other structures where
  the shapes of two (or more) tensors are tied together are reflected in
  grouping their regularizers together.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import logging
import tensorflow as tf
from morph_net.framework import concat_and_slice_regularizers
from morph_net.framework import generic_regularizers
from morph_net.framework import grouping_regularizers
_GROUPING_OPS = ('Add', 'Sub', 'Mul', 'Div', 'Maximum', 'Minimum', 'SquaredDifference', 'RealDiv')
NON_PASS_THROUGH_OPS = ('Conv2D', 'Conv2DBackpropInput', 'MatMul')

def _remove_nones_and_dups(items):
    if False:
        while True:
            i = 10
    result = []
    for i in items:
        if i is not None and i not in result:
            result.append(i)
    return result

def _raise_type_error_if_not_operation(op):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(op, tf.Operation):
        raise TypeError("'op' must be of type tf.Operation, not %s" % str(type(op)))

class OpRegularizerManager(object):
    """A class for managing OpRegularizers."""

    def __init__(self, ops, op_regularizer_factory_dict, create_grouping_regularizer=None):
        if False:
            for i in range(10):
                print('nop')
        "Creates an instance.\n\n    Args:\n      ops: A list of tf.Operation-s. An OpRegularizer will be created for all\n        the ops in `ops`, and recursively for all ops they depend on via data\n        dependency. Typically `ops` would contain a single tf.Operation, which\n        is the output of the network.\n      op_regularizer_factory_dict: A dictionary, where the keys are strings\n        representing TensorFlow Op types, and the values are callables that\n        create the respective OpRegularizers. For every op encountered during\n        the recursion, if op.type is in op_regularizer_factory_dict, the\n        respective callable will be used to create an OpRegularizer. The\n        signature of the callables is the following args;\n          op; a tf.Operation for which to create a regularizer.\n          opreg_manager; A reference to an OpRegularizerManager object. Can be\n            None if the callable does not need access to OpRegularizerManager.\n      create_grouping_regularizer: A callable that has the signature of\n        grouping_regularizers.MaxGroupingRegularizer's constructor. Will be\n        called whenever a grouping op (see _GROUPING_OPS) is encountered.\n        Defaults to MaxGroupingRegularizer if None.\n\n    Raises:\n      ValueError: If ops is not a list.\n    "
        self._constructed = False
        if not isinstance(ops, list):
            raise ValueError('Input %s ops is not a list. Should probably use []' % str(ops))
        self._op_to_regularizer = {}
        self._regularizer_to_ops = collections.defaultdict(list)
        self._op_regularizer_factory_dict = op_regularizer_factory_dict
        for op_type in NON_PASS_THROUGH_OPS:
            if op_type not in self._op_regularizer_factory_dict:
                self._op_regularizer_factory_dict[op_type] = lambda x, y: None
        self._create_grouping_regularizer = create_grouping_regularizer or grouping_regularizers.MaxGroupingRegularizer
        self._visited = set()
        for op in ops:
            self._get_regularizer(op)
        self._constructed = True

    def get_regularizer(self, op):
        if False:
            i = 10
            return i + 15
        'Looks up or creates an OpRegularizer for a tf.Operation.\n\n    Args:\n      op: A tf.Operation.\n\n    - If `self` has an OpRegularizer for `op`, it will be returned.\n      Otherwise:\n    - If called before construction of `self` was completed (that is, from the\n      constructor), an attempt to create an OpRegularizer for `op` will be made.\n      Otherwise:\n    - If called after contstruction of `self` was completed, an exception will\n      be raised.\n\n    Returns:\n      An OpRegularizer for `op`. Can be None if `op` is not regularized (e.g.\n      `op` is a constant).\n\n    Raises:\n      RuntimeError: If `self` object has no OpRegularizer for `op` in its\n        lookup table, and the construction of `self` has already been completed\n        (because them `self` is immutable and an OpRegularizer cannot be\n        created).\n    '
        try:
            return self._op_to_regularizer[op]
        except KeyError:
            if self._constructed:
                raise ValueError('Op %s does not have a regularizer.' % op.name)
            else:
                return self._get_regularizer(op)

    @property
    def ops(self):
        if False:
            for i in range(10):
                print('nop')
        return self._op_to_regularizer.keys()

    def group_and_replace_regularizers(self, regularizers):
        if False:
            i = 10
            return i + 15
        'Groups a list of OpRegularizers and replaces them by the grouped one.\n\n    Args:\n      regularizers: A list of OpRegularizer objects to be grouped.\n\n    Returns:\n      An OpRegularizer object formed by the grouping.\n\n    Raises:\n      RuntimeError: group_and_replace_regularizers was called affter\n         construction of the OpRegularizerManager object was completed.\n    '
        if self._constructed:
            raise RuntimeError('group_and_replace_regularizers can only be called before construction of the OpRegularizerManager was completed.')
        grouped = self._create_grouping_regularizer(regularizers)
        for r in regularizers:
            self._replace_regularizer(r, grouped)
        return grouped

    def _get_regularizer(self, op):
        if False:
            return 10
        "Fetches the regularizer of `op` if exists, creates it otherwise.\n\n    This function calls itself recursively, directly or via _create_regularizer\n    (which in turn calls _get_regularizer). It performs DFS along the data\n    dependencies of the graph, and uses a self._visited set to detect loops. The\n    use of self._visited makes it not thread safe, but _get_regularizer is a\n    private method that is supposed to only be called form the constructor, so\n    execution in multiple threads (for the same object) is not expected.\n\n    Args:\n      op: A Tf.Operation.\n\n    Returns:\n      An OpRegularizer that corresponds to `op`, or None if op does not have\n      a regularizer (e. g. it's a constant op).\n    "
        _raise_type_error_if_not_operation(op)
        if op not in self._op_to_regularizer:
            if op in self._visited:
                return None
            self._visited.add(op)
            regularizer = self._create_regularizer(op)
            self._op_to_regularizer[op] = regularizer
            self._regularizer_to_ops[regularizer].append(op)
            for i in op.inputs:
                self._get_regularizer(i.op)
            self._visited.remove(op)
        return self._op_to_regularizer[op]

    def _create_regularizer(self, op):
        if False:
            print('Hello World!')
        'Creates an OpRegularizer for `op`.\n\n    Args:\n      op: A Tf.Operation.\n\n    Returns:\n      An OpRegularizer that corresponds to `op`, or None if op does not have\n      a regularizer.\n\n    Raises:\n      RuntimeError: Grouping is attempted at op which is not whitelisted for\n        grouping (in _GROUPING_OPS).\n    '
        if op.type in self._op_regularizer_factory_dict:
            regularizer = self._op_regularizer_factory_dict[op.type](op, self)
            if regularizer is None:
                logging.warning('Failed to create regularizer for %s.', op.name)
            else:
                logging.info('Created regularizer for %s.', op.name)
            return regularizer
        if not op.inputs:
            return None
        if op.type == 'ConcatV2':
            return self._create_concat_regularizer(op)
        inputs_regularizers = _remove_nones_and_dups([self._get_regularizer(i.op) for i in op.inputs])
        if not inputs_regularizers:
            return None
        elif len(inputs_regularizers) == 1:
            return inputs_regularizers[0]
        elif op.type in _GROUPING_OPS:
            return self.group_and_replace_regularizers(inputs_regularizers)
        raise RuntimeError('Grouping is attempted at op which is not whitelisted for grouping: %s' % str(op.type))

    def _create_concat_regularizer(self, concat_op):
        if False:
            i = 10
            return i + 15
        'Creates an OpRegularizer for a concat op.\n\n    Args:\n      concat_op: A tf.Operation of type ConcatV2.\n\n    Returns:\n      An OpRegularizer for `concat_op`.\n    '
        input_ops = [i.op for i in concat_op.inputs[:-1]]
        regularizers_to_concat = [self._get_regularizer(op) for op in input_ops]
        if regularizers_to_concat == [None] * len(regularizers_to_concat):
            return None
        offset = 0
        ops_to_concat = []
        for (r, op) in zip(regularizers_to_concat, input_ops):
            if r is None:
                length = op.outputs[0].shape.as_list()[-1]
                offset += length
                ops_to_concat.append(self._ConstantOpReg(length))
            else:
                length = tf.shape(r.alive_vector)[0]
                slice_ref = concat_and_slice_regularizers.SlicingReferenceRegularizer(lambda : self._get_regularizer(concat_op), offset, length)
                offset += length
                self._replace_regularizer(r, slice_ref)
                ops_to_concat.append(r)
        return concat_and_slice_regularizers.ConcatRegularizer(ops_to_concat)

    def _replace_regularizer(self, source, target):
        if False:
            i = 10
            return i + 15
        "Replaces `source` by 'target' in self's lookup tables."
        for op in self._regularizer_to_ops[source]:
            assert self._op_to_regularizer[op] is source
            self._op_to_regularizer[op] = target
            self._regularizer_to_ops[target].append(op)
        del self._regularizer_to_ops[source]

    class _ConstantOpReg(generic_regularizers.OpRegularizer):
        """A class with the constant alive property, and zero regularization."""

        def __init__(self, size):
            if False:
                print('Hello World!')
            self._regularization_vector = tf.zeros(size)
            self._alive_vector = tf.cast(tf.ones(size), tf.bool)

        @property
        def regularization_vector(self):
            if False:
                print('Hello World!')
            return self._regularization_vector

        @property
        def alive_vector(self):
            if False:
                for i in range(10):
                    print('nop')
            return self._alive_vector