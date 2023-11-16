"""Tests for warm_starting_util."""
import os
import numpy as np
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import warm_starting_util as ws_util
ones = init_ops.ones_initializer
norms = init_ops.truncated_normal_initializer
rand = init_ops.random_uniform_initializer
zeros = init_ops.zeros_initializer

class WarmStartingUtilTest(test.TestCase):

    def _write_vocab(self, string_values, file_name):
        if False:
            i = 10
            return i + 15
        vocab_file = os.path.join(self.get_temp_dir(), file_name)
        with open(vocab_file, 'w') as f:
            f.write('\n'.join(string_values))
        return vocab_file

    def _write_checkpoint(self, sess):
        if False:
            while True:
                i = 10
        self.evaluate(variables.global_variables_initializer())
        saver = saver_lib.Saver()
        ckpt_prefix = os.path.join(self.get_temp_dir(), 'model')
        saver.save(sess, ckpt_prefix, global_step=0)

    def _create_prev_run_var(self, var_name, shape=None, initializer=None, partitioner=None):
        if False:
            return 10
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                var = variable_scope.get_variable(var_name, shape=shape, initializer=initializer, partitioner=partitioner)
                self._write_checkpoint(sess)
                if partitioner:
                    self.assertTrue(isinstance(var, variables.PartitionedVariable))
                    var = var._get_variable_list()
                return (var, self.evaluate(var))

    def _create_prev_run_vars(self, var_names, shapes, initializers):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                all_vars = []
                for (var_name, shape, initializer) in zip(var_names, shapes, initializers):
                    all_vars.append(variable_scope.get_variable(var_name, shape=shape, initializer=initializer))
                self._write_checkpoint(sess)
                return [self.evaluate(var) for var in all_vars]

    def _create_dummy_inputs(self):
        if False:
            return 10
        return {'sc_int': array_ops.sparse_placeholder(dtypes.int32), 'sc_hash': array_ops.sparse_placeholder(dtypes.string), 'sc_keys': array_ops.sparse_placeholder(dtypes.string), 'sc_vocab': array_ops.sparse_placeholder(dtypes.string), 'real': array_ops.placeholder(dtypes.float32)}

    def _create_linear_model(self, feature_cols, partitioner):
        if False:
            i = 10
            return i + 15
        cols_to_vars = {}
        with variable_scope.variable_scope('', partitioner=partitioner):
            fc.linear_model(features=self._create_dummy_inputs(), feature_columns=feature_cols, units=1, cols_to_vars=cols_to_vars)
        return cols_to_vars

    def _assert_cols_to_vars(self, cols_to_vars, cols_to_expected_values, sess):
        if False:
            i = 10
            return i + 15
        for (col, expected_values) in cols_to_expected_values.items():
            for (i, var) in enumerate(cols_to_vars[col]):
                self.assertAllClose(expected_values[i], var.eval(sess))

    def testWarmStartVar(self):
        if False:
            for i in range(10):
                print('nop')
        (_, prev_val) = self._create_prev_run_var('fruit_weights', initializer=[[0.5], [1.0], [1.5], [2.0]])
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_weights = variable_scope.get_variable('fruit_weights', initializer=[[0.0], [0.0], [0.0], [0.0]])
                (prev_tensor_name, var) = ws_util._get_var_info(fruit_weights)
                checkpoint_utils.init_from_checkpoint(self.get_temp_dir(), {prev_tensor_name: var})
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose(prev_val, fruit_weights.eval(sess))

    def testWarmStartVarPrevVarPartitioned(self):
        if False:
            return 10
        (_, weights) = self._create_prev_run_var('fruit_weights', shape=[4, 1], initializer=[[0.5], [1.0], [1.5], [2.0]], partitioner=lambda shape, dtype: [2, 1])
        prev_val = np.concatenate([weights[0], weights[1]], axis=0)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_weights = variable_scope.get_variable('fruit_weights', initializer=[[0.0], [0.0], [0.0], [0.0]])
                (prev_tensor_name, var) = ws_util._get_var_info(fruit_weights)
                checkpoint_utils.init_from_checkpoint(self.get_temp_dir(), {prev_tensor_name: var})
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose(prev_val, fruit_weights.eval(sess))

    def testWarmStartVarCurrentVarPartitioned(self):
        if False:
            return 10
        (_, prev_val) = self._create_prev_run_var('fruit_weights', initializer=[[0.5], [1.0], [1.5], [2.0]])
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_weights = variable_scope.get_variable('fruit_weights', shape=[4, 1], initializer=[[0.0], [0.0], [0.0], [0.0]], partitioner=lambda shape, dtype: [2, 1])
                self.assertTrue(isinstance(fruit_weights, variables.PartitionedVariable))
                (prev_tensor_name, var) = ws_util._get_var_info(fruit_weights)
                checkpoint_utils.init_from_checkpoint(self.get_temp_dir(), {prev_tensor_name: var})
                self.evaluate(variables.global_variables_initializer())
                fruit_weights = fruit_weights._get_variable_list()
                new_val = np.concatenate([fruit_weights[0].eval(sess), fruit_weights[1].eval(sess)], axis=0)
                self.assertAllClose(prev_val, new_val)

    def testWarmStartVarBothVarsPartitioned(self):
        if False:
            return 10
        (_, weights) = self._create_prev_run_var('old_scope/fruit_weights', shape=[4, 1], initializer=[[0.5], [1.0], [1.5], [2.0]], partitioner=lambda shape, dtype: [2, 1])
        prev_val = np.concatenate([weights[0], weights[1]], axis=0)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_weights = variable_scope.get_variable('new_scope/fruit_weights', shape=[4, 1], initializer=[[0.0], [0.0], [0.0], [0.0]], partitioner=lambda shape, dtype: [2, 1])
                self.assertTrue(isinstance(fruit_weights, variables.PartitionedVariable))
                (prev_tensor_name, var) = ws_util._get_var_info(fruit_weights, prev_tensor_name='old_scope/fruit_weights')
                checkpoint_utils.init_from_checkpoint(self.get_temp_dir(), {prev_tensor_name: var})
                self.evaluate(variables.global_variables_initializer())
                fruit_weights = fruit_weights._get_variable_list()
                new_val = np.concatenate([fruit_weights[0].eval(sess), fruit_weights[1].eval(sess)], axis=0)
                self.assertAllClose(prev_val, new_val)

    def testWarmStartVarWithVocab(self):
        if False:
            i = 10
            return i + 15
        prev_vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'old_vocab')
        self._create_prev_run_var('fruit_weights', initializer=[[0.5], [1.0], [1.5], [2.0]])
        new_vocab_path = self._write_vocab(['orange', 'guava', 'banana', 'apple', 'raspberry'], 'new_vocab')
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_weights = variable_scope.get_variable('fruit_weights', initializer=[[0.0], [0.0], [0.0], [0.0], [0.0]])
                ws_util._warm_start_var_with_vocab(fruit_weights, new_vocab_path, 5, self.get_temp_dir(), prev_vocab_path)
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose([[2.0], [1.5], [1.0], [0.5], [0.0]], fruit_weights.eval(sess))

    def testWarmStartVarWithColumnVocab(self):
        if False:
            while True:
                i = 10
        prev_vocab_path = self._write_vocab(['apple', 'orange'], 'old_vocab')
        self._create_prev_run_var('fruit_output_layer', initializer=[[0.5, 0.3], [1.0, 0.8], [1.5, 1.2], [2.0, 2.3]])
        new_vocab_path = self._write_vocab(['orange', 'apple', 'banana'], 'new_vocab')
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_output_layer = variable_scope.get_variable('fruit_output_layer', initializer=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                ws_util._warm_start_var_with_vocab(fruit_output_layer, new_vocab_path, current_vocab_size=3, prev_ckpt=self.get_temp_dir(), prev_vocab_path=prev_vocab_path, axis=1)
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose([[0.3, 0.5, 0.0], [0.8, 1.0, 0.0], [1.2, 1.5, 0.0], [2.3, 2.0, 0.0]], fruit_output_layer.eval(sess))

    def testWarmStartVarWithVocabConstrainedOldVocabSize(self):
        if False:
            for i in range(10):
                print('nop')
        prev_vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'old_vocab')
        self._create_prev_run_var('fruit_weights', initializer=[[0.5], [1.0], [1.5], [2.0]])
        new_vocab_path = self._write_vocab(['orange', 'guava', 'banana', 'apple', 'raspberry'], 'new_vocab')
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_weights = variable_scope.get_variable('fruit_weights', initializer=[[0.0], [0.0], [0.0], [0.0], [0.0]])
                ws_util._warm_start_var_with_vocab(fruit_weights, new_vocab_path, 5, self.get_temp_dir(), prev_vocab_path, previous_vocab_size=2)
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose([[0.0], [0.0], [1.0], [0.5], [0.0]], fruit_weights.eval(sess))

    def testWarmStartVarWithVocabPrevVarPartitioned(self):
        if False:
            while True:
                i = 10
        prev_vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'old_vocab')
        self._create_prev_run_var('fruit_weights', shape=[4, 1], initializer=[[0.5], [1.0], [1.5], [2.0]], partitioner=lambda shape, dtype: [2, 1])
        new_vocab_path = self._write_vocab(['orange', 'guava', 'banana', 'apple', 'raspberry'], 'new_vocab')
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_weights = variable_scope.get_variable('fruit_weights', initializer=[[0.0], [0.0], [0.0], [0.0], [0.0]])
                ws_util._warm_start_var_with_vocab(fruit_weights, new_vocab_path, 5, self.get_temp_dir(), prev_vocab_path)
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose([[2.0], [1.5], [1.0], [0.5], [0.0]], fruit_weights.eval(sess))

    def testWarmStartVarWithColumnVocabPrevVarPartitioned(self):
        if False:
            print('Hello World!')
        prev_vocab_path = self._write_vocab(['apple', 'orange'], 'old_vocab')
        self._create_prev_run_var('fruit_output_layer', shape=[4, 2], initializer=[[0.5, 0.3], [1.0, 0.8], [1.5, 1.2], [2.0, 2.3]], partitioner=lambda shape, dtype: [2, 1])
        new_vocab_path = self._write_vocab(['orange', 'apple', 'banana'], 'new_vocab')
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_output_layer = variable_scope.get_variable('fruit_output_layer', initializer=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                ws_util._warm_start_var_with_vocab(fruit_output_layer, new_vocab_path, current_vocab_size=3, prev_ckpt=self.get_temp_dir(), prev_vocab_path=prev_vocab_path, axis=1)
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose([[0.3, 0.5, 0.0], [0.8, 1.0, 0.0], [1.2, 1.5, 0.0], [2.3, 2.0, 0.0]], fruit_output_layer.eval(sess))

    def testWarmStartVarWithVocabCurrentVarPartitioned(self):
        if False:
            return 10
        prev_vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'old_vocab')
        self._create_prev_run_var('fruit_weights', initializer=[[0.5], [1.0], [1.5], [2.0]])
        new_vocab_path = self._write_vocab(['orange', 'guava', 'banana', 'apple', 'raspberry'], 'new_vocab')
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_weights = variable_scope.get_variable('fruit_weights', shape=[6, 1], initializer=[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], partitioner=lambda shape, dtype: [2, 1])
                ws_util._warm_start_var_with_vocab(fruit_weights, new_vocab_path, 5, self.get_temp_dir(), prev_vocab_path, current_oov_buckets=1)
                self.evaluate(variables.global_variables_initializer())
                self.assertTrue(isinstance(fruit_weights, variables.PartitionedVariable))
                fruit_weights_vars = fruit_weights._get_variable_list()
                self.assertAllClose([[2.0], [1.5], [1.0]], fruit_weights_vars[0].eval(sess))
                self.assertAllClose([[0.5], [0.0], [0.0]], fruit_weights_vars[1].eval(sess))

    def testWarmStartVarWithColumnVocabCurrentVarPartitioned(self):
        if False:
            print('Hello World!')
        prev_vocab_path = self._write_vocab(['apple', 'orange'], 'old_vocab')
        self._create_prev_run_var('fruit_output_layer', initializer=[[0.5, 0.3], [1.0, 0.8], [1.5, 1.2], [2.0, 2.3]])
        new_vocab_path = self._write_vocab(['orange', 'apple', 'banana'], 'new_vocab')
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_output_layer = variable_scope.get_variable('fruit_output_layer', shape=[4, 3], initializer=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], partitioner=lambda shape, dtype: [2, 1])
                ws_util._warm_start_var_with_vocab(fruit_output_layer, new_vocab_path, current_vocab_size=3, prev_ckpt=self.get_temp_dir(), prev_vocab_path=prev_vocab_path, axis=1)
                self.evaluate(variables.global_variables_initializer())
                self.assertTrue(isinstance(fruit_output_layer, variables.PartitionedVariable))
                fruit_output_layer_vars = fruit_output_layer._get_variable_list()
                self.assertAllClose([[0.3, 0.5, 0.0], [0.8, 1.0, 0.0]], fruit_output_layer_vars[0].eval(sess))
                self.assertAllClose([[1.2, 1.5, 0.0], [2.3, 2.0, 0.0]], fruit_output_layer_vars[1].eval(sess))

    def testWarmStartVarWithVocabBothVarsPartitioned(self):
        if False:
            for i in range(10):
                print('nop')
        prev_vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'old_vocab')
        self._create_prev_run_var('fruit_weights', shape=[4, 1], initializer=[[0.5], [1.0], [1.5], [2.0]], partitioner=lambda shape, dtype: [2, 1])
        new_vocab_path = self._write_vocab(['orange', 'guava', 'banana', 'apple', 'raspberry', 'blueberry'], 'new_vocab')
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_weights = variable_scope.get_variable('fruit_weights', shape=[6, 1], initializer=[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], partitioner=lambda shape, dtype: [2, 1])
                ws_util._warm_start_var_with_vocab(fruit_weights, new_vocab_path, 6, self.get_temp_dir(), prev_vocab_path)
                self.evaluate(variables.global_variables_initializer())
                self.assertTrue(isinstance(fruit_weights, variables.PartitionedVariable))
                fruit_weights_vars = fruit_weights._get_variable_list()
                self.assertAllClose([[2.0], [1.5], [1.0]], fruit_weights_vars[0].eval(sess))
                self.assertAllClose([[0.5], [0.0], [0.0]], fruit_weights_vars[1].eval(sess))

    def testWarmStartVarWithColumnVocabBothVarsPartitioned(self):
        if False:
            print('Hello World!')
        prev_vocab_path = self._write_vocab(['apple', 'orange'], 'old_vocab')
        self._create_prev_run_var('fruit_output_layer', shape=[4, 2], initializer=[[0.5, 0.3], [1.0, 0.8], [1.5, 1.2], [2.0, 2.3]], partitioner=lambda shape, dtype: [2, 1])
        new_vocab_path = self._write_vocab(['orange', 'apple', 'banana'], 'new_vocab')
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                fruit_output_layer = variable_scope.get_variable('fruit_output_layer', shape=[4, 3], initializer=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], partitioner=lambda shape, dtype: [2, 1])
                ws_util._warm_start_var_with_vocab(fruit_output_layer, new_vocab_path, current_vocab_size=3, prev_ckpt=self.get_temp_dir(), prev_vocab_path=prev_vocab_path, axis=1)
                self.evaluate(variables.global_variables_initializer())
                self.assertTrue(isinstance(fruit_output_layer, variables.PartitionedVariable))
                fruit_output_layer_vars = fruit_output_layer._get_variable_list()
                self.assertAllClose([[0.3, 0.5, 0.0], [0.8, 1.0, 0.0]], fruit_output_layer_vars[0].eval(sess))
                self.assertAllClose([[1.2, 1.5, 0.0], [2.3, 2.0, 0.0]], fruit_output_layer_vars[1].eval(sess))

    def testWarmStart_ListOfVariables(self):
        if False:
            return 10
        (_, prev_int_val) = self._create_prev_run_var('v1', shape=[10, 1], initializer=ones())
        self.assertAllEqual(np.ones([10, 1]), prev_int_val)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                var = variable_scope.get_variable('v1', shape=[10, 1], initializer=zeros())
                ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start=[var])
                self.evaluate(variables.global_variables_initializer())
                self.assertAllEqual(var, prev_int_val)

    def testWarmStart_ListOfStrings(self):
        if False:
            return 10
        (_, prev_int_val) = self._create_prev_run_var('v1', shape=[10, 1], initializer=ones())
        self.assertAllEqual(np.ones([10, 1]), prev_int_val)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                var = variable_scope.get_variable('v1', shape=[10, 1], initializer=zeros())
                ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start=['v1'])
                self.evaluate(variables.global_variables_initializer())
                self.assertAllEqual(var, prev_int_val)

    def testWarmStart_TwoVarsFromTheSamePrevVar(self):
        if False:
            i = 10
            return i + 15
        (_, prev_int_val) = self._create_prev_run_var('v1', shape=[10, 1], initializer=ones())
        self.assertAllEqual(np.ones([10, 1]), prev_int_val)
        with ops.Graph().as_default() as g:
            with self.session(graph=g):
                var = variable_scope.get_variable('v1', shape=[10, 1], initializer=zeros())
                var2 = variable_scope.get_variable('v2', shape=[10, 1], initializer=zeros())
                ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start=['v1', 'v2'], var_name_to_prev_var_name=dict(v2='v1'))
                self.evaluate(variables.global_variables_initializer())
                self.assertAllEqual(var, prev_int_val)
                self.assertAllEqual(var2, prev_int_val)

    def testWarmStart_ListOfRegexes(self):
        if False:
            return 10
        [prev_v1_val, prev_v1_momentum_val, prev_v2_val, _] = self._create_prev_run_vars(var_names=['v1', 'v1/Momentum', 'v2', 'v2/Momentum'], shapes=[[10, 1]] * 4, initializers=[ones()] * 4)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                v1 = variable_scope.get_variable('v1', shape=[10, 1], initializer=zeros())
                v1_momentum = variable_scope.get_variable('v1/Momentum', shape=[10, 1], initializer=zeros())
                v2 = variable_scope.get_variable('v2', shape=[10, 1], initializer=zeros())
                v2_momentum = variable_scope.get_variable('v2/Momentum', shape=[10, 1], initializer=zeros())
                ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start=['v1', 'v2[^/]'])
                self.evaluate(variables.global_variables_initializer())
                self.assertAllEqual(v1, prev_v1_val)
                self.assertAllEqual(v1_momentum, prev_v1_momentum_val)
                self.assertAllEqual(v2, prev_v2_val)
                self.assertAllEqual(v2_momentum, np.zeros([10, 1]))

    def testWarmStart_SparseColumnIntegerized(self):
        if False:
            return 10
        sc_int = fc.categorical_column_with_identity('sc_int', num_buckets=10)
        (_, prev_int_val) = self._create_prev_run_var('linear_model/sc_int/weights', shape=[10, 1], initializer=ones())
        self.assertAllEqual(np.ones([10, 1]), prev_int_val)
        partitioner = lambda shape, dtype: [1] * len(shape)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model([sc_int], partitioner)
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_int: [np.zeros([10, 1])]}, sess)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model([sc_int], partitioner)
                ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start='.*sc_int.*')
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_int: [prev_int_val]}, sess)

    def testWarmStart_SparseColumnHashed(self):
        if False:
            i = 10
            return i + 15
        sc_hash = fc.categorical_column_with_hash_bucket('sc_hash', hash_bucket_size=15)
        (_, prev_hash_val) = self._create_prev_run_var('linear_model/sc_hash/weights', shape=[15, 1], initializer=norms())
        partitioner = lambda shape, dtype: [1] * len(shape)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model([sc_hash], partitioner)
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_hash: [np.zeros([15, 1])]}, sess)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model([sc_hash], partitioner)
                ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start='.*sc_hash.*')
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_hash: [prev_hash_val]}, sess)

    def testWarmStart_SparseColumnVocabulary(self):
        if False:
            print('Hello World!')
        vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'vocab')
        sc_vocab = fc.categorical_column_with_vocabulary_file('sc_vocab', vocabulary_file=vocab_path, vocabulary_size=4)
        (_, prev_vocab_val) = self._create_prev_run_var('linear_model/sc_vocab/weights', shape=[4, 1], initializer=ones())
        partitioner = lambda shape, dtype: [1] * len(shape)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [np.zeros([4, 1])]}, sess)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
                ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start='.*sc_vocab.*')
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [prev_vocab_val]}, sess)

    def testWarmStart_ExplicitCheckpointFile(self):
        if False:
            for i in range(10):
                print('nop')
        vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'vocab')
        sc_vocab = fc.categorical_column_with_vocabulary_file('sc_vocab', vocabulary_file=vocab_path, vocabulary_size=4)
        (_, prev_vocab_val) = self._create_prev_run_var('linear_model/sc_vocab/weights', shape=[4, 1], initializer=ones())
        partitioner = lambda shape, dtype: [1] * len(shape)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [np.zeros([4, 1])]}, sess)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
                ws_util.warm_start(os.path.join(self.get_temp_dir(), 'model-0'), vars_to_warm_start='.*sc_vocab.*')
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [prev_vocab_val]}, sess)

    def testWarmStart_SparseColumnVocabularyConstrainedVocabSizes(self):
        if False:
            i = 10
            return i + 15
        old_vocab_path = self._write_vocab(['apple', 'guava', 'banana'], 'old_vocab')
        old_vocab_size = 2
        current_vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'current_vocab')
        sc_vocab = fc.categorical_column_with_vocabulary_file('sc_vocab', vocabulary_file=current_vocab_path, vocabulary_size=2)
        self._create_prev_run_var('linear_model/sc_vocab/weights', shape=[2, 1], initializer=ones())
        partitioner = lambda shape, dtype: [1] * len(shape)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [np.zeros([2, 1])]}, sess)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
                vocab_info = ws_util.VocabInfo(new_vocab=sc_vocab.vocabulary_file, new_vocab_size=sc_vocab.vocabulary_size, num_oov_buckets=sc_vocab.num_oov_buckets, old_vocab=old_vocab_path, old_vocab_size=old_vocab_size)
                ws_util.warm_start(ckpt_to_initialize_from=self.get_temp_dir(), vars_to_warm_start='.*sc_vocab.*', var_name_to_vocab_info={'linear_model/sc_vocab/weights': vocab_info})
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [[[1], [0]]]}, sess)

    def testWarmStart_BucketizedColumn(self):
        if False:
            return 10
        real = fc.numeric_column('real')
        real_bucket = fc.bucketized_column(real, boundaries=[0.0, 1.0, 2.0, 3.0])
        (_, prev_bucket_val) = self._create_prev_run_var('linear_model/real_bucketized/weights', shape=[5, 1], initializer=norms())
        partitioner = lambda shape, dtype: [1] * len(shape)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model([real_bucket], partitioner)
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {real_bucket: [np.zeros([5, 1])]}, sess)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model([real_bucket], partitioner)
                ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start='.*real_bucketized.*')
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {real_bucket: [prev_bucket_val]}, sess)

    def testWarmStart_MultipleCols(self):
        if False:
            for i in range(10):
                print('nop')
        vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'vocab')
        sc_int = fc.categorical_column_with_identity('sc_int', num_buckets=10)
        sc_hash = fc.categorical_column_with_hash_bucket('sc_hash', hash_bucket_size=15)
        sc_keys = fc.categorical_column_with_vocabulary_list('sc_keys', vocabulary_list=['a', 'b', 'c', 'e'])
        sc_vocab = fc.categorical_column_with_vocabulary_file('sc_vocab', vocabulary_file=vocab_path, vocabulary_size=4)
        real = fc.numeric_column('real')
        real_bucket = fc.bucketized_column(real, boundaries=[0.0, 1.0, 2.0, 3.0])
        cross = fc.crossed_column([sc_keys, sc_vocab], hash_bucket_size=20)
        all_linear_cols = [sc_int, sc_hash, sc_keys, sc_vocab, real_bucket, cross]
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                sc_int_weights = variable_scope.get_variable('linear_model/sc_int/weights', shape=[10, 1], initializer=ones())
                sc_hash_weights = variable_scope.get_variable('linear_model/sc_hash/weights', shape=[15, 1], initializer=norms())
                sc_keys_weights = variable_scope.get_variable('linear_model/sc_keys/weights', shape=[4, 1], initializer=rand())
                sc_vocab_weights = variable_scope.get_variable('linear_model/sc_vocab/weights', shape=[4, 1], initializer=ones())
                real_bucket_weights = variable_scope.get_variable('linear_model/real_bucketized/weights', shape=[5, 1], initializer=norms())
                cross_weights = variable_scope.get_variable('linear_model/sc_keys_X_sc_vocab/weights', shape=[20, 1], initializer=rand())
                bias = variable_scope.get_variable('linear_model/bias_weights', shape=[1], initializer=rand())
                self._write_checkpoint(sess)
                (prev_int_val, prev_hash_val, prev_keys_val, prev_vocab_val, prev_bucket_val, prev_cross_val, prev_bias_val) = sess.run([sc_int_weights, sc_hash_weights, sc_keys_weights, sc_vocab_weights, real_bucket_weights, cross_weights, bias])
        partitioner = lambda shape, dtype: [1] * len(shape)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model(all_linear_cols, partitioner)
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_int: [np.zeros([10, 1])], sc_hash: [np.zeros([15, 1])], sc_keys: [np.zeros([4, 1])], sc_vocab: [np.zeros([4, 1])], real_bucket: [np.zeros([5, 1])], cross: [np.zeros([20, 1])]}, sess)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model(all_linear_cols, partitioner)
                vocab_info = ws_util.VocabInfo(new_vocab=sc_vocab.vocabulary_file, new_vocab_size=sc_vocab.vocabulary_size, num_oov_buckets=sc_vocab.num_oov_buckets, old_vocab=vocab_path)
                ws_util.warm_start(self.get_temp_dir(), var_name_to_vocab_info={'linear_model/sc_vocab/weights': vocab_info})
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_int: [prev_int_val], sc_hash: [prev_hash_val], sc_keys: [prev_keys_val], sc_vocab: [prev_vocab_val], real_bucket: [prev_bucket_val], cross: [prev_cross_val], 'bias': [prev_bias_val]}, sess)

    def testWarmStartMoreSettings(self):
        if False:
            print('Hello World!')
        prev_vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'old_vocab')
        new_vocab_path = self._write_vocab(['orange', 'guava', 'banana', 'apple', 'raspberry', 'blueberry'], 'new_vocab')
        sc_hash = fc.categorical_column_with_hash_bucket('sc_hash', hash_bucket_size=15)
        sc_keys = fc.categorical_column_with_vocabulary_list('sc_keys', vocabulary_list=['a', 'b', 'c', 'e'])
        sc_vocab = fc.categorical_column_with_vocabulary_file('sc_vocab', vocabulary_file=new_vocab_path, vocabulary_size=6)
        all_linear_cols = [sc_hash, sc_keys, sc_vocab]
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                variable_scope.get_variable('linear_model/sc_hash/weights', shape=[15, 1], initializer=norms())
                sc_keys_weights = variable_scope.get_variable('some_other_name', shape=[4, 1], initializer=rand())
                variable_scope.get_variable('linear_model/sc_vocab/weights', initializer=[[0.5], [1.0], [2.0], [3.0]])
                self._write_checkpoint(sess)
                prev_keys_val = self.evaluate(sc_keys_weights)

        def _partitioner(shape, dtype):
            if False:
                return 10
            partitions = [1] * len(shape)
            partitions[0] = min(2, shape.dims[0].value)
            return partitions
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model(all_linear_cols, _partitioner)
                vocab_info = ws_util.VocabInfo(new_vocab=sc_vocab.vocabulary_file, new_vocab_size=sc_vocab.vocabulary_size, num_oov_buckets=sc_vocab.num_oov_buckets, old_vocab=prev_vocab_path)
                ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start='.*(sc_keys|sc_vocab).*', var_name_to_vocab_info={ws_util._infer_var_name(cols_to_vars[sc_vocab]): vocab_info}, var_name_to_prev_var_name={ws_util._infer_var_name(cols_to_vars[sc_keys]): 'some_other_name'})
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_keys: np.split(prev_keys_val, 2), sc_hash: [np.zeros([8, 1]), np.zeros([7, 1])], sc_vocab: [np.array([[3.0], [2.0], [1.0]]), np.array([[0.5], [0.0], [0.0]])]}, sess)

    def testWarmStartMoreSettingsNoPartitioning(self):
        if False:
            for i in range(10):
                print('nop')
        prev_vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'old_vocab')
        new_vocab_path = self._write_vocab(['orange', 'guava', 'banana', 'apple', 'raspberry', 'blueberry'], 'new_vocab')
        sc_hash = fc.categorical_column_with_hash_bucket('sc_hash', hash_bucket_size=15)
        sc_keys = fc.categorical_column_with_vocabulary_list('sc_keys', vocabulary_list=['a', 'b', 'c', 'e'])
        sc_vocab = fc.categorical_column_with_vocabulary_file('sc_vocab', vocabulary_file=new_vocab_path, vocabulary_size=6)
        all_linear_cols = [sc_hash, sc_keys, sc_vocab]
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                variable_scope.get_variable('linear_model/sc_hash/weights', shape=[15, 1], initializer=norms())
                sc_keys_weights = variable_scope.get_variable('some_other_name', shape=[4, 1], initializer=rand())
                variable_scope.get_variable('linear_model/sc_vocab/weights', initializer=[[0.5], [1.0], [2.0], [3.0]])
                self._write_checkpoint(sess)
                prev_keys_val = self.evaluate(sc_keys_weights)
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model(all_linear_cols, partitioner=None)
                vocab_info = ws_util.VocabInfo(new_vocab=sc_vocab.vocabulary_file, new_vocab_size=sc_vocab.vocabulary_size, num_oov_buckets=sc_vocab.num_oov_buckets, old_vocab=prev_vocab_path)
                ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start='.*(sc_keys|sc_vocab).*', var_name_to_vocab_info={ws_util._infer_var_name(cols_to_vars[sc_vocab]): vocab_info}, var_name_to_prev_var_name={ws_util._infer_var_name(cols_to_vars[sc_keys]): 'some_other_name'})
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_keys: [prev_keys_val], sc_hash: [np.zeros([15, 1])], sc_vocab: [np.array([[3.0], [2.0], [1.0], [0.5], [0.0], [0.0]])]}, sess)

    def testWarmStartVarsToWarmstartIsNone(self):
        if False:
            print('Hello World!')
        prev_vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'old_vocab')
        new_vocab_path = self._write_vocab(['orange', 'guava', 'banana', 'apple', 'raspberry', 'blueberry'], 'new_vocab')
        sc_hash = fc.categorical_column_with_hash_bucket('sc_hash', hash_bucket_size=15)
        sc_keys = fc.categorical_column_with_vocabulary_list('sc_keys', vocabulary_list=['a', 'b', 'c', 'e'])
        sc_vocab = fc.categorical_column_with_vocabulary_file('sc_vocab', vocabulary_file=new_vocab_path, vocabulary_size=6)
        all_linear_cols = [sc_hash, sc_keys, sc_vocab]
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                variable_scope.get_variable('linear_model/sc_hash/weights', shape=[15, 1], initializer=norms())
                variable_scope.get_variable('some_other_name', shape=[4, 1], initializer=rand())
                variable_scope.get_variable('linear_model/sc_vocab/weights', initializer=[[0.5], [1.0], [2.0], [3.0]])
                self._write_checkpoint(sess)

        def _partitioner(shape, dtype):
            if False:
                return 10
            partitions = [1] * len(shape)
            partitions[0] = min(2, shape.dims[0].value)
            return partitions
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = self._create_linear_model(all_linear_cols, _partitioner)
                vocab_info = ws_util.VocabInfo(new_vocab=sc_vocab.vocabulary_file, new_vocab_size=sc_vocab.vocabulary_size, num_oov_buckets=sc_vocab.num_oov_buckets, old_vocab=prev_vocab_path)
                ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start=None, var_name_to_vocab_info={ws_util._infer_var_name(cols_to_vars[sc_vocab]): vocab_info}, var_name_to_prev_var_name={ws_util._infer_var_name(cols_to_vars[sc_keys]): 'some_other_name'})
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {sc_keys: [np.zeros([2, 1]), np.zeros([2, 1])], sc_hash: [np.zeros([8, 1]), np.zeros([7, 1])], sc_vocab: [np.array([[3.0], [2.0], [1.0]]), np.array([[0.5], [0.0], [0.0]])]}, sess)

    def testWarmStartEmbeddingColumn(self):
        if False:
            for i in range(10):
                print('nop')
        prev_vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'old_vocab')
        new_vocab_path = self._write_vocab(['orange', 'guava', 'banana', 'apple', 'raspberry', 'blueberry'], 'new_vocab')
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                variable_scope.get_variable('input_layer/sc_vocab_embedding/embedding_weights', initializer=[[0.5, 0.4], [1.0, 1.1], [2.0, 2.2], [3.0, 3.3]])
                self._write_checkpoint(sess)

        def _partitioner(shape, dtype):
            if False:
                i = 10
                return i + 15
            partitions = [1] * len(shape)
            partitions[0] = min(2, shape.dims[0].value)
            return partitions
        sc_vocab = fc.categorical_column_with_vocabulary_file('sc_vocab', vocabulary_file=new_vocab_path, vocabulary_size=6)
        emb_vocab_column = fc.embedding_column(categorical_column=sc_vocab, dimension=2)
        all_deep_cols = [emb_vocab_column]
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = {}
                with variable_scope.variable_scope('', partitioner=_partitioner):
                    fc.input_layer(features=self._create_dummy_inputs(), feature_columns=all_deep_cols, cols_to_vars=cols_to_vars)
                vocab_info = ws_util.VocabInfo(new_vocab=sc_vocab.vocabulary_file, new_vocab_size=sc_vocab.vocabulary_size, num_oov_buckets=sc_vocab.num_oov_buckets, old_vocab=prev_vocab_path, backup_initializer=init_ops.random_uniform_initializer(minval=0.42, maxval=0.42))
                ws_util.warm_start(self.get_temp_dir(), var_name_to_vocab_info={ws_util._infer_var_name(cols_to_vars[emb_vocab_column]): vocab_info})
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {emb_vocab_column: [np.array([[3.0, 3.3], [2.0, 2.2], [1.0, 1.1]]), np.array([[0.5, 0.4], [0.42, 0.42], [0.42, 0.42]])]}, sess)

    def testWarmStartEmbeddingColumnLinearModel(self):
        if False:
            for i in range(10):
                print('nop')
        prev_vocab_path = self._write_vocab(['apple', 'banana', 'guava', 'orange'], 'old_vocab')
        new_vocab_path = self._write_vocab(['orange', 'guava', 'banana', 'apple', 'raspberry', 'blueberry'], 'new_vocab')
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                variable_scope.get_variable('linear_model/sc_vocab_embedding/embedding_weights', initializer=[[0.5, 0.4], [1.0, 1.1], [2.0, 2.2], [3.0, 3.3]])
                variable_scope.get_variable('linear_model/sc_vocab_embedding/weights', initializer=[[0.69], [0.71]])
                self._write_checkpoint(sess)

        def _partitioner(shape, dtype):
            if False:
                while True:
                    i = 10
            partitions = [1] * len(shape)
            partitions[0] = min(2, shape.dims[0].value)
            return partitions
        sc_vocab = fc.categorical_column_with_vocabulary_file('sc_vocab', vocabulary_file=new_vocab_path, vocabulary_size=6)
        emb_vocab = fc.embedding_column(categorical_column=sc_vocab, dimension=2)
        all_deep_cols = [emb_vocab]
        with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
                cols_to_vars = {}
                with variable_scope.variable_scope('', partitioner=_partitioner):
                    fc.linear_model(features=self._create_dummy_inputs(), feature_columns=all_deep_cols, cols_to_vars=cols_to_vars)
                vocab_info = ws_util.VocabInfo(new_vocab=sc_vocab.vocabulary_file, new_vocab_size=sc_vocab.vocabulary_size, num_oov_buckets=sc_vocab.num_oov_buckets, old_vocab=prev_vocab_path, backup_initializer=init_ops.random_uniform_initializer(minval=0.42, maxval=0.42))
                ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start='.*sc_vocab.*', var_name_to_vocab_info={'linear_model/sc_vocab_embedding/embedding_weights': vocab_info})
                self.evaluate(variables.global_variables_initializer())
                self._assert_cols_to_vars(cols_to_vars, {emb_vocab: [np.array([[0.69]]), np.array([[0.71]]), np.array([[3.0, 3.3], [2.0, 2.2], [1.0, 1.1]]), np.array([[0.5, 0.4], [0.42, 0.42], [0.42, 0.42]])]}, sess)

    def testErrorConditions(self):
        if False:
            i = 10
            return i + 15
        x = variable_scope.get_variable('x', shape=[4, 1], initializer=ones(), partitioner=lambda shape, dtype: [2, 1])
        self.assertRaises(TypeError, ws_util._warm_start_var_with_vocab, [x], '/tmp', 5, '/tmp', '/tmp')
        with ops.Graph().as_default():
            with self.cached_session() as sess:
                x = variable_scope.get_variable('x', shape=[4, 1], initializer=ones(), partitioner=lambda shape, dtype: [2, 1])
                self._write_checkpoint(sess)
        self.assertRaises(ValueError, ws_util.warm_start, self.get_temp_dir(), var_name_to_vocab_info={'y': ws_util.VocabInfo('', 1, 0, '')})
        self.assertRaises(ValueError, ws_util.warm_start, self.get_temp_dir(), var_name_to_prev_var_name={'y': 'y2'})

    def testWarmStartFromObjectBasedCheckpoint(self):
        if False:
            print('Hello World!')
        prev_val = [[0.5], [1.0], [1.5], [2.0]]
        with ops.Graph().as_default() as g:
            with self.session(graph=g):
                prev_var = variable_scope.get_variable('fruit_weights', initializer=prev_val)
                self.evaluate(variables.global_variables_initializer())
                tracking_util.Checkpoint(v=prev_var).save(os.path.join(self.get_temp_dir(), 'checkpoint'))
        with ops.Graph().as_default() as g:
            with self.session(graph=g):
                fruit_weights = variable_scope.get_variable('fruit_weights', initializer=[[0.0], [0.0], [0.0], [0.0]])
                ws_util.warm_start(self.get_temp_dir())
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose(prev_val, self.evaluate(fruit_weights))
if __name__ == '__main__':
    test.main()