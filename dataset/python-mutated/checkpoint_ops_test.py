"""Functional tests for Python wrappers around warm-starting."""
import os
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_ops
from tensorflow.python.training import saver as saver_lib

@test_util.run_v1_only('This is to test V1 name-based checkpoints which is not supported in V2.')
class LoadAndRemapWrappersTest(test.TestCase):
    """Tests for the functionality of the Python wrappers."""

    def setUp(self):
        if False:
            print('Hello World!')
        ops.reset_default_graph()
        checkpoint_prefix = os.path.join(self.get_temp_dir(), 'model')
        initializer = init_ops.constant_initializer(np.reshape(np.linspace(0.0, 79, 5 * 16), (5, 16)))
        with self.cached_session() as sess:
            with variable_scope.variable_scope('some_scope'):
                variable_scope.get_variable(name='embeddings', shape=[5, 16], initializer=initializer)
            self.evaluate(variables.global_variables_initializer())
            saver = saver_lib.Saver()
            saver.save(sess, checkpoint_prefix, global_step=5)
        self.checkpoint_file = '{}-5'.format(checkpoint_prefix)
        self.new_feature_vocab_file = os.path.join(self.get_temp_dir(), 'new_feature_vocab.txt')
        with open(self.new_feature_vocab_file, 'w') as f:
            f.write('\n'.join(['zero', 'one', 'two', 'three', 'four']) + '\n')
        self.old_feature_vocab_file = os.path.join(self.get_temp_dir(), 'old_feature_vocab.txt')
        with open(self.old_feature_vocab_file, 'w') as f:
            f.write('\n'.join(['zero', 'one', 'two', 'three']) + '\n')
        self.new_class_vocab_file = os.path.join(self.get_temp_dir(), 'new_class_vocab.txt')
        with open(self.new_class_vocab_file, 'w') as f:
            f.write('\n'.join(['MISSING', 'knitting', 'flask', 'eminem']) + '\n')
        self.old_class_vocab_file = os.path.join(self.get_temp_dir(), 'old_class_vocab.txt')
        with open(self.old_class_vocab_file, 'w') as f:
            f.write('\n'.join(['knitting', 'eminem', 'MISSING']) + '\n')
        self.init_val = 42

        def _init_val_initializer(shape, dtype=None, partition_info=None):
            if False:
                i = 10
                return i + 15
            del dtype, partition_info
            return array_ops.tile(constant_op.constant([[self.init_val]], dtype=dtypes.float32), shape)
        self.initializer = _init_val_initializer

    def test_load_and_remap_matrix(self):
        if False:
            print('Hello World!')
        'Tests the end-to-end loading / remapping of weights.'
        remapped_matrix = checkpoint_ops._load_and_remap_matrix(new_row_vocab_file=self.new_feature_vocab_file, old_row_vocab_file=self.old_feature_vocab_file, num_rows_to_load=4, new_col_vocab_file=self.new_class_vocab_file, old_col_vocab_file=self.old_class_vocab_file, new_col_vocab_size=4, old_tensor_name='some_scope/embeddings', ckpt_path=[self.checkpoint_file], new_row_vocab_offset=1, initializer=self.initializer, num_row_oov_buckets=1, num_col_oov_buckets=1)
        expected_remapped_matrix = np.concatenate([np.reshape([18, 34, 50, self.init_val, self.init_val], [5, 1]), np.reshape([16, 32, 48, self.init_val, self.init_val], [5, 1]), np.reshape([self.init_val] * 5, [5, 1]), np.reshape([17, 33, 49, self.init_val, self.init_val], [5, 1]), np.reshape([self.init_val] * 5, [5, 1])], axis=1)
        with self.cached_session():
            self.assertAllClose(expected_remapped_matrix, self.evaluate(remapped_matrix))

    def test_load_and_remap_output_layer_weight_initializer_linear(self):
        if False:
            i = 10
            return i + 15
        'Tests for the output layer initializer in the linear multi-class case.'
        loading_initializer = checkpoint_ops._load_and_remap_matrix_initializer(new_row_vocab_size=5, new_col_vocab_file=self.new_class_vocab_file, old_col_vocab_file=self.old_class_vocab_file, new_col_vocab_size=4, old_tensor_name='some_scope/embeddings', ckpt_path=[self.checkpoint_file], new_row_vocab_file=self.new_feature_vocab_file, old_row_vocab_file=self.old_feature_vocab_file, num_row_oov_buckets=1, num_col_oov_buckets=1, initializer=self.initializer)
        expected_remapped_matrix = np.concatenate([np.reshape([2, 18, 34, 50, self.init_val, self.init_val], [6, 1]), np.reshape([0, 16, 32, 48, self.init_val, self.init_val], [6, 1]), np.reshape([self.init_val] * 6, [6, 1]), np.reshape([1, 17, 33, 49, self.init_val, self.init_val], [6, 1]), np.reshape([self.init_val] * 6, [6, 1])], axis=1)
        remapped_matrix = variable_scope.get_variable(name='linear/obtained_weight_matrix', shape=[6, 5], initializer=loading_initializer, partitioner=partitioned_variables.fixed_size_partitioner(2))
        with self.cached_session():
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose(expected_remapped_matrix, remapped_matrix.as_tensor())

    def test_load_and_remap_output_layer_weight_initializer_dnn_output(self):
        if False:
            while True:
                i = 10
        'Tests for the output layer initializer in the DNN output case.'
        loading_initializer = checkpoint_ops._load_and_remap_matrix_initializer(new_row_vocab_size=5, new_col_vocab_file=self.new_class_vocab_file, old_col_vocab_file=self.old_class_vocab_file, new_col_vocab_size=4, old_tensor_name='some_scope/embeddings', ckpt_path=[self.checkpoint_file], num_col_oov_buckets=1, initializer=self.initializer)
        expected_remapped_matrix = np.concatenate([np.reshape([2, 18, 34, 50, 66], [5, 1]), np.reshape([0, 16, 32, 48, 64], [5, 1]), np.reshape([self.init_val] * 5, [5, 1]), np.reshape([1, 17, 33, 49, 65], [5, 1]), np.reshape([self.init_val] * 5, [5, 1])], axis=1)
        remapped_matrix = variable_scope.get_variable(name='dnn_output/obtained_weight_matrix', shape=[5, 5], initializer=loading_initializer, partitioner=partitioned_variables.fixed_size_partitioner(2))
        with self.cached_session():
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose(expected_remapped_matrix, remapped_matrix.as_tensor())

    def test_initializer_with_oov_only_partition(self):
        if False:
            while True:
                i = 10
        'Tests for the output layer initializer where one partition is all OOV.'
        loading_initializer = checkpoint_ops._load_and_remap_matrix_initializer(new_row_vocab_size=5, new_col_vocab_file=self.new_class_vocab_file, old_col_vocab_file=self.old_class_vocab_file, new_col_vocab_size=4, old_tensor_name='some_scope/embeddings', ckpt_path=[self.checkpoint_file], new_row_vocab_file=self.new_feature_vocab_file, old_row_vocab_file=self.old_feature_vocab_file, num_row_oov_buckets=5, num_col_oov_buckets=1, initializer=self.initializer)
        expected_remapped_matrix = np.concatenate([np.reshape([2, 18, 34, 50] + [self.init_val] * 6, [10, 1]), np.reshape([0, 16, 32, 48] + [self.init_val] * 6, [10, 1]), np.reshape([self.init_val] * 10, [10, 1]), np.reshape([1, 17, 33, 49] + [self.init_val] * 6, [10, 1]), np.reshape([self.init_val] * 10, [10, 1])], axis=1)
        remapped_matrix = variable_scope.get_variable(name='linear_all_oov/obtained_weight_matrix', shape=[10, 5], initializer=loading_initializer, partitioner=partitioned_variables.fixed_size_partitioner(2))
        with self.cached_session():
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose(expected_remapped_matrix, remapped_matrix.as_tensor())

    def test_load_and_remap_linear_multiclass_initializer_default_init(self):
        if False:
            print('Hello World!')
        'Tests where the zeros_initializer default is used for linear.'
        loading_initializer = checkpoint_ops._load_and_remap_matrix_initializer(new_row_vocab_size=5, new_col_vocab_file=self.new_class_vocab_file, old_col_vocab_file=self.old_class_vocab_file, new_col_vocab_size=4, old_tensor_name='some_scope/embeddings', ckpt_path=[self.checkpoint_file], new_row_vocab_file=self.new_feature_vocab_file, old_row_vocab_file=self.old_feature_vocab_file, num_row_oov_buckets=1, num_col_oov_buckets=1)
        expected_remapped_matrix = np.concatenate([np.reshape([2, 18, 34, 50, 0, 0], [6, 1]), np.reshape([0, 16, 32, 48, 0, 0], [6, 1]), np.reshape([0] * 6, [6, 1]), np.reshape([1, 17, 33, 49, 0, 0], [6, 1]), np.reshape([0] * 6, [6, 1])], axis=1)
        remapped_matrix = variable_scope.get_variable(name='linear_init_fallback/obtained_weight_matrix', shape=[6, 5], initializer=loading_initializer, partitioner=partitioned_variables.fixed_size_partitioner(2))
        with self.cached_session():
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose(expected_remapped_matrix, remapped_matrix.as_tensor())

    def test_load_embedding_initializer(self):
        if False:
            while True:
                i = 10
        'Tests for the load_embedding_initializer wrapper.'
        embedding_loading_initializer = checkpoint_ops._load_embedding_initializer(new_vocab_file=self.new_feature_vocab_file, old_vocab_file=self.old_feature_vocab_file, new_vocab_size=5, embedding_dim=16, embedding_tensor_name='some_scope/embeddings', ckpt_path=[self.checkpoint_file], num_oov_buckets=1, initializer=self.initializer)
        expected_remapped_embeddings = np.concatenate([np.reshape(range(64), [4, 16]), np.reshape([self.init_val] * 32, [2, 16])], axis=0)
        remapped_embeddings = variable_scope.get_variable(name='embedding/obtained_embedding_matrix', shape=[6, 16], initializer=embedding_loading_initializer, partitioner=partitioned_variables.fixed_size_partitioner(2))
        with self.cached_session():
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose(expected_remapped_embeddings, remapped_embeddings.as_tensor())

    def test_load_embedding_initializer_large_oov(self):
        if False:
            while True:
                i = 10
        'Tests for the large OOV case for load_embedding_initializer wrapper.'
        self.new_feature_vocab_file = os.path.join(self.get_temp_dir(), 'new_feature_vocab.txt')
        with open(self.new_feature_vocab_file, 'w') as f:
            f.write('\n'.join(['one', 'zero', 'two', 'four']) + '\n')
        self.old_feature_vocab_file = os.path.join(self.get_temp_dir(), 'old_feature_vocab.txt')
        with open(self.old_feature_vocab_file, 'w') as f:
            f.write('\n'.join(['zero', 'one']) + '\n')
        embedding_loading_initializer = checkpoint_ops._load_embedding_initializer(new_vocab_file=self.new_feature_vocab_file, old_vocab_file=self.old_feature_vocab_file, new_vocab_size=4, embedding_dim=16, embedding_tensor_name='some_scope/embeddings', ckpt_path=[self.checkpoint_file], num_oov_buckets=5, initializer=self.initializer)
        expected_remapped_embeddings = np.concatenate([np.reshape(range(16, 32), [1, 16]), np.reshape(range(16), [1, 16]), np.reshape([self.init_val] * 112, [7, 16])], axis=0)
        remapped_embeddings = variable_scope.get_variable(name='embedding/obtained_embedding_matrix', shape=[9, 16], initializer=embedding_loading_initializer, partitioner=partitioned_variables.fixed_size_partitioner(2))
        with self.cached_session():
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose(expected_remapped_embeddings, remapped_embeddings.as_tensor())

    def test_load_embedding_initializer_old_row_vocab(self):
        if False:
            while True:
                i = 10
        'Tests for load_embedding_initializer where we constrain old vocab.'
        embedding_loading_initializer = checkpoint_ops._load_embedding_initializer(new_vocab_file=self.new_feature_vocab_file, old_vocab_file=self.old_feature_vocab_file, old_vocab_size=3, new_vocab_size=5, embedding_dim=16, embedding_tensor_name='some_scope/embeddings', ckpt_path=[self.checkpoint_file], num_oov_buckets=1, initializer=self.initializer)
        expected_remapped_embeddings = np.concatenate([np.reshape(range(48), [3, 16]), np.reshape([self.init_val] * 48, [3, 16])], axis=0)
        remapped_embeddings = variable_scope.get_variable(name='embedding/obtained_embedding_matrix', shape=[6, 16], initializer=embedding_loading_initializer, partitioner=partitioned_variables.fixed_size_partitioner(2))
        with self.cached_session():
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose(expected_remapped_embeddings, remapped_embeddings.as_tensor())
if __name__ == '__main__':
    test.main()