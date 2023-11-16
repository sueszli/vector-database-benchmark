"""Tests for tensorflow.python.training.saver.py."""
import glob
import math
import os
import random
import time
import numpy as np
from google.protobuf.any_pb2 import Any
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
import tensorflow.python.ops.nn_grad
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.summary import summary
from tensorflow.python.trackable import base as trackable_base
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training import saver_test_utils
from tensorflow.python.util import compat

class SaverTest(test.TestCase):

    def basicSaveRestore(self, variable_op):
        if False:
            return 10
        save_path = os.path.join(self.get_temp_dir(), 'basic_save_restore')
        with self.session(graph=ops_lib.Graph()) as sess:
            v0 = variable_op(10.0, name='v0')
            v1 = variable_op(20.0, name='v1')
            v2 = saver_test_utils.CheckpointedOp(name='v2')
            v2_init = v2.insert('k1', 30.0)
            if not context.executing_eagerly():
                self.evaluate([variables.global_variables_initializer(), v2_init])
            self.assertEqual(10.0, self.evaluate(v0))
            self.assertEqual(20.0, self.evaluate(v1))
            self.assertEqual(b'k1', self.evaluate(v2.keys()))
            self.assertEqual(30.0, self.evaluate(v2.values()))
            save = saver_module.Saver({'v0': v0, 'v1': v1, 'v2': v2.saveable}, restore_sequentially=True)
            val = save.save(sess, save_path)
            self.assertIsInstance(val, str)
            self.assertEqual(save_path, val)
        with self.session(graph=ops_lib.Graph()) as sess:
            v0 = variable_op(-1.0, name='v0')
            v1 = variable_op(-1.0, name='v1')
            v2 = saver_test_utils.CheckpointedOp(name='v2')
            if not context.executing_eagerly():
                self.assertEqual(len(variables.report_uninitialized_variables().eval()), 2)
                self.assertEqual(0, len(self.evaluate(v2.keys())))
                self.assertEqual(0, len(self.evaluate(v2.values())))
            save = saver_module.Saver({'v0': v0, 'v1': v1, 'v2': v2.saveable})
            save.restore(sess, save_path)
            self.assertEqual(10.0, self.evaluate(v0))
            self.assertEqual(20.0, self.evaluate(v1))
            self.assertEqual(b'k1', self.evaluate(v2.keys()))
            self.assertEqual(30.0, self.evaluate(v2.values()))
        with self.session(graph=ops_lib.Graph()) as sess:
            v0_2 = variable_op(1000.0, name='v0')
            v1_2 = variable_op(2000.0, name='v1')
            v2_2 = saver_test_utils.CheckpointedOp(name='v2')
            v2_init = v2_2.insert('k1000', 3000.0)
            if not context.executing_eagerly():
                init_all_op = [variables.global_variables_initializer(), v2_init]
                self.evaluate(init_all_op)
                self.assertEqual(b'k1000', self.evaluate(v2_2.keys()))
                self.assertEqual(3000.0, self.evaluate(v2_2.values()))
            self.assertEqual(1000.0, self.evaluate(v0_2))
            self.assertEqual(2000.0, self.evaluate(v1_2))
            save2 = saver_module.Saver({'v0': v0_2, 'v1': v1_2, 'v2': v2_2.saveable})
            save2.restore(sess, save_path)
            self.assertEqual(10.0, self.evaluate(v0_2))
            self.assertEqual(20.0, self.evaluate(v1_2))
            self.assertEqual(b'k1', self.evaluate(v2_2.keys()))
            self.assertEqual(30.0, self.evaluate(v2_2.values()))

    def testBasic(self):
        if False:
            while True:
                i = 10
        self.basicSaveRestore(variables.Variable)

    @test_util.run_in_graph_and_eager_modes
    def testResourceBasic(self):
        if False:
            return 10
        self.basicSaveRestore(resource_variable_ops.ResourceVariable)

    def testResourceColocation(self):
        if False:
            print('Hello World!')
        with ops_lib.Graph().as_default():
            partitioner = partitioned_variables.fixed_size_partitioner(num_shards=2)
            with ops_lib.device('/job:ps/device:GPU:0'):
                v = variable_scope.get_variable('v0', shape=[10, 2], partitioner=partitioner, use_resource=True)
            saver_module.Saver({'v0': v}).build()
            save_op = None
            for op in ops_lib.get_default_graph().get_operations():
                if op.type == 'SaveV2':
                    save_op = op
                    break
            assert save_op is not None
            for save_inp in save_op.inputs[3:]:
                self.assertEqual('/job:ps/device:CPU:0', save_inp.device)

    def testResourceVariableReadOpsAddedDeterministically(self):
        if False:
            print('Hello World!')
        graph_defs = []
        num_graphs = 10
        for _ in range(num_graphs):
            with ops_lib.Graph().as_default() as g:
                for i in range(20):
                    resource_variable_ops.ResourceVariable(i, name='var%s' % i)
                saver_module.Saver()
                graph_defs.append(g.as_graph_def())
        for i in range(num_graphs - 1):
            self.assertEqual(graph_defs[i], graph_defs[i + 1])

    def testEagerBasic(self):
        if False:
            i = 10
            return i + 15
        with context.eager_mode():
            ckpt_prefix = os.path.join(self.get_temp_dir(), 'ckpt')
            v1 = resource_variable_ops.ResourceVariable(3.14, name='v1')
            v2 = resource_variable_ops.ResourceVariable([1, 2], name='v2')
            save = saver_module.Saver([v1, v2])
            save.save(None, ckpt_prefix)
            v1.assign(0.0)
            v2.assign([0, 0])
            self.assertNear(0.0, self.evaluate(v1), 1e-05)
            self.assertAllEqual([0, 0], self.evaluate(v2))
            save.restore(None, ckpt_prefix)
            self.assertNear(3.14, self.evaluate(v1), 1e-05)
            self.assertAllEqual([1, 2], self.evaluate(v2))

    def testEagerGraphCompatibility(self):
        if False:
            return 10
        graph_ckpt_prefix = os.path.join(self.get_temp_dir(), 'graph_ckpt')
        with context.graph_mode():
            with self.session(graph=ops_lib.Graph()) as sess:
                w1 = resource_variable_ops.ResourceVariable(1.0, name='w1')
                w2 = resource_variable_ops.ResourceVariable(2.0, name='w2')
                graph_saver = saver_module.Saver([w1, w2])
                self.evaluate(variables.global_variables_initializer())
                graph_saver.save(sess, graph_ckpt_prefix)
        with context.eager_mode():
            ops_lib._default_graph_stack.reset()
            ops_lib.reset_default_graph()
            w1 = resource_variable_ops.ResourceVariable(0.0, name='w1')
            w2 = resource_variable_ops.ResourceVariable(0.0, name='w2')
            graph_saver = saver_module.Saver([w1, w2])
            graph_saver.restore(None, graph_ckpt_prefix)
            self.assertAllEqual(self.evaluate(w1), 1.0)
            self.assertAllEqual(self.evaluate(w2), 2.0)
        eager_ckpt_prefix = os.path.join(self.get_temp_dir(), 'eager_ckpt')
        with context.eager_mode():
            ops_lib._default_graph_stack.reset()
            ops_lib.reset_default_graph()
            w3 = resource_variable_ops.ResourceVariable(3.0, name='w3')
            w4 = resource_variable_ops.ResourceVariable(4.0, name='w4')
            graph_saver = saver_module.Saver([w3, w4])
            graph_saver.save(None, eager_ckpt_prefix)
        with context.graph_mode():
            with self.session(graph=ops_lib.Graph()) as sess:
                w3 = resource_variable_ops.ResourceVariable(0.0, name='w3')
                w4 = resource_variable_ops.ResourceVariable(0.0, name='w4')
                graph_saver = saver_module.Saver([w3, w4])
                self.evaluate(variables.global_variables_initializer())
                graph_saver.restore(sess, eager_ckpt_prefix)
                self.assertAllEqual(w3, 3.0)
                self.assertAllEqual(w4, 4.0)

    @test_util.run_in_graph_and_eager_modes
    def testResourceSaveRestoreCachingDevice(self):
        if False:
            i = 10
            return i + 15
        save_path = os.path.join(self.get_temp_dir(), 'resource_cache')
        with self.session(graph=ops_lib.Graph()) as sess:
            v = resource_variable_ops.ResourceVariable([1], caching_device='/cpu:0', name='v')
            if context.executing_eagerly():
                sess = None
            else:
                self.evaluate(variables.global_variables_initializer())
            save = saver_module.Saver([v])
            save.save(sess, save_path)
            save2 = saver_module.Saver([v])
            save2.restore(sess, save_path)
            self.assertEqual(self.evaluate(v), [1])

    def testNoAdditionalOpsAddedBySaverForResourceVariablesOutsideSaveScope(self):
        if False:
            for i in range(10):
                print('nop')
        with ops_lib.Graph().as_default() as g:
            v = resource_variable_ops.ResourceVariable(1.0, name='v')
            with ops_lib.name_scope('saver1'):
                saver_module.Saver()
            with ops_lib.name_scope('saver2'):
                saver_module.Saver({'name': v})
        ops_in_saver1_scope_but_not_save_scope = [op for op in g.get_operations() if op.name.startswith('saver1/') and (not op.name.startswith('saver1/save/'))]
        self.assertEqual(ops_in_saver1_scope_but_not_save_scope, [])
        ops_in_saver2_scope_but_not_save_scope = [op for op in g.get_operations() if op.name.startswith('saver2/') and (not op.name.startswith('saver2/save/'))]
        self.assertEqual(ops_in_saver2_scope_but_not_save_scope, [])

    def testSaveCopyRestoreWithSaveRelativePaths(self):
        if False:
            return 10
        'Save, copy checkpoint dir and restore from copied dir.\n\n    This only works for save_relative_paths=True.\n    '
        save_dir1 = os.path.join(self.get_temp_dir(), 'save_dir1')
        os.mkdir(save_dir1)
        save_path1 = os.path.join(save_dir1, 'save_copy_restore')
        with ops_lib.Graph().as_default():
            v0 = variable_v1.VariableV1(10.0, name='v0')
            v1 = variable_v1.VariableV1(20.0, name='v1')
            v2 = saver_test_utils.CheckpointedOp(name='v2')
            v2_init = v2.insert('k1', 30.0)
            save = saver_module.Saver(var_list={'v0': v0, 'v1': v1, 'v2': v2.saveable}, restore_sequentially=True, save_relative_paths=True)
            init_all_op = [variables.global_variables_initializer(), v2_init]
            with self.cached_session() as sess:
                self.evaluate(init_all_op)
                self.assertEqual(10.0, self.evaluate(v0))
                self.assertEqual(20.0, self.evaluate(v1))
                self.assertEqual(b'k1', self.evaluate(v2.keys()))
                self.assertEqual(30.0, self.evaluate(v2.values()))
                val = save.save(sess, save_path1)
                self.assertIsInstance(val, str)
                self.assertEqual(save_path1, val)
            self.assertEqual(checkpoint_management.latest_checkpoint(save_dir1), save_path1)
            save_dir2 = os.path.join(self.get_temp_dir(), 'save_dir2')
            os.renames(save_dir1, save_dir2)
            save_path2 = os.path.join(save_dir2, 'save_copy_restore')
            self.assertEqual(checkpoint_management.latest_checkpoint(save_dir2), save_path2)
            with self.cached_session() as sess:
                v0 = variable_v1.VariableV1(-1.0, name='v0')
                v1 = variable_v1.VariableV1(-1.0, name='v1')
                v2 = saver_test_utils.CheckpointedOp(name='v2')
                save = saver_module.Saver({'v0': v0, 'v1': v1, 'v2': v2.saveable})
                self.assertEqual(len(variables.report_uninitialized_variables().eval()), 2)
                self.assertEqual(0, len(self.evaluate(v2.keys())))
                self.assertEqual(0, len(self.evaluate(v2.values())))
                save.restore(sess, save_path2)
                self.assertEqual(10.0, self.evaluate(v0))
                self.assertEqual(20.0, self.evaluate(v1))
                self.assertEqual(b'k1', self.evaluate(v2.keys()))
                self.assertEqual(30.0, self.evaluate(v2.values()))

    def testFilenameTensor(self):
        if False:
            for i in range(10):
                print('nop')
        with ops_lib.Graph().as_default():
            v0 = variable_v1.VariableV1(0, name='v0')
            filename = b'somerandomfilename'
            save = saver_module.Saver({'v0': v0}, filename=filename)
            with self.cached_session() as sess:
                tensor = sess.graph.get_tensor_by_name(save.saver_def.filename_tensor_name)
                self.assertEqual(self.evaluate(tensor), filename)

    def testInvalidPath(self):
        if False:
            while True:
                i = 10
        v0 = variable_v1.VariableV1(0, name='v0')
        for ver in (saver_pb2.SaverDef.V1, saver_pb2.SaverDef.V2):
            with self.cached_session() as sess:
                save = saver_module.Saver({'v0': v0}, write_version=ver)
                with self.assertRaisesRegex(ValueError, 'The passed save_path is not a valid checkpoint:'):
                    save.restore(sess, 'invalid path')

    @test_util.run_v1_only('train.Saver is V1 only API.')
    def testInt64(self):
        if False:
            return 10
        save_path = os.path.join(self.get_temp_dir(), 'int64')
        with self.cached_session() as sess:
            v = variable_v1.VariableV1(np.int64(15), name='v')
            save = saver_module.Saver({'v': v}, restore_sequentially=True)
            self.evaluate(variables.global_variables_initializer())
            val = save.save(sess, save_path)
            self.assertIsInstance(val, str)
            self.assertEqual(save_path, val)
            with self.cached_session() as sess:
                v = variable_v1.VariableV1(np.int64(-1), name='v')
                save = saver_module.Saver({'v': v})
            with self.assertRaisesWithPredicateMatch(errors_impl.OpError, lambda e: 'uninitialized value v' in e.message):
                self.evaluate(v)
            save.restore(sess, save_path)
            self.assertEqual(np.int64(15), self.evaluate(v))

    def testSomeErrors(self):
        if False:
            for i in range(10):
                print('nop')
        with ops_lib.Graph().as_default():
            v0 = variable_v1.VariableV1([10.0], name='v0')
            v1 = variable_v1.VariableV1([20.0], name='v1')
            v2 = variable_v1.VariableV1([20.0], name='v2')
            v2._set_save_slice_info(variables.Variable.SaveSliceInfo('v1', [1], [0], [1]))
            with self.assertRaisesRegex(ValueError, 'same name: v1'):
                saver_module.Saver([v0, v1, v2])
            saver_module.Saver({'vee1': v1, 'other': [v2]})
            p_v1 = variable_scope.get_variable('p_v1', shape=[4, 5], partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
            p_v2 = variable_scope.get_variable('p_v2', shape=[4, 5], partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
            p_v2._name = 'p_v1'
            with self.assertRaisesRegex(ValueError, 'same name: p_v1'):
                saver_module.Saver([p_v1, p_v2])

    def testSameName(self):
        if False:
            for i in range(10):
                print('nop')
        with ops_lib.Graph().as_default():
            v0 = variable_v1.VariableV1([10.0], name='v0')
            v2 = saver_test_utils.CheckpointedOp(name='v2')
            with self.assertRaisesRegex(ValueError, 'The same saveable will be restored with two names: v0'):
                saver_module.Saver({'v0': v0, 'v0too': v0})
            with self.assertRaisesRegex(ValueError, 'The same saveable will be restored with two names: v2'):
                saver_module.Saver({'v2': v2.saveable, 'v2too': v2.saveable})
            saver_module.Saver({'v0': v0, 'v2': v2.saveable})

    @test_util.run_v1_only('train.Saver and VariableV1 are V1 only APIs.')
    def testBasicsWithListOfVariables(self):
        if False:
            i = 10
            return i + 15
        save_path = os.path.join(self.get_temp_dir(), 'basics_with_list')
        with self.session(graph=ops_lib.Graph()) as sess:
            v0 = variable_v1.VariableV1(10.0, name='v0')
            v1 = variable_v1.VariableV1(20.0, name='v1')
            v2 = saver_test_utils.CheckpointedOp(name='v2')
            v2_init = v2.insert('k1', 30.0)
            save = saver_module.Saver([v0, v1, v2.saveable])
            self.evaluate(variables.global_variables_initializer())
            v2_init.run()
            self.assertEqual(10.0, self.evaluate(v0))
            self.assertEqual(20.0, self.evaluate(v1))
            self.assertEqual(b'k1', self.evaluate(v2.keys()))
            self.assertEqual(30.0, self.evaluate(v2.values()))
            val = save.save(sess, save_path)
            self.assertIsInstance(val, str)
            self.assertEqual(save_path, val)
        with self.session(graph=ops_lib.Graph()) as sess:
            v0 = variable_v1.VariableV1(-1.0, name='v0')
            v1 = variable_v1.VariableV1(-1.0, name='v1')
            v2 = saver_test_utils.CheckpointedOp(name='v2')
            save = saver_module.Saver([v0, v1, v2.saveable])
            with self.assertRaisesWithPredicateMatch(errors_impl.OpError, lambda e: 'uninitialized value v0' in e.message):
                self.evaluate(v0)
            with self.assertRaisesWithPredicateMatch(errors_impl.OpError, lambda e: 'uninitialized value v1' in e.message):
                self.evaluate(v1)
            self.assertEqual(0, len(self.evaluate(v2.keys())))
            self.assertEqual(0, len(self.evaluate(v2.values())))
            save.restore(sess, save_path)
            self.assertEqual(10.0, self.evaluate(v0))
            self.assertEqual(20.0, self.evaluate(v1))
            self.assertEqual(b'k1', self.evaluate(v2.keys()))
            self.assertEqual(30.0, self.evaluate(v2.values()))
        with self.session(graph=ops_lib.Graph()) as sess:
            v0_2 = variable_v1.VariableV1(1000.0, name='v0')
            v1_2 = variable_v1.VariableV1(2000.0, name='v1')
            v2_2 = saver_test_utils.CheckpointedOp(name='v2')
            save2 = saver_module.Saver([v0_2, v1_2, v2_2.saveable])
            v2_2.insert('k1000', 3000.0).run()
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(1000.0, self.evaluate(v0_2))
            self.assertEqual(2000.0, self.evaluate(v1_2))
            self.assertEqual(b'k1000', self.evaluate(v2_2.keys()))
            self.assertEqual(3000.0, self.evaluate(v2_2.values()))
            save2.restore(sess, save_path)
            self.assertEqual(10.0, self.evaluate(v0_2))
            self.assertEqual(20.0, self.evaluate(v1_2))
            self.assertEqual(b'k1', self.evaluate(v2_2.keys()))
            self.assertEqual(30.0, self.evaluate(v2_2.values()))

    def _SaveAndLoad(self, var_name, var_value, other_value, save_path):
        if False:
            i = 10
            return i + 15
        with self.session(graph=ops_lib.Graph()) as sess:
            var = resource_variable_ops.ResourceVariable(var_value, name=var_name)
            save = saver_module.Saver({var_name: var})
            if not context.executing_eagerly():
                self.evaluate(var.initializer)
            val = save.save(sess, save_path)
            self.assertEqual(save_path, val)
        with self.session(graph=ops_lib.Graph()) as sess:
            var = resource_variable_ops.ResourceVariable(other_value, name=var_name)
            save = saver_module.Saver({var_name: var})
            save.restore(sess, save_path)
            self.assertAllClose(var_value, self.evaluate(var))

    def testCacheRereadsFile(self):
        if False:
            while True:
                i = 10
        save_path = os.path.join(self.get_temp_dir(), 'cache_rereads')
        self._SaveAndLoad('var0', 0.0, 1.0, save_path)
        self._SaveAndLoad('var1', 1.1, 2.2, save_path)

    def testAllowEmpty(self):
        if False:
            print('Hello World!')
        save_path = os.path.join(self.get_temp_dir(), 'allow_empty')
        with ops_lib.Graph().as_default(), self.cached_session() as sess:
            _ = constant_op.constant(1)
            save = saver_module.Saver(allow_empty=True)
            val = save.save(sess, save_path)
            self.assertIsNone(val)
        with ops_lib.Graph().as_default(), self.cached_session() as sess:
            save = saver_module.Saver(allow_empty=True)
            save.restore(sess, save_path)

    def testGPU(self):
        if False:
            for i in range(10):
                print('nop')
        if not test.is_gpu_available():
            return
        save_path = os.path.join(self.get_temp_dir(), 'gpu')
        with session.Session('', graph=ops_lib.Graph()) as sess:
            with sess.graph.device(test.gpu_device_name()):
                v0_1 = variable_v1.VariableV1(123.45)
            save = saver_module.Saver({'v0': v0_1})
            self.evaluate(variables.global_variables_initializer())
            save.save(sess, save_path)
        with session.Session('', graph=ops_lib.Graph()) as sess:
            with sess.graph.device(test.gpu_device_name()):
                v0_2 = variable_v1.VariableV1(543.21)
            save = saver_module.Saver({'v0': v0_2})
            self.evaluate(variables.global_variables_initializer())

    def testSharedServerOnGPU(self):
        if False:
            return 10
        if not test.is_gpu_available():
            return
        save_path = os.path.join(self.get_temp_dir(), 'gpu')
        with session.Session('', graph=ops_lib.Graph()) as sess:
            with sess.graph.device(test.gpu_device_name()):
                v0_1 = variable_v1.VariableV1(123.45)
            save = saver_module.Saver({'v0': v0_1}, sharded=True, allow_empty=True)
            self.evaluate(variables.global_variables_initializer())
            save.save(sess, save_path)
        with session.Session('', graph=ops_lib.Graph()) as sess:
            with sess.graph.device(test.gpu_device_name()):
                v0_2 = variable_v1.VariableV1(543.21)
            save = saver_module.Saver({'v0': v0_2}, sharded=True, allow_empty=True)
            self.evaluate(variables.global_variables_initializer())

    def testVariables(self):
        if False:
            for i in range(10):
                print('nop')
        save_path = os.path.join(self.get_temp_dir(), 'variables')
        with session.Session('', graph=ops_lib.Graph()) as sess:
            one = variable_v1.VariableV1(1.0)
            twos = variable_v1.VariableV1([2.0, 2.0, 2.0])
            v2 = saver_test_utils.CheckpointedOp(name='v2')
            init = variables.global_variables_initializer()
            save = saver_module.Saver()
            init.run()
            v2.insert('k1', 3.0).run()
            save.save(sess, save_path)
        with session.Session('', graph=ops_lib.Graph()) as sess:
            one = variable_v1.VariableV1(0.0)
            twos = variable_v1.VariableV1([0.0, 0.0, 0.0])
            v2 = saver_test_utils.CheckpointedOp(name='v2')
            save = saver_module.Saver()
            save.restore(sess, save_path)
            self.assertAllClose(1.0, self.evaluate(one))
            self.assertAllClose([2.0, 2.0, 2.0], self.evaluate(twos))
            self.assertEqual(b'k1', self.evaluate(v2.keys()))
            self.assertEqual(3.0, self.evaluate(v2.values()))

    def testVarListShouldBeEmptyInDeferredBuild(self):
        if False:
            return 10
        with ops_lib.Graph().as_default():
            v = variable_v1.VariableV1(1.0)
            with self.assertRaisesRegex(ValueError, 'defer_build'):
                saver_module.Saver([v], defer_build=True)

    def testBuildShouldBeCalledBeforeSaveInCaseOfDeferBuild(self):
        if False:
            while True:
                i = 10
        save_path = os.path.join(self.get_temp_dir(), 'error_deferred_build')
        with ops_lib.Graph().as_default(), session.Session() as sess:
            variable_v1.VariableV1(1.0)
            saver = saver_module.Saver(defer_build=True)
            with self.assertRaisesRegex(RuntimeError, 'build'):
                saver.save(sess, save_path)

    def testDeferredBuild(self):
        if False:
            while True:
                i = 10
        save_path = os.path.join(self.get_temp_dir(), 'deferred_build')
        with session.Session('', graph=ops_lib.Graph()) as sess:
            one = variable_v1.VariableV1(1.0)
            save = saver_module.Saver(defer_build=True)
            twos = variable_v1.VariableV1([2.0, 2.0, 2.0])
            init = variables.global_variables_initializer()
            save.build()
            init.run()
            save.save(sess, save_path)
        with session.Session('', graph=ops_lib.Graph()) as sess:
            one = variable_v1.VariableV1(0.0)
            twos = variable_v1.VariableV1([0.0, 0.0, 0.0])
            save = saver_module.Saver()
            save.restore(sess, save_path)
            self.assertAllClose(1.0, self.evaluate(one))
            self.assertAllClose([2.0, 2.0, 2.0], self.evaluate(twos))

    @test_util.run_v1_only('train.Saver is V1 only API.')
    def testReshape(self):
        if False:
            return 10
        save_path = os.path.join(self.get_temp_dir(), 'variables_reshape')
        with session.Session('', graph=ops_lib.Graph()) as sess:
            var = variable_v1.VariableV1([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            init = variables.global_variables_initializer()
            save = saver_module.Saver()
            init.run()
            save.save(sess, save_path)
        with session.Session('', graph=ops_lib.Graph()) as sess:
            var = variable_v1.VariableV1([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            save = saver_module.Saver()
            with self.assertRaisesRegex(errors_impl.InvalidArgumentError, 'Assign requires shapes of both tensors to match.'):
                save.restore(sess, save_path)
        with session.Session('', graph=ops_lib.Graph()) as sess:
            var = variable_v1.VariableV1([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            save = saver_module.Saver(reshape=True)
            save.restore(sess, save_path)
            self.assertAllClose([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], self.evaluate(var))

    @test_util.run_in_graph_and_eager_modes
    def testSaveWithGlobalStep(self, pad_step_number=False):
        if False:
            return 10
        save_path = os.path.join(self.get_temp_dir(), 'ckpt_with_global_step')
        global_step_int = 5
        self._SaveAndLoad('var0', 0.0, 1.0, save_path)
        for use_tensor in [True, False]:
            with self.session(graph=ops_lib.Graph()):
                var = resource_variable_ops.ResourceVariable(1.0, name='var0')
                save = saver_module.Saver({var._shared_name: var}, pad_step_number=pad_step_number)
                if context.executing_eagerly():
                    sess = None
                else:
                    self.evaluate(var.initializer)
                    sess = ops_lib.get_default_session()
                if use_tensor:
                    global_step = constant_op.constant(global_step_int)
                    val = save.save(sess, save_path, global_step=global_step)
                else:
                    val = save.save(sess, save_path, global_step=global_step_int)
                if pad_step_number:
                    expected_save_path = '%s-%s' % (save_path, '{:08d}'.format(global_step_int))
                else:
                    expected_save_path = '%s-%d' % (save_path, global_step_int)
                self.assertEqual(expected_save_path, val)

    def testSaveWithGlobalStepWithPadding(self):
        if False:
            while True:
                i = 10
        self.testSaveWithGlobalStep(pad_step_number=True)

    def testSaveToNonexistingPath(self):
        if False:
            return 10
        file_io.write_string_to_file(os.path.join(self.get_temp_dir(), 'actually_a_file'), '')
        paths = [os.path.join(self.get_temp_dir(), 'nonexisting_dir/path'), os.path.join(self.get_temp_dir(), 'other_nonexisting_dir/path1/path2'), os.path.join(self.get_temp_dir(), 'actually_a_file/path')]
        for save_path in paths:
            v0 = variable_v1.VariableV1(10.0, name='v0')
            v1 = variable_v1.VariableV1(20.0, name='v1')
            save = saver_module.Saver({'v0': v0, 'v1': v1}, restore_sequentially=True)
            init_all_op = variables.global_variables_initializer()
            try:
                with self.cached_session() as sess:
                    self.evaluate(init_all_op)
                    self.assertEqual(10.0, self.evaluate(v0))
                    self.assertEqual(20.0, self.evaluate(v1))
                    save.save(sess, save_path)
                with self.cached_session() as sess:
                    save.restore(sess, save_path)
                    self.assertEqual(10.0, self.evaluate(v0))
                    self.assertEqual(20.0, self.evaluate(v1))
            except ValueError as exc:
                error_msg_template = "Parent directory of {} doesn't exist, can't save."
                self.assertEqual(error_msg_template.format(save_path), str(exc))

    def testSaveToURI(self):
        if False:
            print('Hello World!')
        if os.name == 'nt':
            self.skipTest("Local URI support doesn't work on Windows")
        save_path = 'file://' + os.path.join(self.get_temp_dir(), 'uri')
        v0 = variable_v1.VariableV1(10.0, name='v0')
        v1 = variable_v1.VariableV1(20.0, name='v1')
        save = saver_module.Saver({'v0': v0, 'v1': v1}, restore_sequentially=True)
        init_all_op = variables.global_variables_initializer()
        with self.cached_session() as sess:
            self.evaluate(init_all_op)
            self.assertEqual(10.0, self.evaluate(v0))
            self.assertEqual(20.0, self.evaluate(v1))
            save.save(sess, save_path)

    def testSaveRestoreAndValidateVariableDtype(self):
        if False:
            print('Hello World!')
        for variable_op in [variables.Variable, resource_variable_ops.ResourceVariable]:
            save_path = os.path.join(self.get_temp_dir(), 'basic_save_restore')
            with self.session(graph=ops_lib.Graph()) as sess:
                v0 = variable_op(10.0, name='v0', dtype=dtypes.float32)
                if not context.executing_eagerly():
                    self.evaluate([variables.global_variables_initializer()])
                save = saver_module.Saver({'v0': v0})
                save.save(sess, save_path)
            with self.session(graph=ops_lib.Graph()) as sess:
                v0_wrong_dtype = variable_op(1, name='v0', dtype=dtypes.int32)
                save = saver_module.Saver({'v0': v0_wrong_dtype})
                with self.assertRaisesRegex(errors.InvalidArgumentError, 'original dtype'):
                    save.restore(sess, save_path)

    def testRestoreLargeTensors(self):
        if False:
            for i in range(10):
                print('nop')
        save_dir = self.get_temp_dir()

        def _model():
            if False:
                while True:
                    i = 10
            small_v = [variable_scope.get_variable('small%d' % i, shape=[10, 2], use_resource=True) for i in range(5)]
            large_v = [variable_scope.get_variable('large%d' % i, shape=[32000, 1000], use_resource=True) for i in range(3)]
            return small_v + large_v
        save_graph = ops_lib.Graph()
        with save_graph.as_default(), self.session(graph=save_graph) as sess:
            orig_vars = _model()
            self.evaluate(variables.global_variables_initializer())
            save = saver_module.Saver(max_to_keep=1)
            self.evaluate(variables.global_variables_initializer())
            save.save(sess, save_dir)
            orig_vals = self.evaluate(orig_vars)
        restore_graph = ops_lib.Graph()
        with restore_graph.as_default(), self.session(graph=restore_graph) as sess:
            restored_vars = _model()
            save = saver_module.Saver(max_to_keep=1)
            save.restore(sess, save_dir)
            restored_vals = self.evaluate(restored_vars)
        for (orig, restored) in zip(orig_vals, restored_vals):
            self.assertAllEqual(orig, restored)

    def test_metrics_save_restore(self):
        if False:
            return 10
        api_label = saver_module._SAVER_LABEL

        def _get_write_histogram_proto():
            if False:
                while True:
                    i = 10
            proto_bytes = metrics.GetCheckpointWriteDurations(api_label=api_label)
            histogram_proto = summary_pb2.HistogramProto()
            histogram_proto.ParseFromString(proto_bytes)
            return histogram_proto

        def _get_read_histogram_proto():
            if False:
                return 10
            proto_bytes = metrics.GetCheckpointReadDurations(api_label=api_label)
            histogram_proto = summary_pb2.HistogramProto()
            histogram_proto.ParseFromString(proto_bytes)
            return histogram_proto
        save_path = os.path.join(self.get_temp_dir(), 'metrics_save_restore')
        time_start = metrics.GetTrainingTimeSaved(api_label=api_label)
        num_writes_start = _get_write_histogram_proto().num
        num_reads_start = _get_read_histogram_proto().num
        with self.session(graph=ops_lib.Graph()) as sess:
            v0 = resource_variable_ops.ResourceVariable(10.0, name='v0')
            v1 = resource_variable_ops.ResourceVariable(20.0, name='v1')
            v2 = saver_test_utils.CheckpointedOp(name='v2')
            if not context.executing_eagerly():
                self.evaluate([variables.global_variables_initializer()])
            save = saver_module.Saver({'v0': v0, 'v1': v1, 'v2': v2.saveable}, restore_sequentially=True)
            ckpt_prefix = save.save(sess, save_path)
            filesize = saver_module._get_checkpoint_size(ckpt_prefix)
            count_after_one_save = metrics.GetCheckpointSize(api_label=api_label, filesize=filesize)
            self.assertEqual(_get_write_histogram_proto().num, num_writes_start + 1)
            time_after_one_save = metrics.GetTrainingTimeSaved(api_label=api_label)
            self.assertGreater(time_after_one_save, time_start)
        with self.session(graph=ops_lib.Graph()) as sess:
            v0 = resource_variable_ops.ResourceVariable(-1.0, name='v0')
            v1 = resource_variable_ops.ResourceVariable(-1.0, name='v1')
            v2 = saver_test_utils.CheckpointedOp(name='v2')
            save = saver_module.Saver({'v0': v0, 'v1': v1, 'v2': v2.saveable})
            save.restore(sess, save_path)
            self.assertEqual(_get_write_histogram_proto().num, num_writes_start + 1)
            self.assertEqual(_get_read_histogram_proto().num, num_reads_start + 1)
            self.assertEqual(metrics.GetTrainingTimeSaved(api_label=api_label), time_after_one_save)
            save.save(sess, save_path)
            self.assertEqual(_get_write_histogram_proto().num, num_writes_start + 2)
            self.assertEqual(_get_read_histogram_proto().num, num_reads_start + 1)
            self.assertGreater(metrics.GetTrainingTimeSaved(api_label=api_label), time_after_one_save)
            self.assertEqual(metrics.GetCheckpointSize(api_label=api_label, filesize=filesize), count_after_one_save + 1)

class SaveRestoreShardedTest(test.TestCase):
    _WRITE_VERSION = saver_pb2.SaverDef.V1

    def _get_test_dir(self, dirname):
        if False:
            while True:
                i = 10
        test_dir = os.path.join(self.get_temp_dir(), dirname)
        gfile.MakeDirs(test_dir)
        return test_dir

    def testBasics(self):
        if False:
            return 10
        save_path = os.path.join(self.get_temp_dir(), 'sharded_basics')
        with session.Session(target='', config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
            with sess.graph.device('/cpu:0'):
                v0 = variable_v1.VariableV1(10, name='v0')
                t0 = saver_test_utils.CheckpointedOp(name='t0')
            with sess.graph.device('/cpu:1'):
                v1 = variable_v1.VariableV1(20, name='v1')
                t1 = saver_test_utils.CheckpointedOp(name='t1')
            save = saver_module.Saver({'v0': v0, 'v1': v1, 't0': t0.saveable, 't1': t1.saveable}, write_version=self._WRITE_VERSION, sharded=True)
            self.evaluate(variables.global_variables_initializer())
            t0.insert('k1', 30.0).run()
            t1.insert('k2', 40.0).run()
            val = save.save(sess, save_path)
            if save._write_version is saver_pb2.SaverDef.V1:
                self.assertEqual(save_path + '-?????-of-00002', val)
            else:
                self.assertEqual(save_path, val)
            meta_graph_filename = checkpoint_management.meta_graph_filename(val)
            self.assertEqual(save_path + '.meta', meta_graph_filename)
        if save._write_version is saver_pb2.SaverDef.V1:
            with session.Session(target='', config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
                with sess.graph.device('/cpu:0'):
                    v0 = variable_v1.VariableV1(111, name='v0')
                    t0 = saver_test_utils.CheckpointedOp(name='t0')
                save = saver_module.Saver({'v0': v0, 't0': t0.saveable}, write_version=self._WRITE_VERSION, sharded=True)
                self.evaluate(variables.global_variables_initializer())
                t0.insert('k11', 33.0).run()
                self.assertEqual(111, self.evaluate(v0))
                self.assertEqual(b'k11', self.evaluate(t0.keys()))
                self.assertEqual(33.0, self.evaluate(t0.values()))
                save.restore(sess, save_path + '-00000-of-00002')
                self.assertEqual(10, self.evaluate(v0))
                self.assertEqual(b'k1', self.evaluate(t0.keys()))
                self.assertEqual(30.0, self.evaluate(t0.values()))
            with session.Session(target='', config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
                with sess.graph.device('/cpu:0'):
                    v1 = variable_v1.VariableV1(222)
                    t1 = saver_test_utils.CheckpointedOp(name='t1')
                save = saver_module.Saver({'v1': v1, 't1': t1.saveable}, write_version=self._WRITE_VERSION, sharded=True)
                self.evaluate(variables.global_variables_initializer())
                t1.insert('k22', 44.0).run()
                self.assertEqual(222, self.evaluate(v1))
                self.assertEqual(b'k22', self.evaluate(t1.keys()))
                self.assertEqual(44.0, self.evaluate(t1.values()))
                save.restore(sess, save_path + '-00001-of-00002')
                self.assertEqual(20, self.evaluate(v1))
                self.assertEqual(b'k2', self.evaluate(t1.keys()))
                self.assertEqual(40.0, self.evaluate(t1.values()))
        with session.Session(target='', config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
            with sess.graph.device('/cpu:0'):
                v0 = variable_v1.VariableV1(111, name='v0')
                t0 = saver_test_utils.CheckpointedOp(name='t0')
            with sess.graph.device('/cpu:1'):
                v1 = variable_v1.VariableV1(222, name='v1')
                t1 = saver_test_utils.CheckpointedOp(name='t1')
            save = saver_module.Saver({'v0': v0, 'v1': v1, 't0': t0.saveable, 't1': t1.saveable}, write_version=self._WRITE_VERSION, sharded=True)
            self.evaluate(variables.global_variables_initializer())
            t0.insert('k11', 33.0).run()
            t1.insert('k22', 44.0).run()
            self.assertEqual(111, self.evaluate(v0))
            self.assertEqual(222, self.evaluate(v1))
            self.assertEqual(b'k11', self.evaluate(t0.keys()))
            self.assertEqual(33.0, self.evaluate(t0.values()))
            self.assertEqual(b'k22', self.evaluate(t1.keys()))
            self.assertEqual(44.0, self.evaluate(t1.values()))
            save_path = os.path.join(self.get_temp_dir(), 'sharded_basics')
            if save._write_version is saver_pb2.SaverDef.V1:
                save.restore(sess, save_path + '-?????-of-?????')
            else:
                save.restore(sess, save_path)
            self.assertEqual(10, self.evaluate(v0))
            self.assertEqual(20, self.evaluate(v1))
            self.assertEqual(b'k1', self.evaluate(t0.keys()))
            self.assertEqual(30.0, self.evaluate(t0.values()))
            self.assertEqual(b'k2', self.evaluate(t1.keys()))
            self.assertEqual(40.0, self.evaluate(t1.values()))
        if save._write_version is saver_pb2.SaverDef.V1:
            self.assertEqual(checkpoint_management.latest_checkpoint(self.get_temp_dir()), os.path.join(self.get_temp_dir(), 'sharded_basics-?????-of-00002'))
        else:
            self.assertEqual(checkpoint_management.latest_checkpoint(self.get_temp_dir()), os.path.join(self.get_temp_dir(), 'sharded_basics'))

    def testSaverDef(self):
        if False:
            for i in range(10):
                print('nop')
        with ops_lib.Graph().as_default(), self.cached_session():
            v0 = variable_v1.VariableV1(123, name='v0')
            save = saver_module.Saver({'v0': v0}, sharded=True)
            sd = save.as_saver_def()
            self.assertTrue(sd.sharded)

    def _testPartitionedVariables(self, use_resource):
        if False:
            i = 10
            return i + 15
        var_full_shape = [10, 3]
        var_name = 'my_var'
        saved_dir = self._get_test_dir('partitioned_variables')
        saved_path = os.path.join(saved_dir, 'ckpt')
        call_saver_with_dict = False

        def _save(partitioner=None):
            if False:
                i = 10
                return i + 15
            with ops_lib.Graph().as_default(), self.session() as sess:
                rnd = random_ops.random_uniform(var_full_shape).eval()
                if partitioner:
                    vs = [variable_scope.get_variable(var_name, shape=var_full_shape, initializer=rnd, partitioner=partitioner, use_resource=use_resource)]
                elif use_resource:
                    vs = [resource_variable_ops.ResourceVariable(rnd, name=var_name)]
                else:
                    vs = [variable_v1.VariableV1(rnd, name=var_name)]
                self.evaluate(variables.global_variables_initializer())
                if call_saver_with_dict:
                    saver = saver_module.Saver({var_name: vs[0]})
                else:
                    saver = saver_module.Saver(vs)
                actual_path = saver.save(sess, saved_path)
                self.assertEqual(saved_path, actual_path)
                return rnd

        def _restore(partitioner=None):
            if False:
                while True:
                    i = 10
            with ops_lib.Graph().as_default(), self.session() as sess:
                if partitioner:
                    new_vs = [variable_scope.get_variable(var_name, shape=var_full_shape, initializer=array_ops.zeros(var_full_shape), partitioner=partitioner)]
                else:
                    new_vs = [variable_v1.VariableV1(array_ops.zeros(shape=var_full_shape), name=var_name)]
                self.evaluate(variables.global_variables_initializer())
                if call_saver_with_dict:
                    saver = saver_module.Saver({var_name: new_vs[0]})
                else:
                    saver = saver_module.Saver(new_vs)
                saver.restore(sess, saved_path)
                if partitioner:
                    return new_vs[0].as_tensor().eval()
                else:
                    return new_vs[0].eval()
        for call_saver_with_dict in {False, True}:
            saved_full = _save(partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
            restored_full = _restore()
            self.assertAllEqual(saved_full, restored_full)
            restored_full = _restore(partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
            self.assertAllEqual(saved_full, restored_full)
            restored_full = _restore(partitioner=partitioned_variables.fixed_size_partitioner(num_shards=3))
            self.assertAllEqual(saved_full, restored_full)
            saved_full = _save()
            restored_full = _restore(partitioner=partitioned_variables.fixed_size_partitioner(num_shards=3))
            self.assertAllEqual(saved_full, restored_full)

    def testPartitionedVariable(self):
        if False:
            return 10
        self._testPartitionedVariables(use_resource=False)

    def testPartitionedResourceVariable(self):
        if False:
            for i in range(10):
                print('nop')
        self._testPartitionedVariables(use_resource=True)

class SaveRestoreShardedTestV2(SaveRestoreShardedTest):
    _WRITE_VERSION = saver_pb2.SaverDef.V2

    def testIterators(self):
        if False:
            return 10
        save_path = os.path.join(self.get_temp_dir(), 'sharded_iterators')
        with session.Session(target='', config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
            with sess.graph.device('/cpu:0'):
                ds0 = dataset_ops.Dataset.range(10)
                it0 = dataset_ops.make_initializable_iterator(ds0)
                get_next0 = it0.get_next()
            saveable0 = iterator_ops._IteratorSaveable(it0._iterator_resource, name='saveable_it0')
            with sess.graph.device('/cpu:1'):
                ds1 = dataset_ops.Dataset.range(20)
                it1 = dataset_ops.make_initializable_iterator(ds1)
                get_next1 = it1.get_next()
            saveable1 = iterator_ops._IteratorSaveable(it1._iterator_resource, name='saveable_it1')
            saver = saver_module.Saver({'it0': saveable0, 'it1': saveable1}, write_version=self._WRITE_VERSION, sharded=True)
            self.evaluate(it0.initializer)
            self.evaluate(it1.initializer)
            self.assertEqual(0, self.evaluate(get_next0))
            self.assertEqual(1, self.evaluate(get_next0))
            self.assertEqual(0, self.evaluate(get_next1))
            val = saver.save(sess, save_path)
            self.assertEqual(save_path, val)
            data_files = glob.glob(save_path + '.data*')
            self.assertEqual(2, len(data_files))
        with session.Session(target='', config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
            with sess.graph.device('/cpu:0'):
                ds0 = dataset_ops.Dataset.range(10)
                it0 = dataset_ops.make_initializable_iterator(ds0)
                get_next0 = it0.get_next()
            saveable0 = iterator_ops._IteratorSaveable(it0._iterator_resource, name='saveable_it0')
            with sess.graph.device('/cpu:1'):
                ds1 = dataset_ops.Dataset.range(20)
                it1 = dataset_ops.make_initializable_iterator(ds1)
                get_next1 = it1.get_next()
            saveable1 = iterator_ops._IteratorSaveable(it1._iterator_resource, name='saveable_it1')
            saver = saver_module.Saver({'it0': saveable0, 'it1': saveable1}, write_version=self._WRITE_VERSION, sharded=True)
            self.evaluate(it0.initializer)
            self.evaluate(it1.initializer)
            saver.restore(sess, save_path)
            self.assertEqual(2, self.evaluate(get_next0))
            self.assertEqual(1, self.evaluate(get_next1))

    def testIteratorsUnshardedRestore(self):
        if False:
            return 10
        save_path = os.path.join(self.get_temp_dir(), 'restore_unsharded_iterators')
        with session.Session(target='', config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
            with sess.graph.device('/cpu:0'):
                ds0 = dataset_ops.Dataset.range(10)
                it0 = dataset_ops.make_initializable_iterator(ds0)
                get_next0 = it0.get_next()
            saveable0 = iterator_ops._IteratorSaveable(it0._iterator_resource, name='saveable_it0')
            with sess.graph.device('/cpu:1'):
                ds1 = dataset_ops.Dataset.range(20)
                it1 = dataset_ops.make_initializable_iterator(ds1)
                get_next1 = it1.get_next()
            saveable1 = iterator_ops._IteratorSaveable(it1._iterator_resource, name='saveable_it1')
            saver = saver_module.Saver({'it0': saveable0, 'it1': saveable1}, write_version=self._WRITE_VERSION, sharded=True)
            self.evaluate(it0.initializer)
            self.evaluate(it1.initializer)
            self.assertEqual(0, self.evaluate(get_next0))
            self.assertEqual(1, self.evaluate(get_next0))
            self.assertEqual(0, self.evaluate(get_next1))
            val = saver.save(sess, save_path)
            self.assertEqual(save_path, val)
            data_files = glob.glob(save_path + '.data*')
            self.assertEqual(2, len(data_files))
        with session.Session(target='', config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
            with sess.graph.device('/cpu:0'):
                ds0 = dataset_ops.Dataset.range(10)
                it0 = dataset_ops.make_initializable_iterator(ds0)
                get_next0 = it0.get_next()
            saveable0 = iterator_ops._IteratorSaveable(it0._iterator_resource, name='saveable_it0')
            with sess.graph.device('/cpu:1'):
                ds1 = dataset_ops.Dataset.range(20)
                it1 = dataset_ops.make_initializable_iterator(ds1)
                get_next1 = it1.get_next()
            saveable1 = iterator_ops._IteratorSaveable(it1._iterator_resource, name='saveable_it1')
            saver = saver_module.Saver({'it0': saveable0, 'it1': saveable1}, write_version=self._WRITE_VERSION, sharded=False)
            self.evaluate(it0.initializer)
            self.evaluate(it1.initializer)
            saver.restore(sess, save_path)
            self.assertEqual(2, self.evaluate(get_next0))
            self.assertEqual(1, self.evaluate(get_next1))

class MaxToKeepTest(test.TestCase):

    def _get_test_dir(self, dirname):
        if False:
            while True:
                i = 10
        test_dir = os.path.join(self.get_temp_dir(), dirname)
        gfile.MakeDirs(test_dir)
        return test_dir

    def assertCheckpointState(self, model_checkpoint_path, all_model_checkpoint_paths, save_dir):
        if False:
            while True:
                i = 10
        checkpoint_state = checkpoint_management.get_checkpoint_state(save_dir)
        self.assertEqual(checkpoint_state.model_checkpoint_path, model_checkpoint_path)
        self.assertEqual(checkpoint_state.all_model_checkpoint_paths, all_model_checkpoint_paths)

    def testMaxToKeepEager(self):
        if False:
            return 10
        with context.eager_mode():
            save_dir = self._get_test_dir('max_to_keep_eager')
            v = variable_v1.VariableV1(10.0, name='v')
            save = saver_module.Saver({'v': v}, max_to_keep=2)
            self.evaluate(variables.global_variables_initializer())
            if not context.executing_eagerly():
                self.assertEqual([], save.last_checkpoints)
            s1 = save.save(None, os.path.join(save_dir, 's1'))
            self.assertEqual([s1], save.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertCheckpointState(model_checkpoint_path=s1, all_model_checkpoint_paths=[s1], save_dir=save_dir)
            s2 = save.save(None, os.path.join(save_dir, 's2'))
            self.assertEqual([s1, s2], save.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertCheckpointState(model_checkpoint_path=s2, all_model_checkpoint_paths=[s1, s2], save_dir=save_dir)
            s3 = save.save(None, os.path.join(save_dir, 's3'))
            self.assertEqual([s2, s3], save.last_checkpoints)
            self.assertFalse(checkpoint_management.checkpoint_exists(s1))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertTrue(checkpoint_management.checkpoint_exists(s3))
            self.assertCheckpointState(model_checkpoint_path=s3, all_model_checkpoint_paths=[s2, s3], save_dir=save_dir)
            save2 = saver_module.Saver({'v': v}, max_to_keep=2)
            save2.set_last_checkpoints(save.last_checkpoints)
            s2 = save.save(None, os.path.join(save_dir, 's2'))
            self.assertEqual([s3, s2], save.last_checkpoints)
            self.assertFalse(checkpoint_management.checkpoint_exists(s1))
            self.assertTrue(checkpoint_management.checkpoint_exists(s3))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertCheckpointState(model_checkpoint_path=s2, all_model_checkpoint_paths=[s3, s2], save_dir=save_dir)
            s1 = save.save(None, os.path.join(save_dir, 's1'))
            self.assertEqual([s2, s1], save.last_checkpoints)
            self.assertFalse(checkpoint_management.checkpoint_exists(s3))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertCheckpointState(model_checkpoint_path=s1, all_model_checkpoint_paths=[s2, s1], save_dir=save_dir)
            s2 = save2.save(None, os.path.join(save_dir, 's2'))
            self.assertEqual([s3, s2], save2.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertFalse(checkpoint_management.checkpoint_exists(s3))

    def testNonSharded(self):
        if False:
            print('Hello World!')
        save_dir = self._get_test_dir('max_to_keep_non_sharded')
        with ops_lib.Graph().as_default(), self.cached_session() as sess:
            v = variable_v1.VariableV1(10.0, name='v')
            save = saver_module.Saver({'v': v}, max_to_keep=2)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual([], save.last_checkpoints)
            s1 = save.save(sess, os.path.join(save_dir, 's1'))
            self.assertEqual([s1], save.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertCheckpointState(model_checkpoint_path=s1, all_model_checkpoint_paths=[s1], save_dir=save_dir)
            s2 = save.save(sess, os.path.join(save_dir, 's2'))
            self.assertEqual([s1, s2], save.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertCheckpointState(model_checkpoint_path=s2, all_model_checkpoint_paths=[s1, s2], save_dir=save_dir)
            s3 = save.save(sess, os.path.join(save_dir, 's3'))
            self.assertEqual([s2, s3], save.last_checkpoints)
            self.assertFalse(checkpoint_management.checkpoint_exists(s1))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertTrue(checkpoint_management.checkpoint_exists(s3))
            self.assertCheckpointState(model_checkpoint_path=s3, all_model_checkpoint_paths=[s2, s3], save_dir=save_dir)
            save2 = saver_module.Saver(saver_def=save.as_saver_def())
            save2.set_last_checkpoints(save.last_checkpoints)
            save3 = saver_module.Saver(saver_def=save.as_saver_def())
            s2 = save.save(sess, os.path.join(save_dir, 's2'))
            self.assertEqual([s3, s2], save.last_checkpoints)
            self.assertFalse(checkpoint_management.checkpoint_exists(s1))
            self.assertFalse(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s1)))
            self.assertTrue(checkpoint_management.checkpoint_exists(s3))
            self.assertTrue(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s3)))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertTrue(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s2)))
            self.assertCheckpointState(model_checkpoint_path=s2, all_model_checkpoint_paths=[s3, s2], save_dir=save_dir)
            s1 = save.save(sess, os.path.join(save_dir, 's1'))
            self.assertEqual([s2, s1], save.last_checkpoints)
            self.assertFalse(checkpoint_management.checkpoint_exists(s3))
            self.assertFalse(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s3)))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertTrue(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s2)))
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertTrue(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s1)))
            self.assertCheckpointState(model_checkpoint_path=s1, all_model_checkpoint_paths=[s2, s1], save_dir=save_dir)
            s2 = save2.save(sess, os.path.join(save_dir, 's2'))
            self.assertEqual([s3, s2], save2.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertTrue(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s1)))
            self.assertFalse(checkpoint_management.checkpoint_exists(s3))
            self.assertFalse(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s3)))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertTrue(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s2)))
            self.assertCheckpointState(model_checkpoint_path=s2, all_model_checkpoint_paths=[s3, s2], save_dir=save_dir)
            s1 = save2.save(sess, os.path.join(save_dir, 's1'))
            self.assertEqual([s2, s1], save2.last_checkpoints)
            self.assertFalse(checkpoint_management.checkpoint_exists(s3))
            self.assertFalse(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s3)))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertTrue(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s2)))
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertTrue(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s1)))
            self.assertCheckpointState(model_checkpoint_path=s1, all_model_checkpoint_paths=[s2, s1], save_dir=save_dir)
            s2 = save3.save(sess, os.path.join(save_dir, 's2'))
            self.assertEqual([s2], save3.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertTrue(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s1)))
            self.assertFalse(checkpoint_management.checkpoint_exists(s3))
            self.assertFalse(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s3)))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertTrue(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s2)))
            self.assertCheckpointState(model_checkpoint_path=s2, all_model_checkpoint_paths=[s2], save_dir=save_dir)
            s1 = save3.save(sess, os.path.join(save_dir, 's1'))
            self.assertEqual([s2, s1], save3.last_checkpoints)
            self.assertFalse(checkpoint_management.checkpoint_exists(s3))
            self.assertFalse(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s3)))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertTrue(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s2)))
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertTrue(checkpoint_management.checkpoint_exists(checkpoint_management.meta_graph_filename(s1)))
            self.assertCheckpointState(model_checkpoint_path=s1, all_model_checkpoint_paths=[s2, s1], save_dir=save_dir)

    def testSharded(self):
        if False:
            print('Hello World!')
        save_dir = self._get_test_dir('max_to_keep_sharded')
        with session.Session(target='', config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
            with sess.graph.device('/cpu:0'):
                v0 = variable_v1.VariableV1(111, name='v0')
            with sess.graph.device('/cpu:1'):
                v1 = variable_v1.VariableV1(222, name='v1')
            save = saver_module.Saver({'v0': v0, 'v1': v1}, sharded=True, max_to_keep=2)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual([], save.last_checkpoints)
            s1 = save.save(sess, os.path.join(save_dir, 's1'))
            self.assertEqual([s1], save.last_checkpoints)
            if save._write_version is saver_pb2.SaverDef.V1:
                self.assertEqual(2, len(gfile.Glob(s1)))
            else:
                self.assertEqual(4, len(gfile.Glob(s1 + '*')))
            self.assertTrue(gfile.Exists(checkpoint_management.meta_graph_filename(s1)))
            s2 = save.save(sess, os.path.join(save_dir, 's2'))
            self.assertEqual([s1, s2], save.last_checkpoints)
            if save._write_version is saver_pb2.SaverDef.V1:
                self.assertEqual(2, len(gfile.Glob(s1)))
            else:
                self.assertEqual(4, len(gfile.Glob(s1 + '*')))
            self.assertTrue(gfile.Exists(checkpoint_management.meta_graph_filename(s1)))
            if save._write_version is saver_pb2.SaverDef.V1:
                self.assertEqual(2, len(gfile.Glob(s2)))
            else:
                self.assertEqual(4, len(gfile.Glob(s2 + '*')))
            self.assertTrue(gfile.Exists(checkpoint_management.meta_graph_filename(s2)))
            s3 = save.save(sess, os.path.join(save_dir, 's3'))
            self.assertEqual([s2, s3], save.last_checkpoints)
            self.assertEqual(0, len(gfile.Glob(s1 + '*')))
            self.assertFalse(gfile.Exists(checkpoint_management.meta_graph_filename(s1)))
            if save._write_version is saver_pb2.SaverDef.V1:
                self.assertEqual(2, len(gfile.Glob(s2)))
            else:
                self.assertEqual(4, len(gfile.Glob(s2 + '*')))
            self.assertTrue(gfile.Exists(checkpoint_management.meta_graph_filename(s2)))
            if save._write_version is saver_pb2.SaverDef.V1:
                self.assertEqual(2, len(gfile.Glob(s3)))
            else:
                self.assertEqual(4, len(gfile.Glob(s3 + '*')))
            self.assertTrue(gfile.Exists(checkpoint_management.meta_graph_filename(s3)))

    def testNoMaxToKeep(self):
        if False:
            while True:
                i = 10
        save_dir = self._get_test_dir('no_max_to_keep')
        save_dir2 = self._get_test_dir('max_to_keep_0')
        with self.cached_session() as sess:
            v = variable_v1.VariableV1(10.0, name='v')
            self.evaluate(variables.global_variables_initializer())
            save = saver_module.Saver({'v': v}, max_to_keep=None)
            self.assertEqual([], save.last_checkpoints)
            s1 = save.save(sess, os.path.join(save_dir, 's1'))
            self.assertEqual([], save.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            s2 = save.save(sess, os.path.join(save_dir, 's2'))
            self.assertEqual([], save.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            save2 = saver_module.Saver({'v': v}, max_to_keep=0)
            self.assertEqual([], save2.last_checkpoints)
            s1 = save2.save(sess, os.path.join(save_dir2, 's1'))
            self.assertEqual([], save2.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            s2 = save2.save(sess, os.path.join(save_dir2, 's2'))
            self.assertEqual([], save2.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))

    def testNoMetaGraph(self):
        if False:
            while True:
                i = 10
        save_dir = self._get_test_dir('no_meta_graph')
        with self.cached_session() as sess:
            v = variable_v1.VariableV1(10.0, name='v')
            save = saver_module.Saver({'v': v})
            self.evaluate(variables.global_variables_initializer())
            s1 = save.save(sess, os.path.join(save_dir, 's1'), write_meta_graph=False)
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertFalse(gfile.Exists(checkpoint_management.meta_graph_filename(s1)))

class RecoverLastCheckpointsTest(test.TestCase):

    def _get_test_dir(self, dirname):
        if False:
            print('Hello World!')
        test_dir = os.path.join(self.get_temp_dir(), dirname)
        gfile.MakeDirs(test_dir)
        return test_dir

    def assertCheckpointState(self, model_checkpoint_path, all_model_checkpoint_paths, save_dir):
        if False:
            return 10
        checkpoint_state = checkpoint_management.get_checkpoint_state(save_dir)
        self.assertEqual(checkpoint_state.model_checkpoint_path, model_checkpoint_path)
        self.assertEqual(checkpoint_state.all_model_checkpoint_paths, all_model_checkpoint_paths)

    def test_recover_last_checkpoints(self):
        if False:
            for i in range(10):
                print('nop')
        with context.eager_mode():
            save_dir = self._get_test_dir('recover_last_checkpoints')
            v = variable_v1.VariableV1(10.0, name='v')
            save = saver_module.Saver({'v': v}, max_to_keep=10)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual([], save.last_checkpoints)
            s1 = save.save(None, os.path.join(save_dir, 'ckpt-1'))
            s2 = save.save(None, os.path.join(save_dir, 'ckpt-2'))
            s3 = save.save(None, os.path.join(save_dir, 'ckpt-3'))
            self.assertEqual([s1, s2, s3], save.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertTrue(checkpoint_management.checkpoint_exists(s2))
            self.assertTrue(checkpoint_management.checkpoint_exists(s3))
            self.assertCheckpointState(model_checkpoint_path=s3, all_model_checkpoint_paths=[s1, s2, s3], save_dir=save_dir)
            save2 = saver_module.Saver({'v': v}, max_to_keep=10)
            self.assertEqual([], save2.last_checkpoints)
            save2.recover_last_checkpoints([s1, s2, s3])
            self.assertEqual([s1, s2, s3], save2.last_checkpoints)
            for fname in gfile.Glob('{}*'.format(s1)):
                gfile.Remove(fname)
            self.assertFalse(checkpoint_management.checkpoint_exists(s1))
            save3 = saver_module.Saver({'v': v}, max_to_keep=10)
            self.assertEqual([], save3.last_checkpoints)
            save3.recover_last_checkpoints([s1, s2, s3])
            self.assertEqual([s2, s3], save3.last_checkpoints)
            s4 = save3.save(None, os.path.join(save_dir, 'ckpt-4'))
            self.assertCheckpointState(model_checkpoint_path=s4, all_model_checkpoint_paths=[s2, s3, s4], save_dir=save_dir)

class KeepCheckpointEveryNHoursTest(test.TestCase):

    def _get_test_dir(self, dirname):
        if False:
            i = 10
            return i + 15
        test_dir = os.path.join(self.get_temp_dir(), dirname)
        gfile.MakeDirs(test_dir)
        return test_dir

    @test_util.run_in_graph_and_eager_modes
    @test.mock.patch.object(saver_module, 'time')
    def testNonSharded(self, mock_time):
        if False:
            for i in range(10):
                print('nop')
        save_dir = self._get_test_dir('keep_checkpoint_every_n_hours')
        with self.cached_session() as sess:
            v = variable_v1.VariableV1([10.0], name='v')
            self.evaluate(variables.global_variables_initializer())
            start_time = time.time()
            mock_time.time.return_value = start_time
            save = saver_module.Saver({'v': v}, max_to_keep=2, keep_checkpoint_every_n_hours=0.7 / 3600)
            self.assertEqual([], save.last_checkpoints)
            mock_time.time.return_value = start_time + 1.0
            s1 = save.save(sess, os.path.join(save_dir, 's1'))
            self.assertEqual([s1], save.last_checkpoints)
            s2 = save.save(sess, os.path.join(save_dir, 's2'))
            self.assertEqual([s1, s2], save.last_checkpoints)
            s3 = save.save(sess, os.path.join(save_dir, 's3'))
            self.assertEqual([s2, s3], save.last_checkpoints)
            s4 = save.save(sess, os.path.join(save_dir, 's4'))
            self.assertEqual([s3, s4], save.last_checkpoints)
            self.assertTrue(checkpoint_management.checkpoint_exists(s1))
            self.assertFalse(checkpoint_management.checkpoint_exists(s2))
            self.assertTrue(checkpoint_management.checkpoint_exists(s3))
            self.assertTrue(checkpoint_management.checkpoint_exists(s4))

class SaveRestoreWithVariableNameMap(test.TestCase):

    def _testNonReshape(self, variable_op):
        if False:
            return 10
        save_path = os.path.join(self.get_temp_dir(), 'non_reshape')
        with self.session(graph=ops_lib.Graph()) as sess:
            v0 = variable_op(10.0, name='v0')
            v1 = variable_op(20.0, name='v1')
            save = saver_module.Saver({'save_prefix/v0': v0, 'save_prefix/v1': v1})
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(10.0, self.evaluate(v0))
            self.assertEqual(20.0, self.evaluate(v1))
            val = save.save(sess, save_path)
            self.assertIsInstance(val, str)
            self.assertEqual(save_path, val)
            save = saver_module.Saver({'v0': v0, 'v1': v1})
            with self.assertRaisesOpError('not found in checkpoint'):
                save.restore(sess, save_path)
        with self.session(graph=ops_lib.Graph()) as sess:
            v0 = variable_op(-1.0, name='v0')
            v1 = variable_op(-1.0, name='v1')
            if not context.executing_eagerly():
                with self.assertRaisesOpError('uninitialized'):
                    self.evaluate(v0)
                with self.assertRaisesOpError('uninitialized'):
                    self.evaluate(v1)
            save = saver_module.Saver({'save_prefix/v0': v0, 'save_prefix/v1': v1})
            save.restore(sess, save_path)
            if not context.executing_eagerly():
                self.assertEqual(10.0, self.evaluate(v0))
                self.assertEqual(20.0, self.evaluate(v1))
        with self.session(graph=ops_lib.Graph()) as sess:
            v0 = variable_op(-1.0, name='restore_prefix/v0')
            v1 = variable_op(-1.0, name='restore_prefix/v1')
            if not context.executing_eagerly():
                with self.assertRaisesOpError('uninitialized'):
                    self.evaluate(v0)
                with self.assertRaisesOpError('uninitialized'):
                    self.evaluate(v1)
            save = saver_module.Saver({'save_prefix/v0': v0, 'save_prefix/v1': v1})
            save.restore(sess, save_path)
            self.assertEqual(10.0, self.evaluate(v0))
            self.assertEqual(20.0, self.evaluate(v1))

    @test_util.run_in_graph_and_eager_modes
    def testNonReshapeResourceVariable(self):
        if False:
            while True:
                i = 10
        self._testNonReshape(resource_variable_ops.ResourceVariable)

    def testNonReshapeVariable(self):
        if False:
            for i in range(10):
                print('nop')
        self._testNonReshape(variables.Variable)

class MetaGraphTest(test.TestCase):

    def _get_test_dir(self, dirname):
        if False:
            return 10
        test_dir = os.path.join(self.get_temp_dir(), dirname)
        gfile.MakeDirs(test_dir)
        return test_dir

    @test_util.run_v1_only('Queue-based input pipelines have been replaced by `tf.data` and not supported in V2.')
    def testAddCollectionDef(self):
        if False:
            for i in range(10):
                print('nop')
        test_dir = self._get_test_dir('good_collection')
        filename = os.path.join(test_dir, 'metafile')
        with self.cached_session():
            v0 = variable_v1.VariableV1(1.0, name='v0')
            cond.cond(math_ops.less(v0, 10), lambda : math_ops.add(v0, 1), lambda : math_ops.subtract(v0, 1))
            while_loop.while_loop(lambda i: math_ops.less(i, 10), lambda i: math_ops.add(i, 1), [v0])
            var = variable_v1.VariableV1(constant_op.constant(0, dtype=dtypes.int64))
            count_up_to = var.count_up_to(3)
            input_queue = data_flow_ops.FIFOQueue(30, dtypes.float32, shared_name='collection_queue')
            qr = queue_runner_impl.QueueRunner(input_queue, [count_up_to])
            variables.global_variables_initializer()
            save = saver_module.Saver({'v0': v0})
            ops_lib.add_to_collection('int_collection', 3)
            ops_lib.add_to_collection('float_collection', 3.5)
            ops_lib.add_to_collection('string_collection', 'hello')
            ops_lib.add_to_collection('variable_collection', v0)
            queue_runner_impl.add_queue_runner(qr)
            queue_runner = queue_runner_pb2.QueueRunnerDef(queue_name='test_queue')
            ops_lib.add_to_collection('user_defined_string_collection', str(queue_runner))
            ops_lib.add_to_collection('user_defined_bytes_collection', queue_runner.SerializeToString())
            any_buf = Any()
            any_buf.Pack(queue_runner)
            ops_lib.add_to_collection('user_defined_any_collection', any_buf)
            meta_graph_def = save.export_meta_graph(filename)
            self.assertTrue(meta_graph_def.HasField('saver_def'))
            self.assertTrue(meta_graph_def.HasField('graph_def'))
            self.assertTrue(meta_graph_def.HasField('meta_info_def'))
            self.assertNotEqual(meta_graph_def.meta_info_def.tensorflow_version, '')
            self.assertNotEqual(meta_graph_def.meta_info_def.tensorflow_git_version, '')
            collection_def = meta_graph_def.collection_def
            self.assertEqual(len(collection_def), 12)
        with ops_lib.Graph().as_default():
            new_saver = saver_module.import_meta_graph(filename)
            new_meta_graph_def = new_saver.export_meta_graph()
        test_util.assert_meta_graph_protos_equal(self, meta_graph_def, new_meta_graph_def)

    def testAddCollectionDefFails(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            v0 = variable_v1.VariableV1(10.0, name='v0')
            save = saver_module.Saver({'v0': v0})
            meta_graph_def = meta_graph_pb2.MetaGraphDef()
            ops_lib.add_to_collection(save, 3)
            save._add_collection_def(meta_graph_def, save)
            self.assertEqual(len(meta_graph_def.collection_def), 0)
            ops_lib.add_to_collection('int_collection', 3)
            ops_lib.add_to_collection('int_collection', 3.5)
            save._add_collection_def(meta_graph_def, 'int_collection')
            self.assertEqual(len(meta_graph_def.collection_def), 0)

    def _testMultiSaverCollectionSave(self, test_dir):
        if False:
            for i in range(10):
                print('nop')
        filename = os.path.join(test_dir, 'metafile')
        saver0_ckpt = os.path.join(test_dir, 'saver0.ckpt')
        saver1_ckpt = os.path.join(test_dir, 'saver1.ckpt')
        with self.session(graph=ops_lib.Graph()) as sess:
            v0 = variable_v1.VariableV1([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='v0')
            v1 = variable_v1.VariableV1(11.0, name='v1')
            saver0 = saver_module.Saver({'v0': v0}, name='saver0')
            saver1 = saver_module.Saver({'v1': v1}, name='saver1')
            ops_lib.add_to_collection('savers', saver0)
            ops_lib.add_to_collection('savers', saver1)
            self.evaluate(variables.global_variables_initializer())
            saver0.save(sess, saver0_ckpt)
            saver1.save(sess, saver1_ckpt)
            meta_graph_def = saver_module.export_meta_graph(filename)
            meta_graph_def0 = saver0.export_meta_graph()
            meta_graph_def1 = saver1.export_meta_graph()
            self.assertFalse(meta_graph_def.HasField('saver_def'))
            self.assertTrue(meta_graph_def0.HasField('saver_def'))
            self.assertTrue(meta_graph_def1.HasField('saver_def'))
            collection_def = meta_graph_def.collection_def['savers']
            kind = collection_def.WhichOneof('kind')
            self.assertEqual(kind, 'bytes_list')
            savers = getattr(collection_def, kind)
            self.assertEqual(2, len(savers.value))
            collection_def = meta_graph_def0.collection_def['savers']
            kind = collection_def.WhichOneof('kind')
            self.assertEqual(kind, 'bytes_list')
            savers = getattr(collection_def, kind)
            self.assertEqual(2, len(savers.value))

    def _testMultiSaverCollectionRestore(self, test_dir):
        if False:
            for i in range(10):
                print('nop')
        filename = os.path.join(test_dir, 'metafile')
        saver0_ckpt = os.path.join(test_dir, 'saver0.ckpt')
        saver1_ckpt = os.path.join(test_dir, 'saver1.ckpt')
        with self.session(graph=ops_lib.Graph()) as sess:
            saver_module.import_meta_graph(filename)
            savers = ops_lib.get_collection('savers')
            self.assertEqual(2, len(savers))
            new_saver0 = savers[0]
            new_saver0.restore(sess, saver0_ckpt)
            v0 = sess.graph.get_tensor_by_name('v0:0')
            v1 = sess.graph.get_tensor_by_name('v1:0')
            self.assertAllEqual([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], self.evaluate(v0))
            self.assertEqual([3, 2], v0.get_shape())
            self.assertEqual([], v1.get_shape())
            with self.assertRaisesWithPredicateMatch(errors_impl.OpError, lambda e: 'uninitialized value v1' in e.message):
                self.evaluate(v1)
            new_saver1 = savers[1]
            new_saver1.restore(sess, saver1_ckpt)
            v1 = sess.graph.get_tensor_by_name('v1:0')
            self.assertEqual(11.0, self.evaluate(v1))

    @test_util.run_v1_only('Exporting/importing meta graphs is only supported in V1.')
    def testMultiSaverCollection(self):
        if False:
            for i in range(10):
                print('nop')
        test_dir = self._get_test_dir('saver_collection')
        self._testMultiSaverCollectionSave(test_dir)
        self._testMultiSaverCollectionRestore(test_dir)

    @test_util.run_v1_only('Exporting/importing meta graphs is only supported in V1.')
    def testClearExtraneousSavers(self):
        if False:
            return 10
        test_dir = self._get_test_dir('clear_extraneous_savers')
        filename = os.path.join(test_dir, 'metafile')
        saver0_ckpt = os.path.join(test_dir, 'saver0.ckpt')
        saver1_ckpt = os.path.join(test_dir, 'saver1.ckpt')
        with self.session(graph=ops_lib.Graph()) as sess:
            v0 = variable_v1.VariableV1([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='v0')
            v1 = variable_v1.VariableV1(11.0, name='v1')
            saver0 = saver_module.Saver({'v0': v0}, name='saver0')
            saver1 = saver_module.Saver({'v1': v1}, name='saver1')
            ops_lib.add_to_collection('savers', saver0)
            ops_lib.add_to_collection('savers', saver1)
            self.evaluate(variables.global_variables_initializer())
            saver0.save(sess, saver0_ckpt)
            saver1.save(sess, saver1_ckpt)
            meta_graph_def = saver_module.export_meta_graph(filename)
            meta_graph_def0 = saver0.export_meta_graph()
            meta_graph_def1 = saver1.export_meta_graph(clear_extraneous_savers=True)
            self.assertFalse(meta_graph_def.HasField('saver_def'))
            self.assertTrue(meta_graph_def0.HasField('saver_def'))
            self.assertTrue(meta_graph_def1.HasField('saver_def'))
            collection_def = meta_graph_def.collection_def['savers']
            kind = collection_def.WhichOneof('kind')
            self.assertEqual(kind, 'bytes_list')
            savers = getattr(collection_def, kind)
            self.assertEqual(2, len(savers.value))
            collection_def = meta_graph_def1.collection_def['savers']
            kind = collection_def.WhichOneof('kind')
            self.assertEqual(kind, 'bytes_list')
            savers = getattr(collection_def, kind)
            self.assertEqual(1, len(savers.value))
            self.assertEqual(33, len(meta_graph_def0.graph_def.node))
            self.assertEqual(21, len(meta_graph_def1.graph_def.node))

    def testBinaryAndTextFormat(self):
        if False:
            return 10
        test_dir = self._get_test_dir('binary_and_text')
        filename = os.path.join(test_dir, 'metafile')
        with ops_lib.Graph().as_default(), self.session():
            variable_v1.VariableV1(10.0, name='v0')
            saver_module.export_meta_graph(filename, as_text=False)
        with ops_lib.Graph().as_default(), self.session():
            saver = saver_module.import_meta_graph(filename)
            self.assertIsNotNone(saver)
            saver.export_meta_graph(filename, as_text=True)
        with ops_lib.Graph().as_default(), self.session():
            saver_module.import_meta_graph(filename)
            graph_io.write_graph(saver.as_saver_def(), os.path.dirname(filename), os.path.basename(filename))
        with ops_lib.Graph().as_default(), self.session():
            with self.assertRaisesWithPredicateMatch(IOError, lambda e: 'Cannot parse file'):
                saver_module.import_meta_graph(filename)
            gfile.Remove(filename)
            with self.assertRaisesWithPredicateMatch(IOError, lambda e: 'does not exist'):
                saver_module.import_meta_graph(filename)

    @test_util.run_v1_only('Exporting/importing meta graphs is only supported in V1.')
    def testSliceVariable(self):
        if False:
            i = 10
            return i + 15
        test_dir = self._get_test_dir('slice_saver')
        filename = os.path.join(test_dir, 'metafile')
        with self.cached_session():
            v1 = variable_v1.VariableV1([20.0], name='v1')
            v2 = variable_v1.VariableV1([20.0], name='v2')
            v2._set_save_slice_info(variables.Variable.SaveSliceInfo('v1', [1], [0], [1]))
            slice_saver = saver_module.Saver({'first': v1, 'second': v2})
            self.evaluate(variables.global_variables_initializer())
            meta_graph_def = slice_saver.export_meta_graph(filename)
        with ops_lib.Graph().as_default():
            new_saver = saver_module.import_meta_graph(filename)
            self.assertIsNotNone(new_saver)
            new_meta_graph_def = new_saver.export_meta_graph()
            test_util.assert_meta_graph_protos_equal(self, meta_graph_def, new_meta_graph_def)

    def _testGraphExtensionSave(self, test_dir):
        if False:
            print('Hello World!')
        filename = os.path.join(test_dir, 'metafile')
        saver0_ckpt = os.path.join(test_dir, 'saver0.ckpt')
        images = constant_op.constant(1.2, dtypes.float32, shape=[100, 28])
        with ops_lib.name_scope('hidden1'):
            weights = variable_v1.VariableV1(random_ops.truncated_normal([28, 128], stddev=1.0 / math.sqrt(float(28))), name='weights')
            biases = variable_v1.VariableV1(cond.cond(math_ops.less(random.random(), 0.5), lambda : array_ops.ones([128]), lambda : array_ops.zeros([128])), name='biases')
            hidden1 = nn_ops.relu(math_ops.matmul(images, weights) + biases)
        with ops_lib.name_scope('hidden2'):
            weights = variable_v1.VariableV1(random_ops.truncated_normal([128, 32], stddev=1.0 / math.sqrt(float(128))), name='weights')

            def loop_cond(it, _):
                if False:
                    return 10
                return it < 2

            def loop_body(it, biases):
                if False:
                    i = 10
                    return i + 15
                biases += constant_op.constant(0.1, shape=[32])
                return (it + 1, biases)
            (_, biases) = while_loop.while_loop(loop_cond, loop_body, [constant_op.constant(0), variable_v1.VariableV1(array_ops.zeros([32]))])
            hidden2 = nn_ops.relu(math_ops.matmul(hidden1, weights) + biases)
        with ops_lib.name_scope('softmax_linear'):
            weights = variable_v1.VariableV1(random_ops.truncated_normal([32, 10], stddev=1.0 / math.sqrt(float(32))), name='weights')
            biases = variable_v1.VariableV1(array_ops.zeros([10]), name='biases')
            logits = math_ops.matmul(hidden2, weights) + biases
            ops_lib.add_to_collection('logits', logits)
        init_all_op = variables.global_variables_initializer()
        with self.cached_session() as sess:
            self.evaluate(init_all_op)
            self.evaluate(logits)
            saver0 = saver_module.Saver()
            saver0.save(sess, saver0_ckpt)
            saver0.export_meta_graph(filename)

    def _testGraphExtensionRestore(self, test_dir):
        if False:
            for i in range(10):
                print('nop')
        filename = os.path.join(test_dir, 'metafile')
        train_filename = os.path.join(test_dir, 'train_metafile')
        saver0_ckpt = os.path.join(test_dir, 'saver0.ckpt')
        with self.session(graph=ops_lib.Graph()) as sess:
            new_saver = saver_module.import_meta_graph(filename)
            new_saver.export_meta_graph()
            new_saver.restore(sess, saver0_ckpt)
            labels = constant_op.constant(0, dtypes.int32, shape=[100], name='labels')
            batch_size = array_ops.size(labels)
            labels = array_ops.expand_dims(labels, 1)
            indices = array_ops.expand_dims(math_ops.range(0, batch_size), 1)
            concated = array_ops.concat([indices, labels], 1)
            onehot_labels = sparse_ops.sparse_to_dense(concated, array_ops_stack.stack([batch_size, 10]), 1.0, 0.0)
            logits = ops_lib.get_collection('logits')[0]
            cross_entropy = nn_ops.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits, name='xentropy')
            loss = math_ops.reduce_mean(cross_entropy, name='xentropy_mean')
            summary.scalar('loss', loss)
            optimizer = gradient_descent.GradientDescentOptimizer(0.01)
            train_op = optimizer.minimize(loss)
            ops_lib.add_to_collection('train_op', train_op)
            self.evaluate(train_op)
            saver_module.export_meta_graph(train_filename)

    def _testRestoreFromTrainGraphWithControlContext(self, test_dir):
        if False:
            while True:
                i = 10
        train_filename = os.path.join(test_dir, 'train_metafile')
        saver0_ckpt = os.path.join(test_dir, 'saver0.ckpt')
        with self.session(graph=ops_lib.Graph()) as sess:
            new_saver = saver_module.import_meta_graph(train_filename)
            new_saver.restore(sess, saver0_ckpt)
            train_op = ops_lib.get_collection('train_op')[0]
            self.evaluate(train_op)

    def testGraphExtension(self):
        if False:
            for i in range(10):
                print('nop')
        test_dir = self._get_test_dir('graph_extension')
        with ops_lib.Graph().as_default():
            self._testGraphExtensionSave(test_dir)
            self._testGraphExtensionRestore(test_dir)
            self._testRestoreFromTrainGraphWithControlContext(test_dir)

    def _testGradientSerDes(self, graph_fn):
        if False:
            i = 10
            return i + 15
        'Tests that gradients can be computed after exporting and importing.\n\n    Builds a graph, exports it, and verifies that it can be imported and the\n    gradient can be built and run correctly.\n\n    Args:\n      graph_fn: takes a single float Tensor argument as input, outputs a single\n        Tensor\n    '
        test_dir = self._get_test_dir('nested_control_flow')
        filename = os.path.join(test_dir, 'metafile')
        saver_ckpt = os.path.join(test_dir, 'saver.ckpt')
        with ops_lib.Graph().as_default():
            var = variable_v1.VariableV1(0.0)
            var_name = var.name
            output = graph_fn(var)
            output_name = output.name
            init_op = variables.global_variables_initializer()
            with session.Session() as sess:
                self.evaluate(init_op)
                self.evaluate(output)
                saver = saver_module.Saver()
                saver.save(sess, saver_ckpt)
                saver.export_meta_graph(filename)
            grad = gradients_impl.gradients([output], [var])
            no_constfold_config = config_pb2.ConfigProto()
            no_constfold_config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
            with session.Session(config=no_constfold_config) as sess:
                self.evaluate(init_op)
                expected_grad_value = self.evaluate(grad)
        context._reset_context()
        with ops_lib.Graph().as_default():
            with session.Session() as sess:
                saver = saver_module.import_meta_graph(filename)
                saver.restore(sess, saver_ckpt)
            var = ops_lib.get_default_graph().get_tensor_by_name(var_name)
            output = ops_lib.get_default_graph().get_tensor_by_name(output_name)
            grad = gradients_impl.gradients([output], [var])
            init_op = variables.global_variables_initializer()
            with session.Session(config=no_constfold_config) as sess:
                self.evaluate(init_op)
                actual_grad_value = self.evaluate(grad)
                self.assertEqual(expected_grad_value, actual_grad_value)

    def _testWhileLoopAndGradientSerDes(self, outer_body_fn):
        if False:
            print('Hello World!')
        return self._testGradientSerDes(lambda x: while_loop.while_loop(lambda i, y: i < 5, outer_body_fn, [0, x])[1])

    def testNestedWhileLoopsSerDes(self):
        if False:
            print('Hello World!')

        def body(i, x):
            if False:
                return 10
            (_, r) = while_loop.while_loop(lambda j, y: j < 3, lambda j, y: (j + 1, y + x), [0, 0.0])
            return (i + 1, x + r)
        self._testWhileLoopAndGradientSerDes(body)

    def testNestedControlFlowSerDes(self):
        if False:
            for i in range(10):
                print('nop')

        def body(i, x):
            if False:
                while True:
                    i = 10
            cond_result = cond.cond(i > 0, lambda : while_loop.while_loop(lambda j, y: j < 3, lambda j, y: (j + 1, y + x), [0, 0.0])[1], lambda : x)
            return (i + 1, cond_result)
        self._testWhileLoopAndGradientSerDes(body)

    def testNestedCondsSerDes(self):
        if False:
            for i in range(10):
                print('nop')
        self._testGradientSerDes(lambda x: cond.cond(x > 0, lambda : cond.cond(x > 3, lambda : array_ops.identity(x), lambda : math_ops.multiply(x, 2.0)), lambda : cond.cond(x < -3, lambda : constant_op.constant(1.0), lambda : math_ops.multiply(x, -1.0))))

    @test_util.run_v1_only('This exercises Tensor.op which is meaningless in V2.')
    def testStrippedOpListDef(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            v0 = variable_v1.VariableV1(0.0)
            var = variable_v1.VariableV1(10.0)
            math_ops.add(v0, var)

            @function.Defun(dtypes.float32)
            def minus_one(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x - 1
            minus_one(array_ops.identity(v0))
            save = saver_module.Saver({'v0': v0})
            variables.global_variables_initializer()
            meta_graph_def = save.export_meta_graph()
            ops = [o.name for o in meta_graph_def.meta_info_def.stripped_op_list.op]
            if save._write_version is saver_pb2.SaverDef.V1:
                self.assertEqual(ops, ['AddV2', 'Assign', 'Const', 'Identity', 'NoOp', 'PlaceholderWithDefault', 'RestoreV2', 'SaveSlices', 'Sub', 'VariableV2'])
            else:
                self.assertEqual(ops, ['AddV2', 'Assign', 'Const', 'Identity', 'NoOp', 'PlaceholderWithDefault', 'RestoreV2', 'SaveV2', 'Sub', 'VariableV2'])
            op_list = meta_graph.stripped_op_list_for_graph(meta_graph_def.graph_def)
            self.assertEqual(ops, [o.name for o in op_list.op])
            for o in op_list.op:
                self.assertEqual(o.summary, '')
                self.assertEqual(o.description, '')

    def testStripDefaultValuedAttrs(self):
        if False:
            i = 10
            return i + 15
        'Verifies that default valued attrs are stripped, unless disabled.'
        with ops_lib.Graph().as_default(), self.cached_session():
            real_num = variable_v1.VariableV1(1.0, dtype=dtypes.float32, name='real')
            imag_num = variable_v1.VariableV1(2.0, dtype=dtypes.float32, name='imag')
            math_ops.complex(real_num, imag_num, name='complex')
            save = saver_module.Saver({'real_num': real_num, 'imag_num': imag_num})
            variables.global_variables_initializer()
            meta_graph_def = save.export_meta_graph(strip_default_attrs=True)
            node_def = test_util.get_node_def_from_graph('complex', meta_graph_def.graph_def)
            self.assertNotIn('T', node_def.attr)
            self.assertNotIn('Tout', node_def.attr)
        with ops_lib.Graph().as_default(), self.session():
            real_num = variable_v1.VariableV1(1.0, dtype=dtypes.float32, name='real')
            imag_num = variable_v1.VariableV1(2.0, dtype=dtypes.float32, name='imag')
            math_ops.complex(real_num, imag_num, name='complex')
            save = saver_module.Saver({'real_num': real_num, 'imag_num': imag_num})
            variables.global_variables_initializer()
            meta_graph_def = save.export_meta_graph(strip_default_attrs=False)
            node_def = test_util.get_node_def_from_graph('complex', meta_graph_def.graph_def)
            self.assertIn('T', node_def.attr)
            self.assertIn('Tout', node_def.attr)

    def testImportIntoNamescope(self):
        if False:
            i = 10
            return i + 15
        test_dir = self._get_test_dir('import_into_namescope')
        filename = os.path.join(test_dir, 'ckpt')
        with ops_lib.Graph().as_default():
            image = array_ops.placeholder(dtypes.float32, [None, 784], name='image')
            label = array_ops.placeholder(dtypes.float32, [None, 10], name='label')
            with session.Session() as sess:
                weights = variable_v1.VariableV1(random_ops.random_uniform([784, 10]), name='weights')
                bias = variable_v1.VariableV1(array_ops.zeros([10]), name='bias')
                logit = nn_ops.relu(math_ops.matmul(image, weights) + bias, name='logits')
                nn_ops.softmax(logit, name='prediction')
                cost = nn_ops.softmax_cross_entropy_with_logits(labels=label, logits=logit, name='cost')
                adam.AdamOptimizer().minimize(cost, name='optimize')
                saver = saver_module.Saver()
                self.evaluate(variables.global_variables_initializer())
                saver.save(sess, filename)
        graph = ops_lib.Graph()
        with session.Session(graph=graph) as sess:
            new_saver = saver_module.import_meta_graph(filename + '.meta', graph=graph, import_scope='new_model')
            new_saver.restore(sess, filename)
            sess.run(['new_model/optimize'], {'new_model/image:0': np.random.random([1, 784]), 'new_model/label:0': np.random.randint(10, size=[1, 10])})

    def testImportIntoNamescopeWithoutVariables(self):
        if False:
            while True:
                i = 10
        test_dir = self._get_test_dir('no_vars_graph')
        filename = os.path.join(test_dir, 'ckpt')
        graph_1 = ops_lib.Graph()
        with session.Session(graph=graph_1) as sess:
            constant_op.constant([1, 2, 3], name='x')
            constant_op.constant([1, 2, 3], name='y')
            saver = saver_module.Saver(allow_empty=True)
            saver.save(sess, filename)
        graph_2 = ops_lib.Graph()
        with session.Session(graph=graph_2) as sess:
            new_saver_1 = saver_module.import_meta_graph(filename + '.meta', graph=graph_2, import_scope='subgraph_1')
            self.assertIsNone(new_saver_1)
            variable_v1.VariableV1(array_ops.zeros([10]), name='my_scope/my_var')
            self.evaluate(variables.global_variables_initializer())
            new_saver_2 = saver_module.import_meta_graph(filename + '.meta', graph=graph_2, import_scope='subgraph_2')
            self.assertIsNone(new_saver_2)
            new_saver_3 = saver_module.import_meta_graph(filename + '.meta', graph=graph_2, import_scope='my_scope')
            self.assertIsInstance(new_saver_3, saver_module.Saver)

    def testImportIntoImplicitNamescope(self):
        if False:
            while True:
                i = 10
        test_dir = self._get_test_dir('import_into_namescope')
        filename = os.path.join(test_dir, 'ckpt')
        with ops_lib.Graph().as_default():
            image = array_ops.placeholder(dtypes.float32, [None, 784], name='image')
            label = array_ops.placeholder(dtypes.float32, [None, 10], name='label')
            with session.Session() as sess:
                weights = variable_v1.VariableV1(random_ops.random_uniform([784, 10]), name='weights')
                bias = variable_v1.VariableV1(array_ops.zeros([10]), name='bias')
                logit = nn_ops.relu(math_ops.matmul(image, weights) + bias, name='logits')
                nn_ops.softmax(logit, name='prediction')
                cost = nn_ops.softmax_cross_entropy_with_logits(labels=label, logits=logit, name='cost')
                adam.AdamOptimizer().minimize(cost, name='optimize')
                saver = saver_module.Saver()
                self.evaluate(variables.global_variables_initializer())
                saver.save(sess, filename)
        graph = ops_lib.Graph()
        with session.Session(graph=graph) as sess:
            with ops_lib.name_scope('new_model'):
                new_saver = saver_module.import_meta_graph(filename + '.meta', graph=graph)
            new_saver.restore(sess, filename)
            sess.run(['new_model/optimize'], {'new_model/image:0': np.random.random([1, 784]), 'new_model/label:0': np.random.randint(10, size=[1, 10])})

    def testClearDevicesOnImport(self):
        if False:
            print('Hello World!')
        with ops_lib.Graph().as_default():
            with ops_lib.device('/job:ps/replica:0/task:0/device:GPU:0'):
                image = array_ops.placeholder(dtypes.float32, [None, 784], name='image')
                label = array_ops.placeholder(dtypes.float32, [None, 10], name='label')
                weights = variable_v1.VariableV1(random_ops.random_uniform([784, 10]), name='weights')
                bias = variable_v1.VariableV1(array_ops.zeros([10]), name='bias')
                logit = nn_ops.relu(math_ops.matmul(image, weights) + bias)
                nn_ops.softmax(logit, name='prediction')
                cost = nn_ops.softmax_cross_entropy_with_logits(labels=label, logits=logit)
                adam.AdamOptimizer().minimize(cost, name='optimize')
            meta_graph_def = saver_module.export_meta_graph()
        with session.Session(graph=ops_lib.Graph()) as sess:
            saver_module.import_meta_graph(meta_graph_def, clear_devices=False, import_scope='new_model')
            with self.assertRaises(errors_impl.InvalidArgumentError):
                self.evaluate(variables.global_variables_initializer())
        with session.Session(graph=ops_lib.Graph()) as sess:
            saver_module.import_meta_graph(meta_graph_def, clear_devices=True, import_scope='new_model')
            self.evaluate(variables.global_variables_initializer())
            sess.run(['new_model/optimize'], {'new_model/image:0': np.random.random([1, 784]), 'new_model/label:0': np.random.randint(10, size=[1, 10])})

    def testClearDevicesOnExport(self):
        if False:
            for i in range(10):
                print('nop')
        with ops_lib.Graph().as_default():
            with ops_lib.device('/job:ps/replica:0/task:0/device:GPU:0'):
                image = array_ops.placeholder(dtypes.float32, [None, 784], name='image')
                label = array_ops.placeholder(dtypes.float32, [None, 10], name='label')
                weights = variable_v1.VariableV1(random_ops.random_uniform([784, 10]), name='weights')
                bias = variable_v1.VariableV1(array_ops.zeros([10]), name='bias')
                logit = nn_ops.relu(math_ops.matmul(image, weights) + bias)
                nn_ops.softmax(logit, name='prediction')
                cost = nn_ops.softmax_cross_entropy_with_logits(labels=label, logits=logit)
                adam.AdamOptimizer().minimize(cost, name='optimize')
            meta_graph_def = saver_module.export_meta_graph(clear_devices=True)
            graph_io.write_graph(meta_graph_def, self.get_temp_dir(), 'meta_graph.pbtxt')
        with session.Session(graph=ops_lib.Graph()) as sess:
            saver_module.import_meta_graph(meta_graph_def, import_scope='new_model')
            self.evaluate(variables.global_variables_initializer())
            sess.run(['new_model/optimize'], {'new_model/image:0': np.random.random([1, 784]), 'new_model/label:0': np.random.randint(10, size=[1, 10])})

    def testPreserveDatasetAndFunctions(self):
        if False:
            while True:
                i = 10
        with ops_lib.Graph().as_default() as g:
            dataset = dataset_ops.Dataset.range(10).map(lambda x: x * x)
            iterator = dataset_ops.make_one_shot_iterator(dataset)
            next_element = iterator.get_next()
            _ = array_ops.identity(next_element, name='output')
            meta_graph_def_simple = saver_module.export_meta_graph()
            meta_graph_def_devices_cleared = saver_module.export_meta_graph(clear_devices=True)
            meta_graph_def_from_graph_def = saver_module.export_meta_graph(clear_devices=True, graph_def=g.as_graph_def())
        for meta_graph_def in [meta_graph_def_simple, meta_graph_def_devices_cleared, meta_graph_def_from_graph_def]:
            with session.Session(graph=ops_lib.Graph()) as sess:
                saver_module.import_meta_graph(meta_graph_def, import_scope='new_model')
                self.evaluate(variables.global_variables_initializer())
                for i in range(10):
                    self.assertEqual(i * i, sess.run('new_model/output:0'))
                with self.assertRaises(errors.OutOfRangeError):
                    sess.run('new_model/output:0')

class CheckpointReaderTest(test.TestCase):
    _WRITE_VERSION = saver_pb2.SaverDef.V1

    def testDebugString(self):
        if False:
            print('Hello World!')
        v0 = variable_v1.VariableV1([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32, name='v0')
        v1 = variable_v1.VariableV1([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=dtypes.float32, name='v1')
        init_all_op = variables.global_variables_initializer()
        save = saver_module.Saver({'v0': v0, 'v1': v1}, write_version=self._WRITE_VERSION)
        save_path = os.path.join(self.get_temp_dir(), 'ckpt_for_debug_string' + str(self._WRITE_VERSION))
        with self.cached_session() as sess:
            self.evaluate(init_all_op)
            save.save(sess, save_path)
            reader = py_checkpoint_reader.NewCheckpointReader(save_path)
            self.assertTrue(reader.has_tensor('v0'))
            self.assertTrue(reader.has_tensor('v1'))
            debug_string = reader.debug_string()
            self.assertIn(compat.as_bytes('v0 (DT_FLOAT) [2,3]'), debug_string)
            self.assertIn(compat.as_bytes('v1 (DT_FLOAT) [3,2,1]'), debug_string)
            var_map = reader.get_variable_to_shape_map()
            self.assertEqual([2, 3], var_map['v0'])
            self.assertEqual([3, 2, 1], var_map['v1'])
            v0_tensor = reader.get_tensor('v0')
            v1_tensor = reader.get_tensor('v1')
            self.assertAllEqual(v0, v0_tensor)
            self.assertAllEqual(v1, v1_tensor)
            with self.assertRaisesRegex(errors.NotFoundError, 'v3 not found in checkpoint'):
                reader.get_tensor('v3')

    def testNonexistentPath(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(errors.NotFoundError, 'Unsuccessful TensorSliceReader'):
            py_checkpoint_reader.NewCheckpointReader('non-existent')

class CheckpointReaderForV2Test(CheckpointReaderTest):
    _WRITE_VERSION = saver_pb2.SaverDef.V2

class WriteGraphTest(test.TestCase):

    def _get_test_dir(self, dirname):
        if False:
            i = 10
            return i + 15
        test_dir = os.path.join(self.get_temp_dir(), dirname)
        gfile.MakeDirs(test_dir)
        return test_dir

    def testWriteGraph(self):
        if False:
            print('Hello World!')
        test_dir = self._get_test_dir('write_graph_dir')
        variable_v1.VariableV1([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32, name='v0')
        path = graph_io.write_graph(ops_lib.get_default_graph(), os.path.join(test_dir, 'l1'), 'graph.pbtxt')
        truth = os.path.join(test_dir, 'l1', 'graph.pbtxt')
        self.assertEqual(path, truth)
        self.assertTrue(os.path.exists(path))

    def testRecursiveCreate(self):
        if False:
            while True:
                i = 10
        test_dir = self._get_test_dir('deep_dir')
        variable_v1.VariableV1([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32, name='v0')
        path = graph_io.write_graph(ops_lib.get_default_graph().as_graph_def(), os.path.join(test_dir, 'l1', 'l2', 'l3'), 'graph.pbtxt')
        truth = os.path.join(test_dir, 'l1', 'l2', 'l3', 'graph.pbtxt')
        self.assertEqual(path, truth)
        self.assertTrue(os.path.exists(path))

class ScopedGraphTest(test.TestCase):

    def _get_test_dir(self, dirname):
        if False:
            print('Hello World!')
        test_dir = os.path.join(self.get_temp_dir(), dirname)
        gfile.MakeDirs(test_dir)
        return test_dir

    def _testScopedSave(self, test_dir, exported_filename, ckpt_filename):
        if False:
            return 10
        graph = ops_lib.Graph()
        with graph.as_default():
            images = constant_op.constant(1.2, dtypes.float32, shape=[100, 28], name='images')
            with ops_lib.name_scope('hidden1'):
                weights1 = variable_v1.VariableV1(random_ops.truncated_normal([28, 128], stddev=1.0 / math.sqrt(float(28))), name='weights')
                biases1 = variable_v1.VariableV1(cond.cond(math_ops.less(random.random(), 0.5), lambda : array_ops.ones([128]), lambda : array_ops.zeros([128])), name='biases')
                hidden1 = nn_ops.relu(math_ops.matmul(images, weights1) + biases1)
            with ops_lib.name_scope('hidden2'):
                weights2 = variable_v1.VariableV1(random_ops.truncated_normal([128, 32], stddev=1.0 / math.sqrt(float(128))), name='weights')

                def loop_cond(it, _):
                    if False:
                        for i in range(10):
                            print('nop')
                    return it < 2

                def loop_body(it, biases2):
                    if False:
                        print('Hello World!')
                    biases2 += constant_op.constant(0.1, shape=[32])
                    return (it + 1, biases2)
                (_, biases2) = while_loop.while_loop(loop_cond, loop_body, [constant_op.constant(0), variable_v1.VariableV1(array_ops.zeros([32]))])
                hidden2 = nn_ops.relu(math_ops.matmul(hidden1, weights2) + biases2)
            with ops_lib.name_scope('softmax_linear'):
                weights3 = variable_v1.VariableV1(random_ops.truncated_normal([32, 10], stddev=1.0 / math.sqrt(float(32))), name='weights')
                biases3 = variable_v1.VariableV1(array_ops.zeros([10]), name='biases')
                logits = math_ops.matmul(hidden2, weights3) + biases3
                ops_lib.add_to_collection('logits', logits)
                queue_runner = queue_runner_pb2.QueueRunnerDef(queue_name='test_queue')
                ops_lib.add_to_collection('user_defined_string_collection', str(queue_runner))
                ops_lib.add_to_collection('user_defined_bytes_collection', queue_runner.SerializeToString())
                any_buf = Any()
                any_buf.Pack(queue_runner)
                ops_lib.add_to_collection('user_defined_any_collection', any_buf)
            (_, var_list) = meta_graph.export_scoped_meta_graph(filename=os.path.join(test_dir, exported_filename), graph=ops_lib.get_default_graph(), export_scope='hidden1')
            self.assertEqual(['biases:0', 'weights:0'], sorted(var_list.keys()))
        with graph.as_default(), self.session() as sess:
            self.evaluate(variables.global_variables_initializer())
            saver = saver_module.Saver(var_list=var_list, max_to_keep=1)
            saver.save(sess, os.path.join(test_dir, ckpt_filename), write_state=False)

    def _testScopedRestore(self, test_dir, exported_filename, new_exported_filename, ckpt_filename):
        if False:
            print('Hello World!')
        graph = ops_lib.Graph()
        with graph.as_default():
            new_image = constant_op.constant(1.2, dtypes.float32, shape=[100, 28], name='images')
            var_list = meta_graph.import_scoped_meta_graph(os.path.join(test_dir, exported_filename), graph=graph, input_map={'$unbound_inputs_images': new_image}, import_scope='new_hidden1')
            self.assertEqual(['biases:0', 'weights:0'], sorted(var_list.keys()))
            hidden1 = graph.as_graph_element('new_hidden1/Relu:0')
            weights1 = graph.as_graph_element('new_hidden1/weights:0')
            biases1 = graph.as_graph_element('new_hidden1/biases:0')
        with graph.as_default():
            with ops_lib.name_scope('hidden2'):
                weights = variable_v1.VariableV1(random_ops.truncated_normal([128, 32], stddev=1.0 / math.sqrt(float(128))), name='weights')

                def loop_cond(it, _):
                    if False:
                        for i in range(10):
                            print('nop')
                    return it < 2

                def loop_body(it, biases):
                    if False:
                        return 10
                    biases += constant_op.constant(0.1, shape=[32])
                    return (it + 1, biases)
                (_, biases) = while_loop.while_loop(loop_cond, loop_body, [constant_op.constant(0), variable_v1.VariableV1(array_ops.zeros([32]))])
                hidden2 = nn_ops.relu(math_ops.matmul(hidden1, weights) + biases)
            with ops_lib.name_scope('softmax_linear'):
                weights = variable_v1.VariableV1(random_ops.truncated_normal([32, 10], stddev=1.0 / math.sqrt(float(32))), name='weights')
                biases = variable_v1.VariableV1(array_ops.zeros([10]), name='biases')
                logits = math_ops.matmul(hidden2, weights) + biases
                ops_lib.add_to_collection('logits', logits)
            rest_variables = list(set(variables.global_variables()) - set(var_list.keys()))
            init_rest_op = variables.variables_initializer(rest_variables)
        with graph.as_default(), self.session() as sess:
            saver = saver_module.Saver(var_list=var_list, max_to_keep=1)
            saver.restore(sess, os.path.join(test_dir, ckpt_filename))
            self.evaluate([weights1, biases1])
            self.evaluate(init_rest_op)
            self.evaluate(logits)

    def testScopedSaveAndRestore(self):
        if False:
            for i in range(10):
                print('nop')
        test_dir = self._get_test_dir('scoped_export_import')
        ckpt_filename = 'ckpt'
        self._testScopedSave(test_dir, 'exported_hidden1.pbtxt', ckpt_filename)
        self._testScopedRestore(test_dir, 'exported_hidden1.pbtxt', 'exported_new_hidden1.pbtxt', ckpt_filename)

    def testCopyScopedGraph(self):
        if False:
            return 10
        test_dir = self._get_test_dir('scoped_copy')
        saver0_ckpt = os.path.join(test_dir, 'saver0.ckpt')
        graph1 = ops_lib.Graph()
        with graph1.as_default():
            with ops_lib.name_scope('hidden1'):
                images = constant_op.constant(1.0, dtypes.float32, shape=[3, 2], name='images')
                weights1 = variable_v1.VariableV1([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='weights')
                biases1 = variable_v1.VariableV1([0.1] * 3, name='biases')
                nn_ops.relu(math_ops.matmul(images, weights1) + biases1, name='relu')
        with graph1.as_default(), self.session(graph=graph1) as sess:
            self.evaluate(variables.global_variables_initializer())
            (_, var_list_1) = meta_graph.export_scoped_meta_graph(export_scope='hidden1')
            saver = saver_module.Saver(var_list=var_list_1, max_to_keep=1)
            saver.save(sess, saver0_ckpt, write_state=False)
        expected = np.reshape([[5.0999999, 7.0999999, 9.10000038] * 3], (3, 3))
        with graph1.as_default():
            with self.assertRaisesWithPredicateMatch(ValueError, lambda e: 'need to be different' in str(e)):
                meta_graph.copy_scoped_meta_graph(from_scope='hidden1', to_scope='hidden1')
        with graph1.as_default():
            var_list_2 = meta_graph.copy_scoped_meta_graph(from_scope='hidden1', to_scope='hidden2')
        with graph1.as_default(), self.session(graph=graph1) as sess:
            saver1 = saver_module.Saver(var_list=var_list_1, max_to_keep=1)
            saver1.restore(sess, saver0_ckpt)
            saver2 = saver_module.Saver(var_list=var_list_2, max_to_keep=1)
            saver2.restore(sess, saver0_ckpt)
            self.assertAllClose(expected, sess.run('hidden1/relu:0'))
            self.assertAllClose(expected, sess.run('hidden2/relu:0'))
        graph2 = ops_lib.Graph()
        with graph2.as_default():
            new_var_list_1 = meta_graph.copy_scoped_meta_graph(from_scope='hidden1', to_scope='new_hidden1', from_graph=graph1, to_graph=graph2)
            with self.session() as sess:
                saver3 = saver_module.Saver(var_list=new_var_list_1, max_to_keep=1)
                saver3.restore(sess, saver0_ckpt)
                self.assertAllClose(expected, sess.run('new_hidden1/relu:0'))

    def testExportGraphDefWithScope(self):
        if False:
            i = 10
            return i + 15
        test_dir = self._get_test_dir('export_graph_def')
        saver0_ckpt = os.path.join(test_dir, 'saver0.ckpt')
        graph1 = ops_lib.Graph()
        with graph1.as_default():
            with ops_lib.name_scope('hidden1'):
                images = constant_op.constant(1.0, dtypes.float32, shape=[3, 2], name='images')
                weights1 = variable_v1.VariableV1([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='weights')
                biases1 = variable_v1.VariableV1([0.1] * 3, name='biases')
                nn_ops.relu(math_ops.matmul(images, weights1) + biases1, name='relu')
            with self.session(graph=graph1) as sess:
                self.evaluate(variables.global_variables_initializer())
                (_, var_list_1) = meta_graph.export_scoped_meta_graph(graph_def=graph1.as_graph_def(), export_scope='hidden1')
                saver = saver_module.Saver(var_list=var_list_1, max_to_keep=1)
                saver.save(sess, saver0_ckpt, write_state=False)
        expected = np.reshape([[5.0999999, 7.0999999, 9.10000038] * 3], (3, 3))
        graph2 = ops_lib.Graph()
        with graph2.as_default():
            new_var_list_1 = meta_graph.copy_scoped_meta_graph(from_scope='hidden1', to_scope='new_hidden1', from_graph=graph1, to_graph=graph2)
            with self.session(graph=graph2) as sess:
                saver3 = saver_module.Saver(var_list=new_var_list_1, max_to_keep=1)
                saver3.restore(sess, saver0_ckpt)
                self.assertAllClose(expected, sess.run('new_hidden1/relu:0'))

    def testSerializeSaverWithScope(self):
        if False:
            for i in range(10):
                print('nop')
        test_dir = self._get_test_dir('export_graph_def')
        saver1_ckpt = os.path.join(test_dir, 'saver1.ckpt')
        saver2_ckpt = os.path.join(test_dir, 'saver2.ckpt')
        graph = ops_lib.Graph()
        with graph.as_default():
            with ops_lib.name_scope('hidden1'):
                variable1 = variable_v1.VariableV1([1.0], name='variable1')
                saver1 = saver_module.Saver(var_list=[variable1])
                graph.add_to_collection(ops_lib.GraphKeys.SAVERS, saver1)
            with ops_lib.name_scope('hidden2'):
                variable2 = variable_v1.VariableV1([2.0], name='variable2')
            saver2 = saver_module.Saver(var_list=[variable2], name='hidden2/')
            graph.add_to_collection(ops_lib.GraphKeys.SAVERS, saver2)
            with self.session(graph=graph) as sess:
                self.evaluate(variables.global_variables_initializer())
                saver1.save(sess, saver1_ckpt, write_state=False)
                saver2.save(sess, saver2_ckpt, write_state=False)
        graph1 = ops_lib.Graph()
        with graph1.as_default():
            var_dict1 = meta_graph.copy_scoped_meta_graph(from_scope='hidden1', to_scope='new_hidden1', from_graph=graph, to_graph=graph1)
            self.assertEqual(1, len(var_dict1))
            saver_list1 = graph1.get_collection(ops_lib.GraphKeys.SAVERS)
            self.assertEqual(1, len(saver_list1))
            with self.session(graph=graph1) as sess:
                saver_list1[0].restore(sess, saver1_ckpt)
                self.assertEqual(1.0, self.evaluate(var_dict1['variable1:0']))
        graph2 = ops_lib.Graph()
        with graph2.as_default():
            var_dict2 = meta_graph.copy_scoped_meta_graph(from_scope='hidden2', to_scope='new_hidden2', from_graph=graph, to_graph=graph2)
            self.assertEqual(1, len(var_dict2))
            saver_list2 = graph2.get_collection(ops_lib.GraphKeys.SAVERS)
            self.assertEqual(1, len(saver_list2))
            with self.session(graph=graph2) as sess:
                saver_list2[0].restore(sess, saver2_ckpt)
                self.assertEqual(2.0, self.evaluate(var_dict2['variable2:0']))

class _OwnsAVariableSimple(trackable_base.Trackable):
    """A Trackable object which can be saved using a tf.train.Saver."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.non_dep_variable = variable_scope.get_variable(name='non_dep_variable', initializer=6.0, use_resource=True)

    def _gather_saveables_for_checkpoint(self):
        if False:
            while True:
                i = 10
        return {trackable_base.VARIABLE_VALUE_KEY: self.non_dep_variable}

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self.non_dep_variable.name

class _MirroringSaveable(saver_module.BaseSaverBuilder.ResourceVariableSaveable):

    def __init__(self, primary_variable, mirrored_variable, name):
        if False:
            while True:
                i = 10
        self._primary_variable = primary_variable
        self._mirrored_variable = mirrored_variable
        super(_MirroringSaveable, self).__init__(self._primary_variable, '', name)

    def restore(self, restored_tensors, restored_shapes):
        if False:
            print('Hello World!')
        'Restore the same value into both variables.'
        (tensor,) = restored_tensors
        return control_flow_ops.group(self._primary_variable.assign(tensor), self._mirrored_variable.assign(tensor))

class _OwnsMirroredVariables(trackable_base.Trackable):
    """A Trackable object which returns a more complex SaveableObject."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.non_dep_variable = variable_scope.get_variable(name='non_dep_variable', initializer=6.0, use_resource=True)
        self.mirrored = variable_scope.get_variable(name='mirrored', initializer=15.0, use_resource=True)

    def _gather_saveables_for_checkpoint(self):
        if False:
            for i in range(10):
                print('nop')

        def _saveable_factory(name=self.non_dep_variable.name):
            if False:
                i = 10
                return i + 15
            return _MirroringSaveable(primary_variable=self.non_dep_variable, mirrored_variable=self.mirrored, name=name)
        return {trackable_base.VARIABLE_VALUE_KEY: _saveable_factory}

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self.non_dep_variable.name

class TrackableCompatibilityTests(test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def testNotSaveableButIsTrackable(self):
        if False:
            print('Hello World!')
        v = _OwnsAVariableSimple()
        test_dir = self.get_temp_dir()
        prefix = os.path.join(test_dir, 'ckpt')
        for saver in (saver_module.Saver(var_list=[v]), saver_module.Saver(var_list={'v': v})):
            with self.cached_session() as sess:
                self.evaluate(v.non_dep_variable.assign(42.0))
                save_path = saver.save(sess, prefix)
                self.evaluate(v.non_dep_variable.assign(43.0))
                saver.restore(sess, save_path)
                self.assertEqual(42.0, self.evaluate(v.non_dep_variable))

    @test_util.run_in_graph_and_eager_modes
    def testMoreComplexSaveableReturned(self):
        if False:
            print('Hello World!')
        v = _OwnsMirroredVariables()
        test_dir = self.get_temp_dir()
        prefix = os.path.join(test_dir, 'ckpt')
        self.evaluate(v.non_dep_variable.assign(42.0))
        for saver in (saver_module.Saver(var_list=[v]), saver_module.Saver(var_list={'v': v})):
            with self.cached_session() as sess:
                save_path = saver.save(sess, prefix)
                self.evaluate(v.non_dep_variable.assign(43.0))
                self.evaluate(v.mirrored.assign(44.0))
                saver.restore(sess, save_path)
                self.assertEqual(42.0, self.evaluate(v.non_dep_variable))
                self.assertEqual(42.0, self.evaluate(v.mirrored))

    def testSingleTensorEvaluation(self):
        if False:
            i = 10
            return i + 15

        class _CountingSaveable(saver_module.BaseSaverBuilder.SaveableObject):

            def __init__(self, name):
                if False:
                    while True:
                        i = 10
                self.eval_count = 0

                def _tensor():
                    if False:
                        while True:
                            i = 10
                    self.eval_count += 1
                    return constant_op.constant([1.0])
                dummy_op = constant_op.constant([2.0])
                super(_CountingSaveable, self).__init__(dummy_op, [saver_module.BaseSaverBuilder.SaveSpec(_tensor, '', name, dtype=dummy_op.dtype, device=dummy_op.device)], name)

            def restore(self, restored_tensors, restored_shapes):
                if False:
                    return 10
                'Restore the same value into both variables.'
                pass
        with context.eager_mode():
            v = _CountingSaveable('foo')
            saver = saver_module.Saver(var_list=[v])
            test_dir = self.get_temp_dir()
            prefix = os.path.join(test_dir, 'ckpt')
            with self.cached_session() as sess:
                save_path = saver.save(sess, prefix)
                self.assertEqual(1, v.eval_count)
                saver.restore(sess, save_path)
                self.assertEqual(1, v.eval_count)

    def testVariableNotFoundErrorRaised(self):
        if False:
            print('Hello World!')
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
        a = resource_variable_ops.ResourceVariable(1.0, name='a')
        b = resource_variable_ops.ResourceVariable(1.0, name='b')
        a_saver = saver_module.Saver([a])
        b_saver = saver_module.Saver([b])
        with self.cached_session() as sess:
            self.evaluate(a.initializer)
            save_path = a_saver.save(sess=sess, save_path=checkpoint_prefix)
            with self.assertRaisesRegex(errors.NotFoundError, 'Key b not found in checkpoint'):
                b_saver.restore(sess=sess, save_path=save_path)
            with self.assertRaises(errors.NotFoundError) as cs:
                b_saver.restore(sess=sess, save_path=save_path)
            self.assertNotIn('NewCheckpointReader', cs.exception.message)

    @test_util.run_v1_only('train.Saver is V1 only API.')
    def testGraphChangedForRestoreErrorRaised(self):
        if False:
            for i in range(10):
                print('nop')
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
        with ops_lib.Graph().as_default() as g:
            a = variable_v1.VariableV1(1.0, name='a')
            a_saver = saver_module.Saver([a])
            with self.session(graph=g) as sess:
                self.evaluate(a.initializer)
                save_path = a_saver.save(sess=sess, save_path=checkpoint_prefix)
        with ops_lib.Graph().as_default() as g:
            a = variable_v1.VariableV1([1.0], name='a')
            a_saver = saver_module.Saver([a])
            with self.session(graph=g) as sess:
                with self.assertRaisesRegex(errors.InvalidArgumentError, 'a mismatch between the current graph and the graph'):
                    a_saver.restore(sess=sess, save_path=save_path)
if __name__ == '__main__':
    test.main()