"""Tests for sync_replicas_optimizer.py."""
import time
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.test_util import create_local_cluster
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import training

def get_workers(num_workers, replicas_to_aggregate, workers):
    if False:
        while True:
            i = 10
    sessions = []
    graphs = []
    train_ops = []
    for worker_id in range(num_workers):
        graph = ops.Graph()
        is_chief = worker_id == 0
        with graph.as_default():
            with ops.device('/job:ps/task:0'):
                global_step = variable_v1.VariableV1(0, name='global_step', trainable=False)
                var_0 = variable_v1.VariableV1(0.0, name='v0')
            with ops.device('/job:ps/task:1'):
                var_1 = variable_v1.VariableV1(1.0, name='v1')
                var_sparse = variable_v1.VariableV1([[3.0], [4.0]], name='v_sparse')
            with ops.device('/job:worker/task:' + str(worker_id)):
                grads_0 = constant_op.constant(0.1 + worker_id * 0.2)
                grads_1 = constant_op.constant(0.9 + worker_id * 0.2)
                grads_sparse = indexed_slices.IndexedSlices(constant_op.constant([0.1 + worker_id * 0.2], shape=[1, 1]), constant_op.constant([1]), constant_op.constant([2, 1]))
                sgd_opt = gradient_descent.GradientDescentOptimizer(2.0)
                sync_rep_opt = training.SyncReplicasOptimizer(sgd_opt, replicas_to_aggregate=replicas_to_aggregate, total_num_replicas=num_workers)
                train_op = [sync_rep_opt.apply_gradients(zip([grads_0, grads_1, grads_sparse], [var_0, var_1, var_sparse]), global_step=global_step)]
                sync_replicas_hook = sync_rep_opt.make_session_run_hook(is_chief, num_tokens=num_workers)
            session = training.MonitoredTrainingSession(master=workers[worker_id].target, is_chief=is_chief, hooks=[sync_replicas_hook])
        sessions.append(session)
        graphs.append(graph)
        train_ops.append(train_op)
    return (sessions, graphs, train_ops)

class SyncReplicasOptimizerTest(test.TestCase):

    def _run(self, train_op, sess):
        if False:
            while True:
                i = 10
        sess.run(train_op)

    @test_util.run_v1_only('This exercises tensor lookup via names which is not supported in V2.')
    def test2Workers(self):
        if False:
            i = 10
            return i + 15
        num_workers = 2
        replicas_to_aggregate = 2
        num_ps = 2
        (workers, _) = create_local_cluster(num_workers=num_workers, num_ps=num_ps)
        (sessions, graphs, train_ops) = get_workers(num_workers, replicas_to_aggregate, workers)
        var_0_g_0 = graphs[0].get_tensor_by_name('v0:0')
        var_1_g_0 = graphs[0].get_tensor_by_name('v1:0')
        local_step_0 = graphs[0].get_tensor_by_name('sync_rep_local_step:0')
        self.assertAllEqual(0.0, sessions[0].run(var_0_g_0))
        self.assertAllEqual(1.0, sessions[0].run(var_1_g_0))
        self.assertAllEqual(0, sessions[0].run(local_step_0))
        var_0_g_1 = graphs[1].get_tensor_by_name('v0:0')
        var_1_g_1 = graphs[1].get_tensor_by_name('v1:0')
        var_sparse_g_1 = graphs[1].get_tensor_by_name('v_sparse:0')
        local_step_1 = graphs[1].get_tensor_by_name('sync_rep_local_step:0')
        global_step = graphs[1].get_tensor_by_name('global_step:0')
        self.assertAllEqual(0, sessions[1].run(global_step))
        self.assertAllEqual(0, sessions[1].run(local_step_1))
        self.assertAllClose([[3.0], [4.0]], sessions[1].run(var_sparse_g_1))
        sessions[0].run(train_ops[0])
        sessions[1].run(train_ops[1])
        while sessions[1].run(global_step) != 1:
            time.sleep(0.01)
        self.assertAllClose(0 - (0.1 + 0.3) / 2 * 2.0, sessions[1].run(var_0_g_1))
        self.assertAllClose(1 - (0.9 + 1.1) / 2 * 2.0, sessions[1].run(var_1_g_1))
        self.assertAllClose([[3.0], [4.0 - (0.1 + 0.3) / 2 * 2.0]], sessions[1].run(var_sparse_g_1))
        self.assertAllEqual(0, sessions[0].run(local_step_0))
        self.assertAllEqual(0, sessions[1].run(local_step_1))
        sessions[0].run(train_ops[0])
        sessions[1].run(train_ops[1])
        self.assertAllEqual(1, sessions[1].run(global_step))
        self.assertAllEqual(1, sessions[0].run(local_step_0))
        self.assertAllEqual(1, sessions[1].run(local_step_1))
        self.assertAllClose(0 - (0.1 + 0.3) / 2 * 2.0, sessions[1].run(var_0_g_1))
        self.assertAllClose(1 - (0.9 + 1.1) / 2 * 2.0, sessions[1].run(var_1_g_1))
        threads = []
        threads.append(self.checkedThread(target=self._run, args=(train_ops[0], sessions[0])))
        threads.append(self.checkedThread(target=self._run, args=(train_ops[1], sessions[1])))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertAllEqual(2, sessions[1].run(global_step))
        self.assertAllClose(0 - 2 * (0.1 + 0.3) / 2 * 2.0, sessions[1].run(var_0_g_1))
        self.assertAllClose(1 - 2 * (0.9 + 1.1) / 2 * 2.0, sessions[1].run(var_1_g_1))

    @test_util.run_v1_only('This exercises tensor lookup via names which is not supported in V2.')
    def test3Workers1Backup(self):
        if False:
            i = 10
            return i + 15
        num_workers = 3
        replicas_to_aggregate = 2
        num_ps = 2
        (workers, _) = create_local_cluster(num_workers=num_workers, num_ps=num_ps)
        (sessions, graphs, train_ops) = get_workers(num_workers, replicas_to_aggregate, workers)
        var_0_g_1 = graphs[1].get_tensor_by_name('v0:0')
        var_1_g_1 = graphs[1].get_tensor_by_name('v1:0')
        local_step_1 = graphs[1].get_tensor_by_name('sync_rep_local_step:0')
        global_step = graphs[1].get_tensor_by_name('global_step:0')
        self.assertAllEqual(0, sessions[1].run(global_step))
        self.assertAllEqual(0, sessions[1].run(local_step_1))
        sessions[0].run(train_ops[0])
        sessions[2].run(train_ops[2])
        while sessions[1].run(global_step) != 1:
            time.sleep(0.01)
        self.assertAllEqual(1, sessions[1].run(global_step))
        self.assertAllClose(0 - (0.1 + 0.5) / 2 * 2.0, sessions[1].run(var_0_g_1))
        self.assertAllClose(1 - (0.9 + 1.3) / 2 * 2.0, sessions[1].run(var_1_g_1))
        sessions[1].run(train_ops[1])
        sessions[0].run(train_ops[0])
        sessions[1].run(train_ops[1])
        sessions[2].run(train_ops[2])
        self.assertAllEqual(1, sessions[1].run(global_step))
        self.assertAllEqual(1, sessions[1].run(local_step_1))
        thread_0 = self.checkedThread(target=self._run, args=(train_ops[0], sessions[0]))
        thread_1 = self.checkedThread(target=self._run, args=(train_ops[1], sessions[1]))
        thread_0.start()
        self.assertAllEqual(1, sessions[1].run(global_step))
        thread_1.start()
        thread_1.join()
        thread_0.join()
        self.assertAllEqual(2, sessions[1].run(global_step))
        self.assertAllClose(-0.6 - (0.1 + 0.3) / 2 * 2.0, sessions[1].run(var_0_g_1))
        self.assertAllClose(-1.2 - (0.9 + 1.1) / 2 * 2.0, sessions[1].run(var_1_g_1))

class SyncReplicasOptimizerHookTest(test.TestCase):

    def testErrorIfUsedBeforeMinimizeCalled(self):
        if False:
            return 10
        opt = training.SyncReplicasOptimizer(opt=gradient_descent.GradientDescentOptimizer(1.0), replicas_to_aggregate=1, total_num_replicas=1)
        hook = opt.make_session_run_hook(True)
        with self.assertRaisesRegex(ValueError, 'apply_gradient should be called'):
            hook.begin()

    @test_util.run_v1_only('train.SyncReplicasOptimizer and train.GradientDescentOptimizer are V1 only APIs.')
    def testCanCreatedBeforeMinimizeCalled(self):
        if False:
            while True:
                i = 10
        'This behavior is required to be integrated with Estimators.'
        opt = training.SyncReplicasOptimizer(opt=gradient_descent.GradientDescentOptimizer(1.0), replicas_to_aggregate=1, total_num_replicas=1)
        hook = opt.make_session_run_hook(True)
        v = variable_v1.VariableV1([0.0])
        global_step = variable_v1.VariableV1(0, name='global_step', trainable=False)
        opt.minimize(v, global_step=global_step)
        hook.begin()

    @test_util.run_v1_only('train.SyncReplicasOptimizer and train.AdamOptimizer are V1 only APIs.')
    def testFetchVariableList(self):
        if False:
            print('Hello World!')
        opt = training.SyncReplicasOptimizer(opt=adam.AdamOptimizer(0.01), replicas_to_aggregate=1, total_num_replicas=1)
        v = variable_v1.VariableV1([0.0], name='fetch_variable_test')
        global_step = variable_v1.VariableV1(0, name='global_step', trainable=False)
        opt.minimize(v, global_step=global_step)
        opt_variables = opt.variables()
        (beta1_power, beta2_power) = opt._opt._get_beta_accumulators()
        self.assertIn(beta1_power, opt_variables)
        self.assertIn(beta2_power, opt_variables)
if __name__ == '__main__':
    test.main()