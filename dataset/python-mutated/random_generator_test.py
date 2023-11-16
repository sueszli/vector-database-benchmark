"""Tests tf.random.Generator with distribution strategies."""
import functools
import os
from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values as dist_values
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import stateful_random_ops as rng
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.util import deprecation

def get_num_local_replicas(strat, values=None):
    if False:
        return 10
    strat_name = type(strat).__name__
    if 'MultiWorker' in strat_name or 'CollectiveAllReduceStrategy' in strat_name:
        if values is None:
            values = strat.run(lambda : constant_op.constant(0))
            values = strat.experimental_local_results(values)
        return len(values)
    else:
        return strat.num_replicas_in_sync
ps_strategies = [strategy_combinations.parameter_server_strategy_3worker_2ps_cpu, strategy_combinations.parameter_server_strategy_1worker_2ps_cpu, strategy_combinations.parameter_server_strategy_3worker_2ps_1gpu, strategy_combinations.parameter_server_strategy_1worker_2ps_1gpu]
all_strategies = strategy_combinations.all_strategies + strategy_combinations.multiworker_strategies + ps_strategies

def run_on_strategy(replica_fn, strat, coord):
    if False:
        for i in range(10):
            print('nop')

    def distributed_fn():
        if False:
            i = 10
            return i + 15
        return strat.run(replica_fn)
    if coord is not None:
        results = coord.schedule(def_function.function(distributed_fn)).fetch()
    else:
        results = distributed_fn()
    return results

class GeneratorTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(GeneratorTest, self).setUp()
        v2_compat.enable_v2_behavior()

    def assertAllDifferent(self, tensors):
        if False:
            for i in range(10):
                print('nop')
        'Checks that there are no duplicate elements anywhere among the tensors.\n\n    Args:\n      tensors: a list of tensors. They can have different shapes.\n    '
        values = [array_ops.reshape(t, shape=[-1]) for t in tensors]
        values = array_ops.concat(values, axis=0)
        values = self.evaluate(values)
        values = values.tolist()
        self.assertAllEqual(len(values), len(set(values)))

    @test_util.run_v2_only
    def testCreateOutsideMirroredStrat(self):
        if False:
            i = 10
            return i + 15
        'Tests RNG/MirrorStrategy interaction #1.\n\n    If an RNG is created outside a DS scope, all replicas will access the\n    same RNG object, and accesses are serialized.\n    '
        shape = [3, 4]
        dtype = dtypes.int32
        gen = rng.Generator.from_seed(1234)
        strat = MirroredStrategy(devices=['cpu:0', 'cpu:1'])
        with strat.scope():

            def f():
                if False:
                    print('Hello World!')
                t1 = gen.uniform_full_int(shape=shape, dtype=dtype)
                t2 = gen.uniform_full_int(shape=shape, dtype=dtype)
                t = array_ops_stack.stack([t1, t2])
                return t
            results = strat.extended.call_for_each_replica(fn=f)
            values = results.values
            self.assertAllEqual(2, len(values))
            self.assertAllDifferent(values)

    @test_util.run_v2_only
    def testMirroredStratParaAsync(self):
        if False:
            print('Hello World!')
        'Tests RNG/MirrorStrategy interaction #2.\n\n    The user can create n independent RNGs outside strategy.scope(), where n\n    is the number of replicas, and give one to each replica. The replicas can\n    thus get different random-number streams.\n    '
        shape = [3, 4]
        dtype = dtypes.int32
        gens = rng.get_global_generator().split(count=2)
        devices = ['cpu:0', 'cpu:1']
        strat = MirroredStrategy(devices=devices)
        gens = dist_values.PerReplica([[g] for g in gens])
        with strat.scope():

            def f(gen):
                if False:
                    for i in range(10):
                        print('nop')
                t1 = gen.uniform_full_int(shape=shape, dtype=dtype)
                t2 = gen.uniform_full_int(shape=shape, dtype=dtype)
                t = array_ops_stack.stack([t1, t2])
                return t
            results = strat.extended.call_for_each_replica(fn=f, args=gens)
            local_results = strat.experimental_local_results(results)
            self.assertAllEqual(2, len(local_results))
            self.assertAllDifferent(local_results)

    @ds_combinations.generate(combinations.combine(strat=all_strategies, mode=['eager']))
    def testCrossReplica(self, strat):
        if False:
            print('Hello World!')
        'Tests that RNG can be properly advanced in cross-replica context.'

        def read_values(dv):
            if False:
                while True:
                    i = 10
            return [v.read_value() for v in strat.experimental_local_results(dv)]
        with strat.scope():
            g = rng.Generator.from_seed(1)
            s1 = read_values(g.state)
            g.normal([3])
            g.skip(4)
            s2 = read_values(g.state)
        self.assertNotAllEqual(s1[0], s2[0])
        self.assertEqual(len(s1), len(s2))
        for i in range(1, len(s1)):
            self.assertAllEqual(s1[0], s1[i])
            self.assertAllEqual(s2[0], s2[i])

    @ds_combinations.generate(combinations.combine(strat=all_strategies, mode=['eager'], jit_replica_fn=[False, True], seeded=[True, False]))
    def testDistStrat(self, strat, jit_replica_fn, seeded):
        if False:
            while True:
                i = 10
        'Tests RNG with distribution strategies.'
        strat_name = type(strat).__name__
        if 'TPU' in strat_name and (not jit_replica_fn):
            self.skipTest('TPUStrategy requires the replica function (the function passed to strategy.run) to be decorated with tf.function')
        coord = None
        if 'ParameterServer' in strat_name:
            coord = coordinator_lib.ClusterCoordinator(strat)
        creators = {True: functools.partial(rng.Generator.from_seed, 1234), False: rng.Generator.from_non_deterministic_state}
        shape = [3, 4]
        dtype = dtypes.int32
        creator = creators[seeded]
        with strat.scope():
            gen = creator()

            def f():
                if False:
                    while True:
                        i = 10
                t1 = gen.uniform_full_int(shape=shape, dtype=dtype)
                t2 = gen.uniform_full_int(shape=shape, dtype=dtype)
                t = array_ops_stack.stack([t1, t2])
                return t
            replica_fn = def_function.function(f) if jit_replica_fn else f
            results = run_on_strategy(replica_fn, strat, coord)
            values = strat.experimental_local_results(results)
            n = get_num_local_replicas(strat, values)
            self.assertAllEqual(n, len(values))
            self.assertAllDifferent(values)

    @ds_combinations.generate(combinations.combine(strat=[strategy_combinations.parameter_server_strategy_fn('ParameterServer1Worker2PSCPUFixedShards', num_workers=1, num_ps=2, variable_partitioner=sharded_variable.FixedShardsPartitioner(2))], mode=['eager']))
    def testShardedError(self, strat):
        if False:
            return 10
        'Tests error about sharding is raised.'
        with strat.scope():
            with self.assertRaisesRegex(ValueError, 'state is sharded, which is not allowed'):
                rng.Generator.from_seed(1234)

    @ds_combinations.generate(combinations.combine(strat=all_strategies, mode=['eager'], jit_replica_fn=[False, True]))
    def testDistVarAsTFFunArg(self, strat, jit_replica_fn):
        if False:
            while True:
                i = 10
        "Tests that RNG with dist variables can be used as tf.function's arg."
        strat_name = type(strat).__name__
        if 'CentralStorage' in strat_name:
            self.skipTest("CentralStorageStrategy wraps variable updates in merge_call which can't be called inside a tf.function that doesn't cover the entire replica function (the function passed to strategy.run).")
        if 'TPU' in strat_name and (not jit_replica_fn):
            self.skipTest('TPUStrategy requires the replica function (the function passed to strategy.run) to be decorated with tf.function')
        coord = None
        if 'ParameterServer' in strat_name:
            coord = coordinator_lib.ClusterCoordinator(strat)
        shape = [3, 4]
        dtype = dtypes.int32
        with strat.scope():
            gen = rng.Generator.from_seed(1234)

            @def_function.function
            def f(gen):
                if False:
                    while True:
                        i = 10
                t1 = gen.uniform_full_int(shape=shape, dtype=dtype)
                t2 = gen.uniform_full_int(shape=shape, dtype=dtype)
                t = array_ops_stack.stack([t1, t2])
                return t

            def g():
                if False:
                    while True:
                        i = 10
                return f(gen)
            replica_fn = def_function.function(g) if jit_replica_fn else g
            for _ in range(2):
                results = run_on_strategy(replica_fn, strat, coord)
                values = strat.experimental_local_results(results)
                n = get_num_local_replicas(strat, values)
                self.assertAllEqual(n, len(values))
                self.assertAllDifferent(values)

    @ds_combinations.generate(combinations.combine(strat1=strategy_combinations.all_strategies, strat2=strategy_combinations.all_strategies, jit_replica_fn=[False, True], mode=['eager']) + combinations.combine(strat1=strategy_combinations.multiworker_strategies + ps_strategies, strat2=[None], jit_replica_fn=[False, True], mode=['eager']))
    def testDistStratRestore(self, strat1, strat2, jit_replica_fn):
        if False:
            print('Hello World!')
        'Tests checkpointing and restoring (to possibly different #replicas).'
        if strat2 is None:
            strat2 = strat1
        strat1_name = type(strat1).__name__
        strat2_name = type(strat2).__name__
        if 'Default' in strat1_name or 'Default' in strat2_name:
            self.skipTest("We don't guarantee consistency between strategy and no-strategy.")
        if ('TPU' in strat1_name or 'TPU' in strat2_name) and (not jit_replica_fn):
            self.skipTest('TPUStrategy requires the replica function (the function passed to strategy.run) to be decorated with tf.function')
        coord1 = None
        if 'ParameterServer' in strat1_name:
            coord1 = coordinator_lib.ClusterCoordinator(strat1)
        coord2 = None
        if 'ParameterServer' in strat2_name:
            coord2 = coordinator_lib.ClusterCoordinator(strat2)
        fname = os.path.join(self.get_temp_dir(), 'checkpoint')

        def uniform(strat, coord, g):
            if False:
                return 10

            def f():
                if False:
                    print('Hello World!')
                return g.uniform_full_int([3], dtype=dtypes.int32)
            replica_fn = def_function.function(f) if jit_replica_fn else f
            result = run_on_strategy(replica_fn, strat, coord)
            return strat.experimental_local_results(result)
        with strat1.scope():
            g1 = rng.Generator.from_seed(1)
        with strat2.scope():
            g2 = rng.Generator.from_seed(10)
        cp1 = tracking_util.Checkpoint(g=g1)
        cp2 = tracking_util.Checkpoint(g=g2)

        def write_restore_compare():
            if False:
                print('Hello World!')
            cp1.write(fname)
            r1 = uniform(strat1, coord1, g1)
            cp2.restore(fname)
            r2 = uniform(strat2, coord2, g2)
            n1 = get_num_local_replicas(strat1)
            n2 = get_num_local_replicas(strat2)
            n = min(n1, n2)
            self.assertAllEqual(r1[:n], r2[:n])
        for _ in range(2):
            write_restore_compare()

    @ds_combinations.generate(combinations.combine(strat=all_strategies, mode=['eager'], is_save_in_scope=[True, False]))
    def testSavedModel(self, strat, is_save_in_scope):
        if False:
            for i in range(10):
                print('nop')

        class CustomModule(module.Module):

            def __init__(self):
                if False:
                    return 10
                super(CustomModule, self).__init__()
                self.g = rng.Generator.from_seed(0)

            @def_function.function
            def __call__(self):
                if False:
                    i = 10
                    return i + 15
                return self.g.state

            @def_function.function
            def mutate(self):
                if False:
                    return 10
                self.g.normal([])
        with strat.scope():
            m = CustomModule()
            m.mutate()
            state_before = m()
            path = os.path.join(self.get_temp_dir(), 'saved_model')
        if is_save_in_scope:
            with strat.scope():
                save.save(m, path)
        else:
            save.save(m, path)
        with strat.scope():
            m.mutate()
            state_before_2 = m()
        imported = load.load(path)
        state_after = imported()
        self.assertAllEqual(state_before, state_after)
        imported.mutate()
        state_after_2 = imported()
        self.assertAllEqual(state_before_2, state_after_2)
if __name__ == '__main__':
    with deprecation.silence():
        multi_process_runner.test_main()