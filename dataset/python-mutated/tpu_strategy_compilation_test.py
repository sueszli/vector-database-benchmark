"""Tests for TPUStrategy in regards to compiling programs."""
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('tpu', '', 'Name of TPU to connect to.')
flags.DEFINE_string('project', None, 'Name of GCP project with TPU.')
flags.DEFINE_string('zone', None, 'Name of GCP zone with TPU.')

def get_tpu_cluster_resolver():
    if False:
        while True:
            i = 10
    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project)
    return resolver

def get_tpu_strategy():
    if False:
        for i in range(10):
            print('nop')
    resolver = get_tpu_cluster_resolver()
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_lib.TPUStrategyV2(resolver)
    return strategy

class TPUStrategyCompilationTest(test.TestCase):

    def test_functions_compile_same_signature(self):
        if False:
            while True:
                i = 10
        'Tests compiling different functions with the same signature.'
        strategy = get_tpu_strategy()

        @def_function.function
        def return_one():
            if False:
                i = 10
                return i + 15

            def computation():
                if False:
                    for i in range(10):
                        print('nop')
                return constant_op.constant(1)
            return strategy.run(computation)

        @def_function.function
        def return_two():
            if False:
                return 10

            def computation():
                if False:
                    i = 10
                    return i + 15
                return constant_op.constant(2)
            return strategy.run(computation)
        expected_result_ones = [1 for _ in range(0, strategy.num_replicas_in_sync)]
        self.assertAllEqual(expected_result_ones, strategy.experimental_local_results(return_one()))
        expected_result_twos = [2 for _ in range(0, strategy.num_replicas_in_sync)]
        self.assertAllEqual(expected_result_twos, strategy.experimental_local_results(return_two()))
if __name__ == '__main__':
    test.main()