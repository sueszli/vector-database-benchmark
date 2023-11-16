from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import numpy as np
import ray
from model import SimpleCNN, download_mnist_retry
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
os.environ['LANG'] = 'C.UTF-8'
parser = argparse.ArgumentParser(description='Run the synchronous parameter server example.')
parser.add_argument('--cluster_mode', type=str, default='local', help='The mode for the Spark cluster. local, yarn or spark-submit.')
parser.add_argument('--num_workers', default=4, type=int, help='The number of workers to use.')
parser.add_argument('--iterations', default=50, type=int, help='Iteration time.')
parser.add_argument('--executor_cores', type=int, default=8, help="The number of driver's cpu cores you want to use.You can change it depending on your own cluster setting.")
parser.add_argument('--executor_memory', type=str, default='10g', help="The size of slave(executor)'s memory you want to use.You can change it depending on your own cluster setting.")
parser.add_argument('--driver_memory', type=str, default='2g', help="The size of driver's memory you want to use.You can change it depending on your own cluster setting.")
parser.add_argument('--driver_cores', type=int, default=8, help="The number of driver's cpu cores you want to use.You can change it depending on your own cluster setting.")
parser.add_argument('--extra_executor_memory_for_ray', type=str, default='20g', help='The extra executor memory to store some data.You can change it depending on your own cluster setting.')
parser.add_argument('--extra_python_lib', type=str, default='python/orca/example/ray_on_spark/parameter_server/model.py', help='The extra python file to import on distribution.You can change it depending on your own cluster setting.')
parser.add_argument('--object_store_memory', type=str, default='4g', help='The memory to store data on local.You can change it depending on your own cluster setting.')

@ray.remote
class ParameterServer(object):

    def __init__(self, learning_rate):
        if False:
            print('Hello World!')
        self.net = SimpleCNN(learning_rate=learning_rate)

    def apply_gradients(self, *gradients):
        if False:
            return 10
        self.net.apply_gradients(np.mean(gradients, axis=0))
        return self.net.variables.get_flat()

    def get_weights(self):
        if False:
            i = 10
            return i + 15
        return self.net.variables.get_flat()

@ray.remote
class Worker(object):

    def __init__(self, worker_index, batch_size=50):
        if False:
            return 10
        self.worker_index = worker_index
        self.batch_size = batch_size
        self.mnist = download_mnist_retry(seed=worker_index)
        self.net = SimpleCNN()

    def compute_gradients(self, weights):
        if False:
            return 10
        self.net.variables.set_flat(weights)
        (xs, ys) = self.mnist.train.next_batch(self.batch_size)
        return self.net.compute_gradients(xs, ys)
if __name__ == '__main__':
    args = parser.parse_args()
    cluster_mode = args.cluster_mode
    if cluster_mode.startswith('yarn'):
        sc = init_orca_context(cluster_mode=cluster_mode, cores=args.executor_cores, memory=args.executor_memory, init_ray_on_spark=True, num_executors=args.num_workers, driver_memory=args.driver_memory, driver_cores=args.driver_cores, extra_executor_memory_for_ray=args.extra_executor_memory_for_ray, object_store_memory=args.object_store_memory, extra_python_lib=args.extra_python_lib, additional_archive='MNIST_data.zip#MNIST_data')
        ray_ctx = OrcaContext.get_ray_context()
    elif cluster_mode == 'local':
        sc = init_orca_context(cores=args.driver_cores)
        ray_ctx = OrcaContext.get_ray_context()
    elif cluster_mode == 'spark-submit':
        sc = init_orca_context(cluster_mode=cluster_mode)
        ray_ctx = OrcaContext.get_ray_context()
    else:
        print("init_orca_context failed. cluster_mode should be one of 'local','yarn' and 'spark-submit' but got " + cluster_mode)
    net = SimpleCNN()
    ps = ParameterServer.remote(0.0001 * args.num_workers)
    workers = [Worker.remote(worker_index) for worker_index in range(args.num_workers)]
    mnist = download_mnist_retry()
    i = 0
    current_weights = ps.get_weights.remote()
    print('Begin iteration')
    while i < args.iterations:
        gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
        current_weights = ps.apply_gradients.remote(*gradients)
        if i % 10 == 0:
            net.variables.set_flat(ray.get(current_weights))
            (test_xs, test_ys) = mnist.test.next_batch(1000)
            accuracy = net.compute_accuracy(test_xs, test_ys)
            print('Iteration {}: accuracy is {}'.format(i, accuracy))
        i += 1
    stop_orca_context()