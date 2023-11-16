import copy
import os
import subprocess
import time
import unittest
from paddle.distributed.utils.launch_utils import TrainerProc, find_free_ports, get_cluster, watch_local_trainers

def get_cluster_from_args(selected_gpus):
    if False:
        for i in range(10):
            print('nop')
    cluster_node_ips = '127.0.0.1'
    node_ip = '127.0.0.1'
    node_ips = [x.strip() for x in cluster_node_ips.split(',')]
    node_ips.index(node_ip)
    free_ports = None
    free_ports = find_free_ports(len(selected_gpus))
    if free_ports is not None:
        free_ports = list(free_ports)
    trainer_endpoints = []
    for ip in node_ips:
        trainer_endpoints.append(['%s:%d' % (ip, port) for port in free_ports])
    return get_cluster(node_ips, node_ip, trainer_endpoints, selected_gpus)

def get_gpus(selected_gpus):
    if False:
        print('Hello World!')
    selected_gpus = [x.strip() for x in selected_gpus.split(',')]
    return selected_gpus

def start_local_trainers(cluster, pod, training_script, training_script_args, log_dir=None):
    if False:
        print('Hello World!')
    current_env = copy.copy(os.environ.copy())
    current_env.pop('http_proxy', None)
    current_env.pop('https_proxy', None)
    procs = []
    for t in pod.trainers:
        proc_env = {'PADDLE_TRAINER_ID': '%d' % t.rank, 'PADDLE_CURRENT_ENDPOINT': '%s' % t.endpoint, 'PADDLE_TRAINERS_NUM': '%d' % cluster.trainers_nranks(), 'PADDLE_TRAINER_ENDPOINTS': ','.join(cluster.trainers_endpoints()), 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '6170', 'NCCL_DEBUG': 'INFO', 'PADDLE_DISTRI_BACKEND': 'gloo'}
        current_env.update(proc_env)
        print(f'trainer proc env:{current_env}')
        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            cmd = 'python -m coverage run --branch -p ' + training_script
        else:
            cmd = 'python -u ' + training_script
        print(f'start trainer proc:{cmd} env:{proc_env}')
        fn = None
        proc = subprocess.Popen(cmd.split(' '), env=current_env)
        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.rank
        tp.log_fn = fn
        tp.cmd = cmd
        procs.append(tp)
    return procs

class TestMultipleGpus(unittest.TestCase):

    def run_mnist_2gpu(self, target_file_name):
        if False:
            for i in range(10):
                print('nop')
        selected_gpus = get_gpus('0,1')
        cluster = None
        pod = None
        (cluster, pod) = get_cluster_from_args(selected_gpus)
        procs = start_local_trainers(cluster, pod, training_script=target_file_name, training_script_args=[])
        while True:
            alive = watch_local_trainers(procs, cluster.trainers_nranks())
            if not alive:
                print(f'Local procs complete, POD info:{pod}')
                break
            time.sleep(3)

class TestDataParallelGradientCheck(TestMultipleGpus):

    def test_multiple_gpus_dynamic(self):
        if False:
            i = 10
            return i + 15
        self.run_mnist_2gpu('parallel_dygraph_gradient_check.py')

class TestDataParallelGradientCheckInEagerMode(TestMultipleGpus):

    def test_multiple_gpus_dynamic(self):
        if False:
            print('Hello World!')
        self.run_mnist_2gpu('parallel_dygraph_gradient_check_in_eager_mode.py')
if __name__ == '__main__':
    unittest.main()