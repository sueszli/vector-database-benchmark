import copy
import logging
import os
import pytest
import tempfile
import unittest
import subprocess
from typing import Any, Dict
import yaml
from ray.tests.kuberay.utils import get_pod, get_pod_names, get_raycluster, ray_client_port_forward, ray_job_submit, switch_to_ray_parent_dir, kubectl_exec_python_script, kubectl_logs, kubectl_delete, wait_for_pods, wait_for_pod_to_start, wait_for_ray_health
from ray.tests.kuberay.scripts import gpu_actor_placement, gpu_actor_validation, non_terminated_nodes_count
logger = logging.getLogger(__name__)
RAY_IMAGE = os.environ.get('RAY_IMAGE', 'rayproject/ray:nightly-py38')
AUTOSCALER_IMAGE = os.environ.get('AUTOSCALER_IMAGE', RAY_IMAGE)
PULL_POLICY = os.environ.get('PULL_POLICY', 'IfNotPresent')
logger.info(f'Using image `{RAY_IMAGE}` for Ray containers.')
logger.info(f'Using image `{AUTOSCALER_IMAGE}` for Autoscaler containers.')
logger.info(f'Using pull policy `{PULL_POLICY}` for all images.')
EXAMPLE_CLUSTER_PATH = 'ray/python/ray/tests/kuberay/test_files/ray-cluster.autoscaler-template.yaml'
HEAD_SERVICE = 'raycluster-autoscaler-head-svc'
HEAD_POD_PREFIX = 'raycluster-autoscaler-head'
CPU_WORKER_PREFIX = 'raycluster-autoscaler-worker-small-group'
RAY_CLUSTER_NAME = 'raycluster-autoscaler'
RAY_CLUSTER_NAMESPACE = 'default'
pytestmark = pytest.mark.timeout(300)

class KubeRayAutoscalingTest(unittest.TestCase):
    """e2e verification of autoscaling following the steps in the Ray documentation.
    kubectl is used throughout, as that reflects the instructions in the docs.
    """

    def _get_ray_cr_config(self, min_replicas=0, cpu_replicas=0, gpu_replicas=0) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Get Ray CR config yaml.\n\n        - Use configurable replica fields for a CPU workerGroup.\n\n        - Add a GPU-annotated group for testing GPU upscaling.\n\n        - Fill in Ray image, autoscaler image, and image pull policies from env\n          variables.\n        '
        with open(EXAMPLE_CLUSTER_PATH) as ray_cr_config_file:
            ray_cr_config_str = ray_cr_config_file.read()
        for k8s_object in yaml.safe_load_all(ray_cr_config_str):
            if k8s_object['kind'] in ['RayCluster', 'RayJob', 'RayService']:
                config = k8s_object
                break
        head_group = config['spec']['headGroupSpec']
        head_group['rayStartParams']['resources'] = '"{\\"Custom1\\": 1, \\"Custom2\\": 5}"'
        cpu_group = config['spec']['workerGroupSpecs'][0]
        cpu_group['replicas'] = cpu_replicas
        cpu_group['minReplicas'] = min_replicas
        cpu_group['maxReplicas'] = 300
        cpu_group['rayStartParams']['resources'] = '"{\\"Custom1\\": 1, \\"Custom2\\": 5}"'
        gpu_group = copy.deepcopy(cpu_group)
        gpu_group['rayStartParams']['num-gpus'] = '1'
        gpu_group['replicas'] = gpu_replicas
        gpu_group['minReplicas'] = 0
        gpu_group['maxReplicas'] = 1
        gpu_group['groupName'] = 'fake-gpu-group'
        config['spec']['workerGroupSpecs'].append(gpu_group)
        for group_spec in config['spec']['workerGroupSpecs'] + [config['spec']['headGroupSpec']]:
            containers = group_spec['template']['spec']['containers']
            ray_container = containers[0]
            assert ray_container['name'] in ['ray-head', 'ray-worker']
            ray_container['image'] = RAY_IMAGE
            for container in containers:
                container['imagePullPolicy'] = PULL_POLICY
        autoscaler_options = {'image': AUTOSCALER_IMAGE, 'imagePullPolicy': PULL_POLICY, 'idleTimeoutSeconds': 10}
        config['spec']['autoscalerOptions'] = autoscaler_options
        return config

    def _apply_ray_cr(self, min_replicas=0, cpu_replicas=0, gpu_replicas=0, validate_replicas: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Apply Ray CR config yaml, with configurable replica fields for the cpu\n        workerGroup.\n\n        If the CR does not yet exist, `replicas` can be set as desired.\n        If the CR does already exist, the recommended usage is this:\n            (1) Set `cpu_replicas` and `gpu_replicas` to what we currently expect them\n                to be.\n            (2) Set `validate_replicas` to True. We will then check that the replicas\n            set on the CR coincides with `replicas`.\n        '
        with tempfile.NamedTemporaryFile('w') as config_file:
            if validate_replicas:
                raycluster = get_raycluster(RAY_CLUSTER_NAME, namespace=RAY_CLUSTER_NAMESPACE)
                assert raycluster['spec']['workerGroupSpecs'][0]['replicas'] == cpu_replicas
                assert raycluster['spec']['workerGroupSpecs'][1]['replicas'] == gpu_replicas
                logger.info(f'Validated that cpu and gpu worker replicas for {RAY_CLUSTER_NAME} are currently {cpu_replicas} and {gpu_replicas}, respectively.')
            cr_config = self._get_ray_cr_config(min_replicas=min_replicas, cpu_replicas=cpu_replicas, gpu_replicas=gpu_replicas)
            yaml.dump(cr_config, config_file)
            config_file.flush()
            subprocess.check_call(['kubectl', 'apply', '-f', config_file.name])

    def _non_terminated_nodes_count(self) -> int:
        if False:
            return 10
        with ray_client_port_forward(head_service=HEAD_SERVICE):
            return non_terminated_nodes_count.main()

    def testAutoscaling(self):
        if False:
            print('Hello World!')
        "Test the following behaviors:\n\n        1. Spinning up a Ray cluster\n        2. Scaling up Ray workers via autoscaler.sdk.request_resources()\n        3. Scaling up by updating the CRD's minReplicas\n        4. Scaling down by removing the resource request and reducing maxReplicas\n        5. Autoscaler recognizes GPU annotations and Ray custom resources.\n        6. Autoscaler and operator ignore pods marked for deletion.\n        7. Autoscaler logs work. Autoscaler events are piped to the driver.\n        8. Ray utils show correct resource limits in the head container.\n\n        Tests the following modes of interaction with a Ray cluster on K8s:\n        1. kubectl exec\n        2. Ray Client\n        3. Ray Job Submission\n\n        TODO (Dmitri): Split up the test logic.\n        Too much is stuffed into this one test case.\n\n        Resources requested by this test are safely within the bounds of an m5.xlarge\n        instance.\n\n        The resource REQUESTS are:\n        - One Ray head pod\n            - Autoscaler: .25 CPU, .5 Gi memory\n            - Ray node: .5 CPU, .5 Gi memeory\n        - Three Worker pods\n            - Ray node: .5 CPU, .5 Gi memory\n        Total: 2.25 CPU, 2.5 Gi memory.\n\n        Including operator and system pods, the total CPU requested is around 3.\n\n        The cpu LIMIT of each Ray container is 1.\n        The `num-cpus` arg to Ray start is 1 for each Ray container; thus Ray accounts\n        1 CPU for each Ray node in the test.\n        "
        switch_to_ray_parent_dir()
        logger.info('Creating a RayCluster with no worker pods.')
        self._apply_ray_cr(min_replicas=0, cpu_replicas=0, gpu_replicas=0)
        logger.info('Confirming presence of head.')
        wait_for_pods(goal_num_pods=1, namespace=RAY_CLUSTER_NAMESPACE)
        logger.info('Waiting for head pod to start Running.')
        wait_for_pod_to_start(pod_name_filter=HEAD_POD_PREFIX, namespace=RAY_CLUSTER_NAMESPACE)
        logger.info('Confirming Ray is up on the head pod.')
        wait_for_ray_health(pod_name_filter=HEAD_POD_PREFIX, namespace=RAY_CLUSTER_NAMESPACE)
        head_pod = get_pod(pod_name_filter=HEAD_POD_PREFIX, namespace=RAY_CLUSTER_NAMESPACE)
        assert head_pod, 'Could not find the Ray head pod.'
        logger.info('Confirming head pod resource allocation.')
        out = kubectl_exec_python_script(script_name='check_cpu_and_memory.py', pod=head_pod, container='ray-head', namespace='default')
        logger.info('Scaling up to one worker via Ray resource request.')
        kubectl_exec_python_script(script_name='scale_up.py', pod=head_pod, container='ray-head', namespace='default')
        logs = kubectl_logs(head_pod, namespace='default', container='autoscaler')
        assert 'Adding 1 node(s) of type small-group.' in logs
        logger.info('Confirming number of workers.')
        wait_for_pods(goal_num_pods=2, namespace=RAY_CLUSTER_NAMESPACE)
        logger.info('Scaling up to two workers by editing minReplicas.')
        self._apply_ray_cr(min_replicas=2, cpu_replicas=1, gpu_replicas=0, validate_replicas=True)
        logger.info('Confirming number of workers.')
        wait_for_pods(goal_num_pods=3, namespace=RAY_CLUSTER_NAMESPACE)
        assert not any(('gpu' in pod_name for pod_name in get_pod_names(namespace=RAY_CLUSTER_NAMESPACE)))
        logger.info('Scheduling an Actor with GPU demands.')
        with ray_client_port_forward(head_service=HEAD_SERVICE, ray_namespace='gpu-test'):
            gpu_actor_placement.main()
        logger.info('Confirming fake GPU worker up-scaling.')
        wait_for_pods(goal_num_pods=4, namespace=RAY_CLUSTER_NAMESPACE)
        gpu_workers = [pod_name for pod_name in get_pod_names(namespace=RAY_CLUSTER_NAMESPACE) if 'gpu' in pod_name]
        assert len(gpu_workers) == 1
        logger.info('Confirming GPU actor placement.')
        with ray_client_port_forward(head_service=HEAD_SERVICE, ray_namespace='gpu-test'):
            out = gpu_actor_validation.main()
        assert 'on-a-gpu-node' in out
        logger.info('Reducing min workers to 0.')
        self._apply_ray_cr(min_replicas=0, cpu_replicas=2, gpu_replicas=1, validate_replicas=True)
        logger.info('Removing resource demands.')
        kubectl_exec_python_script(script_name='scale_down.py', pod=head_pod, container='ray-head', namespace='default')
        logger.info('Confirming workers are gone.')
        logs = kubectl_logs(head_pod, namespace='default', container='autoscaler')
        assert 'Removing 1 nodes of type fake-gpu-group (idle).' in logs
        wait_for_pods(goal_num_pods=1, namespace=RAY_CLUSTER_NAMESPACE)
        logger.info('Scaling up workers with request for custom resources.')
        job_logs = ray_job_submit(script_name='scale_up_custom.py', head_service=HEAD_SERVICE)
        assert 'Submitted custom scale request!' in job_logs, job_logs
        logger.info('Confirming two workers have scaled up.')
        wait_for_pods(goal_num_pods=3, namespace=RAY_CLUSTER_NAMESPACE)
        logger.info('Deleting Ray cluster.')
        kubectl_delete(kind='raycluster', name=RAY_CLUSTER_NAME, namespace=RAY_CLUSTER_NAMESPACE)
        logger.info('Confirming Ray pods are gone.')
        wait_for_pods(goal_num_pods=0, namespace=RAY_CLUSTER_NAMESPACE)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-vv', __file__]))