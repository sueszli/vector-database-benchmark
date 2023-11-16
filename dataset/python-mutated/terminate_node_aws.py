import subprocess
import random
import requests
import time
import logging
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray._private.test_utils import safe_write_to_results_json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def terminate_current_instance():
    if False:
        return 10
    "Use AWS CLI to terminate current instance.\n\n    This requires the 'chaos-test-name': 'tune-chaos-test' tag to be set\n    on the instance."
    token = requests.put('http://169.254.169.254/latest/api/token', headers={'X-aws-ec2-metadata-token-ttl-seconds': '300'}).text
    instance_id = requests.get('http://169.254.169.254/latest/meta-data/instance-id', headers={'X-aws-ec2-metadata-token': token}).text
    region = requests.get('http://169.254.169.254/latest/meta-data/placement/region', headers={'X-aws-ec2-metadata-token': token}).text
    return subprocess.run(['aws', 'ec2', 'terminate-instances', '--instance-ids', instance_id, '--region', region], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def terminate_node(node_id: str):
    if False:
        print('Hello World!')
    killer_task = ray.remote(terminate_current_instance).options(num_cpus=0, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False))
    ray.get(killer_task.remote())

def get_random_node(exclude_current: bool=True):
    if False:
        return 10
    nodes = ray.nodes()
    if exclude_current:
        current_node_ip = ray.util.get_node_ip_address()
    else:
        current_node_ip = 'DUMMY'
    nodes = [node for node in nodes if node['Alive'] and node['NodeManagerAddress'] != current_node_ip and (not node['Resources'].get('head'))]
    if not nodes:
        return None
    random_node = random.choice(nodes)
    return random_node

@ray.remote(num_cpus=0, resources={'head': 0.01})
class InstanceKillerActor:

    def __init__(self, probability: float=0.1, time_between_checks_s: float=60, warmup_time_s: float=0) -> None:
        if False:
            i = 10
            return i + 15
        self.probability = probability
        self.time_between_checks_s = time_between_checks_s
        self.warmup_time_s = warmup_time_s
        self.last_fail_check = None
        self.history = []
        logging.basicConfig(level=logging.INFO)
        self.start_killing()

    def start_killing(self):
        if False:
            while True:
                i = 10
        time.sleep(self.warmup_time_s)
        while True:
            if random.random() < self.probability:
                self.kill()
            time.sleep(self.time_between_checks_s)

    def kill(self):
        if False:
            while True:
                i = 10
        failures = 0
        max_failures = 3
        node = None
        terminated_succesfully = False
        while not terminated_succesfully and failures < max_failures:
            try:
                node = get_random_node()
                if not node:
                    logger.info('No alive worker nodes')
                    continue
                terminate_node(node['NodeID'])
                terminated_succesfully = True
                logger.info(f"Killed node {node['NodeID']} with IP {node['NodeManagerAddress']}")
            except Exception:
                failures += 1
                logger.exception(f'Killing random node failed in attempt {failures}. Retrying {max_failures - failures} more times')
        self.history.append({'timestamp': time.time(), 'node': node, 'terminated_succesfully': terminated_succesfully})
        safe_write_to_results_json(self.history)

def create_instance_killer(probability: float=0.1, time_between_checks_s: float=60, warmup_time_s: float=0):
    if False:
        print('Hello World!')
    killer_actor_cls = InstanceKillerActor.options(scheduling_strategy=NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False))
    actor = killer_actor_cls.remote(probability=probability, time_between_checks_s=time_between_checks_s, warmup_time_s=warmup_time_s)
    return actor