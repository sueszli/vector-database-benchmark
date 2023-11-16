import argparse
import json
import logging
import os
import random
import string
import time
import numpy as np
import ray
from ray._private.test_utils import monitor_memory_usage, wait_for_condition
from ray.data._internal.progress_bar import ProgressBar
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

def run_task_workload(total_num_cpus, smoke):
    if False:
        while True:
            i = 10
    "Run task-based workload that doesn't require object reconstruction."

    @ray.remote(num_cpus=1, max_retries=-1)
    def task():
        if False:
            print('Hello World!')

        def generate_data(size_in_kb=10):
            if False:
                return 10
            return np.zeros(1024 * size_in_kb, dtype=np.uint8)
        a = ''
        for _ in range(100000):
            a = a + random.choice(string.ascii_letters)
        return generate_data(size_in_kb=50)

    @ray.remote(num_cpus=1, max_retries=-1)
    def invoke_nested_task():
        if False:
            while True:
                i = 10
        time.sleep(0.8)
        return ray.get(task.remote())
    multiplier = 75
    if smoke:
        multiplier = 1
    TOTAL_TASKS = int(total_num_cpus * 2 * multiplier)
    pb = ProgressBar('Chaos test', TOTAL_TASKS)
    results = [invoke_nested_task.remote() for _ in range(TOTAL_TASKS)]
    pb.block_until_complete(results)
    pb.close()
    wait_for_condition(lambda : ray.cluster_resources().get('CPU', 0) == ray.available_resources().get('CPU', 0), timeout=60)

def run_actor_workload(total_num_cpus, smoke):
    if False:
        for i in range(10):
            print('nop')
    "Run actor-based workload.\n\n    The test checks if actor restart -1 and task_retries -1 works\n    as expected. It basically requires many actors to report the\n    seqno to the centralized DB actor while there are failures.\n    If at least once is guaranteed upon failures, this test\n    shouldn't fail.\n    "

    @ray.remote(num_cpus=0)
    class DBActor:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.letter_dict = set()

        def add(self, letter):
            if False:
                while True:
                    i = 10
            self.letter_dict.add(letter)

        def get(self):
            if False:
                i = 10
                return i + 15
            return self.letter_dict

    @ray.remote(num_cpus=1, max_restarts=-1, max_task_retries=-1)
    class ReportActor:

        def __init__(self, db_actor):
            if False:
                while True:
                    i = 10
            self.db_actor = db_actor

        def add(self, letter):
            if False:
                return 10
            ray.get(self.db_actor.add.remote(letter))
    NUM_CPUS = int(total_num_cpus)
    multiplier = 2
    if smoke:
        multiplier = 1
    TOTAL_TASKS = int(300 * multiplier)
    head_node_id = ray.get_runtime_context().get_node_id()
    db_actors = [DBActor.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False)).remote() for _ in range(NUM_CPUS)]
    pb = ProgressBar('Chaos test', TOTAL_TASKS * NUM_CPUS)
    actors = []
    for db_actor in db_actors:
        actors.append(ReportActor.remote(db_actor))
    results = []
    highest_reported_num = 0
    for a in actors:
        for _ in range(TOTAL_TASKS):
            results.append(a.add.remote(str(highest_reported_num)))
            highest_reported_num += 1
    pb.fetch_until_complete(results)
    pb.close()
    for actor in actors:
        ray.kill(actor)
    wait_for_condition(lambda : ray.cluster_resources().get('CPU', 0) == ray.available_resources().get('CPU', 0), timeout=60)
    letter_set = set()
    for db_actor in db_actors:
        letter_set.update(ray.get(db_actor.get.remote()))
    for i in range(highest_reported_num):
        assert str(i) in letter_set, i

def run_placement_group_workload(total_num_cpus, smoke):
    if False:
        while True:
            i = 10
    raise NotImplementedError

def parse_script_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-kill-interval', type=int, default=60)
    parser.add_argument('--workload', type=str)
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_known_args()

def main():
    if False:
        for i in range(10):
            print('nop')
    "Test task/actor/placement group basic chaos test.\n\n    Currently, it only tests node failures scenario.\n    Node failures are implemented by an actor that keeps calling\n    Raylet's KillRaylet RPC.\n\n    Ideally, we should setup the infra to cause machine failures/\n    network partitions/etc., but we don't do that for now.\n\n    In the short term, we will only test gRPC network delay +\n    node failures.\n\n    Currently, the test runs 3 steps. Each steps records the\n    peak memory usage to observe the memory usage while there\n    are node failures.\n\n    Step 1: Warm up the cluster. It is needed to pre-start workers\n        if necessary.\n\n    Step 2: Start the test without a failure.\n\n    Step 3: Start the test with constant node failures.\n    "
    (args, unknown) = parse_script_args()
    logging.info('Received arguments: {}'.format(args))
    ray.init(address='auto')
    total_num_cpus = ray.cluster_resources()['CPU']
    total_nodes = 0
    for n in ray.nodes():
        if n['Alive']:
            total_nodes += 1
    monitor_actor = monitor_memory_usage()
    workload = None
    if args.workload == 'tasks':
        workload = run_task_workload
    elif args.workload == 'actors':
        workload = run_actor_workload
    elif args.workload == 'pg':
        workload = run_placement_group_workload
    else:
        assert False
    print('Warm up... Prestarting workers if necessary.')
    start = time.time()
    workload(total_num_cpus, args.smoke)
    print(f'Runtime when warm up: {time.time() - start}')
    print('Running without failures')
    start = time.time()
    workload(total_num_cpus, args.smoke)
    print(f'Runtime when there are no failures: {time.time() - start}')
    (used_gb, usage) = ray.get(monitor_actor.get_peak_memory_info.remote())
    print('Memory usage without failures.')
    print(f'Peak memory usage: {round(used_gb, 2)}GB')
    print(f'Peak memory usage per processes:\n {usage}')
    print('Running with failures')
    start = time.time()
    node_killer = ray.get_actor('node_killer', namespace='release_test_namespace')
    node_killer.run.remote()
    workload(total_num_cpus, args.smoke)
    print(f'Runtime when there are many failures: {time.time() - start}')
    print(f'Total node failures: {ray.get(node_killer.get_total_killed_nodes.remote())}')
    node_killer.stop_run.remote()
    (used_gb, usage) = ray.get(monitor_actor.get_peak_memory_info.remote())
    print('Memory usage with failures.')
    print(f'Peak memory usage: {round(used_gb, 2)}GB')
    print(f'Peak memory usage per processes:\n {usage}')
    ray.get(monitor_actor.stop_run.remote())
    print(f'Total number of killed nodes: {ray.get(node_killer.get_total_killed_nodes.remote())}')
    with open(os.environ['TEST_OUTPUT_JSON'], 'w') as f:
        f.write(json.dumps({'success': 1, '_peak_memory': round(used_gb, 2), '_peak_process_memory': usage}))
main()