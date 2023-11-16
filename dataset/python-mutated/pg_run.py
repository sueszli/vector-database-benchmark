import os
import time
import json
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
RUNTIME = 600
NUM_CPU_BUNDLES = 30

@ray.remote(num_cpus=1)
class Worker(object):

    def __init__(self, i):
        if False:
            for i in range(10):
                print('nop')
        self.i = i

    def work(self):
        if False:
            print('Hello World!')
        time.sleep(0.1)
        print('work ', self.i)

@ray.remote(num_cpus=1, num_gpus=1)
class Trainer(object):

    def __init__(self, i):
        if False:
            while True:
                i = 10
        self.i = i

    def train(self):
        if False:
            return 10
        time.sleep(0.2)
        print('train ', self.i)

def main():
    if False:
        while True:
            i = 10
    ray.init(address='auto')
    bundles = [{'CPU': 1, 'GPU': 1}]
    bundles += [{'CPU': 1} for _ in range(NUM_CPU_BUNDLES)]
    pg = placement_group(bundles, strategy='PACK')
    ray.get(pg.ready())
    workers = [Worker.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote(i) for i in range(NUM_CPU_BUNDLES)]
    trainer = Trainer.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote(0)
    start = time.time()
    while True:
        ray.get([workers[i].work.remote() for i in range(NUM_CPU_BUNDLES)])
        ray.get(trainer.train.remote())
        end = time.time()
        if end - start > RUNTIME:
            break
    if 'TEST_OUTPUT_JSON' in os.environ:
        out_file = open(os.environ['TEST_OUTPUT_JSON'], 'w')
        results = {}
        json.dump(results, out_file)
if __name__ == '__main__':
    main()