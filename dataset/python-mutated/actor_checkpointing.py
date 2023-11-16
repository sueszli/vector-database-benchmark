import os
import sys
import ray
import json
import tempfile
import shutil

@ray.remote(num_cpus=1)
class Worker:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.state = {'num_tasks_executed': 0}

    def execute_task(self, crash=False):
        if False:
            print('Hello World!')
        if crash:
            sys.exit(1)
        self.state['num_tasks_executed'] = self.state['num_tasks_executed'] + 1

    def checkpoint(self):
        if False:
            for i in range(10):
                print('nop')
        return self.state

    def restore(self, state):
        if False:
            return 10
        self.state = state

class Controller:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.worker = Worker.remote()
        self.worker_state = ray.get(self.worker.checkpoint.remote())

    def execute_task_with_fault_tolerance(self):
        if False:
            print('Hello World!')
        i = 0
        while True:
            i = i + 1
            try:
                ray.get(self.worker.execute_task.remote(crash=i % 2 == 1))
                self.worker_state = ray.get(self.worker.checkpoint.remote())
                return
            except ray.exceptions.RayActorError:
                print('Actor crashes, restarting...')
                self.worker = Worker.remote()
                ray.get(self.worker.restore.remote(self.worker_state))
controller = Controller()
controller.execute_task_with_fault_tolerance()
controller.execute_task_with_fault_tolerance()
assert ray.get(controller.worker.checkpoint.remote())['num_tasks_executed'] == 2

@ray.remote(max_restarts=-1, max_task_retries=-1)
class ImmortalActor:

    def __init__(self, checkpoint_file):
        if False:
            print('Hello World!')
        self.checkpoint_file = checkpoint_file
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {}

    def update(self, key, value):
        if False:
            print('Hello World!')
        import random
        if random.randrange(10) < 5:
            sys.exit(1)
        self.state[key] = value
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f)

    def get(self, key):
        if False:
            return 10
        return self.state[key]
checkpoint_dir = tempfile.mkdtemp()
actor = ImmortalActor.remote(os.path.join(checkpoint_dir, 'checkpoint.json'))
ray.get(actor.update.remote('1', 1))
ray.get(actor.update.remote('2', 2))
assert ray.get(actor.get.remote('1')) == 1
shutil.rmtree(checkpoint_dir)