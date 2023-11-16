import ray
import math
import time
import random
ray.init()

@ray.remote
class ProgressActor:

    def __init__(self, total_num_samples: int):
        if False:
            return 10
        self.total_num_samples = total_num_samples
        self.num_samples_completed_per_task = {}

    def report_progress(self, task_id: int, num_samples_completed: int) -> None:
        if False:
            return 10
        self.num_samples_completed_per_task[task_id] = num_samples_completed

    def get_progress(self) -> float:
        if False:
            return 10
        return sum(self.num_samples_completed_per_task.values()) / self.total_num_samples

@ray.remote
def sampling_task(num_samples: int, task_id: int, progress_actor: ray.actor.ActorHandle) -> int:
    if False:
        print('Hello World!')
    num_inside = 0
    for i in range(num_samples):
        (x, y) = (random.uniform(-1, 1), random.uniform(-1, 1))
        if math.hypot(x, y) <= 1:
            num_inside += 1
        if (i + 1) % 1000000 == 0:
            progress_actor.report_progress.remote(task_id, i + 1)
    progress_actor.report_progress.remote(task_id, num_samples)
    return num_inside
NUM_SAMPLING_TASKS = 10
NUM_SAMPLES_PER_TASK = 10000000
TOTAL_NUM_SAMPLES = NUM_SAMPLING_TASKS * NUM_SAMPLES_PER_TASK
progress_actor = ProgressActor.remote(TOTAL_NUM_SAMPLES)
results = [sampling_task.remote(NUM_SAMPLES_PER_TASK, i, progress_actor) for i in range(NUM_SAMPLING_TASKS)]
while True:
    progress = ray.get(progress_actor.get_progress.remote())
    print(f'Progress: {int(progress * 100)}%')
    if progress == 1:
        break
    time.sleep(1)
total_num_inside = sum(ray.get(results))
pi = total_num_inside * 4 / TOTAL_NUM_SAMPLES
print(f'Estimated value of π is: {pi}')
assert str(pi).startswith('3.14')