from collections import namedtuple
import math
import time
import random
from locust import HttpUser, TaskSet, task, constant
from locust import LoadTestShape

class UserTasks(TaskSet):

    @task
    def get_root(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.get('/')

class WebsiteUser(HttpUser):
    wait_time = constant(0.5)
    tasks = [UserTasks]

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        time.sleep(random.randint(0, 5))
        super().__init__(*args, **kwargs)
Step = namedtuple('Step', ['users', 'dwell'])

class StepLoadShape(LoadTestShape):
    """
    A step load shape that waits until the target user count has
    been reached before waiting on a per-step timer.

    The purpose here is to ensure that a target number of users is always reached,
    regardless of how slow the user spawn rate is. The dwell time is there to
    observe the steady state at that number of users.

    Keyword arguments:

        targets_with_times -- iterable of 2-tuples, with the desired user count first,
            and the dwell (hold) time with that user count second

    """
    targets_with_times = (Step(10, 10), Step(20, 15), Step(10, 10))

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.step = 0
        self.time_active = False
        super().__init__(*args, **kwargs)

    def tick(self):
        if False:
            return 10
        if self.step >= len(self.targets_with_times):
            return None
        target = self.targets_with_times[self.step]
        users = self.get_current_user_count()
        if target.users == users:
            if not self.time_active:
                self.reset_time()
                self.time_active = True
            run_time = self.get_run_time()
            if run_time > target.dwell:
                self.step += 1
                self.time_active = False
        return (target.users, 100)