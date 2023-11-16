import random
from time import time

def between(min_wait, max_wait):
    if False:
        i = 10
        return i + 15
    '\n    Returns a function that will return a random number between min_wait and max_wait.\n\n    Example::\n\n        class MyUser(User):\n            # wait between 3.0 and 10.5 seconds after each task\n            wait_time = between(3.0, 10.5)\n    '
    return lambda instance: min_wait + random.random() * (max_wait - min_wait)

def constant(wait_time):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a function that just returns the number specified by the wait_time argument\n\n    Example::\n\n        class MyUser(User):\n            wait_time = constant(3)\n    '
    return lambda instance: wait_time

def constant_pacing(wait_time):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a function that will track the run time of the tasks, and for each time it's\n    called it will return a wait time that will try to make the total time between task\n    execution equal to the time specified by the wait_time argument.\n\n    In the following example the task will always be executed once every 10 seconds, no matter\n    the task execution time::\n\n        class MyUser(User):\n            wait_time = constant_pacing(10)\n            @task\n            def my_task(self):\n                time.sleep(random.random())\n\n    If a task execution exceeds the specified wait_time, the wait will be 0 before starting\n    the next task.\n    "

    def wait_time_func(self):
        if False:
            while True:
                i = 10
        if not hasattr(self, '_cp_last_wait_time'):
            self._cp_last_wait_time = 0
        run_time = time() - self._cp_last_run - self._cp_last_wait_time
        self._cp_last_wait_time = max(0, wait_time - run_time)
        self._cp_last_run = time()
        return self._cp_last_wait_time
    return wait_time_func

def constant_throughput(task_runs_per_second):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a function that will track the run time of the tasks, and for each time it's\n    called it will return a wait time that will try to make the number of task runs per second\n    execution equal to the time specified by the task_runs_per_second argument.\n\n    If you have multiple requests in a task your RPS will of course be higher than the\n    specified throughput.\n\n    This is the mathematical inverse of constant_pacing.\n\n    In the following example the task will always be executed once every 10 seconds, no matter\n    the task execution time::\n\n        class MyUser(User):\n            wait_time = constant_throughput(0.1)\n            @task\n            def my_task(self):\n                time.sleep(random.random())\n\n    If a task execution exceeds the specified wait_time, the wait will be 0 before starting\n    the next task.\n    "
    return constant_pacing(1 / task_runs_per_second)