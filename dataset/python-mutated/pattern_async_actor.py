import ray

@ray.remote
class TaskStore:

    def get_next_task(self):
        if False:
            print('Hello World!')
        return 'task'

@ray.remote
class TaskExecutor:

    def __init__(self, task_store):
        if False:
            return 10
        self.task_store = task_store
        self.num_executed_tasks = 0

    def run(self):
        if False:
            print('Hello World!')
        while True:
            task = ray.get(task_store.get_next_task.remote())
            self._execute_task(task)

    def _execute_task(self, task):
        if False:
            while True:
                i = 10
        self.num_executed_tasks = self.num_executed_tasks + 1

    def get_num_executed_tasks(self):
        if False:
            for i in range(10):
                print('nop')
        return self.num_executed_tasks
task_store = TaskStore.remote()
task_executor = TaskExecutor.remote(task_store)
task_executor.run.remote()
try:
    ray.get(task_executor.get_num_executed_tasks.remote(), timeout=5)
except ray.exceptions.GetTimeoutError:
    print("get_num_executed_tasks didn't finish in 5 seconds")

@ray.remote
class AsyncTaskExecutor:

    def __init__(self, task_store):
        if False:
            print('Hello World!')
        self.task_store = task_store
        self.num_executed_tasks = 0

    async def run(self):
        while True:
            task = await task_store.get_next_task.remote()
            self._execute_task(task)

    def _execute_task(self, task):
        if False:
            for i in range(10):
                print('nop')
        self.num_executed_tasks = self.num_executed_tasks + 1

    def get_num_executed_tasks(self):
        if False:
            for i in range(10):
                print('nop')
        return self.num_executed_tasks
async_task_executor = AsyncTaskExecutor.remote(task_store)
async_task_executor.run.remote()
num_executed_tasks = ray.get(async_task_executor.get_num_executed_tasks.remote())
print(f'num of executed tasks so far: {num_executed_tasks}')