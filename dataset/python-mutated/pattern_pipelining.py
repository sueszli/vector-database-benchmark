import ray

@ray.remote
class WorkQueue:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.queue = list(range(10))

    def get_work_item(self):
        if False:
            return 10
        if self.queue:
            return self.queue.pop(0)
        else:
            return None

@ray.remote
class WorkerWithoutPipelining:

    def __init__(self, work_queue):
        if False:
            while True:
                i = 10
        self.work_queue = work_queue

    def process(self, work_item):
        if False:
            for i in range(10):
                print('nop')
        print(work_item)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        while True:
            work_item = ray.get(self.work_queue.get_work_item.remote())
            if work_item is None:
                break
            self.process(work_item)

@ray.remote
class WorkerWithPipelining:

    def __init__(self, work_queue):
        if False:
            for i in range(10):
                print('nop')
        self.work_queue = work_queue

    def process(self, work_item):
        if False:
            print('Hello World!')
        print(work_item)

    def run(self):
        if False:
            return 10
        self.work_item_ref = self.work_queue.get_work_item.remote()
        while True:
            work_item = ray.get(self.work_item_ref)
            if work_item is None:
                break
            self.work_item_ref = self.work_queue.get_work_item.remote()
            self.process(work_item)
work_queue = WorkQueue.remote()
worker_without_pipelining = WorkerWithoutPipelining.remote(work_queue)
ray.get(worker_without_pipelining.run.remote())
work_queue = WorkQueue.remote()
worker_with_pipelining = WorkerWithPipelining.remote(work_queue)
ray.get(worker_with_pipelining.run.remote())