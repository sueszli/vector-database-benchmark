from .randommini import Random
from .renderworker import RenderWorker
from threading import Thread, active_count

class ThreadedRenderWorker(Thread):

    def __init__(self, rw):
        if False:
            for i in range(10):
                print('nop')
        super(ThreadedRenderWorker, self).__init__()
        self.worker = rw
        self.result = None

    def getWorker(self):
        if False:
            while True:
                i = 10
        return self.worker

    def getResult(self):
        if False:
            return 10
        return self.result

    def run(self):
        if False:
            while True:
                i = 10
        self.result = self.worker.render()

class ThreadRenderWorkerPool:

    def __init__(self, baseExpectedSpeed=1600.0):
        if False:
            for i in range(10):
                print('nop')
        self.rnd = Random()
        self.baseSpeed = baseExpectedSpeed
        self.workers = []

    def createNextWorker(self, taskable_renderer):
        if False:
            print('Hello World!')
        speed = (0.5 + self.rnd.real64()) * self.baseSpeed
        task = taskable_renderer.getNextTask(speed)
        if task:
            worker = ThreadedRenderWorker(RenderWorker(task))
            self.workers.append(worker)
            worker.start()
            return worker
        return None

    def activeCount(self):
        if False:
            for i in range(10):
                print('nop')
        return active_count() - 1

    def joinAll(self):
        if False:
            while True:
                i = 10
        for w in self.workers:
            w.join()