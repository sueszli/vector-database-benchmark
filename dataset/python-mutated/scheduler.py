from celery.beat import Scheduler

class mScheduler(Scheduler):

    def tick(self):
        if False:
            while True:
                i = 10
        raise Exception