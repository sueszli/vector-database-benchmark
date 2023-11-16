from locust import task, HttpUser, events
from locust.runners import STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP, WorkerRunner
import time
import gevent

class MyUser(HttpUser):
    host = 'http://www.google.com'

    @task
    def my_task(self):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(10):
            self.client.get('/')
            time.sleep(1)
        for _ in range(5):
            self.client.get('/error')
            time.sleep(1)

def checker(environment):
    if False:
        for i in range(10):
            print('nop')
    while not environment.runner.state in [STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP]:
        time.sleep(1)
        if environment.runner.stats.total.fail_ratio > 0.2:
            print(f'fail ratio was {environment.runner.stats.total.fail_ratio}, quitting')
            environment.runner.quit()
            return

@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    if False:
        while True:
            i = 10
    if not isinstance(environment.runner, WorkerRunner):
        gevent.spawn(checker, environment)