from locust import HttpUser, TaskSet, task, events, between
from gevent.lock import Semaphore
all_users_spawned = Semaphore()
all_users_spawned.acquire()

@events.init.add_listener
def _(environment, **kw):
    if False:
        return 10

    @environment.events.spawning_complete.add_listener
    def on_spawning_complete(**kw):
        if False:
            i = 10
            return i + 15
        all_users_spawned.release()

class UserTasks(TaskSet):

    def on_start(self):
        if False:
            while True:
                i = 10
        all_users_spawned.wait()
        self.wait()

    @task
    def index(self):
        if False:
            return 10
        self.client.get('/')

class WebsiteUser(HttpUser):
    host = 'http://127.0.0.1:8089'
    wait_time = between(2, 5)
    tasks = [UserTasks]