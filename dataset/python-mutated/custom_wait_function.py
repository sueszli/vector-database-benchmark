from locust import HttpUser, TaskSet, task
import random

def index(l):
    if False:
        print('Hello World!')
    l.client.get('/')

def stats(l):
    if False:
        for i in range(10):
            print('nop')
    l.client.get('/stats/requests')

class UserTasks(TaskSet):
    tasks = [index, stats]

    @task
    def page404(self):
        if False:
            return 10
        self.client.get('/does_not_exist')

class WebsiteUser(HttpUser):
    """
    User class that does requests to the locust web server running on localhost
    """
    host = 'http://127.0.0.1:8089'
    wait_time = lambda self: random.expovariate(1)
    tasks = [UserTasks]

def strictExp(min_wait, max_wait, mu=1):
    if False:
        while True:
            i = 10
    '\n    Returns an exponentially distributed time strictly between two bounds.\n    '
    while True:
        x = random.expovariate(mu)
        increment = (max_wait - min_wait) / (mu * 6.0)
        result = min_wait + x * increment
        if result < max_wait:
            break
    return result

class StrictWebsiteUser(HttpUser):
    """
    User class that makes exponential requests but strictly between two bounds.
    """
    host = 'http://127.0.0.1:8089'
    wait_time = lambda self: strictExp(3, 7)
    tasks = [UserTasks]