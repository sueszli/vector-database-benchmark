import os
from locust import HttpUser, TaskSet, task, between
from locust.clients import HttpSession

class MultipleHostsUser(HttpUser):
    abstract = True

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.api_client = HttpSession(base_url=os.environ['API_HOST'], request_event=self.client.request_event, user=self)

class UserTasks(TaskSet):

    @task
    def index(self):
        if False:
            i = 10
            return i + 15
        self.user.client.get('/')

    @task
    def index_other_host(self):
        if False:
            i = 10
            return i + 15
        self.user.api_client.get('/stats/requests')

class WebsiteUser(MultipleHostsUser):
    """
    User class that does requests to the locust web server running on localhost
    """
    host = 'http://127.0.0.1:8089'
    wait_time = between(2, 5)
    tasks = [UserTasks]