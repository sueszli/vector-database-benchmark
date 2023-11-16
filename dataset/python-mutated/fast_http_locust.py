from locust import FastHttpUser, task

class WebsiteUser(FastHttpUser):
    """
    User class that does requests to the locust web server running on localhost,
    using the fast HTTP client
    """
    host = 'http://127.0.0.1:8089'

    @task
    def index(self):
        if False:
            i = 10
            return i + 15
        self.client.get('/')

    @task
    def stats(self):
        if False:
            while True:
                i = 10
        self.client.get('/stats/requests')