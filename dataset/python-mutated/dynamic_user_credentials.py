from locust import HttpUser, TaskSet, task, between
USER_CREDENTIALS = [('user1', 'password'), ('user2', 'password'), ('user3', 'password')]

class UserBehaviour(TaskSet):

    def on_start(self):
        if False:
            print('Hello World!')
        if len(USER_CREDENTIALS) > 0:
            (user, passw) = USER_CREDENTIALS.pop()
            self.client.post('/login', {'username': user, 'password': passw})

    @task
    def some_task(self):
        if False:
            while True:
                i = 10
        self.client.get('/protected/resource')

class User(HttpUser):
    tasks = [UserBehaviour]
    wait_time = between(5, 60)