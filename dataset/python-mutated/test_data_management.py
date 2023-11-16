from locust.user.wait_time import constant
from locust import HttpUser, task
from locust import events
from locust.runners import MasterRunner
import requests
import datetime

def timestring():
    if False:
        while True:
            i = 10
    now = datetime.datetime.now()
    return datetime.datetime.strftime(now, '%m:%S.%f')[:-5]
print('1. Parsing locustfile, happens before anything else')
global_test_data = requests.post('https://postman-echo.com/post', data='global_test_data_' + timestring()).json()['data']
test_run_specific_data = None

@events.init.add_listener
def init(environment, **_kwargs):
    if False:
        i = 10
        return i + 15
    print('2. Initializing locust, happens after parsing the locustfile but before test start')

@events.quitting.add_listener
def quitting(environment, **_kwargs):
    if False:
        return 10
    print('9. locust is about to shut down')

@events.test_start.add_listener
def test_start(environment, **_kwargs):
    if False:
        for i in range(10):
            print('nop')
    global test_run_specific_data
    print('3. Starting test run')
    if not isinstance(environment.runner, MasterRunner):
        test_run_specific_data = requests.post('https://postman-echo.com/post', data='test-run-specific_' + timestring()).json()['data']

@events.quit.add_listener
def quit(exit_code, **kwargs):
    if False:
        return 10
    print(f'10. Locust has shut down with code {exit_code}')

@events.test_stopping.add_listener
def test_stopping(environment, **_kwargs):
    if False:
        while True:
            i = 10
    print('6. stopping test run')

@events.test_stop.add_listener
def test_stop(environment, **_kwargs):
    if False:
        while True:
            i = 10
    print('8. test run stopped')

class MyUser(HttpUser):
    host = 'https://postman-echo.com'
    wait_time = constant(180)
    first_start = True

    def on_start(self):
        if False:
            for i in range(10):
                print('nop')
        if MyUser.first_start:
            MyUser.first_start = False
            print("X. Here's where you would put things you want to run the first time a User is started")
        print('4. A user was started')
        self.user_specific_testdata = self.client.post('https://postman-echo.com/post', data='user-specific_' + timestring()).json()['data']

    @task
    def t(self):
        if False:
            while True:
                i = 10
        self.client.get(f'/get?{global_test_data}')
        self.client.get(f'/get?{test_run_specific_data}')
        self.client.get(f'/get?{self.user_specific_testdata}')
        print('5. Getting task-run-specific testdata')
        task_run_specific_testdata = self.client.post('https://postman-echo.com/post', data='task_run_specific_testdata_' + timestring()).json()['data']
        self.client.get(f'/get?{task_run_specific_testdata}')

    def on_stop(self):
        if False:
            while True:
                i = 10
        print('7. a user was stopped')