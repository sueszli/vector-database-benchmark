from locust import events, task, User, runners
from locust_plugins import csvreader

class DemoUser(User):
    reader: csvreader.CSVDictReader

    @task
    def t(self):
        if False:
            while True:
                i = 10
        thing = next(self.reader)
        print(thing)

@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    if False:
        while True:
            i = 10
    if not isinstance(environment.runner, runners.MasterRunner):
        DemoUser.reader = csvreader.CSVDictReader(f'mythings_{environment.runner.worker_index}.csv')