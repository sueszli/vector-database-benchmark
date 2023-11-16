import gevent
from locust import HttpUser, task, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
setup_logging('INFO', None)

class MyUser(HttpUser):
    host = 'https://docs.locust.io'

    @task
    def t(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.get('/')
env = Environment(user_classes=[MyUser], events=events)
runner = env.create_local_runner()
web_ui = env.create_web_ui('127.0.0.1', 8089)
env.events.init.fire(environment=env, runner=runner, web_ui=web_ui)
gevent.spawn(stats_printer(env.stats))
gevent.spawn(stats_history, env.runner)
runner.start(1, spawn_rate=10)
gevent.spawn_later(60, lambda : runner.quit())
runner.greenlet.join()
web_ui.stop()