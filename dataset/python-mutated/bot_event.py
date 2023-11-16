import logging
from raven import Client
import raven
import uuid
import requests
import time
from pokemongo_bot.base_dir import _base_dir

class BotEvent(object):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        self.config = config
        self.logger = logging.getLogger(__name__)
        client_uuid = uuid.uuid4()
        self.client_id = str(client_uuid)
        if self.config.health_record:
            self.logger.info('Health check is enabled. For more information:')
            self.logger.info('https://github.com/PokemonGoF/PokemonGo-Bot/tree/dev#analytics')
            self.client = Client(dsn='https://8abac56480f34b998813d831de262514:196ae1d8dced41099f8253ea2c8fe8e6@app.getsentry.com/90254', name='PokemonGof-Bot', processors=('raven.processors.SanitizePasswordsProcessor', 'raven.processors.RemoveStackLocalsProcessor'), install_logging_hook=False, hook_libraries=(), enable_breadcrumbs=False, logging=False, context={})
        self.heartbeat_wait = 15 * 60
        self.last_heartbeat = time.time()

    def capture_error(self):
        if False:
            while True:
                i = 10
        if self.config.health_record:
            self.client.captureException()

    def login_success(self):
        if False:
            for i in range(10):
                print('nop')
        if self.config.health_record:
            self.last_heartbeat = time.time()
            self.track_url('/loggedin')

    def login_failed(self):
        if False:
            print('Hello World!')
        if self.config.health_record:
            self.track_url('/login')

    def login_retry(self):
        if False:
            for i in range(10):
                print('nop')
        if self.config.health_record:
            self.track_url('/relogin')

    def logout(self):
        if False:
            return 10
        if self.config.health_record:
            self.track_url('/logout')

    def heartbeat(self):
        if False:
            for i in range(10):
                print('nop')
        if self.config.health_record:
            current_time = time.time()
            if current_time - self.heartbeat_wait > self.last_heartbeat:
                self.last_heartbeat = current_time
                self.track_url('/heartbeat')

    def track_url(self, path):
        if False:
            while True:
                i = 10
        data = {'v': '1', 'tid': 'UA-81469507-1', 'aip': '1', 'cid': self.client_id, 't': 'pageview', 'dp': path}
        try:
            response = requests.post('http://www.google-analytics.com/collect', data=data)
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            pass