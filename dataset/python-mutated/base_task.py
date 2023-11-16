import logging
import time

class BaseTask(object):
    TASK_API_VERSION = 1

    def __init__(self, bot, config):
        if False:
            print('Hello World!')
        '\n\n    :param bot:\n    :type bot: pokemongo_bot.PokemonGoBot\n    :param config:\n    :return:\n    '
        self.bot = bot
        self.config = config
        self._validate_work_exists()
        self.logger = logging.getLogger(type(self).__name__)
        self.enabled = config.get('enabled', True)
        self.last_log_time = time.time()
        self.initialize()

    def _validate_work_exists(self):
        if False:
            i = 10
            return i + 15
        method = getattr(self, 'work', None)
        if not method or not callable(method):
            raise NotImplementedError('Missing "work" method')

    def emit_event(self, event, sender=None, level='info', formatted='', data={}):
        if False:
            for i in range(10):
                print('nop')
        if not sender:
            sender = self
        try:
            if time.time() - self.last_log_time >= self.config.get('log_interval', 0):
                self.last_log_time = time.time()
                self.bot.event_manager.emit(event, sender=sender, level=level, formatted=formatted, data=data)
        except AttributeError:
            if time.time() - self.last_log_time > 0:
                self.last_log_time = time.time()
                self.bot.event_manager.emit(event, sender=sender, level=level, formatted=formatted, data=data)

    def initialize(self):
        if False:
            return 10
        pass