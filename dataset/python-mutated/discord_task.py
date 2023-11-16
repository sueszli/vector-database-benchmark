from pokemongo_bot.base_task import BaseTask
from pokemongo_bot.event_handlers import DiscordHandler

class FileIOException(Exception):
    pass

class DiscordTask(BaseTask):
    SUPPORTED_TASK_API_VERSION = 1

    def initialize(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.enabled:
            return
        self.bot.event_manager.add_handler(DiscordHandler(self.bot, self.config))

    def work(self):
        if False:
            i = 10
            return i + 15
        if not self.enabled:
            return