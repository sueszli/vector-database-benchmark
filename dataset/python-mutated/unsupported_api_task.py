from pokemongo_bot.base_task import BaseTask

class UnsupportedApiTask(BaseTask):
    SUPPORTED_TASK_API_VERSION = 2

    def work():
        if False:
            print('Hello World!')
        return 2