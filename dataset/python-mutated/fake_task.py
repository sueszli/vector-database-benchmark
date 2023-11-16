from pokemongo_bot.base_task import BaseTask

class FakeTask(BaseTask):
    SUPPORTED_TASK_API_VERSION = 1

    def work(self):
        if False:
            i = 10
            return i + 15
        return 'FakeTask'