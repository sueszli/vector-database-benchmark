import logging
from typing import Optional
from aim.ext.notifier.base_notifier import BaseNotifier

class LoggingNotifier(BaseNotifier):

    def __init__(self, _id: str, config: dict):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(_id)
        self.message_template = config['message']
        self.logger = logging.getLogger('notifier')

    def notify(self, message: Optional[str]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        message_template = message or self.message_template
        msg = message_template.format(**kwargs)
        self.logger.error(msg)