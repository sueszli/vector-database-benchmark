from ..common import current_millis
from ..logging import get_logger
from .middleware import Middleware, SkipMessage

class AgeLimit(Middleware):
    """Middleware that drops messages that have been in the queue for
    too long.

    Parameters:
      max_age(int): The default message age limit in milliseconds.
        Defaults to ``None``, meaning that messages can exist
        indefinitely.
    """

    def __init__(self, *, max_age=None):
        if False:
            print('Hello World!')
        self.logger = get_logger(__name__, type(self))
        self.max_age = max_age

    @property
    def actor_options(self):
        if False:
            print('Hello World!')
        return {'max_age'}

    def before_process_message(self, broker, message):
        if False:
            return 10
        actor = broker.get_actor(message.actor_name)
        max_age = message.options.get('max_age') or actor.options.get('max_age', self.max_age)
        if not max_age:
            return
        if current_millis() - message.message_timestamp >= max_age:
            self.logger.warning('Message %r has exceeded its age limit.', message.message_id)
            message.fail()
            raise SkipMessage('Message age limit exceeded')