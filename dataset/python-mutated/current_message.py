from threading import local
from .middleware import Middleware

class CurrentMessage(Middleware):
    """Middleware that exposes the current message via a thread-local
    variable.

    Example:
      >>> import dramatiq
      >>> from dramatiq.middleware import CurrentMessage

      >>> @dramatiq.actor
      ... def example(x):
      ...     print(CurrentMessage.get_current_message())
      ...
      >>> example.send(1)

    """
    STATE = local()

    @classmethod
    def get_current_message(cls):
        if False:
            i = 10
            return i + 15
        'Get the message that triggered the current actor.  Messages\n        are thread local so this returns ``None`` when called outside\n        of actor code.\n        '
        return getattr(cls.STATE, 'message', None)

    def before_process_message(self, broker, message):
        if False:
            return 10
        setattr(self.STATE, 'message', message)

    def after_process_message(self, broker, message, *, result=None, exception=None):
        if False:
            i = 10
            return i + 15
        delattr(self.STATE, 'message')