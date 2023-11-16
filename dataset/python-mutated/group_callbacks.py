import os
from ..rate_limits import Barrier
from .middleware import Middleware
GROUP_CALLBACK_BARRIER_TTL = int(os.getenv('dramatiq_group_callback_barrier_ttl', '86400000'))

class GroupCallbacks(Middleware):

    def __init__(self, rate_limiter_backend):
        if False:
            while True:
                i = 10
        self.rate_limiter_backend = rate_limiter_backend

    def after_process_message(self, broker, message, *, result=None, exception=None):
        if False:
            for i in range(10):
                print('nop')
        from ..message import Message
        if exception is None:
            group_completion_uuid = message.options.get('group_completion_uuid')
            group_completion_callbacks = message.options.get('group_completion_callbacks')
            if group_completion_uuid and group_completion_callbacks:
                barrier = Barrier(self.rate_limiter_backend, group_completion_uuid, ttl=GROUP_CALLBACK_BARRIER_TTL)
                if barrier.wait(block=False):
                    for message in group_completion_callbacks:
                        broker.enqueue(Message(**message))