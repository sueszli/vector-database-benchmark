from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from airflow.models import BaseOperator
from airflow.providers.redis.hooks.redis import RedisHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class RedisPublishOperator(BaseOperator):
    """
    Publish a message to Redis.

    :param channel: redis channel to which the message is published (templated)
    :param message: the message to publish (templated)
    :param redis_conn_id: redis connection to use
    """
    template_fields: Sequence[str] = ('channel', 'message')

    def __init__(self, *, channel: str, message: str, redis_conn_id: str='redis_default', **kwargs) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        self.redis_conn_id = redis_conn_id
        self.channel = channel
        self.message = message

    def execute(self, context: Context) -> None:
        if False:
            return 10
        '\n        Publish the message to Redis channel.\n\n        :param context: the context object\n        '
        redis_hook = RedisHook(redis_conn_id=self.redis_conn_id)
        self.log.info('Sending message %s to Redis on channel %s', self.message, self.channel)
        result = redis_hook.get_conn().publish(channel=self.channel, message=self.message)
        self.log.info('Result of publishing %s', result)