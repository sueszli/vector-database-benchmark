from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING, Sequence
from airflow.providers.redis.hooks.redis import RedisHook
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class RedisPubSubSensor(BaseSensorOperator):
    """
    Redis sensor for reading a message from pub sub channels.

    :param channels: The channels to be subscribed to (templated)
    :param redis_conn_id: the redis connection id
    """
    template_fields: Sequence[str] = ('channels',)
    ui_color = '#f0eee4'

    def __init__(self, *, channels: list[str] | str, redis_conn_id: str, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.channels = channels
        self.redis_conn_id = redis_conn_id

    @cached_property
    def pubsub(self):
        if False:
            return 10
        hook = RedisHook(redis_conn_id=self.redis_conn_id).get_conn().pubsub()
        hook.subscribe(self.channels)
        return hook

    def poke(self, context: Context) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "\n        Check for message on subscribed channels and write to xcom the message with key ``message``.\n\n        An example of message ``{'type': 'message', 'pattern': None, 'channel': b'test', 'data': b'hello'}``\n\n        :param context: the context object\n        :return: ``True`` if message (with type 'message') is available or ``False`` if not\n        "
        self.log.info('RedisPubSubSensor checking for message on channels: %s', self.channels)
        message = self.pubsub.get_message()
        self.log.info('Message %s from channel %s', message, self.channels)
        if message and message['type'] == 'message':
            context['ti'].xcom_push(key='message', value=message)
            self.pubsub.unsubscribe(self.channels)
            return True
        return False