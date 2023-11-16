from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from airflow.providers.redis.hooks.redis import RedisHook
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class RedisKeySensor(BaseSensorOperator):
    """Checks for the existence of a key in a Redis."""
    template_fields: Sequence[str] = ('key',)
    ui_color = '#f0eee4'

    def __init__(self, *, key: str, redis_conn_id: str, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.redis_conn_id = redis_conn_id
        self.key = key

    def poke(self, context: Context) -> bool:
        if False:
            return 10
        self.log.info('Sensor checks for existence of key: %s', self.key)
        return RedisHook(self.redis_conn_id).get_conn().exists(self.key)