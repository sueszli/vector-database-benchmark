from __future__ import annotations
from typing import Union, cast
from uuid import uuid4
from django.conf import settings
from sentry.utils import json
from sentry.utils.json import JSONData
from sentry.utils.redis import redis_clusters
SLACK_FAILED_MESSAGE = 'The slack resource does not exist or has not been granted access in that workspace.'

class RedisRuleStatus:

    def __init__(self, uuid: str | None=None) -> None:
        if False:
            return 10
        self._uuid = uuid or self._generate_uuid()
        cluster_id = getattr(settings, 'SENTRY_RULE_TASK_REDIS_CLUSTER', 'default')
        self.client = redis_clusters.get(cluster_id)
        self._set_initial_value()

    @property
    def uuid(self) -> str:
        if False:
            return 10
        return self._uuid

    def set_value(self, status: str, rule_id: int | None=None, error_message: str | None=None) -> None:
        if False:
            i = 10
            return i + 15
        value = self._format_value(status, rule_id, error_message)
        self.client.set(self._get_redis_key(), f'{value}', ex=60 * 60)

    def get_value(self) -> JSONData:
        if False:
            while True:
                i = 10
        key = self._get_redis_key()
        value = self.client.get(key)
        return json.loads(cast(Union[str, bytes], value))

    def _generate_uuid(self) -> str:
        if False:
            return 10
        return uuid4().hex

    def _set_initial_value(self) -> None:
        if False:
            i = 10
            return i + 15
        value = json.dumps({'status': 'pending'})
        self.client.set(self._get_redis_key(), f'{value}', ex=60 * 60, nx=True)

    def _get_redis_key(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'slack-channel-task:1:{self.uuid}'

    def _format_value(self, status: str, rule_id: int | None, error_message: str | None) -> str:
        if False:
            print('Hello World!')
        value = {'status': status}
        if rule_id:
            value['rule_id'] = str(rule_id)
        if error_message:
            value['error'] = error_message
        elif status == 'failed':
            value['error'] = SLACK_FAILED_MESSAGE
        return json.dumps(value)