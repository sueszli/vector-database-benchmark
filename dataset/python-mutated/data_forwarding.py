import logging
from typing import Any, MutableMapping
from sentry import ratelimits, tsdb
from sentry.api.serializers import serialize
from sentry.eventstore.models import Event
from sentry.plugins.base import Plugin
from sentry.plugins.base.configuration import react_plugin_config
from sentry.tsdb.base import TSDBModel
logger = logging.getLogger(__name__)

class DataForwardingPlugin(Plugin):

    def configure(self, project, request):
        if False:
            i = 10
            return i + 15
        return react_plugin_config(self, project, request)

    def has_project_conf(self):
        if False:
            return 10
        return True

    def get_rate_limit(self):
        if False:
            while True:
                i = 10
        '\n        Returns a tuple of (Number of Requests, Window in Seconds)\n        '
        return (50, 1)

    def forward_event(self, event: Event, payload: MutableMapping[str, Any]) -> bool:
        if False:
            return 10
        'Forward the event and return a boolean if it was successful.'
        raise NotImplementedError

    def get_event_payload(self, event):
        if False:
            print('Hello World!')
        return serialize(event)

    def get_plugin_type(self):
        if False:
            for i in range(10):
                print('nop')
        return 'data-forwarding'

    def get_rl_key(self, event):
        if False:
            while True:
                i = 10
        return f'{self.conf_key}:{event.project.organization_id}'

    def initialize_variables(self, event):
        if False:
            print('Hello World!')
        return

    def is_ratelimited(self, event):
        if False:
            i = 10
            return i + 15
        self.initialize_variables(event)
        rl_key = self.get_rl_key(event)
        (limit, window) = self.get_rate_limit()
        if limit and window and ratelimits.is_limited(rl_key, limit=limit, window=window):
            logger.info('data_forwarding.skip_rate_limited', extra={'event_id': event.event_id, 'issue_id': event.group_id, 'project_id': event.project_id, 'organization_id': event.project.organization_id})
            return True
        return False

    def post_process(self, event, **kwargs):
        if False:
            return 10
        if self.is_ratelimited(event):
            return
        payload = self.get_event_payload(event)
        success = self.forward_event(event, payload)
        if success is False:
            pass
        tsdb.incr(TSDBModel.project_total_forwarded, event.project.id, count=1)