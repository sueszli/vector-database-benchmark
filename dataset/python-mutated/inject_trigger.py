from __future__ import absolute_import
from st2common.runners.base_action import Action
__all__ = ['InjectTriggerAction']

class InjectTriggerAction(Action):

    def run(self, trigger=None, trigger_name=None, payload=None, trace_tag=None):
        if False:
            print('Hello World!')
        payload = payload or {}
        datastore_service = self.action_service.datastore_service
        client = datastore_service.get_api_client()
        if trigger and trigger_name:
            raise ValueError('Parameters `trigger` and `trigger_name` are mutually exclusive.')
        if not trigger and (not trigger_name):
            raise ValueError('You must include the `trigger_name` parameter.')
        trigger = trigger if trigger else trigger_name
        self.logger.debug('Injecting trigger "%s" with payload="%s"' % (trigger, str(payload)))
        result = client.webhooks.post_generic_webhook(trigger=trigger, payload=payload, trace_tag=trace_tag)
        return result