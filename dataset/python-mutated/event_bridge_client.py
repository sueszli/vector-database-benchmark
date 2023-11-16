import base64
import json
from hashlib import sha1
from metaflow.util import to_bytes, to_unicode

class EventBridgeClient(object):

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        from ..aws_client import get_aws_client
        self._client = get_aws_client('events')
        self.name = format(name)

    def cron(self, cron):
        if False:
            return 10
        self.cron = cron
        return self

    def role_arn(self, role_arn):
        if False:
            i = 10
            return i + 15
        self.role_arn = role_arn
        return self

    def state_machine_arn(self, state_machine_arn):
        if False:
            i = 10
            return i + 15
        self.state_machine_arn = state_machine_arn
        return self

    def schedule(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.cron:
            self._disable()
        else:
            self._set()
        return self.name

    def _disable(self):
        if False:
            i = 10
            return i + 15
        try:
            self._client.disable_rule(Name=self.name)
        except self._client.exceptions.ResourceNotFoundException:
            pass

    def _set(self):
        if False:
            i = 10
            return i + 15
        self._client.put_rule(Name=self.name, ScheduleExpression='cron(%s)' % self.cron, Description='Metaflow generated rule for %s' % self.name, State='ENABLED')
        self._client.put_targets(Rule=self.name, Targets=[{'Id': self.name, 'Arn': self.state_machine_arn, 'Input': json.dumps({'Parameters': json.dumps({})}), 'RoleArn': self.role_arn}])

    def delete(self):
        if False:
            i = 10
            return i + 15
        try:
            response = self._client.remove_targets(Rule=self.name, Ids=[self.name])
            if response.get('FailedEntryCount', 0) > 0:
                raise RuntimeError('Failed to remove targets from rule %s' % self.name)
            return self._client.delete_rule(Name=self.name)
        except self._client.exceptions.ResourceNotFoundException:
            return None

def format(name):
    if False:
        print('Hello World!')
    if len(name) > 64:
        name_hash = to_unicode(base64.b32encode(sha1(to_bytes(name)).digest()))[:16].lower()
        return '%s-%s' % (name[:47], name_hash)
    else:
        return name