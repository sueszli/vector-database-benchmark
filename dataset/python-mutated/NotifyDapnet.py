from json import dumps
import requests
from requests.auth import HTTPBasicAuth
from .NotifyBase import NotifyBase
from ..AppriseLocale import gettext_lazy as _
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import is_call_sign
from ..utils import parse_call_sign
from ..utils import parse_list
from ..utils import parse_bool

class DapnetPriority:
    NORMAL = 0
    EMERGENCY = 1
DAPNET_PRIORITIES = {DapnetPriority.NORMAL: 'normal', DapnetPriority.EMERGENCY: 'emergency'}
DAPNET_PRIORITY_MAP = {'n': DapnetPriority.NORMAL, 'e': DapnetPriority.EMERGENCY, '0': DapnetPriority.NORMAL, '1': DapnetPriority.EMERGENCY}

class NotifyDapnet(NotifyBase):
    """
    A wrapper for DAPNET / Hampager Notifications
    """
    service_name = 'Dapnet'
    service_url = 'https://hampager.de/'
    secure_protocol = 'dapnet'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_dapnet'
    notify_url = 'http://www.hampager.de:8080/calls'
    body_maxlen = 80
    title_maxlen = 0
    default_batch_size = 50
    templates = ('{schema}://{user}:{password}@{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'user': {'name': _('User Name'), 'type': 'string', 'required': True}, 'password': {'name': _('Password'), 'type': 'string', 'private': True, 'required': True}, 'target_callsign': {'name': _('Target Callsign'), 'type': 'string', 'regex': ('^[a-z0-9]{2,5}(-[a-z0-9]{1,2})?$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'name': _('Target Callsign'), 'type': 'string', 'map_to': 'targets'}, 'priority': {'name': _('Priority'), 'type': 'choice:int', 'values': DAPNET_PRIORITIES, 'default': DapnetPriority.NORMAL}, 'txgroups': {'name': _('Transmitter Groups'), 'type': 'string', 'default': 'dl-all', 'private': True}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}})

    def __init__(self, targets=None, priority=None, txgroups=None, batch=False, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize Dapnet Object\n        '
        super().__init__(**kwargs)
        self.targets = list()
        self.priority = int(NotifyDapnet.template_args['priority']['default'] if priority is None else next((v for (k, v) in DAPNET_PRIORITY_MAP.items() if str(priority).lower().startswith(k)), NotifyDapnet.template_args['priority']['default']))
        if not (self.user and self.password):
            msg = 'A Dapnet user/pass was not provided.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.txgroups = parse_list(NotifyDapnet.template_args['txgroups']['default'] if not txgroups else txgroups)
        self.batch = batch
        for target in parse_call_sign(targets):
            result = is_call_sign(target)
            if not result:
                self.logger.warning('Dropping invalid Amateur radio call sign ({}).'.format(target))
                continue
            if result['callsign'] not in self.targets:
                self.targets.append(result['callsign'])
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform Dapnet Notification\n        '
        if not self.targets:
            self.logger.warning('There are no Amateur radio callsigns to notify')
            return False
        batch_size = 1 if not self.batch else self.default_batch_size
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json; charset=utf-8'}
        has_error = False
        targets = list(self.targets)
        for index in range(0, len(targets), batch_size):
            payload = {'text': body, 'callSignNames': targets[index:index + batch_size], 'transmitterGroupNames': self.txgroups, 'emergency': self.priority == DapnetPriority.EMERGENCY}
            self.logger.debug('DAPNET POST URL: %s' % self.notify_url)
            self.logger.debug('DAPNET Payload: %s' % dumps(payload))
            self.throttle()
            try:
                r = requests.post(self.notify_url, data=dumps(payload), headers=headers, auth=HTTPBasicAuth(username=self.user, password=self.password), verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.created:
                    self.logger.warning('Failed to send DAPNET notification {} to {}: error={}.'.format(payload['text'], ' to {}'.format(self.targets), r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                else:
                    self.logger.info("Sent '{}' DAPNET notification {}".format(payload['text'], 'to {}'.format(self.targets)))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending DAPNET notification to {}'.format(self.targets))
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'priority': DAPNET_PRIORITIES[self.template_args['priority']['default']] if self.priority not in DAPNET_PRIORITIES else DAPNET_PRIORITIES[self.priority], 'batch': 'yes' if self.batch else 'no', 'txgroups': ','.join(self.txgroups)}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = '{user}:{password}@'.format(user=NotifyDapnet.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        return '{schema}://{auth}{targets}?{params}'.format(schema=self.secure_protocol, auth=auth, targets='/'.join([self.pprint(x, privacy, safe='') for x in self.targets]), params=NotifyDapnet.urlencode(params))

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of targets associated with this notification\n        '
        batch_size = 1 if not self.batch else self.default_batch_size
        targets = len(self.targets)
        if batch_size > 1:
            targets = int(targets / batch_size) + (1 if targets % batch_size else 0)
        return targets

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = [NotifyDapnet.unquote(results['host'])]
        results['targets'].extend(NotifyDapnet.split_path(results['fullpath']))
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyDapnet.parse_list(results['qsd']['to'])
        if 'priority' in results['qsd'] and len(results['qsd']['priority']):
            results['priority'] = NotifyDapnet.unquote(results['qsd']['priority'])
        if 'txgroups' in results['qsd']:
            results['txgroups'] = [x.lower() for x in NotifyDapnet.parse_list(results['qsd']['txgroups'])]
        results['batch'] = parse_bool(results['qsd'].get('batch', NotifyDapnet.template_args['batch']['default']))
        return results