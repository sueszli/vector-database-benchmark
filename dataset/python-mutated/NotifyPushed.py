import re
import requests
from json import dumps
from itertools import chain
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
IS_CHANNEL = re.compile('^#?(?P<name>[A-Za-z0-9]+)$')
IS_USER_PUSHED_ID = re.compile('^@(?P<name>[A-Za-z0-9]+)$')

class NotifyPushed(NotifyBase):
    """
    A wrapper to Pushed Notifications

    """
    service_name = 'Pushed'
    service_url = 'https://pushed.co/'
    secure_protocol = 'pushed'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_pushed'
    notify_url = 'https://api.pushed.co/1/push'
    title_maxlen = 0
    body_maxlen = 160
    templates = ('{schema}://{app_key}/{app_secret}', '{schema}://{app_key}/{app_secret}@{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'app_key': {'name': _('Application Key'), 'type': 'string', 'private': True, 'required': True}, 'app_secret': {'name': _('Application Secret'), 'type': 'string', 'private': True, 'required': True}, 'target_user': {'name': _('Target User'), 'prefix': '@', 'type': 'string', 'map_to': 'targets'}, 'target_channel': {'name': _('Target Channel'), 'type': 'string', 'prefix': '#', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}})

    def __init__(self, app_key, app_secret, targets=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Pushed Object\n\n        '
        super().__init__(**kwargs)
        self.app_key = validate_regex(app_key)
        if not self.app_key:
            msg = 'An invalid Pushed Application Key ({}) was specified.'.format(app_key)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.app_secret = validate_regex(app_secret)
        if not self.app_secret:
            msg = 'An invalid Pushed Application Secret ({}) was specified.'.format(app_secret)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.channels = list()
        self.users = list()
        targets = parse_list(targets)
        if targets:
            for target in targets:
                result = IS_CHANNEL.match(target)
                if result:
                    self.channels.append(result.group('name'))
                    continue
                result = IS_USER_PUSHED_ID.match(target)
                if result:
                    self.users.append(result.group('name'))
                    continue
                self.logger.warning('Dropped invalid channel/userid (%s) specified.' % target)
            if len(self.channels) + len(self.users) == 0:
                msg = 'No Pushed targets to notify.'
                self.logger.warning(msg)
                raise TypeError(msg)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform Pushed Notification\n        '
        has_error = False
        payload = {'app_key': self.app_key, 'app_secret': self.app_secret, 'target_type': 'app', 'content': body}
        if len(self.channels) + len(self.users) == 0:
            return self._send(payload=payload, notify_type=notify_type, **kwargs)
        channels = list(self.channels)
        users = list(self.users)
        _payload = dict(payload)
        _payload['target_type'] = 'channel'
        while len(channels) > 0:
            _payload['target_alias'] = channels.pop(0)
            if not self._send(payload=_payload, notify_type=notify_type, **kwargs):
                has_error = True
        _payload = dict(payload)
        _payload['target_type'] = 'pushed_id'
        while len(users):
            _payload['pushed_id'] = users.pop(0)
            if not self._send(payload=_payload, notify_type=notify_type, **kwargs):
                has_error = True
        return not has_error

    def _send(self, payload, notify_type, **kwargs):
        if False:
            print('Hello World!')
        '\n        A lower level call that directly pushes a payload to the Pushed\n        Notification servers.  This should never be called directly; it is\n        referenced automatically through the send() function.\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        self.logger.debug('Pushed POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
        self.logger.debug('Pushed Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(self.notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyPushed.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Pushed notification:{}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Pushed notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Pushed notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        return '{schema}://{app_key}/{app_secret}/{targets}/?{params}'.format(schema=self.secure_protocol, app_key=self.pprint(self.app_key, privacy, safe=''), app_secret=self.pprint(self.app_secret, privacy, mode=PrivacyMode.Secret, safe=''), targets='/'.join([NotifyPushed.quote(x) for x in chain(['#{}'.format(x) for x in self.channels], ['@{}'.format(x) for x in self.users])]), params=NotifyPushed.urlencode(params))

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.channels) + len(self.users)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        app_key = NotifyPushed.unquote(results['host'])
        entries = NotifyPushed.split_path(results['fullpath'])
        try:
            app_secret = entries.pop(0)
        except IndexError:
            app_secret = None
            app_key = None
        results['targets'] = entries
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyPushed.parse_list(results['qsd']['to'])
        results['app_key'] = app_key
        results['app_secret'] = app_secret
        return results