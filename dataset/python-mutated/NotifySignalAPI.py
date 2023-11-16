import re
import requests
from json import dumps
import base64
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import parse_phone_no
from ..utils import parse_bool
from ..URLBase import PrivacyMode
from ..AppriseLocale import gettext_lazy as _
GROUP_REGEX = re.compile('^\\s*((\\@|\\%40)?(group\\.)|\\@|\\%40)(?P<group>[a-z0-9_=-]+)', re.I)

class NotifySignalAPI(NotifyBase):
    """
    A wrapper for SignalAPI Notifications
    """
    service_name = 'Signal API'
    service_url = 'https://bbernhard.github.io/signal-cli-rest-api/'
    protocol = 'signal'
    secure_protocol = 'signals'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_signal'
    attachment_support = True
    default_batch_size = 10
    title_maxlen = 0
    templates = ('{schema}://{host}/{from_phone}', '{schema}://{host}:{port}/{from_phone}', '{schema}://{user}@{host}/{from_phone}', '{schema}://{user}@{host}:{port}/{from_phone}', '{schema}://{user}:{password}@{host}/{from_phone}', '{schema}://{user}:{password}@{host}:{port}/{from_phone}', '{schema}://{host}/{from_phone}/{targets}', '{schema}://{host}:{port}/{from_phone}/{targets}', '{schema}://{user}@{host}/{from_phone}/{targets}', '{schema}://{user}@{host}:{port}/{from_phone}/{targets}', '{schema}://{user}:{password}@{host}/{from_phone}/{targets}', '{schema}://{user}:{password}@{host}:{port}/{from_phone}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}, 'from_phone': {'name': _('From Phone No'), 'type': 'string', 'required': True, 'regex': ('^\\+?[0-9\\s)(+-]+$', 'i'), 'map_to': 'source'}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'target_channel': {'name': _('Target Group ID'), 'type': 'string', 'prefix': '@', 'regex': ('^[a-z0-9_=-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'from': {'alias_of': 'from_phone'}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}, 'status': {'name': _('Show Status'), 'type': 'bool', 'default': False}})

    def __init__(self, source=None, targets=None, batch=False, status=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize SignalAPI Object\n        '
        super().__init__(**kwargs)
        self.batch = batch
        self.status = status
        self.targets = list()
        self.invalid_targets = list()
        result = is_phone_no(source)
        if not result:
            msg = 'An invalid Signal API Source Phone No ({}) was provided.'.format(source)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.source = '+{}'.format(result['full'])
        if targets:
            for target in parse_phone_no(targets):
                result = is_phone_no(target)
                if result:
                    self.targets.append('+{}'.format(result['full']))
                    continue
                result = GROUP_REGEX.match(target)
                if result:
                    self.targets.append('group.{}'.format(result.group('group')))
                    continue
                self.logger.warning('Dropped invalid phone/group ({}) specified.'.format(target))
                self.invalid_targets.append(target)
                continue
        else:
            self.targets.append(self.source)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform Signal API Notification\n        '
        if len(self.targets) == 0:
            self.logger.warning('There were no Signal API targets to notify.')
            return False
        has_error = False
        attachments = []
        if attach and self.attachment_support:
            for attachment in attach:
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                try:
                    with open(attachment.path, 'rb') as f:
                        attachments.append(base64.b64encode(f.read()).decode('utf-8'))
                except (OSError, IOError) as e:
                    self.logger.warning('An I/O error occurred while reading {}.'.format(attachment.name if attachment else 'attachment'))
                    self.logger.debug('I/O Exception: %s' % str(e))
                    return False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        payload = {'message': '{}{}'.format('' if not self.status else '{} '.format(self.asset.ascii(notify_type)), body).rstrip(), 'number': self.source, 'recipients': []}
        if attachments:
            payload['base64_attachments'] = attachments
        auth = None
        if self.user:
            auth = (self.user, self.password)
        schema = 'https' if self.secure else 'http'
        notify_url = '%s://%s' % (schema, self.host)
        if isinstance(self.port, int):
            notify_url += ':%d' % self.port
        notify_url += '/v2/send'
        batch_size = 1 if not self.batch else self.default_batch_size
        for index in range(0, len(self.targets), batch_size):
            payload['recipients'] = self.targets[index:index + batch_size]
            self.logger.debug('Signal API POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
            self.logger.debug('Signal API Payload: %s' % str(payload))
            self.throttle()
            try:
                r = requests.post(notify_url, auth=auth, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code not in (requests.codes.ok, requests.codes.created):
                    status_str = NotifySignalAPI.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send {} Signal API notification{}: {}{}error={}.'.format(len(self.targets[index:index + batch_size]), ' to {}'.format(self.targets[index]) if batch_size == 1 else '(s)', status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent {} Signal API notification{}.'.format(len(self.targets[index:index + batch_size]), ' to {}'.format(self.targets[index]) if batch_size == 1 else '(s)'))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occured sending {} Signal API notification(s).'.format(len(self.targets[index:index + batch_size])))
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'batch': 'yes' if self.batch else 'no', 'status': 'yes' if self.status else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifySignalAPI.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=NotifySignalAPI.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        if len(self.targets) == 1 and self.source in self.targets:
            targets = []
        elif len(self.targets) == 0:
            targets = self.invalid_targets
        else:
            targets = ['@{}'.format(x[6:]) if x[0] != '+' else x for x in self.targets]
        return '{schema}://{auth}{hostname}{port}/{src}/{dst}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), src=self.source, dst='/'.join([NotifySignalAPI.quote(x, safe='@+') for x in targets]), params=NotifySignalAPI.urlencode(params))

    def __len__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of targets associated with this notification\n        '
        batch_size = 1 if not self.batch else self.default_batch_size
        targets = len(self.targets)
        if batch_size > 1:
            targets = int(targets / batch_size) + (1 if targets % batch_size else 0)
        return targets

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifySignalAPI.split_path(results['fullpath'])
        results['apikey'] = NotifySignalAPI.unquote(results['host'])
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['source'] = NotifySignalAPI.unquote(results['qsd']['from'])
        elif results['targets']:
            results['source'] = results['targets'].pop(0)
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifySignalAPI.parse_phone_no(results['qsd']['to'])
        results['batch'] = parse_bool(results['qsd'].get('batch', False))
        results['status'] = parse_bool(results['qsd'].get('status', False))
        return results