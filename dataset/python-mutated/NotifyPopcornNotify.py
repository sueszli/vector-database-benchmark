import requests
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import is_email
from ..utils import is_phone_no
from ..utils import parse_list
from ..utils import parse_bool
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotifyPopcornNotify(NotifyBase):
    """
    A wrapper for PopcornNotify Notifications
    """
    service_name = 'PopcornNotify'
    service_url = 'https://popcornnotify.com/'
    secure_protocol = 'popcorn'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_popcornnotify'
    notify_url = 'https://popcornnotify.com/notify'
    default_batch_size = 10
    templates = ('{schema}://{apikey}/{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('API Key'), 'type': 'string', 'regex': ('^[a-z0-9]+$', 'i'), 'required': True}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'target_email': {'name': _('Target Email'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}})

    def __init__(self, apikey, targets=None, batch=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize PopcornNotify Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey, *self.template_tokens['apikey']['regex'])
        if not self.apikey:
            msg = 'An invalid PopcornNotify API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.batch = batch
        self.targets = list()
        for target in parse_list(targets):
            result = is_phone_no(target)
            if result:
                self.targets.append(result['full'])
                continue
            result = is_email(target)
            if result:
                self.targets.append(result['full_email'])
                continue
            self.logger.warning('Dropped invalid target ({}) specified.'.format(target))

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform PopcornNotify Notification\n        '
        if len(self.targets) == 0:
            self.logger.warning('There were no PopcornNotify targets to notify.')
            return False
        has_error = False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'message': body, 'subject': title}
        auth = (self.apikey, None)
        batch_size = 1 if not self.batch else self.default_batch_size
        for index in range(0, len(self.targets), batch_size):
            payload['recipients'] = ','.join(self.targets[index:index + batch_size])
            self.logger.debug('PopcornNotify POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
            self.logger.debug('PopcornNotify Payload: %s' % str(payload))
            self.throttle()
            try:
                r = requests.post(self.notify_url, auth=auth, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyPopcornNotify.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send {} PopcornNotify notification{}: {}{}error={}.'.format(len(self.targets[index:index + batch_size]), ' to {}'.format(self.targets[index]) if batch_size == 1 else '(s)', status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent {} PopcornNotify notification{}.'.format(len(self.targets[index:index + batch_size]), ' to {}'.format(self.targets[index]) if batch_size == 1 else '(s)'))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occured sending {} PopcornNotify notification(s).'.format(len(self.targets[index:index + batch_size])))
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'batch': 'yes' if self.batch else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{apikey}/{targets}/?{params}'.format(schema=self.secure_protocol, apikey=self.pprint(self.apikey, privacy, safe=''), targets='/'.join([NotifyPopcornNotify.quote(x, safe='') for x in self.targets]), params=NotifyPopcornNotify.urlencode(params))

    def __len__(self):
        if False:
            return 10
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
        results['targets'] = NotifyPopcornNotify.split_path(results['fullpath'])
        results['apikey'] = NotifyPopcornNotify.unquote(results['host'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyPopcornNotify.parse_list(results['qsd']['to'])
        results['batch'] = parse_bool(results['qsd'].get('batch', False))
        return results