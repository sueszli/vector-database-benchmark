import requests
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import parse_list
from ..AppriseLocale import gettext_lazy as _

class NotifyNextcloud(NotifyBase):
    """
    A wrapper for Nextcloud Notifications
    """
    service_name = 'Nextcloud'
    service_url = 'https://nextcloud.com/'
    protocol = 'ncloud'
    secure_protocol = 'nclouds'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_nextcloud'
    title_maxlen = 255
    body_maxlen = 4000
    templates = ('{schema}://{host}/{targets}', '{schema}://{host}:{port}/{targets}', '{schema}://{user}:{password}@{host}/{targets}', '{schema}://{user}:{password}@{host}:{port}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}, 'target_user': {'name': _('Target User'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'version': {'name': _('Version'), 'type': 'int', 'min': 1, 'default': 21}, 'url_prefix': {'name': _('URL Prefix'), 'type': 'string'}, 'to': {'alias_of': 'targets'}})
    template_kwargs = {'headers': {'name': _('HTTP Header'), 'prefix': '+'}}

    def __init__(self, targets=None, version=None, headers=None, url_prefix=None, **kwargs):
        if False:
            return 10
        '\n        Initialize Nextcloud Object\n        '
        super().__init__(**kwargs)
        self.targets = parse_list(targets)
        self.version = self.template_args['version']['default']
        if version is not None:
            try:
                self.version = int(version)
                if self.version < self.template_args['version']['min']:
                    raise ValueError()
            except (ValueError, TypeError):
                msg = 'At invalid Nextcloud version ({}) was specified.'.format(version)
                self.logger.warning(msg)
                raise TypeError(msg)
        self.url_prefix = '' if not url_prefix else url_prefix.strip('/')
        self.headers = {}
        if headers:
            self.headers.update(headers)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform Nextcloud Notification\n        '
        if len(self.targets) == 0:
            self.logger.warning('There were no Nextcloud targets to notify.')
            return False
        headers = {'User-Agent': self.app_id, 'OCS-APIREQUEST': 'true'}
        headers.update(self.headers)
        has_error = False
        targets = list(self.targets)
        while len(targets):
            target = targets.pop(0)
            payload = {'shortMessage': title if title else self.app_desc}
            if body:
                payload['longMessage'] = body
            auth = None
            if self.user:
                auth = (self.user, self.password)
            notify_url = '{schema}://{host}/{url_prefix}/ocs/v2.php/apps/admin_notifications/api/v1/notifications/{target}' if self.version < 21 else '{schema}://{host}/{url_prefix}/ocs/v2.php/apps/notifications/api/v2/admin_notifications/{target}'
            notify_url = notify_url.format(schema='https' if self.secure else 'http', host=self.host if not isinstance(self.port, int) else '{}:{}'.format(self.host, self.port), url_prefix=self.url_prefix, target=target)
            self.logger.debug('Nextcloud v%d POST URL: %s (cert_verify=%r)', self.version, notify_url, self.verify_certificate)
            self.logger.debug('Nextcloud v%d Payload: %s', self.version, str(payload))
            self.throttle()
            try:
                r = requests.post(notify_url, data=payload, headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyNextcloud.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send Nextcloud v{} notification:{}{}error={}.'.format(self.version, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Nextcloud %d notification.', self.version)
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Nextcloud v%dnotification.', self.version)
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'+{}'.format(k): v for (k, v) in self.headers.items()}
        params['version'] = str(self.version)
        if self.url_prefix:
            params['url_prefix'] = self.url_prefix
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyNextcloud.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=NotifyNextcloud.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}/{targets}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), targets='/'.join([NotifyNextcloud.quote(x) for x in self.targets]), params=NotifyNextcloud.urlencode(params))

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
        return targets if targets else 1

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        results['targets'] = NotifyNextcloud.split_path(results['fullpath'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyNextcloud.parse_list(results['qsd']['to'])
        if 'version' in results['qsd'] and len(results['qsd']['version']):
            results['version'] = NotifyNextcloud.unquote(results['qsd']['version'])
        if 'url_prefix' in results['qsd'] and len(results['qsd']['url_prefix']):
            results['url_prefix'] = NotifyNextcloud.unquote(results['qsd']['url_prefix'])
        results['headers'] = {NotifyNextcloud.unquote(x): NotifyNextcloud.unquote(y) for (x, y) in results['qsd+'].items()}
        return results