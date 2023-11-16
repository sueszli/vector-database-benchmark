import requests
from json import dumps
from uuid import uuid4
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotifyHomeAssistant(NotifyBase):
    """
    A wrapper for Home Assistant Notifications
    """
    service_name = 'HomeAssistant'
    service_url = 'https://www.home-assistant.io/'
    protocol = 'hassio'
    secure_protocol = 'hassios'
    default_insecure_port = 8123
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_homeassistant'
    templates = ('{schema}://{host}/{accesstoken}', '{schema}://{host}:{port}/{accesstoken}', '{schema}://{user}@{host}/{accesstoken}', '{schema}://{user}@{host}:{port}/{accesstoken}', '{schema}://{user}:{password}@{host}/{accesstoken}', '{schema}://{user}:{password}@{host}:{port}/{accesstoken}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}, 'accesstoken': {'name': _('Long-Lived Access Token'), 'type': 'string', 'private': True, 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'nid': {'name': _('Notification ID'), 'type': 'string', 'regex': ('^[a-z0-9_-]+$', 'i')}})

    def __init__(self, accesstoken, nid=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize Home Assistant Object\n        '
        super().__init__(**kwargs)
        self.fullpath = kwargs.get('fullpath', '')
        if not (self.secure or self.port):
            self.port = self.default_insecure_port
        self.accesstoken = validate_regex(accesstoken)
        if not self.accesstoken:
            msg = 'An invalid Home Assistant Long-Lived Access Token ({}) was specified.'.format(accesstoken)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.nid = None
        if nid:
            self.nid = validate_regex(nid, *self.template_args['nid']['regex'])
            if not self.nid:
                msg = 'An invalid Home Assistant Notification Identifier ({}) was specified.'.format(nid)
                self.logger.warning(msg)
                raise TypeError(msg)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Sends Message\n        '
        payload = {'title': title, 'message': body, 'notification_id': self.nid if self.nid else str(uuid4())}
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'Authorization': 'Bearer {}'.format(self.accesstoken)}
        auth = None
        if self.user:
            auth = (self.user, self.password)
        schema = 'https' if self.secure else 'http'
        url = '{}://{}'.format(schema, self.host)
        if isinstance(self.port, int):
            url += ':%d' % self.port
        url += '' if not self.fullpath else '/' + self.fullpath.strip('/')
        url += '/api/services/persistent_notification/create'
        self.logger.debug('Home Assistant POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('Home Assistant Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(url, data=dumps(payload), headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyHomeAssistant.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Home Assistant notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Home Assistant notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Home Assistant notification to %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {}
        if self.nid:
            params['nid'] = self.nid
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyHomeAssistant.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=NotifyHomeAssistant.quote(self.user, safe=''))
        default_port = 443 if self.secure else self.default_insecure_port
        url = '{schema}://{auth}{hostname}{port}{fullpath}{accesstoken}/?{params}'
        return url.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if not self.port or self.port == default_port else ':{}'.format(self.port), fullpath='/' if not self.fullpath else '/{}/'.format(NotifyHomeAssistant.quote(self.fullpath.strip('/'), safe='/')), accesstoken=self.pprint(self.accesstoken, privacy, safe=''), params=NotifyHomeAssistant.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        if 'accesstoken' in results['qsd'] and len(results['qsd']['accesstoken']):
            results['accesstoken'] = NotifyHomeAssistant.unquote(results['qsd']['accesstoken'])
        else:
            fullpath = NotifyHomeAssistant.split_path(results['fullpath'])
            results['accesstoken'] = fullpath.pop() if fullpath else None
            results['fullpath'] = '/'.join(fullpath)
        if 'nid' in results['qsd'] and len(results['qsd']['nid']):
            results['nid'] = NotifyHomeAssistant.unquote(results['qsd']['nid'])
        return results