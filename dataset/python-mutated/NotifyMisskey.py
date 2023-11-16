import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class MisskeyVisibility:
    """
    The visibility of any note created
    """
    PUBLIC = 'public'
    HOME = 'home'
    FOLLOWERS = 'followers'
    PRIVATE = 'private'
    SPECIFIED = 'specified'
MISSKEY_VISIBILITIES = (MisskeyVisibility.PUBLIC, MisskeyVisibility.HOME, MisskeyVisibility.FOLLOWERS, MisskeyVisibility.PRIVATE, MisskeyVisibility.SPECIFIED)

class NotifyMisskey(NotifyBase):
    """
    A wrapper for Misskey Notifications
    """
    service_name = 'Misskey'
    service_url = 'https://misskey-hub.net/'
    protocol = 'misskey'
    secure_protocol = 'misskeys'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_misskey'
    title_maxlen = 0
    body_maxlen = 512
    templates = ('{schema}://{project_id}/{msghook}',)
    templates = ('{schema}://{token}@{host}', '{schema}://{token}@{host}:{port}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'token': {'name': _('Access Token'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}})
    template_args = dict(NotifyBase.template_args, **{'token': {'alias_of': 'token'}, 'visibility': {'name': _('Visibility'), 'type': 'choice:string', 'values': MISSKEY_VISIBILITIES, 'default': MisskeyVisibility.PUBLIC}})

    def __init__(self, token=None, visibility=None, **kwargs):
        if False:
            return 10
        '\n        Initialize Misskey Object\n        '
        super().__init__(**kwargs)
        self.token = validate_regex(token)
        if not self.token:
            msg = 'An invalid Misskey Access Token was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        if visibility:
            vis = 'invalid' if not isinstance(visibility, str) else visibility.lower().strip()
            self.visibility = next((v for v in MISSKEY_VISIBILITIES if v.startswith(vis)), None)
            if self.visibility not in MISSKEY_VISIBILITIES:
                msg = 'The Misskey visibility specified ({}) is invalid.'.format(visibility)
                self.logger.warning(msg)
                raise TypeError(msg)
        else:
            self.visibility = self.template_args['visibility']['default']
        self.schema = 'https' if self.secure else 'http'
        self.api_url = '%s://%s' % (self.schema, self.host)
        if isinstance(self.port, int):
            self.api_url += ':%d' % self.port
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'visibility': self.visibility}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        host = self.host
        if isinstance(self.port, int):
            host += ':%d' % self.port
        return '{schema}://{token}@{host}/?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, host=host, token=self.pprint(self.token, privacy, safe=''), params=NotifyMisskey.urlencode(params))

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        wrapper to _send since we can alert more then one channel\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        payload = {'i': self.token, 'text': body, 'visibility': self.visibility}
        api_url = f'{self.api_url}/api/notes/create'
        self.logger.debug('Misskey GET URL: %s (cert_verify=%r)' % (api_url, self.verify_certificate))
        self.logger.debug('Misskey Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(api_url, headers=headers, data=dumps(payload), verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyMisskey.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Misskey notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Misskey notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Misskey notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            results['token'] = NotifyMisskey.unquote(results['qsd']['token'])
        elif not results['password'] and results['user']:
            results['token'] = NotifyMisskey.unquote(results['user'])
        if 'visibility' in results['qsd'] and len(results['qsd']['visibility']):
            results['visibility'] = NotifyMisskey.unquote(results['qsd']['visibility'])
        return results