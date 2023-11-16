import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyImageSize
from ..common import NotifyType
from ..utils import parse_bool
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotifyMattermost(NotifyBase):
    """
    A wrapper for Mattermost Notifications
    """
    service_name = 'Mattermost'
    service_url = 'https://mattermost.com/'
    protocol = 'mmost'
    secure_protocol = 'mmosts'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_mattermost'
    default_port = 8065
    image_size = NotifyImageSize.XY_72
    body_maxlen = 4000
    title_maxlen = 0
    templates = ('{schema}://{host}/{token}', '{schema}://{host}:{port}/{token}', '{schema}://{host}/{fullpath}/{token}', '{schema}://{host}:{port}/{fullpath}/{token}', '{schema}://{botname}@{host}/{token}', '{schema}://{botname}@{host}:{port}/{token}', '{schema}://{botname}@{host}/{fullpath}/{token}', '{schema}://{botname}@{host}:{port}/{fullpath}/{token}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'token': {'name': _('Webhook Token'), 'type': 'string', 'private': True, 'required': True}, 'fullpath': {'name': _('Path'), 'type': 'string'}, 'botname': {'name': _('Bot Name'), 'type': 'string', 'map_to': 'user'}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}})
    template_args = dict(NotifyBase.template_args, **{'channels': {'name': _('Channels'), 'type': 'list:string'}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}, 'to': {'alias_of': 'channels'}})

    def __init__(self, token, fullpath=None, channels=None, include_image=False, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize Mattermost Object\n        '
        super().__init__(**kwargs)
        if self.secure:
            self.schema = 'https'
        else:
            self.schema = 'http'
        self.fullpath = '' if not isinstance(fullpath, str) else fullpath.strip()
        self.token = validate_regex(token)
        if not self.token:
            msg = 'An invalid Mattermost Authorization Token ({}) was specified.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.channels = [x.lstrip('#') for x in parse_list(channels)]
        if not self.port:
            self.port = self.default_port
        self.include_image = include_image
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform Mattermost Notification\n        '
        channels = list(self.channels) if self.channels else [None]
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        payload = {'text': body, 'icon_url': None}
        image_url = None if not self.include_image else self.image_url(notify_type)
        if image_url:
            payload['icon_url'] = image_url
        payload['username'] = self.user if self.user else self.app_id
        has_error = False
        while len(channels):
            channel = channels.pop(0)
            if channel:
                payload['channel'] = channel
            url = '{}://{}:{}{}/hooks/{}'.format(self.schema, self.host, self.port, self.fullpath, self.token)
            self.logger.debug('Mattermost POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
            self.logger.debug('Mattermost Payload: %s' % str(payload))
            self.throttle()
            try:
                r = requests.post(url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyMattermost.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send Mattermost notification{}: {}{}error={}.'.format('' if not channel else ' to channel {}'.format(channel), status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Mattermost notification{}.'.format('' if not channel else ' to channel {}'.format(channel)))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Mattermost notification{}.'.format('' if not channel else ' to channel {}'.format(channel)))
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        if self.channels:
            params['channel'] = ','.join([NotifyMattermost.quote(x, safe='') for x in self.channels])
        default_port = 443 if self.secure else self.default_port
        default_schema = self.secure_protocol if self.secure else self.protocol
        botname = ''
        if self.user:
            botname = '{botname}@'.format(botname=NotifyMattermost.quote(self.user, safe=''))
        return '{schema}://{botname}{hostname}{port}{fullpath}{token}/?{params}'.format(schema=default_schema, botname=botname, hostname=self.host, port='' if not self.port or self.port == default_port else ':{}'.format(self.port), fullpath='/' if not self.fullpath else '{}/'.format(NotifyMattermost.quote(self.fullpath, safe='/')), token=self.pprint(self.token, privacy, safe=''), params=NotifyMattermost.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        tokens = NotifyMattermost.split_path(results['fullpath'])
        results['token'] = None if not tokens else tokens.pop()
        results['fullpath'] = '' if not tokens else '/{}'.format('/'.join(tokens))
        results['channels'] = list()
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['channels'].append(NotifyMattermost.parse_list(results['qsd']['to']))
        if 'channel' in results['qsd'] and len(results['qsd']['channel']):
            results['channels'].append(NotifyMattermost.parse_list(results['qsd']['channel']))
        results['include_image'] = parse_bool(results['qsd'].get('image', False))
        return results