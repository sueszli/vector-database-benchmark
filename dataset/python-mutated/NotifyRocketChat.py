import re
import requests
from json import loads
from json import dumps
from itertools import chain
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyImageSize
from ..common import NotifyFormat
from ..common import NotifyType
from ..utils import parse_list
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
IS_CHANNEL = re.compile('^#(?P<name>[A-Za-z0-9_-]+)$')
IS_USER = re.compile('^@(?P<name>[A-Za-z0-9._-]+)$')
IS_ROOM_ID = re.compile('^(?P<name>[A-Za-z0-9]+)$')
RC_HTTP_ERROR_MAP = {400: 'Channel/RoomId is wrong format, or missing from server.', 401: 'Authentication tokens provided is invalid or missing.'}

class RocketChatAuthMode:
    """
    The Chat Authentication mode is detected
    """
    WEBHOOK = 'webhook'
    BASIC = 'basic'
ROCKETCHAT_AUTH_MODES = (RocketChatAuthMode.WEBHOOK, RocketChatAuthMode.BASIC)

class NotifyRocketChat(NotifyBase):
    """
    A wrapper for Notify Rocket.Chat Notifications
    """
    service_name = 'Rocket.Chat'
    service_url = 'https://rocket.chat/'
    protocol = 'rocket'
    secure_protocol = 'rockets'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_rocketchat'
    image_size = NotifyImageSize.XY_128
    title_maxlen = 0
    body_maxlen = 1000
    notify_format = NotifyFormat.MARKDOWN
    templates = ('{schema}://{user}:{password}@{host}:{port}/{targets}', '{schema}://{user}:{password}@{host}/{targets}', '{schema}://{webhook}@{host}', '{schema}://{webhook}@{host}:{port}', '{schema}://{webhook}@{host}/{targets}', '{schema}://{webhook}@{host}:{port}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}, 'webhook': {'name': _('Webhook'), 'type': 'string'}, 'target_channel': {'name': _('Target Channel'), 'type': 'string', 'prefix': '#', 'map_to': 'targets'}, 'target_user': {'name': _('Target User'), 'type': 'string', 'prefix': '@', 'map_to': 'targets'}, 'target_room': {'name': _('Target Room ID'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'mode': {'name': _('Webhook Mode'), 'type': 'choice:string', 'values': ROCKETCHAT_AUTH_MODES}, 'avatar': {'name': _('Use Avatar'), 'type': 'bool', 'default': False}, 'webhook': {'alias_of': 'webhook'}, 'to': {'alias_of': 'targets'}})

    def __init__(self, webhook=None, targets=None, mode=None, avatar=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize Notify Rocket.Chat Object\n        '
        super().__init__(**kwargs)
        self.schema = 'https' if self.secure else 'http'
        self.api_url = '%s://%s' % (self.schema, self.host)
        if isinstance(self.port, int):
            self.api_url += ':%d' % self.port
        self.channels = list()
        self.rooms = list()
        self.users = list()
        self.webhook = webhook
        self.headers = {}
        self.mode = None if not isinstance(mode, str) else mode.lower()
        if self.mode and self.mode not in ROCKETCHAT_AUTH_MODES:
            msg = 'The authentication mode specified ({}) is invalid.'.format(mode)
            self.logger.warning(msg)
            raise TypeError(msg)
        if not self.mode:
            if self.webhook is not None:
                self.mode = RocketChatAuthMode.WEBHOOK
            else:
                self.mode = RocketChatAuthMode.BASIC
        if self.mode == RocketChatAuthMode.BASIC and (not (self.user and self.password)):
            msg = 'No Rocket.Chat user/pass combo was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        elif self.mode == RocketChatAuthMode.WEBHOOK and (not self.webhook):
            msg = 'No Rocket.Chat Incoming Webhook was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        for recipient in parse_list(targets):
            result = IS_CHANNEL.match(recipient)
            if result:
                self.channels.append(result.group('name'))
                continue
            result = IS_ROOM_ID.match(recipient)
            if result:
                self.rooms.append(result.group('name'))
                continue
            result = IS_USER.match(recipient)
            if result:
                self.users.append(result.group('name'))
                continue
            self.logger.warning('Dropped invalid channel/room/user ({}) specified.'.format(recipient))
        if self.mode == RocketChatAuthMode.BASIC and len(self.rooms) == 0 and (len(self.channels) == 0):
            msg = 'No Rocket.Chat room and/or channels specified to notify.'
            self.logger.warning(msg)
            raise TypeError(msg)
        if self.mode == RocketChatAuthMode.BASIC:
            self.avatar = False if avatar is None else avatar
        else:
            self.avatar = True if avatar is None else avatar
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'avatar': 'yes' if self.avatar else 'no', 'mode': self.mode}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        if self.mode == RocketChatAuthMode.BASIC:
            auth = '{user}:{password}@'.format(user=NotifyRocketChat.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        else:
            auth = '{user}{webhook}@'.format(user='{}:'.format(NotifyRocketChat.quote(self.user, safe='')) if self.user else '', webhook=self.pprint(self.webhook, privacy, mode=PrivacyMode.Secret, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}/{targets}/?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), targets='/'.join([NotifyRocketChat.quote(x, safe='@#') for x in chain(['#{}'.format(x) for x in self.channels], self.rooms, ['@{}'.format(x) for x in self.users])]), params=NotifyRocketChat.urlencode(params))

    def __len__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.channels) + len(self.rooms) + len(self.users)
        return targets if targets > 0 else 1

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        wrapper to _send since we can alert more then one channel\n        '
        return getattr(self, '_send_{}_notification'.format(self.mode))(body=body, title=title, notify_type=notify_type, **kwargs)

    def _send_webhook_notification(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sends a webhook notification\n        '
        payload = self._payload(body, title, notify_type)
        path = 'hooks/{}'.format(self.webhook)
        targets = ['@{}'.format(u) for u in self.users]
        targets.extend(['#{}'.format(c) for c in self.channels])
        targets.extend(['{}'.format(r) for r in self.rooms])
        if len(targets) == 0:
            return self._send(payload, notify_type=notify_type, path=path, **kwargs)
        has_error = False
        while len(targets):
            target = targets.pop(0)
            payload['channel'] = target
            if not self._send(payload, notify_type=notify_type, path=path, **kwargs):
                has_error = True
        return not has_error

    def _send_basic_notification(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Authenticates with the server using a user/pass combo for\n        notifications.\n        '
        if not self.login():
            return False
        _payload = self._payload(body, title, notify_type)
        has_error = False
        channels = ['@{}'.format(u) for u in self.users]
        channels.extend(['#{}'.format(c) for c in self.channels])
        payload = _payload.copy()
        while len(channels) > 0:
            channel = channels.pop(0)
            payload['channel'] = channel
            if not self._send(payload, notify_type=notify_type, headers=self.headers, **kwargs):
                has_error = True
        rooms = list(self.rooms)
        payload = _payload.copy()
        while len(rooms):
            room = rooms.pop(0)
            payload['roomId'] = room
            if not self._send(payload, notify_type=notify_type, headers=self.headers, **kwargs):
                has_error = True
        self.logout()
        return not has_error

    def _payload(self, body, title='', notify_type=NotifyType.INFO):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prepares a payload object\n        '
        payload = {'text': body}
        image_url = self.image_url(notify_type)
        if self.avatar and image_url:
            payload['avatar'] = image_url
        return payload

    def _send(self, payload, notify_type, path='api/v1/chat.postMessage', headers={}, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform Notify Rocket.Chat Notification\n        '
        api_url = '{}/{}'.format(self.api_url, path)
        self.logger.debug('Rocket.Chat POST URL: %s (cert_verify=%r)' % (api_url, self.verify_certificate))
        self.logger.debug('Rocket.Chat Payload: %s' % str(payload))
        headers.update({'User-Agent': self.app_id, 'Content-Type': 'application/json'})
        self.throttle()
        try:
            r = requests.post(api_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyRocketChat.http_response_code_lookup(r.status_code, RC_HTTP_ERROR_MAP)
                self.logger.warning('Failed to send Rocket.Chat {}:notification: {}{}error={}.'.format(self.mode, status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Rocket.Chat {}:notification.'.format(self.mode))
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Rocket.Chat {}:notification.'.format(self.mode))
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def login(self):
        if False:
            while True:
                i = 10
        '\n        login to our server\n\n        '
        payload = {'username': self.user, 'password': self.password}
        api_url = '{}/{}'.format(self.api_url, 'api/v1/login')
        try:
            r = requests.post(api_url, data=payload, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyRocketChat.http_response_code_lookup(r.status_code, RC_HTTP_ERROR_MAP)
                self.logger.warning('Failed to authenticate {} with Rocket.Chat: {}{}error={}.'.format(self.user, status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.debug('Rocket.Chat authentication successful')
                response = loads(r.content)
                if response.get('status') != 'success':
                    self.logger.warning('Could not authenticate {} with Rocket.Chat.'.format(self.user))
                    return False
                self.headers['X-Auth-Token'] = response.get('data', {'authToken': None}).get('authToken')
                self.headers['X-User-Id'] = response.get('data', {'userId': None}).get('userId')
        except (AttributeError, TypeError, ValueError):
            self.logger.warning('A commuication error occurred authenticating {} on Rocket.Chat.'.format(self.user))
            return False
        except requests.RequestException as e:
            self.logger.warning('A connection error occurred authenticating {} on Rocket.Chat.'.format(self.user))
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def logout(self):
        if False:
            print('Hello World!')
        '\n        logout of our server\n        '
        api_url = '{}/{}'.format(self.api_url, 'api/v1/logout')
        try:
            r = requests.post(api_url, headers=self.headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyRocketChat.http_response_code_lookup(r.status_code, RC_HTTP_ERROR_MAP)
                self.logger.warning('Failed to logoff {} from Rocket.Chat: {}{}error={}.'.format(self.user, status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.debug('Rocket.Chat log off successful; response %s.' % r.content)
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred logging off the Rocket.Chat server')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        try:
            match = re.match('^\\s*(?P<schema>[^:]+://)((?P<user>[^:]+):)?(?P<webhook>[a-z0-9]+(/|%2F)[a-z0-9]+)\\@(?P<url>.+)$', url, re.I)
        except TypeError:
            return None
        if match:
            url = '{schema}{user}{url}'.format(schema=match.group('schema'), user='{}@'.format(match.group('user')) if match.group('user') else '', url=match.group('url'))
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        if match:
            results['webhook'] = NotifyRocketChat.unquote(match.group('webhook'))
            results['password'] = match.group('webhook')
        results['targets'] = NotifyRocketChat.split_path(results['fullpath'])
        if 'mode' in results['qsd'] and len(results['qsd']['mode']):
            results['mode'] = NotifyRocketChat.unquote(results['qsd']['mode'])
        if 'avatar' in results['qsd'] and len(results['qsd']['avatar']):
            results['avatar'] = parse_bool(results['qsd'].get('avatar', True))
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyRocketChat.parse_list(results['qsd']['to'])
        if 'webhook' in results['qsd'] and len(results['qsd']['webhook']):
            results['webhook'] = NotifyRocketChat.unquote(results['qsd']['webhook'])
        return results