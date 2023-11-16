import re
import requests
from markdown import markdown
from json import dumps
from json import loads
from time import time
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..common import NotifyImageSize
from ..common import NotifyFormat
from ..utils import parse_bool
from ..utils import parse_list
from ..utils import is_hostname
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
MATRIX_V1_WEBHOOK_PATH = '/api/v1/matrix/hook'
MATRIX_V2_API_PATH = '/_matrix/client/r0'
MATRIX_V3_API_PATH = '/_matrix/client/v3'
MATRIX_V3_MEDIA_PATH = '/_matrix/media/v3'
MATRIX_V2_MEDIA_PATH = '/_matrix/media/r0'
MATRIX_HTTP_ERROR_MAP = {403: 'Unauthorized - Invalid Token.', 429: 'Rate limit imposed; wait 2s and try again'}
IS_ROOM_ALIAS = re.compile('^\\s*(#|%23)?(?P<room>[a-z0-9-]+)((:|%3A)(?P<home_server>[a-z0-9.-]+))?\\s*$', re.I)
IS_ROOM_ID = re.compile('^\\s*(!|&#33;|%21)(?P<room>[a-z0-9-]+)((:|%3A)(?P<home_server>[a-z0-9.-]+))?\\s*$', re.I)

class MatrixMessageType:
    """
    The Matrix Message types
    """
    TEXT = 'text'
    NOTICE = 'notice'
MATRIX_MESSAGE_TYPES = (MatrixMessageType.TEXT, MatrixMessageType.NOTICE)

class MatrixVersion:
    V2 = '2'
    V3 = '3'
MATRIX_VERSIONS = (MatrixVersion.V2, MatrixVersion.V3)

class MatrixWebhookMode:
    DISABLED = 'off'
    MATRIX = 'matrix'
    SLACK = 'slack'
    T2BOT = 't2bot'
MATRIX_WEBHOOK_MODES = (MatrixWebhookMode.DISABLED, MatrixWebhookMode.MATRIX, MatrixWebhookMode.SLACK, MatrixWebhookMode.T2BOT)

class NotifyMatrix(NotifyBase):
    """
    A wrapper for Matrix Notifications
    """
    service_name = 'Matrix'
    service_url = 'https://matrix.org/'
    protocol = 'matrix'
    secure_protocol = 'matrixs'
    attachment_support = True
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_matrix'
    image_size = NotifyImageSize.XY_32
    body_maxlen = 65000
    request_rate_per_sec = 0.5
    matrix_api_version = '3'
    default_retries = 2
    default_wait_ms = 1000
    templates = ('{schema}://{token}', '{schema}://{user}@{token}', '{schema}://{user}:{password}@{host}/{targets}', '{schema}://{user}:{password}@{host}:{port}/{targets}', '{schema}://{user}:{token}@{host}/{targets}', '{schema}://{user}:{token}@{host}:{port}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string'}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}, 'token': {'name': _('Access Token'), 'private': True, 'map_to': 'password'}, 'target_user': {'name': _('Target User'), 'type': 'string', 'prefix': '@', 'map_to': 'targets'}, 'target_room_id': {'name': _('Target Room ID'), 'type': 'string', 'prefix': '!', 'map_to': 'targets'}, 'target_room_alias': {'name': _('Target Room Alias'), 'type': 'string', 'prefix': '!', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'image': {'name': _('Include Image'), 'type': 'bool', 'default': False, 'map_to': 'include_image'}, 'mode': {'name': _('Webhook Mode'), 'type': 'choice:string', 'values': MATRIX_WEBHOOK_MODES, 'default': MatrixWebhookMode.DISABLED}, 'version': {'name': _('Matrix API Verion'), 'type': 'choice:string', 'values': MATRIX_VERSIONS, 'default': MatrixVersion.V3}, 'msgtype': {'name': _('Message Type'), 'type': 'choice:string', 'values': MATRIX_MESSAGE_TYPES, 'default': MatrixMessageType.TEXT}, 'to': {'alias_of': 'targets'}, 'token': {'alias_of': 'token'}})

    def __init__(self, targets=None, mode=None, msgtype=None, version=None, include_image=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Matrix Object\n        '
        super().__init__(**kwargs)
        self.rooms = parse_list(targets)
        self.home_server = None
        self.user_id = None
        self.access_token = None
        self.include_image = include_image
        self._room_cache = {}
        self.mode = self.template_args['mode']['default'] if not isinstance(mode, str) else mode.lower()
        if self.mode and self.mode not in MATRIX_WEBHOOK_MODES:
            msg = 'The mode specified ({}) is invalid.'.format(mode)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.version = self.template_args['version']['default'] if not isinstance(version, str) else version
        if self.version not in MATRIX_VERSIONS:
            msg = 'The version specified ({}) is invalid.'.format(version)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.msgtype = self.template_args['msgtype']['default'] if not isinstance(msgtype, str) else msgtype.lower()
        if self.msgtype and self.msgtype not in MATRIX_MESSAGE_TYPES:
            msg = 'The msgtype specified ({}) is invalid.'.format(msgtype)
            self.logger.warning(msg)
            raise TypeError(msg)
        if self.mode == MatrixWebhookMode.T2BOT:
            self.access_token = validate_regex(self.password, '^[a-z0-9]{64}$', 'i')
            if not self.access_token:
                msg = 'An invalid T2Bot/Matrix Webhook ID ({}) was specified.'.format(self.password)
                self.logger.warning(msg)
                raise TypeError(msg)
        elif not is_hostname(self.host):
            msg = 'An invalid Matrix Hostname ({}) was specified'.format(self.host)
            self.logger.warning(msg)
            raise TypeError(msg)
        elif self.port is not None and (not (isinstance(self.port, int) and self.port >= self.template_tokens['port']['min'] and (self.port <= self.template_tokens['port']['max']))):
            msg = 'An invalid Matrix Port ({}) was specified'.format(self.port)
            self.logger.warning(msg)
            raise TypeError(msg)

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform Matrix Notification\n        '
        return getattr(self, '_send_{}_notification'.format('webhook' if self.mode != MatrixWebhookMode.DISABLED else 'server'))(body=body, title=title, notify_type=notify_type, **kwargs)

    def _send_webhook_notification(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform Matrix Notification as a webhook\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        if self.mode != MatrixWebhookMode.T2BOT:
            access_token = self.password if self.password else self.user
            default_port = 443 if self.secure else 80
            url = '{schema}://{hostname}:{port}{webhook_path}/{token}'.format(schema='https' if self.secure else 'http', hostname=self.host, port='' if self.port is None or self.port == default_port else self.port, webhook_path=MATRIX_V1_WEBHOOK_PATH, token=access_token)
        else:
            url = 'https://webhooks.t2bot.io/api/v1/matrix/hook/{token}'.format(token=self.access_token)
        payload = getattr(self, '_{}_webhook_payload'.format(self.mode))(body=body, title=title, notify_type=notify_type, **kwargs)
        self.logger.debug('Matrix POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('Matrix Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyMatrix.http_response_code_lookup(r.status_code, MATRIX_HTTP_ERROR_MAP)
                self.logger.warning('Failed to send Matrix notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Matrix notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Matrix notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def _slack_webhook_payload(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Format the payload for a Slack based message\n\n        '
        if not hasattr(self, '_re_slack_formatting_rules'):
            self._re_slack_formatting_map = {'\\r\\*\\n': '\\n', '&': '&amp;', '<': '&lt;', '>': '&gt;'}
            self._re_slack_formatting_rules = re.compile('(' + '|'.join(self._re_slack_formatting_map.keys()) + ')', re.IGNORECASE)
        title = self._re_slack_formatting_rules.sub(lambda x: self._re_slack_formatting_map[x.group()], title)
        body = self._re_slack_formatting_rules.sub(lambda x: self._re_slack_formatting_map[x.group()], body)
        payload = {'username': self.user if self.user else self.app_id, 'mrkdwn': self.notify_format == NotifyFormat.MARKDOWN, 'attachments': [{'title': title, 'text': body, 'color': self.color(notify_type), 'ts': time(), 'footer': self.app_id}]}
        return payload

    def _matrix_webhook_payload(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Format the payload for a Matrix based message\n\n        '
        payload = {'displayName': self.user if self.user else self.app_id, 'format': 'plain' if self.notify_format == NotifyFormat.TEXT else 'html', 'text': ''}
        if self.notify_format == NotifyFormat.HTML:
            payload['text'] = '{title}{body}'.format(title='' if not title else '<h1>{}</h1>'.format(NotifyMatrix.escape_html(title)), body=body)
        elif self.notify_format == NotifyFormat.MARKDOWN:
            payload['text'] = '{title}{body}'.format(title='' if not title else '<h1>{}</h1>'.format(NotifyMatrix.escape_html(title)), body=markdown(body))
        else:
            payload['text'] = body if not title else '{}\r\n{}'.format(title, body)
        return payload

    def _t2bot_webhook_payload(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Format the payload for a T2Bot Matrix based messages\n\n        '
        payload = self._matrix_webhook_payload(body=body, title=title, notify_type=notify_type, **kwargs)
        image_url = None if not self.include_image else self.image_url(notify_type)
        if image_url:
            payload['avatarUrl'] = image_url
        return payload

    def _send_server_notification(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            return 10
        '\n        Perform Direct Matrix Server Notification (no webhook)\n        '
        if self.access_token is None:
            if not self._login():
                if not self._register():
                    return False
        if len(self.rooms) == 0:
            self.rooms = self._joined_rooms()
            if len(self.rooms) == 0:
                self.logger.warning('There were no Matrix rooms specified to notify.')
                return False
        rooms = list(self.rooms)
        has_error = False
        attachments = None
        if attach and self.attachment_support:
            attachments = self._send_attachments(attach)
            if attachments is False:
                return False
        while len(rooms) > 0:
            room = rooms.pop(0)
            room_id = self._room_join(room)
            if not room_id:
                self.logger.warning('Could not join Matrix room {}.'.format(room))
                has_error = True
                continue
            image_url = None if not self.include_image else self.image_url(notify_type)
            if self.version == MatrixVersion.V3:
                path = '/rooms/{}/send/m.room.message/0'.format(NotifyMatrix.quote(room_id))
            else:
                path = '/rooms/{}/send/m.room.message'.format(NotifyMatrix.quote(room_id))
            if self.version == MatrixVersion.V2:
                if image_url:
                    image_payload = {'msgtype': 'm.image', 'url': image_url, 'body': '{}'.format(notify_type if not title else title)}
                    (postokay, response) = self._fetch(path, payload=image_payload)
                    if not postokay:
                        has_error = True
                        continue
                if attachments:
                    for attachment in attachments:
                        attachment['room_id'] = room_id
                        attachment['type'] = 'm.room.message'
                        (postokay, response) = self._fetch(path, payload=attachment)
                        if not postokay:
                            has_error = True
                            continue
            payload = {'msgtype': 'm.{}'.format(self.msgtype), 'body': '{title}{body}'.format(title='' if not title else '# {}\r\n'.format(title), body=body)}
            if self.notify_format == NotifyFormat.HTML:
                payload.update({'format': 'org.matrix.custom.html', 'formatted_body': '{title}{body}'.format(title='' if not title else '<h1>{}</h1>'.format(title), body=body)})
            elif self.notify_format == NotifyFormat.MARKDOWN:
                payload.update({'format': 'org.matrix.custom.html', 'formatted_body': '{title}{body}'.format(title='' if not title else '<h1>{}</h1>'.format(NotifyMatrix.escape_html(title, whitespace=False)), body=markdown(body))})
            method = 'PUT' if self.version == MatrixVersion.V3 else 'POST'
            (postokay, response) = self._fetch(path, payload=payload, method=method)
            if not postokay:
                self.logger.warning('Could not send notification Matrix room {}.'.format(room))
                has_error = True
                continue
        return not has_error

    def _send_attachments(self, attach):
        if False:
            i = 10
            return i + 15
        '\n        Posts all of the provided attachments\n        '
        payloads = []
        if self.version != MatrixVersion.V2:
            self.logger.warning('Add ?v=2 to Apprise URL to support Attachments')
            return next((False for a in attach if not a), [])
        for attachment in attach:
            if not attachment:
                return False
            if not re.match('^image/', attachment.mimetype, re.I):
                continue
            (postokay, response) = self._fetch('/upload', attachment=attachment)
            if not (postokay and isinstance(response, dict)):
                return False
            if self.version == MatrixVersion.V3:
                payloads.append({'body': attachment.name, 'info': {'mimetype': attachment.mimetype, 'size': len(attachment)}, 'msgtype': 'm.image', 'url': response.get('content_uri')})
            else:
                payloads.append({'info': {'mimetype': attachment.mimetype}, 'msgtype': 'm.image', 'body': 'tta.webp', 'url': response.get('content_uri')})
        return payloads

    def _register(self):
        if False:
            return 10
        '\n        Register with the service if possible.\n        '
        payload = {'kind': 'user', 'auth': {'type': 'm.login.dummy'}}
        params = {'kind': 'user'}
        if self.user:
            payload['username'] = self.user
        if self.password:
            payload['password'] = self.password
        (postokay, response) = self._fetch('/register', payload=payload, params=params)
        if not (postokay and isinstance(response, dict)):
            return False
        self.access_token = response.get('access_token')
        self.home_server = response.get('home_server')
        self.user_id = response.get('user_id')
        if self.access_token is not None:
            self.logger.debug('Registered successfully with Matrix server.')
            return True
        return False

    def _login(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Acquires the matrix token required for making future requests. If we\n        fail we return False, otherwise we return True\n        '
        if self.access_token:
            return True
        if not (self.user and self.password):
            self.logger.warning('Failed to login to Matrix server: user/pass combo is missing.')
            return False
        if self.version == MatrixVersion.V3:
            payload = {'type': 'm.login.password', 'identifier': {'type': 'm.id.user', 'user': self.user}, 'password': self.password}
        else:
            payload = {'type': 'm.login.password', 'user': self.user, 'password': self.password}
        (postokay, response) = self._fetch('/login', payload=payload)
        if not (postokay and isinstance(response, dict)):
            return False
        self.access_token = response.get('access_token')
        self.home_server = response.get('home_server')
        self.user_id = response.get('user_id')
        if not self.access_token:
            return False
        self.logger.debug('Authenticated successfully with Matrix server.')
        return True

    def _logout(self):
        if False:
            print('Hello World!')
        '\n        Relinquishes token from remote server\n        '
        if not self.access_token:
            return True
        payload = {}
        (postokay, response) = self._fetch('/logout', payload=payload)
        if not postokay:
            if response.get('errcode') != u'M_UNKNOWN_TOKEN':
                return False
        self.access_token = None
        self.home_server = None
        self.user_id = None
        self._room_cache = {}
        self.logger.debug('Unauthenticated successfully with Matrix server.')
        return True

    def _room_join(self, room):
        if False:
            print('Hello World!')
        "\n        Joins a matrix room if we're not already in it. Otherwise it attempts\n        to create it if it doesn't exist and always returns\n        the room_id if it was successful, otherwise it returns None\n\n        "
        if not self.access_token:
            return None
        if not isinstance(room, str):
            return None
        payload = {}
        result = IS_ROOM_ID.match(room)
        if result:
            home_server = result.group('home_server') if result.group('home_server') else self.home_server
            room_id = '!{}:{}'.format(result.group('room'), home_server)
            if room_id in self._room_cache:
                return self._room_cache[room_id]['id']
            path = '/join/{}'.format(NotifyMatrix.quote(room_id))
            (postokay, _) = self._fetch(path, payload=payload)
            if postokay:
                self._room_cache[room_id] = {'id': room_id, 'home_server': home_server}
            return room_id if postokay else None
        result = IS_ROOM_ALIAS.match(room)
        if not result:
            self.logger.warning('Ignoring illegally formed room {} from Matrix server list.'.format(room))
            return None
        home_server = self.home_server if not result.group('home_server') else result.group('home_server')
        room = '#{}:{}'.format(result.group('room'), home_server)
        if room in self._room_cache:
            return self._room_cache[room]['id']
        path = '/join/{}'.format(NotifyMatrix.quote(room))
        (postokay, response) = self._fetch(path, payload=payload)
        if postokay:
            self._room_cache[room] = {'id': response.get('room_id'), 'home_server': home_server}
            return self._room_cache[room]['id']
        return self._room_create(room)

    def _room_create(self, room):
        if False:
            print('Hello World!')
        "\n        Creates a matrix room and return it's room_id if successful\n        otherwise None is returned.\n        "
        if not self.access_token:
            return None
        if not isinstance(room, str):
            return None
        result = IS_ROOM_ALIAS.match(room)
        if not result:
            return None
        home_server = result.group('home_server') if result.group('home_server') else self.home_server
        room = '#{}:{}'.format(result.group('room'), home_server)
        payload = {'room_alias_name': result.group('room'), 'name': '#{} - {}'.format(result.group('room'), self.app_desc), 'visibility': 'private', 'preset': 'trusted_private_chat'}
        (postokay, response) = self._fetch('/createRoom', payload=payload)
        if not postokay:
            if response and response.get('errcode') == 'M_ROOM_IN_USE':
                return self._room_id(room)
            return None
        self._room_cache[response.get('room_alias')] = {'id': response.get('room_id'), 'home_server': home_server}
        return response.get('room_id')

    def _joined_rooms(self):
        if False:
            return 10
        '\n        Returns a list of the current rooms the logged in user\n        is a part of.\n        '
        if not self.access_token:
            return list()
        (postokay, response) = self._fetch('/joined_rooms', payload=None, method='GET')
        if not postokay:
            return list()
        return response.get('joined_rooms', list())

    def _room_id(self, room):
        if False:
            return 10
        'Get room id from its alias.\n        Args:\n            room (str): The room alias name.\n\n        Returns:\n            returns the room id if it can, otherwise it returns None\n        '
        if not self.access_token:
            return None
        if not isinstance(room, str):
            return None
        result = IS_ROOM_ALIAS.match(room)
        if not result:
            return None
        home_server = result.group('home_server') if result.group('home_server') else self.home_server
        room = '#{}:{}'.format(result.group('room'), home_server)
        (postokay, response) = self._fetch('/directory/room/{}'.format(NotifyMatrix.quote(room)), payload=None, method='GET')
        if postokay:
            return response.get('room_id')
        return None

    def _fetch(self, path, payload=None, params=None, attachment=None, method='POST'):
        if False:
            print('Hello World!')
        "\n        Wrapper to request.post() to manage it's response better and make\n        the send() function cleaner and easier to maintain.\n\n        This function returns True if the _post was successful and False\n        if it wasn't.\n        "
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'Accept': 'application/json'}
        if self.access_token is not None:
            headers['Authorization'] = 'Bearer %s' % self.access_token
        default_port = 443 if self.secure else 80
        url = '{schema}://{hostname}{port}'.format(schema='https' if self.secure else 'http', hostname=self.host, port='' if self.port is None or self.port == default_port else f':{self.port}')
        if path == '/upload':
            if self.version == MatrixVersion.V3:
                url += MATRIX_V3_MEDIA_PATH + path
            else:
                url += MATRIX_V2_MEDIA_PATH + path
            params = {'filename': attachment.name}
            with open(attachment.path, 'rb') as fp:
                payload = fp.read()
            headers['Content-Type'] = attachment.mimetype
        elif self.version == MatrixVersion.V3:
            url += MATRIX_V3_API_PATH + path
        else:
            url += MATRIX_V2_API_PATH + path
        response = {}
        fn = requests.post if method == 'POST' else requests.put if method == 'PUT' else requests.get
        retries = self.default_retries if self.default_retries > 0 else 1
        while retries > 0:
            retries -= 1
            self.logger.debug('Matrix POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
            self.logger.debug('Matrix Payload: %s' % str(payload))
            r = None
            try:
                r = fn(url, data=dumps(payload) if not attachment else payload, params=params, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                self.logger.debug('Matrix Response: code=%d, %s' % (r.status_code, str(r.content)))
                response = loads(r.content)
                if r.status_code == 429:
                    wait = self.default_wait_ms / 1000
                    try:
                        wait = response['retry_after_ms'] / 1000
                    except KeyError:
                        try:
                            errordata = response['error']
                            wait = errordata['retry_after_ms'] / 1000
                        except KeyError:
                            pass
                    self.logger.warning('Matrix server requested we throttle back {}ms; retries left {}.'.format(wait, retries))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    self.throttle(wait=wait)
                    continue
                elif r.status_code != requests.codes.ok:
                    status_str = NotifyMatrix.http_response_code_lookup(r.status_code, MATRIX_HTTP_ERROR_MAP)
                    self.logger.warning('Failed to handshake with Matrix server: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    return (False, response)
            except (AttributeError, TypeError, ValueError):
                self.logger.warning('Invalid response from Matrix server.')
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, {})
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred while registering with Matrix server.')
                self.logger.debug('Socket Exception: %s' % str(e))
                return (False, response)
            except (OSError, IOError) as e:
                self.logger.warning('An I/O error occurred while reading {}.'.format(attachment.name if attachment else 'unknown file'))
                self.logger.debug('I/O Exception: %s' % str(e))
                return (False, {})
            return (True, response)
        return (False, {})

    def __del__(self):
        if False:
            return 10
        '\n        Ensure we relinquish our token\n        '
        if self.mode == MatrixWebhookMode.T2BOT:
            return
        try:
            self._logout()
        except LookupError:
            pass
        except ImportError:
            pass

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no', 'mode': self.mode, 'version': self.version, 'msgtype': self.msgtype}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = ''
        if self.mode != MatrixWebhookMode.T2BOT:
            if self.user and self.password:
                auth = '{user}:{password}@'.format(user=NotifyMatrix.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
            elif self.user:
                auth = '{user}@'.format(user=NotifyMatrix.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}/{rooms}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=NotifyMatrix.quote(self.host, safe='') if self.mode != MatrixWebhookMode.T2BOT else self.pprint(self.access_token, privacy, safe=''), port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), rooms=NotifyMatrix.quote('/'.join(self.rooms)), params=NotifyMatrix.urlencode(params))

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.rooms)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        if not results.get('host'):
            return None
        results['targets'] = NotifyMatrix.split_path(results['fullpath'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyMatrix.parse_list(results['qsd']['to'])
        results['include_image'] = parse_bool(results['qsd'].get('image', NotifyMatrix.template_args['image']['default']))
        results['mode'] = results['qsd'].get('mode')
        if results['mode'] is None and (not results['password']) and (not results['targets']):
            results['mode'] = MatrixWebhookMode.T2BOT
        if results['mode'] and results['mode'].lower() == MatrixWebhookMode.T2BOT:
            results['password'] = NotifyMatrix.unquote(results['host'])
        if 'msgtype' in results['qsd'] and len(results['qsd']['msgtype']):
            results['msgtype'] = NotifyMatrix.unquote(results['qsd']['msgtype'])
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            results['password'] = NotifyMatrix.unquote(results['qsd']['token'])
        if 'version' in results['qsd'] and len(results['qsd']['version']):
            results['version'] = NotifyMatrix.unquote(results['qsd']['version'])
        elif 'v' in results['qsd'] and len(results['qsd']['v']):
            results['version'] = NotifyMatrix.unquote(results['qsd']['v'])
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            print('Hello World!')
        '\n        Support https://webhooks.t2bot.io/api/v1/matrix/hook/WEBHOOK_TOKEN/\n        '
        result = re.match('^https?://webhooks\\.t2bot\\.io/api/v[0-9]+/matrix/hook/(?P<webhook_token>[A-Z0-9_-]+)/?(?P<params>\\?.+)?$', url, re.I)
        if result:
            mode = 'mode={}'.format(MatrixWebhookMode.T2BOT)
            return NotifyMatrix.parse_url('{schema}://{webhook_token}/{params}'.format(schema=NotifyMatrix.secure_protocol, webhook_token=result.group('webhook_token'), params='?{}'.format(mode) if not result.group('params') else '{}&{}'.format(result.group('params'), mode)))
        return None