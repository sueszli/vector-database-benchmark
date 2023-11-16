import re
import requests
from json import loads
from json import dumps
from os.path import basename
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..common import NotifyImageSize
from ..AppriseLocale import gettext_lazy as _
from ..utils import parse_list
from ..utils import parse_bool
from ..utils import is_hostname
from ..utils import is_ipaddr
from ..utils import validate_regex
from ..URLBase import PrivacyMode
from ..attachment.AttachBase import AttachBase

class NtfyMode:
    """
    Define ntfy Notification Modes
    """
    CLOUD = 'cloud'
    PRIVATE = 'private'
NTFY_MODES = (NtfyMode.CLOUD, NtfyMode.PRIVATE)
NTFY_AUTH_DETECT_RE = re.compile('tk_[^ \t]+', re.IGNORECASE)

class NtfyAuth:
    """
    Define ntfy Authentication Modes
    """
    BASIC = 'basic'
    TOKEN = 'token'
NTFY_AUTH = (NtfyAuth.BASIC, NtfyAuth.TOKEN)

class NtfyPriority:
    """
    Ntfy Priority Definitions
    """
    MAX = 'max'
    HIGH = 'high'
    NORMAL = 'default'
    LOW = 'low'
    MIN = 'min'
NTFY_PRIORITIES = (NtfyPriority.MAX, NtfyPriority.HIGH, NtfyPriority.NORMAL, NtfyPriority.LOW, NtfyPriority.MIN)
NTFY_PRIORITY_MAP = {'l': NtfyPriority.LOW, 'mo': NtfyPriority.LOW, 'n': NtfyPriority.NORMAL, 'h': NtfyPriority.HIGH, 'e': NtfyPriority.MAX, 'mi': NtfyPriority.MIN, 'ma': NtfyPriority.MAX, 'd': NtfyPriority.NORMAL, '1': NtfyPriority.MIN, '2': NtfyPriority.LOW, '3': NtfyPriority.NORMAL, '4': NtfyPriority.HIGH, '5': NtfyPriority.MAX}

class NotifyNtfy(NotifyBase):
    """
    A wrapper for ntfy Notifications
    """
    service_name = 'ntfy'
    service_url = 'https://ntfy.sh/'
    protocol = 'ntfy'
    secure_protocol = 'ntfys'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_ntfy'
    cloud_notify_url = 'https://ntfy.sh'
    attachment_support = True
    image_size = NotifyImageSize.XY_256
    time_to_live = 2419200
    __auto_cloud_host = re.compile('ntfy\\.sh', re.IGNORECASE)
    templates = ('{schema}://{topic}', '{schema}://{host}/{targets}', '{schema}://{host}:{port}/{targets}', '{schema}://{user}@{host}/{targets}', '{schema}://{user}@{host}:{port}/{targets}', '{schema}://{user}:{password}@{host}/{targets}', '{schema}://{user}:{password}@{host}:{port}/{targets}', '{schema}://{token}@{host}/{targets}', '{schema}://{token}@{host}:{port}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string'}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}, 'token': {'name': _('Token'), 'type': 'string', 'private': True}, 'topic': {'name': _('Topic'), 'type': 'string', 'map_to': 'targets', 'regex': ('^[a-z0-9_-]{1,64}$', 'i')}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'attach': {'name': _('Attach'), 'type': 'string'}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}, 'avatar_url': {'name': _('Avatar URL'), 'type': 'string'}, 'filename': {'name': _('Attach Filename'), 'type': 'string'}, 'click': {'name': _('Click'), 'type': 'string'}, 'delay': {'name': _('Delay'), 'type': 'string'}, 'email': {'name': _('Email'), 'type': 'string'}, 'priority': {'name': _('Priority'), 'type': 'choice:string', 'values': NTFY_PRIORITIES, 'default': NtfyPriority.NORMAL}, 'tags': {'name': _('Tags'), 'type': 'string'}, 'mode': {'name': _('Mode'), 'type': 'choice:string', 'values': NTFY_MODES, 'default': NtfyMode.PRIVATE}, 'token': {'alias_of': 'token'}, 'auth': {'name': _('Authentication Type'), 'type': 'choice:string', 'values': NTFY_AUTH, 'default': NtfyAuth.BASIC}, 'to': {'alias_of': 'targets'}})

    def __init__(self, targets=None, attach=None, filename=None, click=None, delay=None, email=None, priority=None, tags=None, mode=None, include_image=True, avatar_url=None, auth=None, token=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize ntfy Object\n        '
        super().__init__(**kwargs)
        self.mode = mode.strip().lower() if isinstance(mode, str) else self.template_args['mode']['default']
        if self.mode not in NTFY_MODES:
            msg = 'An invalid ntfy Mode ({}) was specified.'.format(mode)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.include_image = include_image
        self.auth = auth.strip().lower() if isinstance(auth, str) else self.template_args['auth']['default']
        if self.auth not in NTFY_AUTH:
            msg = 'An invalid ntfy Authentication type ({}) was specified.'.format(auth)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.attach = attach
        self.filename = filename
        self.click = click
        self.delay = delay
        self.email = email
        self.token = token
        self.priority = NotifyNtfy.template_args['priority']['default'] if not priority else next((v for (k, v) in NTFY_PRIORITY_MAP.items() if str(priority).lower().startswith(k)), NotifyNtfy.template_args['priority']['default'])
        self.__tags = parse_list(tags)
        self.avatar_url = avatar_url
        topics = parse_list(targets)
        self.topics = []
        for _topic in topics:
            topic = validate_regex(_topic, *self.template_tokens['topic']['regex'])
            if not topic:
                self.logger.warning('A specified ntfy topic ({}) is invalid and will be ignored'.format(_topic))
                continue
            self.topics.append(topic)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            return 10
        '\n        Perform ntfy Notification\n        '
        has_error = False
        if not len(self.topics):
            self.logger.warning('There are no ntfy topics to notify')
            return False
        image_url = self.image_url(notify_type)
        if self.include_image and (image_url or self.avatar_url):
            image_url = self.avatar_url if self.avatar_url else image_url
        else:
            image_url = None
        topics = list(self.topics)
        while len(topics) > 0:
            topic = topics.pop()
            if attach and self.attachment_support:
                for (no, attachment) in enumerate(attach):
                    _body = body if not no and body else None
                    _title = title if not no and title else None
                    if not attachment:
                        self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                        return False
                    self.logger.debug('Preparing ntfy attachment {}'.format(attachment.url(privacy=True)))
                    (okay, response) = self._send(topic, body=_body, title=_title, image_url=image_url, attach=attachment)
                    if not okay:
                        return False
            else:
                (okay, response) = self._send(topic, body=body, title=title, image_url=image_url)
                if not okay:
                    has_error = True
        return not has_error

    def _send(self, topic, body=None, title=None, attach=None, image_url=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wrapper to the requests (post) object\n        '
        headers = {'User-Agent': self.app_id}
        data = {}
        params = {}
        auth = None
        if self.mode == NtfyMode.CLOUD:
            notify_url = self.cloud_notify_url
        else:
            if self.auth == NtfyAuth.BASIC and self.user:
                auth = (self.user, self.password)
            elif self.auth == NtfyAuth.TOKEN:
                if not self.token:
                    self.logger.warning('No Ntfy Token was specified')
                    return (False, None)
                headers['Authorization'] = f'Bearer {self.token}'
            schema = 'https' if self.secure else 'http'
            notify_url = '%s://%s' % (schema, self.host)
            if isinstance(self.port, int):
                notify_url += ':%d' % self.port
        if not attach:
            headers['Content-Type'] = 'application/json'
            data['topic'] = topic
            virt_payload = data
            if self.attach:
                virt_payload['attach'] = self.attach
                if self.filename:
                    virt_payload['filename'] = self.filename
        else:
            virt_payload = params
            notify_url += '/{topic}'.format(topic=topic)
            virt_payload['filename'] = attach.name
            with open(attach.path, 'rb') as fp:
                data = fp.read()
        if image_url:
            headers['X-Icon'] = image_url
        if title:
            virt_payload['title'] = title
        if body:
            virt_payload['message'] = body
        if self.priority != NtfyPriority.NORMAL:
            headers['X-Priority'] = self.priority
        if self.delay is not None:
            headers['X-Delay'] = self.delay
        if self.click is not None:
            headers['X-Click'] = self.click
        if self.email is not None:
            headers['X-Email'] = self.email
        if self.__tags:
            headers['X-Tags'] = ','.join(self.__tags)
        self.logger.debug('ntfy POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
        self.logger.debug('ntfy Payload: %s' % str(virt_payload))
        self.logger.debug('ntfy Headers: %s' % str(headers))
        self.throttle()
        response = None
        if not attach:
            data = dumps(data)
        try:
            r = requests.post(notify_url, params=params if params else None, data=data, headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyBase.http_response_code_lookup(r.status_code)
                status_code = r.status_code
                try:
                    response = loads(r.content)
                    status_str = response.get('error', status_str)
                    status_code = int(response.get('code', status_code))
                except (AttributeError, TypeError, ValueError):
                    pass
                self.logger.warning("Failed to send ntfy notification to topic '{}': {}{}error={}.".format(topic, status_str, ', ' if status_str else '', status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, response)
            self.logger.info("Sent ntfy notification to '{}'.".format(notify_url))
            return (True, response)
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending ntfy:%s ' % notify_url + 'notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
        except (OSError, IOError) as e:
            self.logger.warning('An I/O error occurred while handling {}.'.format(attach.name if isinstance(attach, AttachBase) else virt_payload))
            self.logger.debug('I/O Exception: %s' % str(e))
        return (False, response)

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        default_port = 443 if self.secure else 80
        params = {'priority': self.priority, 'mode': self.mode, 'image': 'yes' if self.include_image else 'no', 'auth': self.auth}
        if self.avatar_url:
            params['avatar_url'] = self.avatar_url
        if self.attach is not None:
            params['attach'] = self.attach
        if self.click is not None:
            params['click'] = self.click
        if self.delay is not None:
            params['delay'] = self.delay
        if self.email is not None:
            params['email'] = self.email
        if self.__tags:
            params['tags'] = ','.join(self.__tags)
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = ''
        if self.auth == NtfyAuth.BASIC:
            if self.user and self.password:
                auth = '{user}:{password}@'.format(user=NotifyNtfy.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
            elif self.user:
                auth = '{user}@'.format(user=NotifyNtfy.quote(self.user, safe=''))
        elif self.token:
            auth = '{token}@'.format(token=self.pprint(self.token, privacy, safe=''))
        if self.mode == NtfyMode.PRIVATE:
            return '{schema}://{auth}{host}{port}/{targets}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, host=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), targets='/'.join([NotifyNtfy.quote(x, safe='') for x in self.topics]), params=NotifyNtfy.urlencode(params))
        else:
            return '{schema}://{targets}?{params}'.format(schema=self.secure_protocol, targets='/'.join([NotifyNtfy.quote(x, safe='') for x in self.topics]), params=NotifyNtfy.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.topics)

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        if 'priority' in results['qsd'] and len(results['qsd']['priority']):
            results['priority'] = NotifyNtfy.unquote(results['qsd']['priority'])
        if 'attach' in results['qsd'] and len(results['qsd']['attach']):
            results['attach'] = NotifyNtfy.unquote(results['qsd']['attach'])
            _results = NotifyBase.parse_url(results['attach'])
            if _results:
                results['filename'] = None if _results['fullpath'] else basename(_results['fullpath'])
            if 'filename' in results['qsd'] and len(results['qsd']['filename']):
                results['filename'] = basename(NotifyNtfy.unquote(results['qsd']['filename']))
        if 'click' in results['qsd'] and len(results['qsd']['click']):
            results['click'] = NotifyNtfy.unquote(results['qsd']['click'])
        if 'delay' in results['qsd'] and len(results['qsd']['delay']):
            results['delay'] = NotifyNtfy.unquote(results['qsd']['delay'])
        if 'email' in results['qsd'] and len(results['qsd']['email']):
            results['email'] = NotifyNtfy.unquote(results['qsd']['email'])
        if 'tags' in results['qsd'] and len(results['qsd']['tags']):
            results['tags'] = parse_list(NotifyNtfy.unquote(results['qsd']['tags']))
        results['include_image'] = parse_bool(results['qsd'].get('image', NotifyNtfy.template_args['image']['default']))
        if 'avatar_url' in results['qsd']:
            results['avatar_url'] = NotifyNtfy.unquote(results['qsd']['avatar_url'])
        results['targets'] = NotifyNtfy.split_path(results['fullpath'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyNtfy.parse_list(results['qsd']['to'])
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            results['auth'] = NtfyAuth.TOKEN
            results['token'] = NotifyNtfy.unquote(results['qsd']['token'])
        if 'auth' in results['qsd'] and results['qsd']['auth']:
            results['auth'] = NotifyNtfy.unquote(results['qsd']['auth'].strip().lower())
        if not results.get('auth') and results['user'] and (not results['password']):
            results['auth'] = NtfyAuth.TOKEN if NTFY_AUTH_DETECT_RE.match(results['user']) else NtfyAuth.BASIC
        if results.get('auth') == NtfyAuth.TOKEN and (not results.get('token')):
            if results['user'] and (not results['password']):
                results['token'] = NotifyNtfy.unquote(results['user'])
            elif results['password']:
                results['token'] = NotifyNtfy.unquote(results['password'])
        if 'mode' in results['qsd'] and results['qsd']['mode']:
            results['mode'] = NotifyNtfy.unquote(results['qsd']['mode'].strip().lower())
        else:
            results['mode'] = NtfyMode.PRIVATE if (is_hostname(results['host']) or is_ipaddr(results['host'])) and results['targets'] else NtfyMode.CLOUD
        if results['mode'] == NtfyMode.CLOUD:
            if not NotifyNtfy.__auto_cloud_host.search(results['host']):
                results['targets'].insert(0, results['host'])
        elif results['mode'] == NtfyMode.PRIVATE and (not is_hostname(results['host'] or is_ipaddr(results['host']))):
            return None
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Support https://ntfy.sh/topic\n        '
        result = re.match('^(http|ntfy)s?://ntfy\\.sh(?P<topics>/[^?]+)?(?P<params>\\?.+)?$', url, re.I)
        if result:
            mode = 'mode=%s' % NtfyMode.CLOUD
            return NotifyNtfy.parse_url('{schema}://{topics}{params}'.format(schema=NotifyNtfy.secure_protocol, topics=result.group('topics') if result.group('topics') else '', params='?%s' % mode if not result.group('params') else result.group('params') + '&%s' % mode))
        return None