import re
import requests
from json import dumps
import base64
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class AppriseAPIMethod:
    """
    Defines the method to post data tot he remote server
    """
    JSON = 'json'
    FORM = 'form'
APPRISE_API_METHODS = (AppriseAPIMethod.FORM, AppriseAPIMethod.JSON)

class NotifyAppriseAPI(NotifyBase):
    """
    A wrapper for Apprise (Persistent) API Notifications
    """
    service_name = 'Apprise API'
    service_url = 'https://github.com/caronc/apprise-api'
    protocol = 'apprise'
    secure_protocol = 'apprises'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_apprise_api'
    attachment_support = True
    socket_read_timeout = 30.0
    request_rate_per_sec = 0.0
    templates = ('{schema}://{host}/{token}', '{schema}://{host}:{port}/{token}', '{schema}://{user}@{host}/{token}', '{schema}://{user}@{host}:{port}/{token}', '{schema}://{user}:{password}@{host}/{token}', '{schema}://{user}:{password}@{host}:{port}/{token}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}, 'token': {'name': _('Token'), 'type': 'string', 'required': True, 'private': True, 'regex': ('^[A-Z0-9_-]{1,32}$', 'i')}})
    template_args = dict(NotifyBase.template_args, **{'tags': {'name': _('Tags'), 'type': 'string'}, 'method': {'name': _('Query Method'), 'type': 'choice:string', 'values': APPRISE_API_METHODS, 'default': APPRISE_API_METHODS[0]}, 'to': {'alias_of': 'token'}})
    template_kwargs = {'headers': {'name': _('HTTP Header'), 'prefix': '+'}}

    def __init__(self, token=None, tags=None, method=None, headers=None, **kwargs):
        if False:
            return 10
        '\n        Initialize Apprise API Object\n\n        headers can be a dictionary of key/value pairs that you want to\n        additionally include as part of the server headers to post with\n\n        '
        super().__init__(**kwargs)
        self.token = validate_regex(token, *self.template_tokens['token']['regex'])
        if not self.token:
            msg = 'The Apprise API token specified ({}) is invalid.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.method = self.template_args['method']['default'] if not isinstance(method, str) else method.lower()
        if self.method not in APPRISE_API_METHODS:
            msg = 'The method specified ({}) is invalid.'.format(method)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.__tags = parse_list(tags)
        self.headers = {}
        if headers:
            self.headers.update(headers)
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'method': self.method}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        if self.__tags:
            params['tags'] = ','.join([x for x in self.__tags])
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyAppriseAPI.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=NotifyAppriseAPI.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        fullpath = self.fullpath.strip('/')
        return '{schema}://{auth}{hostname}{port}{fullpath}{token}/?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), fullpath='/{}/'.format(NotifyAppriseAPI.quote(fullpath, safe='/')) if fullpath else '/', token=self.pprint(self.token, privacy, safe=''), params=NotifyAppriseAPI.urlencode(params))

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform Apprise API Notification\n        '
        headers = {'User-Agent': self.app_id}
        headers.update(self.headers)
        attachments = []
        files = []
        if attach and self.attachment_support:
            for (no, attachment) in enumerate(attach, start=1):
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                try:
                    if self.method == AppriseAPIMethod.JSON:
                        with open(attachment.path, 'rb') as f:
                            attachments.append({'filename': attachment.name, 'base64': base64.b64encode(f.read()).decode('utf-8'), 'mimetype': attachment.mimetype})
                    else:
                        files.append(('file{:02d}'.format(no), (attachment.name, open(attachment.path, 'rb'), attachment.mimetype)))
                except (OSError, IOError) as e:
                    self.logger.warning('An I/O error occurred while reading {}.'.format(attachment.name if attachment else 'attachment'))
                    self.logger.debug('I/O Exception: %s' % str(e))
                    return False
        payload = {'title': title, 'body': body, 'type': notify_type, 'format': self.notify_format}
        if self.method == AppriseAPIMethod.JSON:
            headers['Content-Type'] = 'application/json'
            if attachments:
                payload['attachments'] = attachments
            payload = dumps(payload)
        if self.__tags:
            payload['tag'] = self.__tags
        auth = None
        if self.user:
            auth = (self.user, self.password)
        schema = 'https' if self.secure else 'http'
        url = '%s://%s' % (schema, self.host)
        if isinstance(self.port, int):
            url += ':%d' % self.port
        fullpath = self.fullpath.strip('/')
        url += '{}'.format('/' + fullpath) if fullpath else ''
        url += '/notify/{}'.format(self.token)
        headers.update({'Accept': 'application/json', 'X-Apprise-ID': self.asset._uid, 'X-Apprise-Recursion-Count': str(self.asset._recursion + 1)})
        self.logger.debug('Apprise API POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('Apprise API Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(url, data=payload, headers=headers, auth=auth, files=files if files else None, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyAppriseAPI.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Apprise API notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Apprise API notification; method=%s.', self.method)
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Apprise API notification to %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        except (OSError, IOError) as e:
            self.logger.warning('An I/O error occurred while reading one of the attached files.')
            self.logger.debug('I/O Exception: %s' % str(e))
            return False
        finally:
            for file in files:
                file[1][1].close()
        return True

    @staticmethod
    def parse_native_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Support http://hostname/notify/token and\n                http://hostname/path/notify/token\n        '
        result = re.match('^http(?P<secure>s?)://(?P<hostname>[A-Z0-9._-]+)(:(?P<port>[0-9]+))?(?P<path>/[^?]+?)?/notify/(?P<token>[A-Z0-9_-]{1,32})/?(?P<params>\\?.+)?$', url, re.I)
        if result:
            return NotifyAppriseAPI.parse_url('{schema}://{hostname}{port}{path}/{token}/{params}'.format(schema=NotifyAppriseAPI.secure_protocol if result.group('secure') else NotifyAppriseAPI.protocol, hostname=result.group('hostname'), port='' if not result.group('port') else ':{}'.format(result.group('port')), path='' if not result.group('path') else result.group('path'), token=result.group('token'), params='' if not result.group('params') else '?{}'.format(result.group('params'))))
        return None

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        results['headers'] = {NotifyAppriseAPI.unquote(x): NotifyAppriseAPI.unquote(y) for (x, y) in results['qsd+'].items()}
        if 'tags' in results['qsd'] and len(results['qsd']['tags']):
            results['tags'] = NotifyAppriseAPI.parse_list(results['qsd']['tags'])
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            results['token'] = NotifyAppriseAPI.unquote(results['qsd']['token'])
        elif 'to' in results['qsd'] and len(results['qsd']['to']):
            results['token'] = NotifyAppriseAPI.unquote(results['qsd']['to'])
        else:
            entries = NotifyAppriseAPI.split_path(results['fullpath'])
            if entries:
                results['token'] = entries[-1]
                entries = entries[:-1]
                results['fullpath'] = '/'.join(entries)
        if 'method' in results['qsd'] and len(results['qsd']['method']):
            results['method'] = NotifyAppriseAPI.unquote(results['qsd']['method'])
        return results