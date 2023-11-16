import re
import requests
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyImageSize
from ..common import NotifyType
from ..AppriseLocale import gettext_lazy as _

class FORMPayloadField:
    """
    Identifies the fields available in the FORM Payload
    """
    VERSION = 'version'
    TITLE = 'title'
    MESSAGE = 'message'
    MESSAGETYPE = 'type'
METHODS = ('POST', 'GET', 'DELETE', 'PUT', 'HEAD', 'PATCH')

class NotifyForm(NotifyBase):
    """
    A wrapper for Form Notifications
    """
    __attach_as_re = re.compile('((?P<match1>(?P<id1a>[a-z0-9_-]+)?(?P<wc1>[*?+$:.%]+)(?P<id1b>[a-z0-9_-]+))|(?P<match2>(?P<id2>[a-z0-9_-]+)(?P<wc2>[*?+$:.%]?)))', re.IGNORECASE)
    attach_as_count = '{:02d}'
    attach_as_default = f'file{attach_as_count}'
    service_name = 'Form'
    protocol = 'form'
    secure_protocol = 'forms'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_Custom_Form'
    attachment_support = True
    image_size = NotifyImageSize.XY_128
    request_rate_per_sec = 0
    form_version = '1.0'
    templates = ('{schema}://{host}', '{schema}://{host}:{port}', '{schema}://{user}@{host}', '{schema}://{user}@{host}:{port}', '{schema}://{user}:{password}@{host}', '{schema}://{user}:{password}@{host}:{port}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}})
    template_args = dict(NotifyBase.template_args, **{'method': {'name': _('Fetch Method'), 'type': 'choice:string', 'values': METHODS, 'default': METHODS[0]}, 'attach-as': {'name': _('Attach File As'), 'type': 'string', 'default': 'file*', 'map_to': 'attach_as'}})
    template_kwargs = {'headers': {'name': _('HTTP Header'), 'prefix': '+'}, 'payload': {'name': _('Payload Extras'), 'prefix': ':'}, 'params': {'name': _('GET Params'), 'prefix': '-'}}

    def __init__(self, headers=None, method=None, payload=None, params=None, attach_as=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Form Object\n\n        headers can be a dictionary of key/value pairs that you want to\n        additionally include as part of the server headers to post with\n\n        '
        super().__init__(**kwargs)
        self.fullpath = kwargs.get('fullpath')
        if not isinstance(self.fullpath, str):
            self.fullpath = ''
        self.method = self.template_args['method']['default'] if not isinstance(method, str) else method.upper()
        if self.method not in METHODS:
            msg = 'The method specified ({}) is invalid.'.format(method)
            self.logger.warning(msg)
            raise TypeError(msg)
        if not isinstance(attach_as, str):
            self.attach_as = self.attach_as_default
            self.attach_multi_support = True
        else:
            result = self.__attach_as_re.match(attach_as.strip())
            if not result:
                msg = 'The attach-as specified ({}) is invalid.'.format(attach_as)
                self.logger.warning(msg)
                raise TypeError(msg)
            self.attach_as = ''
            self.attach_multi_support = False
            if result.group('match1'):
                if result.group('id1a'):
                    self.attach_as += result.group('id1a')
                self.attach_as += self.attach_as_count
                self.attach_multi_support = True
                self.attach_as += result.group('id1b')
            else:
                self.attach_as += result.group('id2')
                if result.group('wc2'):
                    self.attach_as += self.attach_as_count
                    self.attach_multi_support = True
        self.payload_map = {FORMPayloadField.VERSION: FORMPayloadField.VERSION, FORMPayloadField.TITLE: FORMPayloadField.TITLE, FORMPayloadField.MESSAGE: FORMPayloadField.MESSAGE, FORMPayloadField.MESSAGETYPE: FORMPayloadField.MESSAGETYPE}
        self.params = {}
        if params:
            self.params.update(params)
        self.headers = {}
        if headers:
            self.headers.update(headers)
        self.payload_overrides = {}
        self.payload_extras = {}
        if payload:
            self.payload_extras.update(payload)
            for key in list(self.payload_extras.keys()):
                if key in self.payload_map:
                    self.payload_map[key] = self.payload_extras[key]
                    self.payload_overrides[key] = self.payload_extras[key]
                    del self.payload_extras[key]
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'method': self.method}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        params.update({'-{}'.format(k): v for (k, v) in self.params.items()})
        params.update({':{}'.format(k): v for (k, v) in self.payload_extras.items()})
        params.update({':{}'.format(k): v for (k, v) in self.payload_overrides.items()})
        if self.attach_as != self.attach_as_default:
            params['attach-as'] = self.attach_as
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyForm.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=NotifyForm.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}{fullpath}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), fullpath=NotifyForm.quote(self.fullpath, safe='/') if self.fullpath else '/', params=NotifyForm.urlencode(params))

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform Form Notification\n        '
        headers = {'User-Agent': self.app_id}
        headers.update(self.headers)
        files = []
        if attach and self.attachment_support:
            for (no, attachment) in enumerate(attach, start=1):
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                try:
                    files.append((self.attach_as.format(no) if self.attach_multi_support else self.attach_as, (attachment.name, open(attachment.path, 'rb'), attachment.mimetype)))
                except (OSError, IOError) as e:
                    self.logger.warning('An I/O error occurred while opening {}.'.format(attachment.name if attachment else 'attachment'))
                    self.logger.debug('I/O Exception: %s' % str(e))
                    return False
            if not self.attach_multi_support and no > 1:
                self.logger.warning('Multiple attachments provided while form:// Multi-Attachment Support not enabled')
        payload = {}
        for (key, value) in ((FORMPayloadField.VERSION, self.form_version), (FORMPayloadField.TITLE, title), (FORMPayloadField.MESSAGE, body), (FORMPayloadField.MESSAGETYPE, notify_type)):
            if not self.payload_map[key]:
                continue
            payload[self.payload_map[key]] = value
        payload.update(self.payload_extras)
        auth = None
        if self.user:
            auth = (self.user, self.password)
        schema = 'https' if self.secure else 'http'
        url = '%s://%s' % (schema, self.host)
        if isinstance(self.port, int):
            url += ':%d' % self.port
        url += self.fullpath
        self.logger.debug('Form %s URL: %s (cert_verify=%r)' % (self.method, url, self.verify_certificate))
        self.logger.debug('Form Payload: %s' % str(payload))
        self.throttle()
        if self.method == 'GET':
            method = requests.get
            payload.update(self.params)
        elif self.method == 'PUT':
            method = requests.put
        elif self.method == 'PATCH':
            method = requests.patch
        elif self.method == 'DELETE':
            method = requests.delete
        elif self.method == 'HEAD':
            method = requests.head
        else:
            method = requests.post
        try:
            r = method(url, files=None if not files else files, data=payload if self.method != 'GET' else None, params=payload if self.method == 'GET' else self.params, headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code < 200 or r.status_code >= 300:
                status_str = NotifyForm.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Form %s notification: %s%serror=%s.', self.method, status_str, ', ' if status_str else '', str(r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Form %s notification.', self.method)
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Form notification to %s.' % self.host)
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
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        results['payload'] = {NotifyForm.unquote(x): NotifyForm.unquote(y) for (x, y) in results['qsd:'].items()}
        results['headers'] = {NotifyForm.unquote(x): NotifyForm.unquote(y) for (x, y) in results['qsd+'].items()}
        results['params'] = {NotifyForm.unquote(x): NotifyForm.unquote(y) for (x, y) in results['qsd-'].items()}
        if 'attach-as' in results['qsd'] and len(results['qsd']['attach-as']):
            results['attach_as'] = results['qsd']['attach-as']
        if 'method' in results['qsd'] and len(results['qsd']['method']):
            results['method'] = NotifyForm.unquote(results['qsd']['method'])
        return results