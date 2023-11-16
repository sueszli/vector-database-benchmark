import requests
import base64
from json import dumps
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyImageSize
from ..common import NotifyType
from ..AppriseLocale import gettext_lazy as _

class JSONPayloadField:
    """
    Identifies the fields available in the JSON Payload
    """
    VERSION = 'version'
    TITLE = 'title'
    MESSAGE = 'message'
    ATTACHMENTS = 'attachments'
    MESSAGETYPE = 'type'
METHODS = ('POST', 'GET', 'DELETE', 'PUT', 'HEAD', 'PATCH')

class NotifyJSON(NotifyBase):
    """
    A wrapper for JSON Notifications
    """
    service_name = 'JSON'
    protocol = 'json'
    secure_protocol = 'jsons'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_Custom_JSON'
    attachment_support = True
    image_size = NotifyImageSize.XY_128
    request_rate_per_sec = 0
    json_version = '1.0'
    templates = ('{schema}://{host}', '{schema}://{host}:{port}', '{schema}://{user}@{host}', '{schema}://{user}@{host}:{port}', '{schema}://{user}:{password}@{host}', '{schema}://{user}:{password}@{host}:{port}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}})
    template_args = dict(NotifyBase.template_args, **{'method': {'name': _('Fetch Method'), 'type': 'choice:string', 'values': METHODS, 'default': METHODS[0]}})
    template_kwargs = {'headers': {'name': _('HTTP Header'), 'prefix': '+'}, 'payload': {'name': _('Payload Extras'), 'prefix': ':'}, 'params': {'name': _('GET Params'), 'prefix': '-'}}

    def __init__(self, headers=None, method=None, payload=None, params=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize JSON Object\n\n        headers can be a dictionary of key/value pairs that you want to\n        additionally include as part of the server headers to post with\n\n        '
        super().__init__(**kwargs)
        self.fullpath = kwargs.get('fullpath')
        if not isinstance(self.fullpath, str):
            self.fullpath = ''
        self.method = self.template_args['method']['default'] if not isinstance(method, str) else method.upper()
        if self.method not in METHODS:
            msg = 'The method specified ({}) is invalid.'.format(method)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.params = {}
        if params:
            self.params.update(params)
        self.headers = {}
        if headers:
            self.headers.update(headers)
        self.payload_extras = {}
        if payload:
            self.payload_extras.update(payload)
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
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyJSON.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=NotifyJSON.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}{fullpath}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), fullpath=NotifyJSON.quote(self.fullpath, safe='/') if self.fullpath else '/', params=NotifyJSON.urlencode(params))

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform JSON Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        headers.update(self.headers)
        attachments = []
        if attach and self.attachment_support:
            for attachment in attach:
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                try:
                    with open(attachment.path, 'rb') as f:
                        attachments.append({'filename': attachment.name, 'base64': base64.b64encode(f.read()).decode('utf-8'), 'mimetype': attachment.mimetype})
                except (OSError, IOError) as e:
                    self.logger.warning('An I/O error occurred while reading {}.'.format(attachment.name if attachment else 'attachment'))
                    self.logger.debug('I/O Exception: %s' % str(e))
                    return False
        payload = {JSONPayloadField.VERSION: self.json_version, JSONPayloadField.TITLE: title, JSONPayloadField.MESSAGE: body, JSONPayloadField.ATTACHMENTS: attachments, JSONPayloadField.MESSAGETYPE: notify_type}
        for (key, value) in self.payload_extras.items():
            if key in payload:
                if not value:
                    del payload[key]
                else:
                    payload[value] = payload[key]
                    del payload[key]
            else:
                payload[key] = value
        auth = None
        if self.user:
            auth = (self.user, self.password)
        schema = 'https' if self.secure else 'http'
        url = '%s://%s' % (schema, self.host)
        if isinstance(self.port, int):
            url += ':%d' % self.port
        url += self.fullpath
        self.logger.debug('JSON POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('JSON Payload: %s' % str(payload))
        self.throttle()
        if self.method == 'GET':
            method = requests.get
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
            r = method(url, data=dumps(payload), params=self.params, headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code < 200 or r.status_code >= 300:
                status_str = NotifyJSON.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send JSON %s notification: %s%serror=%s.', self.method, status_str, ', ' if status_str else '', str(r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent JSON %s notification.', self.method)
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending JSON notification to %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        results['payload'] = {NotifyJSON.unquote(x): NotifyJSON.unquote(y) for (x, y) in results['qsd:'].items()}
        results['headers'] = {NotifyJSON.unquote(x): NotifyJSON.unquote(y) for (x, y) in results['qsd+'].items()}
        results['params'] = {NotifyJSON.unquote(x): NotifyJSON.unquote(y) for (x, y) in results['qsd-'].items()}
        if 'method' in results['qsd'] and len(results['qsd']['method']):
            results['method'] = NotifyJSON.unquote(results['qsd']['method'])
        return results