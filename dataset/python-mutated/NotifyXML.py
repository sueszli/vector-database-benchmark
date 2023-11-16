import re
import requests
import base64
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyImageSize
from ..common import NotifyType
from ..AppriseLocale import gettext_lazy as _

class XMLPayloadField:
    """
    Identifies the fields available in the JSON Payload
    """
    VERSION = 'Version'
    TITLE = 'Subject'
    MESSAGE = 'Message'
    MESSAGETYPE = 'MessageType'
METHODS = ('POST', 'GET', 'DELETE', 'PUT', 'HEAD', 'PATCH')

class NotifyXML(NotifyBase):
    """
    A wrapper for XML Notifications
    """
    service_name = 'XML'
    protocol = 'xml'
    secure_protocol = 'xmls'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_Custom_XML'
    attachment_support = True
    image_size = NotifyImageSize.XY_128
    request_rate_per_sec = 0
    xsd_ver = '1.1'
    xsd_default_url = 'https://raw.githubusercontent.com/caronc/apprise/master/apprise/assets/NotifyXML-{version}.xsd'
    templates = ('{schema}://{host}', '{schema}://{host}:{port}', '{schema}://{user}@{host}', '{schema}://{user}@{host}:{port}', '{schema}://{user}:{password}@{host}', '{schema}://{user}:{password}@{host}:{port}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}})
    template_args = dict(NotifyBase.template_args, **{'method': {'name': _('Fetch Method'), 'type': 'choice:string', 'values': METHODS, 'default': METHODS[0]}})
    template_kwargs = {'headers': {'name': _('HTTP Header'), 'prefix': '+'}, 'payload': {'name': _('Payload Extras'), 'prefix': ':'}, 'params': {'name': _('GET Params'), 'prefix': '-'}}

    def __init__(self, headers=None, method=None, payload=None, params=None, **kwargs):
        if False:
            return 10
        '\n        Initialize XML Object\n\n        headers can be a dictionary of key/value pairs that you want to\n        additionally include as part of the server headers to post with\n\n        '
        super().__init__(**kwargs)
        self.payload = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<soapenv:Envelope\n    xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"\n    xmlns:xsd="http://www.w3.org/2001/XMLSchema"\n    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n    <soapenv:Body>\n        <Notification{{XSD_URL}}>\n            {{CORE}}\n            {{ATTACHMENTS}}\n       </Notification>\n    </soapenv:Body>\n</soapenv:Envelope>'
        self.fullpath = kwargs.get('fullpath')
        if not isinstance(self.fullpath, str):
            self.fullpath = ''
        self.method = self.template_args['method']['default'] if not isinstance(method, str) else method.upper()
        if self.method not in METHODS:
            msg = 'The method specified ({}) is invalid.'.format(method)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.payload_map = {XMLPayloadField.VERSION: XMLPayloadField.VERSION, XMLPayloadField.TITLE: XMLPayloadField.TITLE, XMLPayloadField.MESSAGE: XMLPayloadField.MESSAGE, XMLPayloadField.MESSAGETYPE: XMLPayloadField.MESSAGETYPE}
        self.params = {}
        if params:
            self.params.update(params)
        self.headers = {}
        if headers:
            self.headers.update(headers)
        self.payload_overrides = {}
        self.payload_extras = {}
        if payload:
            for (k, v) in payload.items():
                key = re.sub('[^A-Za-z0-9_-]*', '', k)
                if not key:
                    self.logger.warning('Ignoring invalid XML Stanza element name({})'.format(k))
                    continue
                if key in self.payload_map:
                    self.payload_map[key] = v
                    self.payload_overrides[key] = v
                else:
                    self.payload_extras[key] = v
        self.xsd_url = None if self.payload_overrides or self.payload_extras else self.xsd_default_url.format(version=self.xsd_ver)
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'method': self.method}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        params.update({'-{}'.format(k): v for (k, v) in self.params.items()})
        params.update({':{}'.format(k): v for (k, v) in self.payload_extras.items()})
        params.update({':{}'.format(k): v for (k, v) in self.payload_overrides.items()})
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyXML.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=NotifyXML.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}{fullpath}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), fullpath=NotifyXML.quote(self.fullpath, safe='/') if self.fullpath else '/', params=NotifyXML.urlencode(params))

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform XML Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/xml'}
        headers.update(self.headers)
        xml_attachments = ''
        payload_base = {}
        for (key, value) in ((XMLPayloadField.VERSION, self.xsd_ver), (XMLPayloadField.TITLE, NotifyXML.escape_html(title, whitespace=False)), (XMLPayloadField.MESSAGE, NotifyXML.escape_html(body, whitespace=False)), (XMLPayloadField.MESSAGETYPE, NotifyXML.escape_html(notify_type, whitespace=False))):
            if not self.payload_map[key]:
                continue
            payload_base[self.payload_map[key]] = value
        payload_base.update({k: NotifyXML.escape_html(v, whitespace=False) for (k, v) in self.payload_extras.items()})
        xml_base = ''.join(['<{}>{}</{}>'.format(k, v, k) for (k, v) in payload_base.items()])
        attachments = []
        if attach and self.attachment_support:
            for attachment in attach:
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                try:
                    with open(attachment.path, 'rb') as f:
                        entry = '<Attachment filename="{}" mimetype="{}">'.format(NotifyXML.escape_html(attachment.name, whitespace=False), NotifyXML.escape_html(attachment.mimetype, whitespace=False))
                        entry += base64.b64encode(f.read()).decode('utf-8')
                        entry += '</Attachment>'
                        attachments.append(entry)
                except (OSError, IOError) as e:
                    self.logger.warning('An I/O error occurred while reading {}.'.format(attachment.name if attachment else 'attachment'))
                    self.logger.debug('I/O Exception: %s' % str(e))
                    return False
            xml_attachments = '<Attachments format="base64">' + ''.join(attachments) + '</Attachments>'
        re_map = {'{{XSD_URL}}': f' xmlns:xsi="{self.xsd_url}"' if self.xsd_url else '', '{{ATTACHMENTS}}': xml_attachments, '{{CORE}}': xml_base}
        re_table = re.compile('(' + '|'.join(re_map.keys()) + ')', re.IGNORECASE)
        auth = None
        if self.user:
            auth = (self.user, self.password)
        schema = 'https' if self.secure else 'http'
        url = '%s://%s' % (schema, self.host)
        if isinstance(self.port, int):
            url += ':%d' % self.port
        url += self.fullpath
        payload = re_table.sub(lambda x: re_map[x.group()], self.payload)
        self.logger.debug('XML POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('XML Payload: %s' % str(payload))
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
            r = method(url, data=payload, headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code < 200 or r.status_code >= 300:
                status_str = NotifyXML.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send JSON %s notification: %s%serror=%s.', self.method, status_str, ', ' if status_str else '', str(r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent XML %s notification.', self.method)
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending XML notification to %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        results['payload'] = {NotifyXML.unquote(x): NotifyXML.unquote(y) for (x, y) in results['qsd:'].items()}
        results['headers'] = {NotifyXML.unquote(x): NotifyXML.unquote(y) for (x, y) in results['qsd+'].items()}
        results['params'] = {NotifyXML.unquote(x): NotifyXML.unquote(y) for (x, y) in results['qsd-'].items()}
        if 'method' in results['qsd'] and len(results['qsd']['method']):
            results['method'] = NotifyXML.unquote(results['qsd']['method'])
        return results