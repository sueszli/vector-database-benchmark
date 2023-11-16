import ssl
import re
from time import sleep
from datetime import datetime
from os.path import isfile
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import parse_list
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
NOTIFY_MQTT_SUPPORT_ENABLED = False
try:
    import paho.mqtt.client as mqtt
    NOTIFY_MQTT_SUPPORT_ENABLED = True
    MQTT_PROTOCOL_MAP = {'311': mqtt.MQTTv311, '31': mqtt.MQTTv31, '5': mqtt.MQTTv5, '50': mqtt.MQTTv5}
except ImportError:
    MQTT_PROTOCOL_MAP = {}
HUMAN_MQTT_PROTOCOL_MAP = {'v3.1.1': '311', 'v3.1': '31', 'v5.0': '5'}

class NotifyMQTT(NotifyBase):
    """
    A wrapper for MQTT Notifications
    """
    enabled = NOTIFY_MQTT_SUPPORT_ENABLED
    requirements = {'packages_required': 'paho-mqtt'}
    service_name = 'MQTT Notification'
    protocol = 'mqtt'
    secure_protocol = 'mqtts'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_mqtt'
    title_maxlen = 0
    body_maxlen = 268435455
    request_rate_per_sec = 0.5
    mqtt_insecure_port = 1883
    mqtt_secure_port = 8883
    mqtt_keepalive = 30
    mqtt_transport = 'tcp'
    mqtt_block_time_sec = 0.2
    mqtt_inflight_messages = 200
    templates = ('{schema}://{user}@{host}/{topic}', '{schema}://{user}@{host}:{port}/{topic}', '{schema}://{user}:{password}@{host}/{topic}', '{schema}://{user}:{password}@{host}:{port}/{topic}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('User Name'), 'type': 'string', 'required': True}, 'password': {'name': _('Password'), 'type': 'string', 'private': True, 'required': True}, 'topic': {'name': _('Target Queue'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'qos': {'name': _('QOS'), 'type': 'int', 'default': 0, 'min': 0, 'max': 2}, 'version': {'name': _('Version'), 'type': 'choice:string', 'values': HUMAN_MQTT_PROTOCOL_MAP, 'default': 'v3.1.1'}, 'client_id': {'name': _('Client ID'), 'type': 'string'}, 'session': {'name': _('Use Session'), 'type': 'bool', 'default': False}})

    def __init__(self, targets=None, version=None, qos=None, client_id=None, session=None, **kwargs):
        if False:
            return 10
        '\n        Initialize MQTT Object\n        '
        super().__init__(**kwargs)
        self.topics = parse_list(targets)
        if version is None:
            self.version = self.template_args['version']['default']
        else:
            self.version = version
        self.client_id = client_id
        self.session = self.template_args['session']['default'] if session is None or not self.client_id else parse_bool(session)
        try:
            self.qos = self.template_args['qos']['default'] if qos is None else int(qos)
            if self.qos < self.template_args['qos']['min'] or self.qos > self.template_args['qos']['max']:
                raise ValueError('')
        except (ValueError, TypeError):
            msg = 'An invalid MQTT QOS ({}) was specified.'.format(qos)
            self.logger.warning(msg)
            raise TypeError(msg)
        if not self.port:
            self.port = self.mqtt_secure_port if self.secure else self.mqtt_insecure_port
        self.ca_certs = None
        if self.secure:
            self.ca_certs = next((cert for cert in self.CA_CERTIFICATE_FILE_LOCATIONS if isfile(cert)), None)
        try:
            self.mqtt_protocol = MQTT_PROTOCOL_MAP[re.sub('[^0-9]+', '', self.version)]
        except KeyError:
            msg = 'An invalid MQTT Protocol version ({}) was specified.'.format(version)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.client = mqtt.Client(client_id=self.client_id, clean_session=not self.session, userdata=None, protocol=self.mqtt_protocol, transport=self.mqtt_transport)
        self.client.max_inflight_messages_set(self.mqtt_inflight_messages)
        self.__initial_connect = True

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform MQTT Notification\n        '
        if len(self.topics) == 0:
            self.logger.warning('There were no MQTT topics to notify.')
            return False
        url = '{host}:{port}'.format(host=self.host, port=self.port)
        try:
            if self.__initial_connect:
                if self.user:
                    self.client.username_pw_set(self.user, password=self.password)
                if self.secure:
                    if self.ca_certs is None:
                        self.logger.error('MQTT secure communication can not be verified, CA certificates file missing')
                        return False
                    self.client.tls_set(ca_certs=self.ca_certs, certfile=None, keyfile=None, cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS, ciphers=None)
                    self.client.tls_insecure_set(not self.verify_certificate)
                if self.client.connect(self.host, port=self.port, keepalive=self.mqtt_keepalive) != mqtt.MQTT_ERR_SUCCESS:
                    self.logger.warning('An MQTT connection could not be established for {}'.format(url))
                    return False
                self.client.loop_start()
                sleep(0.01)
                self.__initial_connect = False
            topics = list(self.topics)
            has_error = False
            while len(topics) > 0 and (not has_error):
                topic = topics.pop()
                url = '{host}:{port}/{topic}'.format(host=self.host, port=self.port, topic=topic)
                self.throttle()
                if not self.client.is_connected() and self.client.reconnect() != mqtt.MQTT_ERR_SUCCESS:
                    self.logger.warning('An MQTT connection could not be sustained for {}'.format(url))
                    has_error = True
                    break
                self.logger.debug('MQTT POST URL: {} (cert_verify={})'.format(url, self.verify_certificate))
                self.logger.debug('MQTT Payload: %s' % str(body))
                result = self.client.publish(topic, payload=body, qos=self.qos, retain=False)
                if result.rc != mqtt.MQTT_ERR_SUCCESS:
                    self.logger.warning('An error (rc={}) occured when sending MQTT to {}'.format(result.rc, url))
                    has_error = True
                    break
                elif not result.is_published():
                    self.logger.debug('Blocking until MQTT payload is published...')
                    reference = datetime.now()
                    while not has_error and (not result.is_published()):
                        sleep(self.mqtt_block_time_sec)
                        elapsed = (datetime.now() - reference).total_seconds()
                        if elapsed >= self.socket_read_timeout:
                            self.logger.warning('The MQTT message could not be delivered')
                            has_error = True
        except ConnectionError as e:
            self.logger.warning('MQTT Connection Error received from {}'.format(url))
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        except ssl.CertificateError as e:
            self.logger.warning('MQTT SSL Certificate Error received from {}'.format(url))
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        except ValueError as e:
            self.logger.warning('MQTT Publishing error received: from {}'.format(url))
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        if not has_error:
            self.logger.info('Sent MQTT notification')
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'version': self.version, 'qos': str(self.qos), 'session': 'yes' if self.session else 'no'}
        if self.client_id:
            params['client_id'] = self.client_id
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyMQTT.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=NotifyMQTT.quote(self.user, safe=''))
        default_port = self.mqtt_secure_port if self.secure else self.mqtt_insecure_port
        return '{schema}://{auth}{hostname}{port}/{targets}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), targets=','.join([NotifyMQTT.quote(x, safe='/') for x in self.topics]), params=NotifyMQTT.urlencode(params))

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.topics)

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        There are no parameters nessisary for this protocol; simply having\n        windows:// is all you need.  This function just makes sure that\n        is in place.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        try:
            results['targets'] = parse_list(NotifyMQTT.unquote(results['fullpath'].lstrip('/')))
        except AttributeError:
            results['targets'] = []
        if 'version' in results['qsd'] and len(results['qsd']['version']):
            results['version'] = NotifyMQTT.unquote(results['qsd']['version'])
        if 'client_id' in results['qsd'] and len(results['qsd']['client_id']):
            results['client_id'] = NotifyMQTT.unquote(results['qsd']['client_id'])
        if 'session' in results['qsd'] and len(results['qsd']['session']):
            results['session'] = parse_bool(results['qsd']['session'])
        if 'qos' in results['qsd'] and len(results['qsd']['qos']):
            results['qos'] = NotifyMQTT.unquote(results['qsd']['qos'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'].extend(NotifyMQTT.parse_list(results['qsd']['to']))
        return results

    @property
    def CA_CERTIFICATE_FILE_LOCATIONS(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return possible locations to root certificate authority (CA) bundles.\n\n        Taken from https://golang.org/src/crypto/x509/root_linux.go\n        TODO: Maybe refactor to a general utility function?\n        '
        candidates = ['/etc/ssl/certs/ca-certificates.crt', '/etc/pki/tls/certs/ca-bundle.crt', '/etc/ssl/ca-bundle.pem', '/etc/pki/tls/cacert.pem', '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem', '/usr/local/etc/ca-certificates/cert.pem']
        try:
            import certifi
            candidates.append(certifi.where())
        except ImportError:
            pass
        return candidates