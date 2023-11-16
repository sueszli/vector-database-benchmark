import base64
import datetime
import re
import requests
from cryptography import x509
from cryptography.exceptions import InvalidSignature as _InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.hashes import SHA256
from urllib3.util import parse_url
_signing_url_host_re = re.compile('^sns\\.[a-zA-Z0-9\\-]{3,}\\.amazonaws\\.com(\\.cn)?$')

class InvalidMessageError(Exception):
    pass

class MessageVerifier:

    def __init__(self, *, topics, session=None):
        if False:
            return 10
        self.topics = topics
        self.http = session if session is not None else requests.session()

    def verify(self, message):
        if False:
            while True:
                i = 10
        if message.get('SignatureVersion') == '2':
            self._validate_v2_signature(message)
        else:
            raise InvalidMessageError('Unknown SignatureVersion')
        self._validate_timestamp(message['Timestamp'])
        self._validate_topic(message['TopicArn'])

    def _validate_topic(self, topic):
        if False:
            print('Hello World!')
        if topic not in self.topics:
            raise InvalidMessageError('Invalid TopicArn')

    def _validate_timestamp(self, timestamp_str):
        if False:
            i = 10
            return i + 15
        now = datetime.datetime.utcnow()
        try:
            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            raise InvalidMessageError('Unknown Timestamp format')
        age = now - timestamp
        if age > datetime.timedelta(hours=1):
            raise InvalidMessageError('Message has expired')

    def _validate_v2_signature(self, message):
        if False:
            i = 10
            return i + 15
        pubkey = self._get_pubkey(message['SigningCertURL'])
        signature = self._get_signature(message)
        data = self._get_data_to_sign(message)
        try:
            pubkey.verify(signature, data, PKCS1v15(), SHA256())
        except _InvalidSignature:
            raise InvalidMessageError('Invalid Signature') from None

    def _get_pubkey(self, cert_url):
        if False:
            return 10
        cert_url_p = parse_url(cert_url)
        cert_scheme = cert_url_p.scheme
        cert_host = cert_url_p.netloc
        if cert_scheme != 'https':
            raise InvalidMessageError('Invalid scheme for SigningCertURL')
        if _signing_url_host_re.fullmatch(cert_host) is None:
            raise InvalidMessageError('Invalid location for SigningCertURL')
        resp = self.http.get(cert_url)
        resp.raise_for_status()
        cert = x509.load_pem_x509_certificate(resp.content, default_backend())
        return cert.public_key()

    def _get_signature(self, message):
        if False:
            while True:
                i = 10
        return base64.b64decode(message['Signature'])

    def _get_data_to_sign(self, message):
        if False:
            return 10
        if message['Type'] == 'Notification':
            parts = self._get_parts_to_sign_notification(message)
        elif message['Type'] in {'SubscriptionConfirmation', 'UnsubscribeConfirmation'}:
            parts = self._get_parts_to_sign_subscription(message)
        else:
            raise InvalidMessageError('Invalid Type')
        return ('\n'.join(parts) + '\n').encode('utf8')

    def _get_parts_to_sign_notification(self, message):
        if False:
            print('Hello World!')
        parts = ['Message', message['Message'], 'MessageId', message['MessageId']]
        if 'Subject' in message:
            parts.extend(['Subject', message['Subject']])
        parts.extend(['Timestamp', message['Timestamp'], 'TopicArn', message['TopicArn'], 'Type', message['Type']])
        return parts

    def _get_parts_to_sign_subscription(self, message):
        if False:
            for i in range(10):
                print('nop')
        return ['Message', message['Message'], 'MessageId', message['MessageId'], 'SubscribeURL', message['SubscribeURL'], 'Timestamp', message['Timestamp'], 'Token', message['Token'], 'TopicArn', message['TopicArn'], 'Type', message['Type']]