"""
.. module: security_monkey.auditors.iam.iam_ssl
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor::  Patrick Kelley <pkelley@netflix.com> @monkeysecurity

"""
from dateutil.tz import tzutc
from dateutil import parser
from security_monkey.watchers.iam.iam_ssl import IAMSSL
from security_monkey.auditor import Auditor
HEARTBLEED_DATE = '2014-04-01T00:00:00Z'

class IAMSSLAuditor(Auditor):
    index = IAMSSL.index
    i_am_singular = IAMSSL.i_am_singular
    i_am_plural = IAMSSL.i_am_plural

    def __init__(self, accounts=None, debug=False):
        if False:
            i = 10
            return i + 15
        super(IAMSSLAuditor, self).__init__(accounts=accounts, debug=debug)

    def check_issuer(self, cert_item):
        if False:
            for i in range(10):
                print('nop')
        '\n        alert when missing issuer.\n        '
        issuer = cert_item.config.get('issuer', None)
        if issuer and 'ERROR_EXTRACTING_ISSUER' in issuer:
            self.add_issue(10, 'Could not extract valid certificate issuer.', cert_item, notes=issuer)

    def check_cert_size_lt_1024(self, cert_item):
        if False:
            for i in range(10):
                print('nop')
        '\n        alert when a cert is using less than 1024 bits\n        '
        size = cert_item.config.get('size', None)
        if size and size < 1024:
            notes = 'Actual size is {0} bits.'.format(size)
            self.add_issue(10, 'Cert size is less than 1024 bits.', cert_item, notes=notes)

    def check_cert_size_lt_2048(self, cert_item):
        if False:
            return 10
        '\n        alert when a cert is using less than 2048 bits\n        '
        size = cert_item.config.get('size', None)
        if size and 1024 <= size < 2048:
            notes = 'Actual size is {0} bits.'.format(size)
            self.add_issue(3, 'Cert size is less than 2048 bits.', cert_item, notes=notes)

    def check_signature_algorith_for_md5(self, cert_item):
        if False:
            for i in range(10):
                print('nop')
        '\n        alert when a cert is using md5 for the hashing part\n         of the signature algorithm\n        '
        sig_alg = cert_item.config.get('signature_algorithm', None)
        if sig_alg and 'md5' in sig_alg.lower():
            self.add_issue(3, 'Cert uses an MD5 signature Algorithm', cert_item, notes=sig_alg)

    def check_signature_algorith_for_sha1(self, cert_item):
        if False:
            return 10
        '\n        alert when a cert is using sha1 for the hashing part of\n         its signature algorithm.\n        Microsoft and Google are aiming to drop support for sha1 by January 2017.\n        '
        sig_alg = cert_item.config.get('signature_algorithm', None)
        if sig_alg and 'sha1' in sig_alg.lower():
            self.add_issue(1, 'Cert uses an SHA1 signature Algorithm', cert_item, notes=sig_alg)

    def check_upcoming_expiration(self, cert_item):
        if False:
            return 10
        "\n        alert when a cert's expiration is within 30 days\n        "
        expiration = cert_item.config.get('expiration', None)
        if expiration:
            expiration = parser.parse(expiration)
            now = expiration.now(tzutc())
            time_to_expiration = (expiration - now).days
            if 0 <= time_to_expiration <= 30:
                notes = 'Expires on {0}.'.format(str(expiration))
                self.add_issue(10, 'Cert will expire soon.', cert_item, notes=notes)

    def check_future_expiration(self, cert_item):
        if False:
            while True:
                i = 10
        "\n        alert when a cert's expiration is within 60 days\n        "
        expiration = cert_item.config.get('expiration', None)
        if expiration:
            expiration = parser.parse(expiration)
            now = expiration.now(tzutc())
            time_to_expiration = (expiration - now).days
            if 0 <= time_to_expiration <= 60:
                notes = 'Expires on {0}.'.format(str(expiration))
                self.add_issue(5, 'Cert will expire soon.', cert_item, notes=notes)

    def check_expired(self, cert_item):
        if False:
            while True:
                i = 10
        "\n        alert when a cert's expiration is within 30 days\n        "
        expiration = cert_item.config.get('expiration', None)
        if expiration:
            expiration = parser.parse(expiration)
            now = expiration.now(tzutc())
            time_to_expiration = (expiration - now).days
            if time_to_expiration < 0:
                notes = 'Expired on {0}.'.format(str(expiration))
                self.add_issue(10, 'Cert has expired.', cert_item, notes=notes)

    def check_upload_date_for_heartbleed(self, cert_item):
        if False:
            i = 10
            return i + 15
        '\n        alert when a cert was uploaded pre-heartbleed.\n        '
        upload = cert_item.config.get('upload_date', None)
        if upload:
            upload = parser.parse(upload)
            heartbleed = parser.parse(HEARTBLEED_DATE)
            if upload < heartbleed:
                notes = 'Cert was uploaded {0} days before heartbleed.'.format((heartbleed - upload).days)
                self.add_issue(10, 'Cert may have been compromised by heartbleed.', cert_item, notes=notes)