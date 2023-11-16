from __future__ import absolute_import, division, print_function, unicode_literals
from . import utils
from .otp import OTP
from .compat import str

class HOTP(OTP):
    """
    Handler for HMAC-based OTP counters.
    """

    def at(self, count):
        if False:
            while True:
                i = 10
        '\n        Generates the OTP for the given count.\n\n        :param count: the OTP HMAC counter\n        :type count: int\n        :returns: OTP\n        :rtype: str\n        '
        return self.generate_otp(count)

    def verify(self, otp, counter):
        if False:
            return 10
        '\n        Verifies the OTP passed in against the current counter OTP.\n\n        :param otp: the OTP to check against\n        :type otp: str\n        :param count: the OTP HMAC counter\n        :type count: int\n        '
        return utils.strings_equal(str(otp), str(self.at(counter)))

    def provisioning_uri(self, name, initial_count=0, issuer_name=None):
        if False:
            while True:
                i = 10
        '\n        Returns the provisioning URI for the OTP.  This can then be\n        encoded in a QR Code and used to provision an OTP app like\n        Google Authenticator.\n\n        See also:\n            https://github.com/google/google-authenticator/wiki/Key-Uri-Format\n\n        :param name: name of the user account\n        :type name: str\n        :param initial_count: starting HMAC counter value, defaults to 0\n        :type initial_count: int\n        :param issuer_name: the name of the OTP issuer; this will be the\n            organization title of the OTP entry in Authenticator\n        :returns: provisioning URI\n        :rtype: str\n        '
        return utils.build_uri(self.secret, name, initial_count=initial_count, issuer_name=issuer_name, algorithm=self.digest().name, digits=self.digits)