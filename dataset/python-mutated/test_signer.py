from sentry.testutils.cases import TestCase
from sentry.utils.email.signer import _CaseInsensitiveSigner

class CaseInsensitiveSignerTests(TestCase):

    def test_it_works(self):
        if False:
            print('Hello World!')
        with self.settings(SECRET_KEY='a'):
            signer = _CaseInsensitiveSigner(salt='sentry.utils.email._CaseInsensitiveSigner')
            assert signer.unsign(signer.sign('foo')) == 'foo'
            assert signer.sign('foo') == 'foo:wkpxg5djz3d4m0zktktfl9hdzw4'
            assert signer.unsign('foo:WKPXG5DJZ3D4M0ZKTKTFL9HDZW4') == 'foo'