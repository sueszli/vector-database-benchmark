from __future__ import annotations
from .util import common_auth_test

def test_auth():
    if False:
        print('Hello World!')
    from ansible_test._internal.ci.azp import AzurePipelinesAuthHelper

    class TestAzurePipelinesAuthHelper(AzurePipelinesAuthHelper):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.public_key_pem = None
            self.private_key_pem = None

        def publish_public_key(self, public_key_pem):
            if False:
                while True:
                    i = 10
            self.public_key_pem = public_key_pem

        def initialize_private_key(self):
            if False:
                i = 10
                return i + 15
            if not self.private_key_pem:
                self.private_key_pem = self.generate_private_key()
            return self.private_key_pem
    auth = TestAzurePipelinesAuthHelper()
    common_auth_test(auth)