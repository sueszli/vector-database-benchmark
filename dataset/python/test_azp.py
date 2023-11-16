from __future__ import annotations

from .util import common_auth_test


def test_auth():
    # noinspection PyProtectedMember
    from ansible_test._internal.ci.azp import (
        AzurePipelinesAuthHelper,
    )

    class TestAzurePipelinesAuthHelper(AzurePipelinesAuthHelper):
        def __init__(self):
            self.public_key_pem = None
            self.private_key_pem = None

        def publish_public_key(self, public_key_pem):
            # avoid publishing key
            self.public_key_pem = public_key_pem

        def initialize_private_key(self):
            # cache in memory instead of on disk
            if not self.private_key_pem:
                self.private_key_pem = self.generate_private_key()

            return self.private_key_pem

    auth = TestAzurePipelinesAuthHelper()

    common_auth_test(auth)
