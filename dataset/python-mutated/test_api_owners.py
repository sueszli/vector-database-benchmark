from typing import Set
from unittest import TestCase
from sentry.api.api_owners import ApiOwner

class APIOwnersTestCase(TestCase):
    teams: Set[str] = set()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        code_owners_file = open('.github/CODEOWNERS')
        lines = code_owners_file.readlines()
        code_owners_file.close()
        for line in lines:
            if line.startswith('/src/'):
                tokens = [s.strip() for s in line.split('@getsentry/')]
                self.teams.update(tokens[1:])

    def test_api_owner_owns_api(self):
        if False:
            for i in range(10):
                print('nop')
        for owner in ApiOwner:
            if owner != ApiOwner.UNOWNED:
                assert owner.value in self.teams