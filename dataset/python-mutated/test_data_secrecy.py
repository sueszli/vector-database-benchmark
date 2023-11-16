from sentry.testutils.cases import APITestCase
from sentry.testutils.helpers.features import with_feature

class SuperuserDataSecrecyTestCase(APITestCase):
    endpoint = 'sentry-api-0-organization-details'
    method = 'get'

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.login_as(self.user)

    @with_feature('organizations:enterprise-data-secrecy')
    def test_superuser_no_access(self):
        if False:
            return 10
        '\n        Please contact the Enterprise team if your code change causes this test to fail\n        '
        superuser = self.create_user(is_superuser=True)
        self.login_as(superuser, superuser=True)
        self.get_error_response(self.organization.slug, status_code=401)

    def test_superuser_has_access(self):
        if False:
            i = 10
            return i + 15
        superuser = self.create_user(is_superuser=True)
        self.login_as(superuser, superuser=True)
        self.get_success_response(self.organization.slug)

    def test_non_member_no_access(self):
        if False:
            print('Hello World!')
        self.login_as(self.create_user())
        self.get_error_response(self.organization.slug, status_code=403)

    def test_member_has_access(self):
        if False:
            while True:
                i = 10
        self.get_success_response(self.organization.slug)