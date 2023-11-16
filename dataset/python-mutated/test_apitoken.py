from datetime import timedelta
from django.utils import timezone
from sentry.conf.server import SENTRY_SCOPE_HIERARCHY_MAPPING, SENTRY_SCOPES
from sentry.hybridcloud.models import ApiTokenReplica
from sentry.models.apitoken import ApiToken
from sentry.models.integrations.sentry_app_installation import SentryAppInstallation
from sentry.models.integrations.sentry_app_installation_token import SentryAppInstallationToken
from sentry.silo import SiloMode
from sentry.testutils.cases import TestCase
from sentry.testutils.outbox import outbox_runner
from sentry.testutils.silo import assume_test_silo_mode, control_silo_test

@control_silo_test(stable=True)
class ApiTokenTest(TestCase):

    def test_is_expired(self):
        if False:
            i = 10
            return i + 15
        token = ApiToken(expires_at=None)
        assert not token.is_expired()
        token = ApiToken(expires_at=timezone.now() + timedelta(days=1))
        assert not token.is_expired()
        token = ApiToken(expires_at=timezone.now() - timedelta(days=1))
        assert token.is_expired()

    def test_get_scopes(self):
        if False:
            while True:
                i = 10
        token = ApiToken(scopes=1)
        assert token.get_scopes() == ['project:read']
        token = ApiToken(scopes=4, scope_list=['project:read'])
        assert token.get_scopes() == ['project:read']
        token = ApiToken(scope_list=['project:read'])
        assert token.get_scopes() == ['project:read']

    def test_enforces_scope_hierarchy(self):
        if False:
            return 10
        user = self.create_user()
        for scope in SENTRY_SCOPES:
            token = ApiToken.objects.create(user_id=user.id, scope_list=[scope])
            assert set(token.get_scopes()) == SENTRY_SCOPE_HIERARCHY_MAPPING[scope]

    def test_organization_id_for_non_internal(self):
        if False:
            i = 10
            return i + 15
        install = self.create_sentry_app_installation()
        token = install.api_token
        org_id = token.organization_id
        with assume_test_silo_mode(SiloMode.REGION):
            assert ApiTokenReplica.objects.get(apitoken_id=token.id).organization_id == org_id
        with outbox_runner():
            install.delete()
        with assume_test_silo_mode(SiloMode.REGION):
            assert ApiTokenReplica.objects.get(apitoken_id=token.id).organization_id is None
        assert token.organization_id is None

class ApiTokenInternalIntegrationTest(TestCase):

    def setUp(self):
        if False:
            return 10
        self.user = self.create_user()
        self.proxy = self.create_user()
        self.org = self.create_organization()
        self.internal_app = self.create_internal_integration(name='Internal App', organization=self.org)
        self.install = SentryAppInstallation.objects.get(sentry_app=self.internal_app)

    def test_multiple_tokens_have_correct_organization_id(self):
        if False:
            i = 10
            return i + 15
        token_1 = self.internal_app.installations.first().api_token
        token_2 = self.create_internal_integration_token(install=self.install, user=self.user)
        assert token_1.organization_id == self.org.id
        assert token_2.organization_id == self.org.id
        with assume_test_silo_mode(SiloMode.REGION):
            assert ApiTokenReplica.objects.get(apitoken_id=token_1.id).organization_id == self.org.id
            assert ApiTokenReplica.objects.get(apitoken_id=token_2.id).organization_id == self.org.id
        with outbox_runner():
            for install_token in SentryAppInstallationToken.objects.all():
                install_token.delete()
        with assume_test_silo_mode(SiloMode.REGION):
            assert ApiTokenReplica.objects.get(apitoken_id=token_1.id).organization_id is None
            assert ApiTokenReplica.objects.get(apitoken_id=token_2.id).organization_id is None