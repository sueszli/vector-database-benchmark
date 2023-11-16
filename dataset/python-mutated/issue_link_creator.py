from django.db import router
from sentry.coreapi import APIUnauthorized
from sentry.mediators.external_issues.creator import Creator
from sentry.mediators.external_requests.issue_link_requester import IssueLinkRequester
from sentry.mediators.mediator import Mediator
from sentry.mediators.param import Param
from sentry.models.group import Group
from sentry.models.platformexternalissue import PlatformExternalIssue
from sentry.services.hybrid_cloud.app import RpcSentryAppInstallation
from sentry.services.hybrid_cloud.user import RpcUser
from sentry.utils.cache import memoize

class IssueLinkCreator(Mediator):
    install = Param(RpcSentryAppInstallation)
    group = Param(Group)
    action = Param(str)
    fields = Param(object)
    uri = Param(str)
    user = Param(RpcUser)
    using = router.db_for_write(PlatformExternalIssue)

    def call(self):
        if False:
            print('Hello World!')
        self._verify_action()
        self._make_external_request()
        self._create_external_issue()
        return self.external_issue

    def _verify_action(self):
        if False:
            print('Hello World!')
        if self.action not in ['link', 'create']:
            raise APIUnauthorized(f"Invalid action '{self.action}'")

    def _make_external_request(self):
        if False:
            while True:
                i = 10
        self.response = IssueLinkRequester.run(install=self.install, uri=self.uri, group=self.group, fields=self.fields, user=self.user, action=self.action)

    def _create_external_issue(self):
        if False:
            for i in range(10):
                print('nop')
        self.external_issue = Creator.run(install=self.install, group=self.group, web_url=self.response['webUrl'], project=self.response['project'], identifier=self.response['identifier'])

    @memoize
    def sentry_app(self):
        if False:
            for i in range(10):
                print('nop')
        return self.install.sentry_app