from django.http import HttpResponse
from django.urls import reverse
from rest_framework.request import Request
from sentry.models.organizationmember import OrganizationMember
from sentry.services.hybrid_cloud.organization import organization_service
from .react_page import ReactPageView

class DisabledMemberView(ReactPageView):

    def is_member_disabled_from_limit(self, request: Request, organization):
        if False:
            for i in range(10):
                print('nop')
        return False

    def handle(self, request: Request, organization, **kwargs) -> HttpResponse:
        if False:
            i = 10
            return i + 15
        user = request.user
        try:
            member = organization_service.check_membership_by_id(user_id=user.id, organization_id=organization.id)
            if not member.flags['member-limit:restricted']:
                return self.redirect(reverse('sentry-organization-issue-list', args=[organization.slug]))
        except OrganizationMember.DoesNotExist:
            pass
        return super().handle(request, organization, **kwargs)