from functools import cached_property
from sentry.incidents.models import IncidentActivity, IncidentActivityType
from sentry.silo import SiloMode
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import assume_test_silo_mode, region_silo_test

class BaseIncidentCommentDetailsTest(APITestCase):
    method = 'put'
    endpoint = 'sentry-api-0-organization-incident-comment-details'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.create_member(user=self.user, organization=self.organization, role='owner', teams=[self.team])
        self.login_as(self.user)
        self.activity = self.create_incident_comment(self.incident, user_id=self.user.id)
        self.detected_activity = self.create_incident_activity(self.incident, user_id=self.user.id, type=IncidentActivityType.CREATED.value)
        user2 = self.create_user()
        self.user2_activity = self.create_incident_comment(incident=self.incident, user_id=user2.id, comment='hello from another user')

    @cached_property
    def organization(self):
        if False:
            while True:
                i = 10
        return self.create_organization()

    @cached_property
    def project(self):
        if False:
            print('Hello World!')
        return self.create_project(organization=self.organization)

    @cached_property
    def user(self):
        if False:
            print('Hello World!')
        return self.create_user()

    @cached_property
    def incident(self):
        if False:
            print('Hello World!')
        return self.create_incident()

    def test_not_found(self):
        if False:
            return 10
        comment = 'hello'
        with self.feature('organizations:incidents'):
            self.get_error_response(self.organization.slug, self.incident.identifier, 123, comment=comment, status_code=404)

    def test_non_comment_type(self):
        if False:
            for i in range(10):
                print('nop')
        comment = 'hello'
        with self.feature('organizations:incidents'):
            self.get_error_response(self.organization.slug, self.incident.identifier, self.detected_activity.id, comment=comment, status_code=404)

@region_silo_test(stable=True)
class OrganizationIncidentCommentUpdateEndpointTest(BaseIncidentCommentDetailsTest):
    method = 'put'

    def test_simple(self):
        if False:
            print('Hello World!')
        comment = 'hello'
        with self.feature('organizations:incidents'):
            self.get_success_response(self.organization.slug, self.incident.identifier, self.activity.id, comment=comment, status_code=200)
        activity = IncidentActivity.objects.get(id=self.activity.id)
        assert activity.type == IncidentActivityType.COMMENT.value
        assert activity.user_id == self.user.id
        assert activity.comment == comment

    def test_cannot_edit_others_comment(self):
        if False:
            print('Hello World!')
        with self.feature('organizations:incidents'):
            self.get_error_response(self.organization.slug, self.incident.identifier, self.user2_activity.id, comment='edited comment', status_code=404)

    def test_superuser_can_edit(self):
        if False:
            i = 10
            return i + 15
        self.user.is_superuser = True
        with assume_test_silo_mode(SiloMode.CONTROL):
            self.user.save()
        edited_comment = 'this comment has been edited'
        with self.feature('organizations:incidents'):
            self.get_success_response(self.organization.slug, self.incident.identifier, self.user2_activity.id, comment=edited_comment, status_code=200)
        activity = IncidentActivity.objects.get(id=self.user2_activity.id)
        assert activity.user_id != self.user.id
        assert activity.comment == edited_comment

@region_silo_test(stable=True)
class OrganizationIncidentCommentDeleteEndpointTest(BaseIncidentCommentDetailsTest):
    method = 'delete'

    def test_simple(self):
        if False:
            print('Hello World!')
        with self.feature('organizations:incidents'):
            self.get_success_response(self.organization.slug, self.incident.identifier, self.activity.id, status_code=204)
        assert not IncidentActivity.objects.filter(id=self.activity.id).exists()

    def test_cannot_delete_others_comments(self):
        if False:
            return 10
        with self.feature('organizations:incidents'):
            self.get_error_response(self.organization.slug, self.incident.identifier, self.user2_activity.id, status_code=404)

    def test_superuser_can_delete(self):
        if False:
            while True:
                i = 10
        self.user.is_superuser = True
        with assume_test_silo_mode(SiloMode.CONTROL):
            self.user.save()
        with self.feature('organizations:incidents'):
            self.get_success_response(self.organization.slug, self.incident.identifier, self.user2_activity.id, status_code=204)
        assert not IncidentActivity.objects.filter(id=self.user2_activity.id).exists()