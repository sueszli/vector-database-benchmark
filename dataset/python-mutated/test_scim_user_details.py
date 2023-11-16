import unittest
import pytest
from django.urls import reverse
from sentry.models.authidentity import AuthIdentity
from sentry.models.authprovider import AuthProvider
from sentry.models.organizationmember import OrganizationMember
from sentry.scim.endpoints.utils import SCIMFilterError, parse_filter_conditions
from sentry.silo import SiloMode
from sentry.testutils.cases import APITestCase, SCIMAzureTestCase, SCIMTestCase
from sentry.testutils.silo import assume_test_silo_mode, no_silo_test, region_silo_test
CREATE_USER_POST_DATA = {'schemas': ['urn:ietf:params:scim:schemas:core:2.0:User'], 'userName': 'test.user@okta.local', 'name': {'givenName': 'Test', 'familyName': 'User'}, 'emails': [{'primary': True, 'value': 'test.user@okta.local', 'type': 'work'}], 'displayName': 'Test User', 'locale': 'en-US', 'externalId': '00ujl29u0le5T6Aj10h7', 'groups': [], 'password': '1mz050nq', 'active': True}

def generate_put_data(member: OrganizationMember, role: str='') -> dict:
    if False:
        for i in range(10):
            print('nop')
    put_data = CREATE_USER_POST_DATA.copy()
    put_data['userName'] = member.email
    put_data['sentryOrgRole'] = role
    return put_data

@region_silo_test(stable=True)
class SCIMMemberTestsPermissions(APITestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.login_as(user=self.user)

    def test_cant_use_scim(self):
        if False:
            i = 10
            return i + 15
        url = reverse('sentry-api-0-organization-scim-member-index', args=[self.organization.slug])
        response = self.client.get(url)
        assert response.status_code == 403

    def test_cant_use_scim_even_with_authprovider(self):
        if False:
            i = 10
            return i + 15
        with assume_test_silo_mode(SiloMode.CONTROL):
            AuthProvider.objects.create(organization_id=self.organization.id, provider='dummy')
        url = reverse('sentry-api-0-organization-scim-member-index', args=[self.organization.slug])
        response = self.client.get(url)
        assert response.status_code == 403

@region_silo_test(stable=True)
class SCIMMemberRoleUpdateTests(SCIMTestCase):
    endpoint = 'sentry-api-0-organization-scim-member-details'
    method = 'put'

    def setUp(self, provider='dummy'):
        if False:
            while True:
                i = 10
        super().setUp(provider=provider)
        self.unrestricted_default_role_member = self.create_member(user=self.create_user(), organization=self.organization)
        self.unrestricted_custom_role_member = self.create_member(user=self.create_user(), organization=self.organization, role='manager')
        self.restricted_default_role_member = self.create_member(user=self.create_user(), organization=self.organization)
        self.restricted_default_role_member.flags['idp:role-restricted'] = True
        self.restricted_default_role_member.save()
        self.restricted_custom_role_member = self.create_member(user=self.create_user(), organization=self.organization, role='manager')
        self.restricted_custom_role_member.flags['idp:role-restricted'] = True
        self.restricted_custom_role_member.save()

    def test_owner(self):
        if False:
            print('Hello World!')
        'Owners cannot be edited by this API, but we will return a success response'
        self.owner = self.create_member(user=self.create_user(), organization=self.organization, role='owner')
        self.get_success_response(self.organization.slug, self.owner.id, **generate_put_data(self.owner, role='member'))
        self.owner.refresh_from_db()
        assert self.owner.role == 'owner'
        assert self.owner.flags['idp:provisioned']

    def test_owner_blank_role(self):
        if False:
            while True:
                i = 10
        'A PUT request with a blank role should go through'
        self.owner = self.create_member(user=self.create_user(), organization=self.organization, role='owner')
        self.get_success_response(self.organization.slug, self.owner.id, **generate_put_data(self.owner))
        self.owner.refresh_from_db()
        assert self.owner.role == 'owner'
        assert self.owner.flags['idp:provisioned']
        self.owner.flags['idp:role-restricted'] = True
        self.owner.save()
        self.get_success_response(self.organization.slug, self.owner.id, **generate_put_data(self.owner))
        self.owner.refresh_from_db()
        assert self.owner.role == 'owner'
        assert not self.owner.flags['idp:role-restricted']
        assert self.owner.flags['idp:provisioned']

    def test_invalid_role(self):
        if False:
            i = 10
            return i + 15
        self.get_error_response(self.organization.slug, self.unrestricted_default_role_member.id, status_code=400, **generate_put_data(self.unrestricted_default_role_member, role='nonexistant'))
        self.get_error_response(self.organization.slug, self.unrestricted_custom_role_member.id, status_code=400, **generate_put_data(self.unrestricted_custom_role_member, role='nonexistant'))
        self.get_error_response(self.organization.slug, self.restricted_default_role_member.id, status_code=400, **generate_put_data(self.restricted_default_role_member, role='nonexistant'))
        self.get_error_response(self.organization.slug, self.restricted_custom_role_member.id, status_code=400, **generate_put_data(self.restricted_custom_role_member, role='nonexistant'))
        self.get_error_response(self.organization.slug, self.unrestricted_default_role_member.id, status_code=400, **generate_put_data(self.unrestricted_default_role_member, role='owner'))
        self.get_error_response(self.organization.slug, self.unrestricted_custom_role_member.id, status_code=400, **generate_put_data(self.unrestricted_custom_role_member, role='owner'))
        self.get_error_response(self.organization.slug, self.restricted_default_role_member.id, status_code=400, **generate_put_data(self.restricted_default_role_member, role='owner'))
        self.get_error_response(self.organization.slug, self.restricted_custom_role_member.id, status_code=400, **generate_put_data(self.restricted_custom_role_member, role='owner'))

    def test_set_to_blank(self):
        if False:
            i = 10
            return i + 15
        resp = self.get_success_response(self.organization.slug, self.unrestricted_default_role_member.id, **generate_put_data(self.unrestricted_default_role_member))
        self.unrestricted_default_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == self.organization.default_role
        assert self.unrestricted_default_role_member.role == self.organization.default_role
        assert not self.unrestricted_default_role_member.flags['idp:role-restricted']
        resp = self.get_success_response(self.organization.slug, self.unrestricted_custom_role_member.id, **generate_put_data(self.unrestricted_custom_role_member))
        self.unrestricted_custom_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == self.unrestricted_custom_role_member.role
        assert self.unrestricted_custom_role_member.role == 'manager'
        assert not self.unrestricted_custom_role_member.flags['idp:role-restricted']
        resp = self.get_success_response(self.organization.slug, self.restricted_default_role_member.id, **generate_put_data(self.restricted_default_role_member))
        self.restricted_default_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == self.organization.default_role
        assert self.restricted_default_role_member.role == self.organization.default_role
        assert not self.restricted_default_role_member.flags['idp:role-restricted']
        resp = self.get_success_response(self.organization.slug, self.restricted_custom_role_member.id, **generate_put_data(self.restricted_custom_role_member))
        self.restricted_custom_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == self.organization.default_role
        assert self.restricted_custom_role_member.role == self.organization.default_role
        assert not self.restricted_custom_role_member.flags['idp:role-restricted']

    def test_set_to_default(self):
        if False:
            for i in range(10):
                print('nop')
        resp = self.get_success_response(self.organization.slug, self.unrestricted_default_role_member.id, **generate_put_data(self.unrestricted_default_role_member, role=self.organization.default_role))
        self.unrestricted_default_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == self.organization.default_role
        assert self.unrestricted_default_role_member.role == self.organization.default_role
        assert self.unrestricted_default_role_member.flags['idp:role-restricted']
        resp = self.get_success_response(self.organization.slug, self.unrestricted_custom_role_member.id, **generate_put_data(self.unrestricted_custom_role_member, role=self.organization.default_role))
        self.unrestricted_custom_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == self.organization.default_role
        assert self.unrestricted_custom_role_member.role == self.organization.default_role
        assert self.unrestricted_custom_role_member.flags['idp:role-restricted']
        resp = self.get_success_response(self.organization.slug, self.restricted_default_role_member.id, **generate_put_data(self.restricted_default_role_member, role=self.organization.default_role))
        self.restricted_default_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == self.organization.default_role
        assert self.restricted_default_role_member.role == self.organization.default_role
        assert self.restricted_default_role_member.flags['idp:role-restricted']
        resp = self.get_success_response(self.organization.slug, self.restricted_custom_role_member.id, **generate_put_data(self.restricted_custom_role_member, role=self.organization.default_role))
        self.restricted_custom_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == self.organization.default_role
        assert self.restricted_custom_role_member.role == self.organization.default_role
        assert self.restricted_custom_role_member.flags['idp:role-restricted']

    def test_set_to_new_role(self):
        if False:
            i = 10
            return i + 15
        new_role = 'admin'
        resp = self.get_success_response(self.organization.slug, self.unrestricted_default_role_member.id, **generate_put_data(self.unrestricted_default_role_member, role=new_role))
        self.unrestricted_default_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == new_role
        assert self.unrestricted_default_role_member.role == new_role
        assert self.unrestricted_default_role_member.flags['idp:role-restricted']
        resp = self.get_success_response(self.organization.slug, self.unrestricted_custom_role_member.id, **generate_put_data(self.unrestricted_custom_role_member, role=new_role))
        self.unrestricted_custom_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == new_role
        assert self.unrestricted_custom_role_member.role == new_role
        assert self.unrestricted_custom_role_member.flags['idp:role-restricted']
        resp = self.get_success_response(self.organization.slug, self.restricted_default_role_member.id, **generate_put_data(self.restricted_default_role_member, role=new_role))
        self.restricted_default_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == new_role
        assert self.restricted_default_role_member.role == new_role
        assert self.restricted_default_role_member.flags['idp:role-restricted']
        resp = self.get_success_response(self.organization.slug, self.restricted_custom_role_member.id, **generate_put_data(self.restricted_custom_role_member, role=new_role))
        self.restricted_custom_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == new_role
        assert self.restricted_custom_role_member.role == new_role
        assert self.restricted_custom_role_member.flags['idp:role-restricted']

    def test_set_to_same_custom_role(self):
        if False:
            return 10
        same_role = self.unrestricted_custom_role_member.role
        assert not self.unrestricted_custom_role_member.flags['idp:role-restricted']
        resp = self.get_success_response(self.organization.slug, self.unrestricted_custom_role_member.id, **generate_put_data(self.unrestricted_custom_role_member, role=same_role))
        self.unrestricted_custom_role_member.refresh_from_db()
        assert resp.data['sentryOrgRole'] == same_role
        assert self.unrestricted_custom_role_member.role == same_role
        assert self.unrestricted_custom_role_member.flags['idp:role-restricted']

    def test_cannot_set_partnership_member_role(self):
        if False:
            return 10
        self.partnership_member = self.create_member(user=self.create_user(), organization=self.organization, role='manager', flags=OrganizationMember.flags['partnership:restricted'])
        self.get_error_response(self.organization.slug, self.partnership_member.id, status_code=403, **generate_put_data(self.partnership_member, role='member'))

@region_silo_test(stable=True)
class SCIMMemberDetailsTests(SCIMTestCase):
    endpoint = 'sentry-api-0-organization-scim-member-details'

    def test_user_details_get(self):
        if False:
            for i in range(10):
                print('nop')
        member = self.create_member(organization=self.organization, email='test.user@okta.local')
        response = self.get_success_response(self.organization.slug, member.id)
        assert response.data == {'schemas': ['urn:ietf:params:scim:schemas:core:2.0:User'], 'id': str(member.id), 'userName': 'test.user@okta.local', 'emails': [{'primary': True, 'value': 'test.user@okta.local', 'type': 'work'}], 'name': {'familyName': 'N/A', 'givenName': 'N/A'}, 'active': True, 'meta': {'resourceType': 'User'}, 'sentryOrgRole': self.organization.default_role}

    def test_user_details_set_inactive(self):
        if False:
            return 10
        member = self.create_member(user=self.create_user(email='test.user@okta.local'), organization=self.organization)
        with assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.create(user_id=member.user_id, auth_provider=self.auth_provider_inst, ident='test_ident')
        patch_req = {'schemas': ['urn:ietf:params:scim:api:messages:2.0:PatchOp'], 'Operations': [{'op': 'Replace', 'path': 'active', 'value': False}]}
        self.get_success_response(self.organization.slug, member.id, raw_data=patch_req, method='patch')
        with pytest.raises(OrganizationMember.DoesNotExist):
            OrganizationMember.objects.get(organization=self.organization, id=member.id)
        with pytest.raises(AuthIdentity.DoesNotExist), assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.get(auth_provider=self.auth_provider_inst, id=member.id)

    def test_user_details_cannot_set_partnership_member_inactive(self):
        if False:
            print('Hello World!')
        member = self.create_member(user=self.create_user(email='test.user@okta.local'), organization=self.organization, flags=OrganizationMember.flags['partnership:restricted'])
        with assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.create(user_id=member.user_id, auth_provider=self.auth_provider_inst, ident='test_ident')
        patch_req = {'schemas': ['urn:ietf:params:scim:api:messages:2.0:PatchOp'], 'Operations': [{'op': 'Replace', 'path': 'active', 'value': False}]}
        self.get_error_response(self.organization.slug, member.id, raw_data=patch_req, method='patch', status_code=403)

    def test_user_details_set_inactive_dict(self):
        if False:
            return 10
        member = self.create_member(user=self.create_user(email='test.user@okta.local'), organization=self.organization)
        with assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.create(user_id=member.user_id, auth_provider=self.auth_provider_inst, ident='test_ident')
        patch_req = {'schemas': ['urn:ietf:params:scim:api:messages:2.0:PatchOp'], 'Operations': [{'op': 'Replace', 'value': {'active': False}}]}
        self.get_success_response(self.organization.slug, member.id, raw_data=patch_req, method='patch')
        with pytest.raises(OrganizationMember.DoesNotExist):
            OrganizationMember.objects.get(organization=self.organization, id=member.id)
        with pytest.raises(AuthIdentity.DoesNotExist), assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.get(auth_provider=self.auth_provider_inst, id=member.id)

    def test_user_details_set_inactive_with_bool_string(self):
        if False:
            i = 10
            return i + 15
        member = self.create_member(user=self.create_user(email='test.user@okta.local'), organization=self.organization)
        with assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.create(user_id=member.user_id, auth_provider=self.auth_provider_inst, ident='test_ident')
        patch_req = {'schemas': ['urn:ietf:params:scim:api:messages:2.0:PatchOp'], 'Operations': [{'op': 'Replace', 'path': 'active', 'value': 'False'}]}
        self.get_success_response(self.organization.slug, member.id, raw_data=patch_req, method='patch')
        with pytest.raises(OrganizationMember.DoesNotExist):
            OrganizationMember.objects.get(organization=self.organization, id=member.id)
        with pytest.raises(AuthIdentity.DoesNotExist), assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.get(auth_provider=self.auth_provider_inst, id=member.id)

    def test_user_details_set_inactive_with_dict_bool_string(self):
        if False:
            return 10
        member = self.create_member(user=self.create_user(email='test.user@okta.local'), organization=self.organization)
        with assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.create(user_id=member.user_id, auth_provider=self.auth_provider_inst, ident='test_ident')
        patch_req = {'schemas': ['urn:ietf:params:scim:api:messages:2.0:PatchOp'], 'Operations': [{'op': 'Replace', 'value': {'id': 'xxxx', 'active': 'False'}}]}
        self.get_success_response(self.organization.slug, member.id, raw_data=patch_req, method='patch')
        with pytest.raises(OrganizationMember.DoesNotExist):
            OrganizationMember.objects.get(organization=self.organization, id=member.id)
        with pytest.raises(AuthIdentity.DoesNotExist), assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.get(auth_provider=self.auth_provider_inst, id=member.id)

    def test_invalid_patch_op(self):
        if False:
            return 10
        member = self.create_member(user=self.create_user(email='test.user@okta.local'), organization=self.organization)
        with assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.create(user_id=member.user_id, auth_provider=self.auth_provider_inst, ident='test_ident')
        patch_req = {'schemas': ['urn:ietf:params:scim:api:messages:2.0:PatchOp'], 'Operations': [{'op': 'invalid', 'value': {'active': False}}]}
        self.get_error_response(self.organization.slug, member.id, raw_data=patch_req, method='patch', status_code=400)

    def test_invalid_patch_op_value(self):
        if False:
            print('Hello World!')
        member = self.create_member(user=self.create_user(email='test.user@okta.local'), organization=self.organization)
        with assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.create(user_id=member.user_id, auth_provider=self.auth_provider_inst, ident='test_ident')
        patch_req = {'schemas': ['urn:ietf:params:scim:api:messages:2.0:PatchOp'], 'Operations': [{'op': 'REPLACE', 'value': {'active': 'invalid'}}]}
        self.get_error_response(self.organization.slug, member.id, raw_data=patch_req, method='patch', status_code=400)

    def test_user_details_get_404(self):
        if False:
            i = 10
            return i + 15
        self.get_error_response(self.organization.slug, 99999999, status_code=404)

    def test_user_details_patch_404(self):
        if False:
            while True:
                i = 10
        patch_req = {'schemas': ['urn:ietf:params:scim:api:messages:2.0:PatchOp'], 'Operations': [{'op': 'replace', 'value': {'active': False}}]}
        self.get_error_response(self.organization.slug, 99999999, raw_data=patch_req, method='patch', status_code=404)

    def test_delete_route(self):
        if False:
            i = 10
            return i + 15
        member = self.create_member(user=self.create_user(), organization=self.organization)
        with assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.create(user_id=member.user_id, auth_provider=self.auth_provider_inst, ident='test_ident')
        self.get_success_response(self.organization.slug, member.id, method='delete')
        with pytest.raises(OrganizationMember.DoesNotExist):
            OrganizationMember.objects.get(organization=self.organization, id=member.id)
        with pytest.raises(AuthIdentity.DoesNotExist), assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.get(auth_provider=self.auth_provider_inst, id=member.id)

    def test_cannot_delete_partnership_member(self):
        if False:
            for i in range(10):
                print('nop')
        member = self.create_member(user=self.create_user(), organization=self.organization, flags=OrganizationMember.flags['partnership:restricted'])
        with assume_test_silo_mode(SiloMode.CONTROL):
            AuthIdentity.objects.create(user_id=member.user_id, auth_provider=self.auth_provider_inst, ident='test_ident')
        self.get_error_response(self.organization.slug, member.id, method='delete', status_code=403)

    def test_patch_inactive_alternate_schema(self):
        if False:
            i = 10
            return i + 15
        member = self.create_member(user=self.create_user(), organization=self.organization)
        patch_req = {'Operations': [{'op': 'replace', 'path': 'active', 'value': False}]}
        self.get_success_response(self.organization.slug, member.id, raw_data=patch_req, method='patch')
        with pytest.raises(OrganizationMember.DoesNotExist):
            OrganizationMember.objects.get(organization=self.organization, id=member.id)

    def test_patch_bad_schema(self):
        if False:
            return 10
        member = self.create_member(user=self.create_user(), organization=self.organization)
        patch_req = {'Operations': [{'op': 'replace', 'path': 'blahblahbbalh', 'value': False}]}
        response = self.get_error_response(self.organization.slug, member.id, raw_data=patch_req, method='patch', status_code=400)
        assert response.data == {'schemas': ['urn:ietf:params:scim:api:messages:2.0:Error'], 'detail': 'Invalid Patch Operation.'}
        patch_req = {'Operations': [{'op': 'replace', 'value': False}]}
        response = self.get_error_response(self.organization.slug, member.id, raw_data=patch_req, method='patch', status_code=400)
        assert response.data == {'schemas': ['urn:ietf:params:scim:api:messages:2.0:Error'], 'detail': 'Invalid Patch Operation.'}

    def test_member_detail_patch_too_many_ops(self):
        if False:
            for i in range(10):
                print('nop')
        member = self.create_member(user=self.create_user(), organization=self.organization)
        patch_req = {'schemas': ['urn:ietf:params:scim:api:messages:2.0:PatchOp'], 'Operations': [{'op': 'replace', 'path': 'active', 'value': False}] * 101}
        response = self.get_error_response(self.organization.slug, member.id, raw_data=patch_req, method='patch', status_code=400)
        assert response.status_code == 400, response.data
        assert response.data == {'schemas': ['urn:ietf:params:scim:api:messages:2.0:Error'], 'detail': '{"Operations":["Ensure this field has no more than 100 elements."]}'}

    def test_overflow_cases(self):
        if False:
            while True:
                i = 10
        member = self.create_member(user=self.create_user(), organization=self.organization)
        self.get_error_response(self.organization.slug, '010101001010101011001010101011', status_code=404)
        self.get_error_response(self.organization.slug, '010101001010101011001010101011', raw_data={}, method='patch', status_code=404)
        self.get_error_response(self.organization.slug, '010101001010101011001010101011', raw_data=member.id, method='delete', status_code=404)

    def test_cant_delete_only_owner_route(self):
        if False:
            i = 10
            return i + 15
        member_om = OrganizationMember.objects.get(organization=self.organization, user_id=self.user.id)
        self.get_error_response(self.organization.slug, member_om.id, method='delete', status_code=403)

    def test_cant_delete_only_owner_route_patch(self):
        if False:
            while True:
                i = 10
        member_om = OrganizationMember.objects.get(organization=self.organization, user_id=self.user.id)
        patch_req = {'schemas': ['urn:ietf:params:scim:api:messages:2.0:PatchOp'], 'Operations': [{'op': 'replace', 'value': {'active': False}}]}
        self.get_error_response(self.organization.slug, member_om.id, raw_data=patch_req, method='patch', status_code=403)

@region_silo_test(stable=True)
class SCIMMemberDetailsAzureTests(SCIMAzureTestCase):
    endpoint = 'sentry-api-0-organization-scim-member-details'

    def test_user_details_get_no_active(self):
        if False:
            for i in range(10):
                print('nop')
        member = self.create_member(organization=self.organization, email='test.user@okta.local')
        response = self.get_success_response(self.organization.slug, member.id)
        assert response.data == {'schemas': ['urn:ietf:params:scim:schemas:core:2.0:User'], 'id': str(member.id), 'userName': 'test.user@okta.local', 'emails': [{'primary': True, 'value': 'test.user@okta.local', 'type': 'work'}], 'name': {'familyName': 'N/A', 'givenName': 'N/A'}, 'meta': {'resourceType': 'User'}, 'sentryOrgRole': self.organization.default_role}

@no_silo_test(stable=True)
class SCIMUtilsTests(unittest.TestCase):

    def test_parse_filter_conditions_basic(self):
        if False:
            return 10
        fil = parse_filter_conditions('userName eq "user@sentry.io"')
        assert fil == 'user@sentry.io'
        fil = parse_filter_conditions("userName eq 'user@sentry.io'")
        assert fil == 'user@sentry.io'
        fil = parse_filter_conditions('value eq "23"')
        assert fil == 23
        fil = parse_filter_conditions('displayName eq "MyTeamName"')
        assert fil == 'MyTeamName'

    def test_parse_filter_conditions_invalids(self):
        if False:
            print('Hello World!')
        with pytest.raises(SCIMFilterError):
            parse_filter_conditions('userName invalid USER@sentry.io')
        with pytest.raises(SCIMFilterError):
            parse_filter_conditions('blablaba eq USER@sentry.io')

    def test_parse_filter_conditions_single_quote_in_email(self):
        if False:
            while True:
                i = 10
        fil = parse_filter_conditions('userName eq "jos\'h@sentry.io"')
        assert fil == "jos'h@sentry.io"