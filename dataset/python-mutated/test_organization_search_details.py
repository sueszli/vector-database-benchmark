from functools import cached_property
from sentry.models.savedsearch import SavedSearch, Visibility
from sentry.models.search_common import SearchType
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class DeleteOrganizationSearchTest(APITestCase):
    endpoint = 'sentry-api-0-organization-search-details'
    method = 'delete'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.login_as(user=self.user)

    @cached_property
    def member(self):
        if False:
            i = 10
            return i + 15
        user = self.create_user('test@test.com')
        self.create_member(organization=self.organization, user=user)
        return user

    def get_response(self, *args, **params):
        if False:
            return 10
        return super().get_response(*(self.organization.slug,) + args, **params)

    def test_owner_can_delete_org_searches(self):
        if False:
            while True:
                i = 10
        search = SavedSearch.objects.create(organization=self.organization, owner_id=self.create_user().id, name='foo', query='', visibility=Visibility.ORGANIZATION)
        response = self.get_response(search.id)
        assert response.status_code == 204, response.content
        assert not SavedSearch.objects.filter(id=search.id).exists()

    def test_owners_can_delete_their_searches(self):
        if False:
            while True:
                i = 10
        search = SavedSearch.objects.create(organization=self.organization, owner_id=self.user.id, name='foo', query='', visibility=Visibility.OWNER)
        response = self.get_response(search.id)
        assert response.status_code == 204, response.content
        assert not SavedSearch.objects.filter(id=search.id).exists()

    def test_member_can_delete_their_searches(self):
        if False:
            for i in range(10):
                print('nop')
        search = SavedSearch.objects.create(organization=self.organization, owner_id=self.member.id, name='foo', query='', visibility=Visibility.OWNER)
        self.login_as(user=self.member)
        response = self.get_response(search.id)
        assert response.status_code == 204, response.content
        assert not SavedSearch.objects.filter(id=search.id).exists()

    def test_owners_cannot_delete_searches_they_do_not_own(self):
        if False:
            for i in range(10):
                print('nop')
        search = SavedSearch.objects.create(organization=self.organization, owner_id=self.create_user().id, name='foo', query='', visibility=Visibility.OWNER)
        response = self.get_response(search.id)
        assert response.status_code == 404, response.content
        assert SavedSearch.objects.filter(id=search.id).exists()

    def test_owners_cannot_delete_global_searches(self):
        if False:
            while True:
                i = 10
        search = SavedSearch.objects.create(name='foo', query='', is_global=True, visibility=Visibility.ORGANIZATION)
        response = self.get_response(search.id)
        assert response.status_code == 404, response.content
        assert SavedSearch.objects.filter(id=search.id).exists()

    def test_members_cannot_delete_shared_searches(self):
        if False:
            i = 10
            return i + 15
        search = SavedSearch.objects.create(organization=self.organization, owner_id=self.user.id, name='foo', query='', visibility=Visibility.ORGANIZATION)
        self.login_as(user=self.member)
        response = self.get_response(search.id)
        assert response.status_code == 403, response.content
        assert SavedSearch.objects.filter(id=search.id).exists()

@region_silo_test(stable=True)
class PutOrganizationSearchTest(APITestCase):
    endpoint = 'sentry-api-0-organization-search-details'
    method = 'put'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.login_as(user=self.user)

    @cached_property
    def member(self):
        if False:
            for i in range(10):
                print('nop')
        user = self.create_user('test@test.com')
        self.create_member(organization=self.organization, user=user)
        return user

    def test_owner_can_edit_shared_search(self):
        if False:
            return 10
        search = SavedSearch.objects.create(organization=self.organization, owner_id=self.create_user().id, name='foo', query='', visibility=Visibility.ORGANIZATION)
        response = self.get_response(self.organization.slug, search.id, type=SearchType.ISSUE.value, name='foo', query='test', visibility=Visibility.ORGANIZATION)
        assert response.status_code == 200, response.content
        updated_obj = SavedSearch.objects.get(id=search.id)
        assert updated_obj.name == 'foo'
        assert updated_obj.query == 'test'

    def test_member_cannot_edit_org_search(self):
        if False:
            i = 10
            return i + 15
        search = SavedSearch.objects.create(organization=self.organization, owner_id=self.user.id, name='foo', query='', visibility=Visibility.ORGANIZATION)
        self.login_as(user=self.member)
        response = self.get_response(self.organization.slug, search.id, type=SearchType.ISSUE.value, name='foo', query='test', visibility=Visibility.ORGANIZATION)
        assert response.status_code == 403, response.content

    def test_member_can_edit_personal_search(self):
        if False:
            return 10
        search = SavedSearch.objects.create(organization=self.organization, owner_id=self.member.id, name='foo', query='', visibility=Visibility.OWNER)
        self.login_as(user=self.member)
        response = self.get_response(self.organization.slug, search.id, type=SearchType.ISSUE.value, name='foo', query='test', visibility=Visibility.OWNER)
        assert response.status_code == 200, response.content
        updated_obj = SavedSearch.objects.get(id=search.id)
        assert updated_obj.name == 'foo'
        assert updated_obj.query == 'test'

    def test_member_cannot_switch_personal_search_to_org(self):
        if False:
            print('Hello World!')
        search = SavedSearch.objects.create(organization=self.organization, owner_id=self.member.id, name='foo', query='', visibility=Visibility.OWNER)
        self.login_as(user=self.member)
        response = self.get_response(self.organization.slug, search.id, type=SearchType.ISSUE.value, name='foo', query='test', visibility=Visibility.ORGANIZATION)
        assert response.status_code == 400, response.content

    def test_exists(self):
        if False:
            while True:
                i = 10
        SavedSearch.objects.create(type=SearchType.ISSUE.value, name='Some global search', query='is:unresolved', is_global=True, visibility=Visibility.ORGANIZATION)
        SavedSearch.objects.create(owner_id=self.member.id, type=SearchType.ISSUE.value, name='Some other users search', query='is:mine', visibility=Visibility.OWNER)
        search = SavedSearch.objects.create(organization=self.organization, owner_id=self.user.id, name='foo', query='', visibility=Visibility.OWNER)
        response = self.get_response(self.organization.slug, search.id, type=SearchType.ISSUE.value, name='foo', query='is:mine', visibility=Visibility.OWNER)
        assert response.status_code == 200, response.content
        response = self.get_response(self.organization.slug, search.id, type=SearchType.ISSUE.value, name='foo', query='is:unresolved', visibility=Visibility.OWNER)
        assert response.status_code == 400, response.content
        assert 'already exists' in response.data['detail']

    def test_can_edit_without_changing_query(self):
        if False:
            print('Hello World!')
        search = SavedSearch.objects.create(organization=self.organization, owner_id=self.create_user().id, name='foo', query='test123', visibility=Visibility.ORGANIZATION)
        response = self.get_response(self.organization.slug, search.id, type=SearchType.ISSUE.value, name='bar', query='test123', visibility=Visibility.ORGANIZATION)
        assert response.status_code == 200, response.content
        updated_obj = SavedSearch.objects.get(id=search.id)
        assert updated_obj.name == 'bar'
        assert updated_obj.query == 'test123'