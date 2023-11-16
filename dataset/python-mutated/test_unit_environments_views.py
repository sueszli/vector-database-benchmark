import json
from typing import Callable
from unittest import TestCase, mock
import pytest
from core.constants import STRING
from django.contrib.contenttypes.models import ContentType
from django.urls import reverse
from pytest_lazyfixture import lazy_fixture
from rest_framework import status
from rest_framework.test import APIClient
from audit.models import AuditLog, RelatedObjectType
from environments.identities.models import Identity
from environments.identities.traits.models import Trait
from environments.models import Environment, EnvironmentAPIKey, Webhook
from environments.permissions.constants import VIEW_ENVIRONMENT
from environments.permissions.models import UserEnvironmentPermission
from features.models import Feature, FeatureState
from metadata.models import Metadata, MetadataModelField
from organisations.models import Organisation, OrganisationRole
from projects.models import Project, ProjectPermissionModel, UserProjectPermission
from projects.permissions import CREATE_ENVIRONMENT, VIEW_PROJECT
from segments.models import EQUAL, Condition, SegmentRule
from users.models import FFAdminUser
from util.tests import Helper

def test_retrieve_environment(admin_client: APIClient, environment: Environment) -> None:
    if False:
        while True:
            i = 10
    url = reverse('api-v1:environments:environment-detail', args=[environment.api_key])
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json['id'] == environment.id
    assert response_json['name'] == environment.name
    assert response_json['project'] == environment.project_id
    assert response_json['api_key'] == environment.api_key
    assert response_json['allow_client_traits'] == environment.allow_client_traits
    assert response_json['banner_colour'] == environment.banner_colour
    assert response_json['banner_text'] == environment.banner_text
    assert response_json['description'] == environment.description
    assert response_json['hide_disabled_flags'] == environment.hide_disabled_flags
    assert response_json['hide_sensitive_data'] == environment.hide_sensitive_data
    assert response_json['metadata'] == []
    assert response_json['minimum_change_request_approvals'] == environment.minimum_change_request_approvals
    assert response_json['total_segment_overrides'] == environment.feature_segments.count()
    assert response_json['use_identity_composite_key_for_hashing'] == environment.use_identity_composite_key_for_hashing
    assert response_json['use_mv_v2_evaluation'] == environment.use_identity_composite_key_for_hashing

def test_can_clone_environment_with_create_environment_permission(test_user, test_user_client, environment, user_project_permission):
    if False:
        while True:
            i = 10
    env_name = 'Cloned env'
    user_project_permission.permissions.add(CREATE_ENVIRONMENT)
    url = reverse('api-v1:environments:environment-clone', args=[environment.api_key])
    response = test_user_client.post(url, {'name': env_name})
    assert response.status_code == status.HTTP_200_OK

@pytest.mark.django_db
class EnvironmentTestCase(TestCase):
    env_post_template = '{"name": "%s", "project": %d}'
    fs_put_template = '{ "id" : %d, "enabled" : "%r", "feature_state_value" : "%s" }'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.client = APIClient()
        self.user = Helper.create_ffadminuser()
        self.client.force_authenticate(user=self.user)
        create_environment_permission = ProjectPermissionModel.objects.get(key=CREATE_ENVIRONMENT)
        read_project_permission = ProjectPermissionModel.objects.get(key=VIEW_PROJECT)
        self.organisation = Organisation.objects.create(name='ssg')
        self.user.add_organisation(self.organisation, OrganisationRole.ADMIN)
        self.project = Project.objects.create(name='Test project', organisation=self.organisation)
        user_project_permission = UserProjectPermission.objects.create(user=self.user, project=self.project)
        user_project_permission.permissions.add(create_environment_permission, read_project_permission)

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        Environment.objects.all().delete()
        AuditLog.objects.all().delete()

    def test_should_return_identities_for_an_environment(self):
        if False:
            while True:
                i = 10
        identifier_one = 'user1'
        identifier_two = 'user2'
        environment = Environment.objects.create(name='environment1', project=self.project)
        Identity.objects.create(identifier=identifier_one, environment=environment)
        Identity.objects.create(identifier=identifier_two, environment=environment)
        url = reverse('api-v1:environments:environment-identities-list', args=[environment.api_key])
        response = self.client.get(url)
        assert response.data['results'][0]['identifier'] == identifier_one
        assert response.data['results'][1]['identifier'] == identifier_two

    def test_audit_log_entry_created_when_new_environment_created(self):
        if False:
            for i in range(10):
                print('nop')
        url = reverse('api-v1:environments:environment-list')
        data = {'project': self.project.id, 'name': 'Test Environment'}
        self.client.post(url, data=data)
        assert AuditLog.objects.filter(related_object_type=RelatedObjectType.ENVIRONMENT.name).count() == 1

    def test_audit_log_created_when_feature_state_updated(self):
        if False:
            for i in range(10):
                print('nop')
        feature = Feature.objects.create(name='feature', project=self.project)
        environment = Environment.objects.create(name='test env', project=self.project)
        feature_state = FeatureState.objects.get(feature=feature, environment=environment)
        url = reverse('api-v1:environments:environment-featurestates-detail', args=[environment.api_key, feature_state.id])
        data = {'id': feature.id, 'enabled': True}
        self.client.put(url, data=data)
        assert AuditLog.objects.filter(related_object_type=RelatedObjectType.FEATURE_STATE.name).count() == 1
        assert AuditLog.objects.first().author

    def test_delete_trait_keys_deletes_trait_for_all_users_in_that_environment(self):
        if False:
            print('Hello World!')
        environment_one = Environment.objects.create(project=self.project, name='Test Environment 1')
        environment_two = Environment.objects.create(project=self.project, name='Test Environment 2')
        identity_one_environment_one = Identity.objects.create(environment=environment_one, identifier='identity-one-env-one')
        identity_one_environment_two = Identity.objects.create(environment=environment_two, identifier='identity-one-env-two')
        trait_key = 'trait-key'
        Trait.objects.create(identity=identity_one_environment_one, trait_key=trait_key, string_value='blah', value_type=STRING)
        Trait.objects.create(identity=identity_one_environment_two, trait_key=trait_key, string_value='blah', value_type=STRING)
        url = reverse('api-v1:environments:environment-delete-traits', args=[environment_one.api_key])
        response = self.client.post(url, data={'key': trait_key})
        assert response.status_code == status.HTTP_200_OK
        assert not Trait.objects.filter(identity=identity_one_environment_one, trait_key=trait_key).exists()
        assert Trait.objects.filter(identity=identity_one_environment_two, trait_key=trait_key).exists()

    def test_environment_user_can_get_their_permissions(self):
        if False:
            print('Hello World!')
        user = FFAdminUser.objects.create(email='new-test@test.com')
        user.add_organisation(self.organisation)
        environment = Environment.objects.create(name='Test environment', project=self.project)
        user_permission = UserEnvironmentPermission.objects.create(user=user, environment=environment)
        user_permission.add_permission('VIEW_ENVIRONMENT')
        url = reverse('api-v1:environments:environment-my-permissions', args=[environment.api_key])
        self.client.force_authenticate(user)
        response = self.client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert not response.json()['admin']
        assert 'VIEW_ENVIRONMENT' in response.json()['permissions']

@pytest.mark.django_db
class WebhookViewSetTestCase(TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.client = APIClient()
        user = Helper.create_ffadminuser()
        self.client.force_authenticate(user=user)
        organisation = Organisation.objects.create(name='Test organisation')
        user.add_organisation(organisation, OrganisationRole.ADMIN)
        project = Project.objects.create(name='Test project', organisation=organisation)
        self.environment = Environment.objects.create(name='Test environment', project=project)
        self.valid_webhook_url = 'http://my.webhook.com/webhooks'

    def test_can_create_webhook_for_an_environment(self):
        if False:
            for i in range(10):
                print('nop')
        url = reverse('api-v1:environments:environment-webhooks-list', args=[self.environment.api_key])
        data = {'url': self.valid_webhook_url, 'enabled': True}
        res = self.client.post(url, data)
        assert res.status_code == status.HTTP_201_CREATED
        assert Webhook.objects.filter(environment=self.environment, **data).exists()

    def test_can_update_webhook_for_an_environment(self):
        if False:
            return 10
        webhook = Webhook.objects.create(url=self.valid_webhook_url, environment=self.environment)
        url = reverse('api-v1:environments:environment-webhooks-detail', args=[self.environment.api_key, webhook.id])
        data = {'url': 'http://my.new.url.com/wehbooks', 'enabled': False}
        res = self.client.put(url, data=json.dumps(data), content_type='application/json')
        assert res.status_code == status.HTTP_200_OK
        webhook.refresh_from_db()
        assert webhook.url == data['url'] and (not webhook.enabled)

    def test_can_update_secret(self):
        if False:
            for i in range(10):
                print('nop')
        webhook = Webhook.objects.create(url=self.valid_webhook_url, environment=self.environment)
        url = reverse('api-v1:environments:environment-webhooks-detail', args=[self.environment.api_key, webhook.id])
        data = {'secret': 'random_secret'}
        res = self.client.patch(url, data=json.dumps(data), content_type='application/json')
        assert res.status_code == status.HTTP_200_OK
        webhook.refresh_from_db()
        assert webhook.secret == data['secret']

    def test_can_delete_webhook_for_an_environment(self):
        if False:
            while True:
                i = 10
        webhook = Webhook.objects.create(url=self.valid_webhook_url, environment=self.environment)
        url = reverse('api-v1:environments:environment-webhooks-detail', args=[self.environment.api_key, webhook.id])
        res = self.client.delete(url)
        assert res.status_code == status.HTTP_204_NO_CONTENT
        assert not Webhook.objects.filter(id=webhook.id).exists()

    def test_can_list_webhooks_for_an_environment(self):
        if False:
            while True:
                i = 10
        webhook = Webhook.objects.create(url=self.valid_webhook_url, environment=self.environment)
        url = reverse('api-v1:environments:environment-webhooks-list', args=[self.environment.api_key])
        res = self.client.get(url)
        assert res.status_code == status.HTTP_200_OK
        assert res.json()[0]['id'] == webhook.id

    def test_cannot_delete_webhooks_for_environment_user_does_not_belong_to(self):
        if False:
            return 10
        new_organisation = Organisation.objects.create(name='New organisation')
        new_project = Project.objects.create(name='New project', organisation=new_organisation)
        new_environment = Environment.objects.create(name='New Environment', project=new_project)
        webhook = Webhook.objects.create(url=self.valid_webhook_url, environment=new_environment)
        url = reverse('api-v1:environments:environment-webhooks-detail', args=[self.environment.api_key, webhook.id])
        res = self.client.delete(url)
        assert res.status_code == status.HTTP_404_NOT_FOUND
        assert Webhook.objects.filter(id=webhook.id).exists()

    @mock.patch('webhooks.mixins.trigger_sample_webhook')
    def test_trigger_sample_webhook_calls_trigger_sample_webhook_method_with_correct_arguments(self, trigger_sample_webhook):
        if False:
            print('Hello World!')
        mocked_response = mock.MagicMock(status_code=200)
        trigger_sample_webhook.return_value = mocked_response
        url = reverse('api-v1:environments:environment-webhooks-trigger-sample-webhook', args=[self.environment.api_key])
        data = {'url': self.valid_webhook_url}
        response = self.client.post(url, data)
        assert response.json()['message'] == 'Request returned 200'
        assert response.status_code == status.HTTP_200_OK
        (args, _) = trigger_sample_webhook.call_args
        assert args[0].url == self.valid_webhook_url

@pytest.mark.django_db
class EnvironmentAPIKeyViewSetTestCase(TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.organisation = Organisation.objects.create(name='Test Org')
        self.project = Project.objects.create(organisation=self.organisation, name='Test Project')
        self.environment = Environment.objects.create(project=self.project, name='Test Environment')
        user = FFAdminUser.objects.create(email='test@example.com')
        user.add_organisation(self.organisation, OrganisationRole.ADMIN)
        self.client = APIClient()
        self.client.force_authenticate(user)
        self.list_url = reverse('api-v1:environments:api-keys-list', args={self.environment.api_key})

    def test_list_api_keys(self):
        if False:
            print('Hello World!')
        api_key_1 = EnvironmentAPIKey.objects.create(environment=self.environment, name='api key 1')
        api_key_2 = EnvironmentAPIKey.objects.create(environment=self.environment, name='api key 2')
        response = self.client.get(self.list_url)
        assert response.status_code == status.HTTP_200_OK
        response_json = response.json()
        assert len(response_json) == 2
        assert {api_key['id'] for api_key in response_json} == {api_key_1.id, api_key_2.id}

    def test_create_api_key(self):
        if False:
            while True:
                i = 10
        some_key = 'some.key'
        data = {'name': 'Some key', 'key': some_key}
        response = self.client.post(self.list_url, data=json.dumps(data), content_type='application/json')
        assert response.status_code == status.HTTP_201_CREATED
        response_json = response.json()
        assert response_json['key'] and response_json['key'].startswith('ser.')
        assert response_json['key'] != some_key
        assert response_json['active']

    def test_update_api_key(self):
        if False:
            i = 10
            return i + 15
        old_name = 'Some key'
        api_key = EnvironmentAPIKey.objects.create(name=old_name, environment=self.environment)
        update_url = reverse('api-v1:environments:api-keys-detail', args=[self.environment.api_key, api_key.id])
        new_name = 'new name'
        new_key = 'new_key'
        response = self.client.patch(update_url, data={'active': False, 'name': new_name, 'key': new_key})
        assert response.status_code == status.HTTP_200_OK
        api_key.refresh_from_db()
        assert api_key.name == new_name
        assert not api_key.active
        assert api_key.key != new_key

    def test_delete_api_key(self):
        if False:
            return 10
        api_key = EnvironmentAPIKey.objects.create(name='Some key', environment=self.environment)
        delete_url = reverse('api-v1:environments:api-keys-detail', args=[self.environment.api_key, api_key.id])
        self.client.delete(delete_url)
        assert not EnvironmentAPIKey.objects.filter(id=api_key.id)

@pytest.mark.parametrize('client, is_admin_master_api_key_client', [(lazy_fixture('admin_master_api_key_client'), True), (lazy_fixture('admin_client'), False)])
def test_should_create_environments(project, client, admin_user, is_admin_master_api_key_client):
    if False:
        print('Hello World!')
    url = reverse('api-v1:environments:environment-list')
    description = 'This is the description'
    data = {'name': 'Test environment', 'project': project.id, 'description': description}
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()['description'] == description
    assert response.json()['use_mv_v2_evaluation'] is True
    assert response.json()['use_identity_composite_key_for_hashing'] is True
    if not is_admin_master_api_key_client:
        assert UserEnvironmentPermission.objects.filter(user=admin_user, admin=True, environment__id=response.json()['id']).exists()

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_create_environment_without_required_metadata_returns_400(project, client, required_a_environment_metadata_field, optional_b_environment_metadata_field):
    if False:
        print('Hello World!')
    url = reverse('api-v1:environments:environment-list')
    description = 'This is the description'
    data = {'name': 'Test environment', 'project': project.id, 'description': description}
    response = client.post(url, data=data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert 'Missing required metadata field' in response.json()['metadata'][0]

def test_view_environment_with_staff__query_count_is_expected(staff_client: APIClient, environment: Environment, with_environment_permissions: Callable[[list[str], int], None], project: Project, django_assert_num_queries: Callable[[int], None], environment_metadata_a: Metadata, environment_metadata_b: Metadata, required_a_environment_metadata_field: MetadataModelField, environment_content_type: ContentType) -> None:
    if False:
        return 10
    with_environment_permissions([VIEW_ENVIRONMENT])
    url = reverse('api-v1:environments:environment-list')
    data = {'project': project.id}
    expected_query_count = 7
    with django_assert_num_queries(expected_query_count):
        response = staff_client.get(url, data=data, content_type='application/json')
    assert response.status_code == status.HTTP_200_OK
    environment_2 = Environment.objects.create(name='Second Environment', project=project)
    Metadata.objects.create(object_id=environment_2.id, content_type=environment_content_type, model_field=required_a_environment_metadata_field, field_value='10')
    with_environment_permissions([VIEW_ENVIRONMENT], environment_id=environment_2.id)
    expected_query_count += 1
    with django_assert_num_queries(expected_query_count):
        response = staff_client.get(url, data=data, content_type='application/json')
    assert response.status_code == status.HTTP_200_OK

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_create_environment_with_required_metadata_returns_201(project, client, required_a_environment_metadata_field, optional_b_environment_metadata_field):
    if False:
        return 10
    url = reverse('api-v1:environments:environment-list')
    description = 'This is the description'
    field_value = 10
    data = {'name': 'Test environment', 'project': project.id, 'description': description, 'metadata': [{'model_field': required_a_environment_metadata_field.id, 'field_value': field_value}]}
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()['metadata'][0]['model_field'] == required_a_environment_metadata_field.field.id
    assert response.json()['metadata'][0]['field_value'] == str(field_value)

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_update_environment_metadata(project, client, environment, environment_metadata_a, environment_metadata_b):
    if False:
        while True:
            i = 10
    url = reverse('api-v1:environments:environment-detail', args=[environment.api_key])
    updated_field_value = 999
    data = {'project': project.id, 'name': 'New name', 'description': 'new_data', 'metadata': [{'model_field': environment_metadata_a.model_field.id, 'field_value': updated_field_value, 'id': environment_metadata_a.id}]}
    response = client.put(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()['metadata']) == 1
    assert response.json()['metadata'][0]['field_value'] == str(updated_field_value)
    environment_metadata_a.refresh_from_db()
    environment_metadata_a.field_value = str(updated_field_value)
    assert Metadata.objects.filter(id=environment_metadata_b.id).exists() is False

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_audit_log_entry_created_when_environment_updated(environment, project, client):
    if False:
        i = 10
        return i + 15
    environment = Environment.objects.create(name='Test environment', project=project)
    url = reverse('api-v1:environments:environment-detail', args=[environment.api_key])
    banner_text = 'production environment be careful'
    banner_colour = '#FF0000'
    hide_disabled_flags = True
    use_identity_composite_key_for_hashing = True
    hide_sensitive_data = True
    data = {'project': project.id, 'name': 'New name', 'banner_text': banner_text, 'banner_colour': banner_colour, 'hide_disabled_flags': hide_disabled_flags, 'use_identity_composite_key_for_hashing': use_identity_composite_key_for_hashing, 'hide_sensitive_data': hide_sensitive_data}
    response = client.put(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_200_OK
    assert AuditLog.objects.filter(related_object_type=RelatedObjectType.ENVIRONMENT.name).count() == 1
    assert response.json()['banner_text'] == banner_text
    assert response.json()['banner_colour'] == banner_colour
    assert response.json()['hide_disabled_flags'] == hide_disabled_flags
    assert response.json()['hide_sensitive_data'] == hide_sensitive_data
    assert response.json()['use_identity_composite_key_for_hashing'] == use_identity_composite_key_for_hashing

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_get_document(environment, project, client, feature, segment):
    if False:
        for i in range(10):
            print('nop')
    segment_rule = SegmentRule.objects.create(segment=segment, type=SegmentRule.ALL_RULE)
    Condition.objects.create(operator=EQUAL, property='property', value='value', rule=segment_rule)
    url = reverse('api-v1:environments:environment-get-document', args=[environment.api_key])
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_get_all_trait_keys_for_environment_only_returns_distinct_keys(identity, client, trait, environment):
    if False:
        i = 10
        return i + 15
    trait_key_one = trait.trait_key
    trait_key_two = 'trait-key-two'
    identity_two = Identity.objects.create(environment=environment, identifier='identity-two')
    Trait.objects.create(identity=identity, trait_key=trait_key_two, string_value='blah', value_type=STRING)
    Trait.objects.create(identity=identity_two, trait_key=trait_key_one, string_value='blah', value_type=STRING)
    url = reverse('api-v1:environments:environment-trait-keys', args=[environment.api_key])
    res = client.get(url)
    assert res.status_code == status.HTTP_200_OK
    assert len(res.json().get('keys')) == 2

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_delete_trait_keys_deletes_traits_matching_provided_key_only(environment, client, identity, trait):
    if False:
        return 10
    trait_to_delete = trait.trait_key
    trait_to_persist = 'trait-key-to-persist'
    Trait.objects.create(identity=identity, trait_key=trait_to_persist, value_type=STRING, string_value='blah')
    url = reverse('api-v1:environments:environment-delete-traits', args=[environment.api_key])
    client.post(url, data={'key': trait_to_delete})
    assert not Trait.objects.filter(identity=identity, trait_key=trait_to_delete).exists()
    assert Trait.objects.filter(identity=identity, trait_key=trait_to_persist).exists()

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_user_can_list_environment_permission(client, environment):
    if False:
        for i in range(10):
            print('nop')
    url = reverse('api-v1:environments:environment-permissions')
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()) == 7

def test_environment_my_permissions_reruns_400_for_master_api_key(admin_master_api_key_client, environment):
    if False:
        print('Hello World!')
    url = reverse('api-v1:environments:environment-my-permissions', args=[environment.api_key])
    response = admin_master_api_key_client.get(url)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()['detail'] == 'This endpoint can only be used with a user and not Master API Key'

def test_partial_environment_update(admin_client: APIClient, environment: 'Environment') -> None:
    if False:
        while True:
            i = 10
    url = reverse('api-v1:environments:environment-detail', args=[environment.api_key])
    new_name = 'updated!'
    response = admin_client.patch(url, data=json.dumps({'name': new_name}), content_type='application/json')
    assert response.status_code == status.HTTP_200_OK