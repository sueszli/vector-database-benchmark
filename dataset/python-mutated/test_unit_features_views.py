import json
import uuid
import pytest
from django.urls import reverse
from django.utils import timezone
from pytest_lazyfixture import lazy_fixture
from rest_framework import status
from audit.constants import FEATURE_DELETED_MESSAGE
from audit.models import AuditLog, RelatedObjectType
from environments.identities.models import Identity
from environments.models import Environment
from features.feature_types import MULTIVARIATE
from features.models import Feature, FeatureSegment, FeatureState
from features.multivariate.models import MultivariateFeatureOption
from organisations.models import Organisation, OrganisationRole
from projects.models import Project, UserProjectPermission
from projects.permissions import VIEW_PROJECT
from segments.models import Segment
from users.models import FFAdminUser, UserPermissionGroup

def test_list_feature_states_from_simple_view_set(environment, feature, admin_user, admin_client, django_assert_num_queries):
    if False:
        return 10
    base_url = reverse('api-v1:features:featurestates-list')
    url = f'{base_url}?environment={environment.id}'
    Feature.objects.create(name='another_feature', project=environment.project)
    another_organisation = Organisation.objects.create(name='another_organisation')
    admin_user.add_organisation(another_organisation)
    another_project = Project.objects.create(name='another_project', organisation=another_organisation)
    Environment.objects.create(name='another_environment', project=another_project)
    Feature.objects.create(project=another_project, name='another_projects_feature')
    UserProjectPermission.objects.create(user=admin_user, project=another_project, admin=True)
    mv_feature = Feature.objects.create(name='mv_feature', project=environment.project, type=MULTIVARIATE)
    MultivariateFeatureOption.objects.create(feature=mv_feature, default_percentage_allocation=10, type='unicode', string_value='foo')
    with django_assert_num_queries(8):
        response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json['count'] == 3

def test_list_feature_states_nested_environment_view_set(environment, project, feature, admin_client, django_assert_num_queries):
    if False:
        return 10
    base_url = reverse('api-v1:environments:environment-featurestates-list', args=[environment.api_key])
    mv_feature = Feature.objects.create(name='mv_feature', project=project, type=MULTIVARIATE)
    MultivariateFeatureOption.objects.create(feature=mv_feature, default_percentage_allocation=10, type='unicode', string_value='foo')
    Feature.objects.create(name='another_feature', project=project)
    with django_assert_num_queries(8):
        response = admin_client.get(base_url)
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json['count'] == 3

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_environment_feature_states_filter_using_feataure_name(environment, project, feature, client):
    if False:
        print('Hello World!')
    Feature.objects.create(name='another_feature', project=project)
    base_url = reverse('api-v1:environments:environment-featurestates-list', args=[environment.api_key])
    url = f'{base_url}?feature_name={feature.name}'
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()['count'] == 1
    assert response.json()['results'][0]['feature'] == feature.id

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_environment_feature_states_filter_to_show_identity_override_only(environment, feature, client):
    if False:
        print('Hello World!')
    FeatureState.objects.get(environment=environment, feature=feature)
    identifier = 'test-identity'
    identity = Identity.objects.create(identifier=identifier, environment=environment)
    FeatureState.objects.create(environment=environment, feature=feature, identity=identity)
    base_url = reverse('api-v1:environments:environment-featurestates-list', args=[environment.api_key])
    url = base_url + '?anyIdentity&feature=' + str(feature.id)
    res = client.get(url)
    assert res.status_code == status.HTTP_200_OK
    assert len(res.json().get('results')) == 1
    assert res.json()['results'][0]['identity']['identifier'] == identifier

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_environment_feature_states_only_returns_latest_versions(environment, feature, client):
    if False:
        while True:
            i = 10
    feature_state = FeatureState.objects.get(environment=environment, feature=feature)
    feature_state_v2 = feature_state.clone(env=environment, live_from=timezone.now(), version=2)
    url = reverse('api-v1:environments:environment-featurestates-list', args=[environment.api_key])
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert len(response_json['results']) == 1
    assert response_json['results'][0]['id'] == feature_state_v2.id

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_environment_feature_states_does_not_return_null_versions(environment, feature, client):
    if False:
        while True:
            i = 10
    feature_state = FeatureState.objects.get(environment=environment, feature=feature)
    FeatureState.objects.create(environment=environment, feature=feature, version=None)
    url = reverse('api-v1:environments:environment-featurestates-list', args=[environment.api_key])
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert len(response_json['results']) == 1
    assert response_json['results'][0]['id'] == feature_state.id

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_create_feature_default_is_archived_is_false(client, project):
    if False:
        for i in range(10):
            print('nop')
    data = {'name': 'test feature'}
    url = reverse('api-v1:projects:project-features-list', args=[project.id])
    response = client.post(url, data=json.dumps(data), content_type='application/json').json()
    assert response['is_archived'] is False

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_update_feature_is_archived(client, project, feature):
    if False:
        while True:
            i = 10
    feature = Feature.objects.create(name='test feature', project=project)
    url = reverse('api-v1:projects:project-features-detail', args=[project.id, feature.id])
    data = {'name': 'test feature', 'is_archived': True}
    response = client.put(url, data=data).json()
    assert response['is_archived'] is True

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_should_create_feature_states_when_feature_created(client, project, environment):
    if False:
        return 10
    environment_2 = Environment.objects.create(name='Test environment 2', project=project)
    default_value = 'This is a value'
    data = {'name': 'test feature', 'initial_value': default_value, 'project': project.id}
    url = reverse('api-v1:projects:project-features-list', args=[project.id])
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED
    assert Feature.objects.filter(name='test feature', project=project.id).count() == 1
    assert FeatureState.objects.filter(environment=environment).count() == 1
    assert FeatureState.objects.filter(environment=environment_2).count() == 1
    feature_state = FeatureState.objects.filter(environment=environment).first()
    assert feature_state.get_feature_state_value() == default_value

@pytest.mark.parametrize('default_value', [12, True, 'test'])
@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_should_create_feature_states_with_value_when_feature_created(client, project, environment, default_value):
    if False:
        return 10
    url = reverse('api-v1:projects:project-features-list', args=[project.id])
    data = {'name': 'test feature', 'initial_value': default_value, 'project': project.id}
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED
    assert Feature.objects.filter(name='test feature', project=project.id).count() == 1
    assert FeatureState.objects.filter(environment=environment).count() == 1
    feature_state = FeatureState.objects.filter(environment=environment).first()
    assert feature_state.get_feature_state_value() == default_value

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_should_delete_feature_states_when_feature_deleted(client, project, feature, environment):
    if False:
        return 10
    url = reverse('api-v1:projects:project-features-detail', args=[project.id, feature.id])
    response = client.delete(url)
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert Feature.objects.filter(name='test feature', project=project.id).count() == 0
    assert FeatureState.objects.filter(environment=environment, feature=feature).count() == 0
    assert FeatureState.objects.filter(environment=environment, feature=feature).count() == 0

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_create_feature_returns_201_if_name_matches_regex(client, project):
    if False:
        return 10
    project.feature_name_regex = '^[a-z_]{18}$'
    project.save()
    feature_name = 'valid_feature_name'
    url = reverse('api-v1:projects:project-features-list', args=[project.id])
    data = {'name': feature_name, 'type': 'FLAG', 'project': project.id}
    response = client.post(url, data=data)
    assert response.status_code == status.HTTP_201_CREATED

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_create_feature_returns_400_if_name_does_not_matches_regex(client, project):
    if False:
        print('Hello World!')
    project.feature_name_regex = '^[a-z]{18}$'
    project.save()
    feature_name = 'not_a_valid_feature_name'
    url = reverse('api-v1:projects:project-features-list', args=[project.id])
    data = {'name': feature_name, 'type': 'FLAG', 'project': project.id}
    response = client.post(url, data=data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()['name'][0] == f'Feature name must match regex: {project.feature_name_regex}'

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_audit_log_created_when_feature_created(client, project, environment):
    if False:
        print('Hello World!')
    url = reverse('api-v1:projects:project-features-list', args=[project.id])
    data = {'name': 'Test feature flag', 'type': 'FLAG', 'project': project.id}
    response = client.post(url, data=data)
    feature_id = response.json()['id']
    assert AuditLog.objects.filter(related_object_type=RelatedObjectType.FEATURE.name, related_object_id=feature_id).count() == 1
    assert AuditLog.objects.filter(related_object_type=RelatedObjectType.FEATURE_STATE.name, project=project, environment__in=project.environments.all()).count() == len(project.environments.all())

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_audit_log_created_when_feature_updated(client, project, feature):
    if False:
        return 10
    url = reverse('api-v1:projects:project-features-detail', args=[project.id, feature.id])
    data = {'name': 'Test Feature updated', 'type': 'FLAG', 'project': project.id}
    client.put(url, data=data)
    assert AuditLog.objects.filter(related_object_type=RelatedObjectType.FEATURE.name).count() == 1

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_audit_logs_created_when_feature_deleted(client, project, feature):
    if False:
        i = 10
        return i + 15
    url = reverse('api-v1:projects:project-features-detail', args=[project.id, feature.id])
    feature_states_ids = list(feature.feature_states.values_list('id', flat=True))
    client.delete(url)
    assert AuditLog.objects.get(related_object_type=RelatedObjectType.FEATURE.name, related_object_id=feature.id, log=FEATURE_DELETED_MESSAGE % feature.name)
    assert AuditLog.objects.filter(related_object_type=RelatedObjectType.FEATURE_STATE.name, related_object_id__in=feature_states_ids, log=FEATURE_DELETED_MESSAGE % feature.name).count() == len(feature_states_ids)

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_should_create_tags_when_feature_created(client, project, tag_one, tag_two):
    if False:
        return 10
    default_value = 'Test'
    feature_name = 'Test feature'
    data = {'name': feature_name, 'project': project.id, 'initial_value': default_value, 'tags': [tag_one.id, tag_two.id]}
    url = reverse('api-v1:projects:project-features-list', args=[project.id])
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED
    feature = Feature.objects.filter(name=feature_name, project=project.id).first()
    assert feature.tags.count() == 2
    assert list(feature.tags.all()) == [tag_one, tag_two]

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_add_owners_fails_if_user_not_found(client, project):
    if False:
        return 10
    feature = Feature.objects.create(name='Test Feature', project=project)
    user_1 = FFAdminUser.objects.create_user(email='user1@mail.com')
    user_2 = FFAdminUser.objects.create_user(email='user2@mail.com')
    url = reverse('api-v1:projects:project-features-add-owners', args=[project.id, feature.id])
    data = {'user_ids': [user_1.id, user_2.id]}
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.data == ['Some users not found']
    assert feature.owners.filter(id__in=[user_1.id, user_2.id]).count() == 0

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_add_owners_adds_owner(staff_user, admin_user, client, project):
    if False:
        for i in range(10):
            print('nop')
    feature = Feature.objects.create(name='Test Feature', project=project)
    UserProjectPermission.objects.create(user=staff_user, project=project).add_permission(VIEW_PROJECT)
    url = reverse('api-v1:projects:project-features-add-owners', args=[project.id, feature.id])
    data = {'user_ids': [staff_user.id, admin_user.id]}
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    json_response = response.json()
    assert len(json_response['owners']) == 2
    assert json_response['owners'][0] == {'id': staff_user.id, 'email': staff_user.email, 'first_name': staff_user.first_name, 'last_name': staff_user.last_name, 'last_login': None}
    assert json_response['owners'][1] == {'id': admin_user.id, 'email': admin_user.email, 'first_name': admin_user.first_name, 'last_name': admin_user.last_name, 'last_login': None}

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_add_group_owners_adds_group_owner(client, project):
    if False:
        return 10
    feature = Feature.objects.create(name='Test Feature', project=project)
    user_1 = FFAdminUser.objects.create_user(email='user1@mail.com')
    organisation = project.organisation
    group_1 = UserPermissionGroup.objects.create(name='Test Group', organisation=organisation)
    group_2 = UserPermissionGroup.objects.create(name='Second Group', organisation=organisation)
    user_1.add_organisation(organisation, OrganisationRole.ADMIN)
    group_1.users.add(user_1)
    group_2.users.add(user_1)
    url = reverse('api-v1:projects:project-features-add-group-owners', args=[project.id, feature.id])
    data = {'group_ids': [group_1.id, group_2.id]}
    json_response = client.post(url, data=json.dumps(data), content_type='application/json').json()
    assert len(json_response['group_owners']) == 2
    assert json_response['group_owners'][0] == {'id': group_1.id, 'name': group_1.name}
    assert json_response['group_owners'][1] == {'id': group_2.id, 'name': group_2.name}

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_remove_group_owners_removes_group_owner(client, project):
    if False:
        while True:
            i = 10
    feature = Feature.objects.create(name='Test Feature', project=project)
    user_1 = FFAdminUser.objects.create_user(email='user1@mail.com')
    organisation = project.organisation
    group_1 = UserPermissionGroup.objects.create(name='To be removed group', organisation=organisation)
    group_2 = UserPermissionGroup.objects.create(name='To be kept group', organisation=organisation)
    user_1.add_organisation(organisation, OrganisationRole.ADMIN)
    group_1.users.add(user_1)
    group_2.users.add(user_1)
    feature.group_owners.add(group_1.id, group_2.id)
    url = reverse('api-v1:projects:project-features-remove-group-owners', args=[project.id, feature.id])
    data = {'group_ids': [group_1.id]}
    json_response = client.post(url, data=json.dumps(data), content_type='application/json').json()
    assert len(json_response['group_owners']) == 1
    assert json_response['group_owners'][0] == {'id': group_2.id, 'name': group_2.name}

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_remove_group_owners_when_nonexistent(client, project):
    if False:
        for i in range(10):
            print('nop')
    feature = Feature.objects.create(name='Test Feature', project=project)
    user_1 = FFAdminUser.objects.create_user(email='user1@mail.com')
    organisation = project.organisation
    group_1 = UserPermissionGroup.objects.create(name='To be removed group', organisation=organisation)
    user_1.add_organisation(organisation, OrganisationRole.ADMIN)
    group_1.users.add(user_1)
    assert feature.group_owners.count() == 0
    url = reverse('api-v1:projects:project-features-remove-group-owners', args=[project.id, feature.id])
    data = {'group_ids': [group_1.id]}
    json_response = client.post(url, data=json.dumps(data), content_type='application/json').json()
    assert len(json_response['group_owners']) == 0

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_add_group_owners_with_wrong_org_group(client, project):
    if False:
        while True:
            i = 10
    feature = Feature.objects.create(name='Test Feature', project=project)
    user_1 = FFAdminUser.objects.create_user(email='user1@mail.com')
    user_2 = FFAdminUser.objects.create_user(email='user2@mail.com')
    organisation = project.organisation
    other_organisation = Organisation.objects.create(name='Orgy')
    group_1 = UserPermissionGroup.objects.create(name='Valid Group', organisation=organisation)
    group_2 = UserPermissionGroup.objects.create(name='Invalid Group', organisation=other_organisation)
    user_1.add_organisation(organisation, OrganisationRole.ADMIN)
    user_2.add_organisation(other_organisation, OrganisationRole.ADMIN)
    group_1.users.add(user_1)
    group_2.users.add(user_2)
    url = reverse('api-v1:projects:project-features-add-group-owners', args=[project.id, feature.id])
    data = {'group_ids': [group_1.id, group_2.id]}
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == 400
    response.json() == {'non_field_errors': ['Some groups not found']}

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_list_features_return_tags(client, project, feature):
    if False:
        for i in range(10):
            print('nop')
    Feature.objects.create(name='test_feature', project=project)
    url = reverse('api-v1:projects:project-features-list', args=[project.id])
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    feature = response_json['results'][0]
    assert 'tags' in feature

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_project_admin_can_create_mv_options_when_creating_feature(client, project):
    if False:
        return 10
    data = {'name': 'test_feature', 'default_enabled': True, 'multivariate_options': [{'type': 'unicode', 'string_value': 'test-value'}]}
    url = reverse('api-v1:projects:project-features-list', args=[project.id])
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED
    response_json = response.json()
    assert len(response_json['multivariate_options']) == 1

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_get_feature_by_uuid(client, project, feature):
    if False:
        return 10
    url = reverse('api-v1:features:get-feature-by-uuid', args=[feature.uuid])
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()['id'] == feature.id
    assert response.json()['uuid'] == str(feature.uuid)

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_get_feature_by_uuid_returns_404_if_feature_does_not_exists(client, project):
    if False:
        return 10
    url = reverse('api-v1:features:get-feature-by-uuid', args=[uuid.uuid4()])
    response = client.get(url)
    assert response.status_code == status.HTTP_404_NOT_FOUND

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_update_feature_state_value_triggers_dynamo_rebuild(client, project, environment, feature, feature_state, settings, mocker):
    if False:
        return 10
    project.enable_dynamo_db = True
    project.save()
    url = reverse('api-v1:environments:environment-featurestates-detail', args=[environment.api_key, feature_state.id])
    mock_dynamo_environment_wrapper = mocker.patch('environments.models.environment_wrapper')
    response = client.patch(url, data=json.dumps({'feature_state_value': 'new value'}), content_type='application/json')
    assert response.status_code == 200
    mock_dynamo_environment_wrapper.write_environments.assert_called_once()

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_create_segment_overrides_creates_correct_audit_log_messages(client, feature, segment, environment):
    if False:
        i = 10
        return i + 15
    another_segment = Segment.objects.create(name='Another segment', project=segment.project)
    feature_segments_url = reverse('api-v1:features:feature-segment-list')
    feature_states_url = reverse('api-v1:features:featurestates-list')
    for _segment in (segment, another_segment):
        feature_segment_response = client.post(feature_segments_url, data={'feature': feature.id, 'segment': _segment.id, 'environment': environment.id})
        assert feature_segment_response.status_code == status.HTTP_201_CREATED
        feature_segment_id = feature_segment_response.json()['id']
        feature_state_response = client.post(feature_states_url, data={'feature': feature.id, 'feature_segment': feature_segment_id, 'environment': environment.id, 'enabled': True})
        assert feature_state_response.status_code == status.HTTP_201_CREATED
    assert AuditLog.objects.count() == 2
    assert AuditLog.objects.filter(log=f"Flag state / Remote config value updated for feature '{feature.name}' and segment '{segment.name}'").count() == 1
    assert AuditLog.objects.filter(log=f"Flag state / Remote config value updated for feature '{feature.name}' and segment '{another_segment.name}'").count() == 1

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_list_features_provides_information_on_number_of_overrides(feature, segment, segment_featurestate, identity, identity_featurestate, project, environment, client):
    if False:
        for i in range(10):
            print('nop')
    url = '%s?environment=%d' % (reverse('api-v1:projects:project-features-list', args=[project.id]), environment.id)
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json['count'] == 1
    assert response_json['results'][0]['num_segment_overrides'] == 1
    assert response_json['results'][0]['num_identity_overrides'] == 1

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_list_features_provides_segment_overrides_for_dynamo_enabled_project(dynamo_enabled_project, dynamo_enabled_project_environment_one, client):
    if False:
        i = 10
        return i + 15
    feature = Feature.objects.create(name='test_feature', project=dynamo_enabled_project)
    segment = Segment.objects.create(name='test_segment', project=dynamo_enabled_project)
    feature_segment = FeatureSegment.objects.create(feature=feature, segment=segment, environment=dynamo_enabled_project_environment_one)
    FeatureState.objects.create(feature=feature, environment=dynamo_enabled_project_environment_one, feature_segment=feature_segment)
    url = '%s?environment=%d' % (reverse('api-v1:projects:project-features-list', args=[dynamo_enabled_project.id]), dynamo_enabled_project_environment_one.id)
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json['count'] == 1
    assert response_json['results'][0]['num_segment_overrides'] == 1
    assert response_json['results'][0]['num_identity_overrides'] is None

def test_create_segment_override_reaching_max_limit(admin_client, feature, segment, project, environment, settings):
    if False:
        print('Hello World!')
    project.max_segment_overrides_allowed = 1
    project.save()
    url = reverse('api-v1:environments:create-segment-override', args=[environment.api_key, feature.id])
    data = {'feature_state_value': {'string_value': 'value'}, 'enabled': True, 'feature_segment': {'segment': segment.id}}
    response = admin_client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED
    response = admin_client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()['environment'] == 'The environment has reached the maximum allowed segments overrides limit.'
    assert environment.feature_segments.count() == 1

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_create_feature_reaching_max_limit(client, project, settings):
    if False:
        for i in range(10):
            print('nop')
    project.max_features_allowed = 1
    project.save()
    url = reverse('api-v1:projects:project-features-list', args=[project.id])
    response = client.post(url, data={'name': 'test_feature', 'project': project.id})
    assert response.status_code == status.HTTP_201_CREATED
    response = client.post(url, data={'name': 'second_feature', 'project': project.id})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()['project'] == 'The Project has reached the maximum allowed features limit.'

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_create_segment_override_using_environment_viewset(client, environment, feature, feature_segment):
    if False:
        while True:
            i = 10
    url = reverse('api-v1:environments:environment-featurestates-list', args=[environment.api_key])
    new_value = 'new-value'
    data = {'feature_state_value': new_value, 'enabled': False, 'feature': feature.id, 'environment': environment.id, 'identity': None, 'feature_segment': feature_segment.id}
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED
    response.json()['feature_state_value'] == new_value

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_cannot_create_feature_state_for_feature_from_different_project(client, environment, project_two_feature, feature_segment, project_two):
    if False:
        for i in range(10):
            print('nop')
    url = reverse('api-v1:environments:environment-featurestates-list', args=[environment.api_key])
    new_value = 'new-value'
    data = {'feature_state_value': new_value, 'enabled': False, 'feature': project_two_feature.id, 'environment': environment.id, 'identity': None, 'feature_segment': feature_segment.id}
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()['feature'][0] == 'Feature does not exist in project'

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_create_feature_state_environment_is_read_only(client, environment, feature, feature_segment, environment_two):
    if False:
        print('Hello World!')
    url = reverse('api-v1:environments:environment-featurestates-list', args=[environment.api_key])
    new_value = 'new-value'
    data = {'feature_state_value': new_value, 'enabled': False, 'feature': feature.id, 'environment': environment_two.id, 'feature_segment': feature_segment.id}
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()['environment'] == environment.id

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_cannot_create_feature_state_of_feature_from_different_project(client, environment, project_two_feature, feature_segment):
    if False:
        for i in range(10):
            print('nop')
    url = reverse('api-v1:environments:environment-featurestates-list', args=[environment.api_key])
    new_value = 'new-value'
    data = {'feature_state_value': new_value, 'enabled': False, 'feature': project_two_feature.id, 'environment': environment.id, 'identity': None, 'feature_segment': feature_segment.id}
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()['feature'][0] == 'Feature does not exist in project'

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_create_feature_state_environment_field_is_read_only(client, environment, feature, feature_segment, environment_two):
    if False:
        while True:
            i = 10
    url = reverse('api-v1:environments:environment-featurestates-list', args=[environment.api_key])
    new_value = 'new-value'
    data = {'feature_state_value': new_value, 'enabled': False, 'feature': feature.id, 'environment': environment_two.id, 'feature_segment': feature_segment.id}
    response = client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()['environment'] == environment.id

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_cannot_update_environment_of_a_feature_state(client, environment, feature, feature_state, environment_two):
    if False:
        print('Hello World!')
    url = reverse('api-v1:environments:environment-featurestates-detail', args=[environment.api_key, feature_state.id])
    new_value = 'new-value'
    data = {'id': feature_state.id, 'feature_state_value': new_value, 'enabled': False, 'feature': feature.id, 'environment': environment_two.id, 'identity': None, 'feature_segment': None}
    response = client.put(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()['environment'][0] == 'Cannot change the environment of a feature state'

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_cannot_update_feature_of_a_feature_state(client, environment, feature_state, feature, identity, project):
    if False:
        while True:
            i = 10
    another_feature = Feature.objects.create(name='another_feature', project=project, initial_value='initial_value')
    url = reverse('api-v1:features:featurestates-detail', args=[feature_state.id])
    feature_state_value = 'New value'
    data = {'enabled': True, 'feature_state_value': {'type': 'unicode', 'string_value': feature_state_value}, 'environment': environment.id, 'feature': another_feature.id}
    response = client.put(url, data=json.dumps(data), content_type='application/json')
    assert another_feature.feature_states.count() == 1
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()['feature'][0] == 'Cannot change the feature of a feature state'