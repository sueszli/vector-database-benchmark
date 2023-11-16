import uuid
import pytest
from django.urls import reverse
from pytest_lazyfixture import lazy_fixture
from rest_framework import status
from features.multivariate.views import MultivariateFeatureOptionViewSet
from projects.permissions import CREATE_FEATURE, VIEW_PROJECT, NestedProjectPermissions

def test_multivariate_feature_options_view_set_get_permissions():
    if False:
        i = 10
        return i + 15
    view_set = MultivariateFeatureOptionViewSet()
    permissions = view_set.get_permissions()
    assert len(permissions) == 1
    assert isinstance(permissions[0], NestedProjectPermissions)
    assert permissions[0].action_permission_map == {'list': VIEW_PROJECT, 'detail': VIEW_PROJECT, 'create': CREATE_FEATURE, 'update': CREATE_FEATURE, 'partial_update': CREATE_FEATURE, 'destroy': CREATE_FEATURE}

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_get_mv_feature_option_by_uuid(client, project, multivariate_feature):
    if False:
        return 10
    mv_option_uuid = multivariate_feature.multivariate_options.first().uuid
    url = reverse('api-v1:multivariate:get-mv-feature-option-by-uuid', args=[str(mv_option_uuid)])
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.data['uuid'] == str(mv_option_uuid)
    assert response.data['feature'] == multivariate_feature.id

@pytest.mark.parametrize('client', [lazy_fixture('admin_master_api_key_client'), lazy_fixture('admin_client')])
def test_get_mv_feature_option_by_uuid_returns_404_if_mv_option_does_not_exists(client, project):
    if False:
        return 10
    url = reverse('api-v1:multivariate:get-mv-feature-option-by-uuid', args=[str(uuid.uuid4())])
    response = client.get(url)
    assert response.status_code == status.HTTP_404_NOT_FOUND