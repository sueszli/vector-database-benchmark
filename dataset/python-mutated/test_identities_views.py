from django.urls import reverse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from environments.identities.views import IdentityViewSet
from environments.permissions.constants import MANAGE_IDENTITIES, VIEW_IDENTITIES
from environments.permissions.permissions import NestedEnvironmentPermissions

def test_user_with_view_identities_permission_can_retrieve_identity(environment, identity, test_user_client, view_environment_permission, view_identities_permission, view_project_permission, user_environment_permission, user_project_permission):
    if False:
        print('Hello World!')
    user_environment_permission.permissions.add(view_environment_permission, view_identities_permission)
    user_project_permission.permissions.add(view_project_permission)
    url = reverse('api-v1:environments:environment-identities-detail', args=(environment.api_key, identity.id))
    response = test_user_client.get(url)
    assert response.status_code == status.HTTP_200_OK

def test_user_with_view_environment_permission_can_not_list_identities(environment, identity, test_user_client, view_environment_permission, manage_identities_permission, view_project_permission, user_environment_permission, user_project_permission):
    if False:
        print('Hello World!')
    user_environment_permission.permissions.add(view_environment_permission)
    user_project_permission.permissions.add(view_project_permission)
    url = reverse('api-v1:environments:environment-identities-list', args=(environment.api_key,))
    response = test_user_client.get(url)
    assert response.status_code == status.HTTP_403_FORBIDDEN

def test_identity_view_set_get_permissions():
    if False:
        return 10
    view_set = IdentityViewSet()
    permissions = view_set.get_permissions()
    assert isinstance(permissions[0], IsAuthenticated)
    assert isinstance(permissions[1], NestedEnvironmentPermissions)
    assert permissions[1].action_permission_map == {'list': VIEW_IDENTITIES, 'retrieve': VIEW_IDENTITIES, 'create': MANAGE_IDENTITIES, 'update': MANAGE_IDENTITIES, 'partial_update': MANAGE_IDENTITIES, 'destroy': MANAGE_IDENTITIES}