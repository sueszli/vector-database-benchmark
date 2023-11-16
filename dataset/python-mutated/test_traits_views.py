from django.urls import reverse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from environments.identities.traits.models import Trait
from environments.identities.traits.views import TraitViewSet
from environments.permissions.constants import MANAGE_IDENTITIES, VIEW_ENVIRONMENT, VIEW_IDENTITIES
from environments.permissions.models import UserEnvironmentPermission
from environments.permissions.permissions import NestedEnvironmentPermissions
from permissions.models import PermissionModel
from projects.models import UserProjectPermission
from projects.permissions import VIEW_PROJECT

def test_user_with_manage_identities_permission_can_add_trait_for_identity(environment, identity, django_user_model, api_client):
    if False:
        return 10
    user = django_user_model.objects.create(email='user@example.com')
    api_client.force_authenticate(user)
    view_environment_permission = PermissionModel.objects.get(key=VIEW_ENVIRONMENT)
    manage_identities_permission = PermissionModel.objects.get(key=MANAGE_IDENTITIES)
    view_project_permission = PermissionModel.objects.get(key=VIEW_PROJECT)
    user_environment_permission = UserEnvironmentPermission.objects.create(user=user, environment=environment)
    user_environment_permission.permissions.add(view_environment_permission, manage_identities_permission)
    user_project_permission = UserProjectPermission.objects.create(user=user, project=environment.project)
    user_project_permission.permissions.add(view_project_permission)
    url = reverse('api-v1:environments:identities-traits-list', args=(environment.api_key, identity.id))
    response = api_client.post(url, data={'trait_key': 'foo', 'value_type': 'unicode', 'string_value': 'foo'})
    assert response.status_code == status.HTTP_201_CREATED

def test_trait_view_delete_trait(environment, admin_client, identity, trait, mocker):
    if False:
        return 10
    url = reverse('api-v1:environments:identities-traits-detail', args=(environment.api_key, identity.id, trait.id))
    res = admin_client.delete(url)
    assert res.status_code == status.HTTP_204_NO_CONTENT
    assert not Trait.objects.filter(pk=trait.id).exists()

def test_trait_view_set_update(environment, admin_client, identity, trait, mocker):
    if False:
        for i in range(10):
            print('nop')
    url = reverse('api-v1:environments:identities-traits-detail', args=(environment.api_key, identity.id, trait.id))
    new_value = 'updated'
    response = admin_client.patch(url, data={'string_value': new_value})
    assert response.status_code == status.HTTP_200_OK
    assert response.json()['string_value'] == new_value

def test_edge_identity_view_set_get_permissions():
    if False:
        while True:
            i = 10
    view_set = TraitViewSet()
    permissions = view_set.get_permissions()
    assert isinstance(permissions[0], IsAuthenticated)
    assert isinstance(permissions[1], NestedEnvironmentPermissions)
    assert permissions[1].action_permission_map == {'list': VIEW_IDENTITIES, 'retrieve': VIEW_IDENTITIES, 'create': MANAGE_IDENTITIES, 'update': MANAGE_IDENTITIES, 'partial_update': MANAGE_IDENTITIES, 'destroy': MANAGE_IDENTITIES}