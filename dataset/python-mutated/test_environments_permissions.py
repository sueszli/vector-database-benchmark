from environments.permissions.models import UserEnvironmentPermission
from environments.permissions.permissions import NestedEnvironmentPermissions
from permissions.models import ENVIRONMENT_PERMISSION_TYPE, PermissionModel

def test_nested_environment_permissions_has_permission_false_if_no_env_key(rf, mocker, db):
    if False:
        for i in range(10):
            print('nop')
    permissions = NestedEnvironmentPermissions()
    request = rf.get('/')
    view = mocker.MagicMock(action='retrieve', kwargs={})
    result = permissions.has_permission(request, view)
    assert result is False

def test_nested_environment_permissions_has_permission_true_if_action_in_map(rf, mocker, db, environment, django_user_model):
    if False:
        print('Hello World!')
    permission_key = 'SOME_PERMISSION'
    permission = PermissionModel.objects.create(key=permission_key, type=ENVIRONMENT_PERMISSION_TYPE, description='foobar')
    action = 'retrieve'
    permissions = NestedEnvironmentPermissions(action_permission_map={action: permission.key})
    user = django_user_model.objects.create(email='test@example.com')
    user_env_permission = UserEnvironmentPermission.objects.create(user=user, environment=environment)
    user_env_permission.permissions.add(permission)
    request = rf.get('/')
    request.user = user
    view = mocker.MagicMock(action=action, kwargs={'environment_api_key': environment.api_key})
    has_permission = permissions.has_permission(request, view)
    assert has_permission is True

def test_nested_environment_permissions_has_permission_if_create_and_user_is_admin(rf, mocker, db, environment, django_user_model):
    if False:
        i = 10
        return i + 15
    permissions = NestedEnvironmentPermissions()
    user = django_user_model.objects.create(email='test@example.com')
    UserEnvironmentPermission.objects.create(user=user, environment=environment, admin=True)
    request = rf.get('/')
    request.user = user
    view = mocker.MagicMock(action='create', kwargs={'environment_api_key': environment.api_key})
    has_permission = permissions.has_permission(request, view)
    assert has_permission is True

def test_nested_environment_permissions_has_object_permission_true_if_action_in_map(rf, mocker, django_user_model, environment):
    if False:
        return 10
    permission_key = 'SOME_PERMISSION'
    permission = PermissionModel.objects.create(key=permission_key, type=ENVIRONMENT_PERMISSION_TYPE, description='foobar')
    action = 'retrieve'
    permissions = NestedEnvironmentPermissions(action_permission_map={action: permission.key})
    user = django_user_model.objects.create(email='test@example.com')
    user_env_permission = UserEnvironmentPermission.objects.create(user=user, environment=environment)
    user_env_permission.permissions.add(permission)
    request = rf.get('/')
    request.user = user
    view = mocker.MagicMock(action=action, kwargs={'environment_api_key': environment.api_key})
    obj = mocker.MagicMock(environment=environment)
    has_object_permission = permissions.has_object_permission(request, view, obj)
    assert has_object_permission is True

def test_nested_environment_permissions_has_object_permission_true_if_user_is_admin(rf, mocker, django_user_model, environment):
    if False:
        print('Hello World!')
    permissions = NestedEnvironmentPermissions()
    user = django_user_model.objects.create(email='test@example.com')
    UserEnvironmentPermission.objects.create(user=user, environment=environment, admin=True)
    request = rf.get('/')
    request.user = user
    view = mocker.MagicMock(action='action', kwargs={'environment_api_key': environment.api_key})
    obj = mocker.MagicMock(environment=environment)
    has_object_permission = permissions.has_object_permission(request, view, obj)
    assert has_object_permission is True