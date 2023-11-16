import pytest
from rest_framework.exceptions import APIException, PermissionDenied
from projects.permissions import IsProjectAdmin

def test_is_project_admin_has_permission_raises_permission_denied_if_not_found(mocker, admin_user):
    if False:
        print('Hello World!')
    request = mocker.MagicMock(user=admin_user)
    view = mocker.MagicMock(kwargs={'project_pk': 1})
    with pytest.raises(PermissionDenied):
        IsProjectAdmin().has_permission(request, view)

def test_is_project_admin_has_permission_raises_api_exception_if_no_kwarg(mocker, admin_user):
    if False:
        for i in range(10):
            print('nop')
    request = mocker.MagicMock(user=admin_user)
    view = mocker.MagicMock(kwargs={'foo': 'bar'})
    with pytest.raises(APIException):
        IsProjectAdmin().has_permission(request, view)

def test_is_project_admin_has_permission_returns_true_if_project_admin(mocker, admin_user, organisation, project):
    if False:
        return 10
    assert admin_user.is_project_admin(project)
    request = mocker.MagicMock(user=admin_user)
    view = mocker.MagicMock(kwargs={'project_pk': project.id})
    assert IsProjectAdmin().has_permission(request, view)