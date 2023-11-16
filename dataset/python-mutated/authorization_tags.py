from django import template
import crum
from dojo.authorization.roles_permissions import Permissions
from dojo.authorization.authorization import user_has_global_permission, user_has_permission, user_has_configuration_permission as configuration_permission
from dojo.request_cache import cache_for_request
register = template.Library()

@register.filter
def has_object_permission(obj, permission):
    if False:
        return 10
    return user_has_permission(crum.get_current_user(), obj, Permissions[permission])

@register.filter
def has_global_permission(permission):
    if False:
        print('Hello World!')
    return user_has_global_permission(crum.get_current_user(), Permissions[permission])

@register.filter
def has_configuration_permission(permission, request):
    if False:
        print('Hello World!')
    if request is None:
        user = crum.get_current_user()
    else:
        user = crum.get_current_user() or request.user
    return configuration_permission(user, permission)

@cache_for_request
def get_user_permissions(user):
    if False:
        for i in range(10):
            print('nop')
    return user.user_permissions.all()

@register.filter
def user_has_configuration_permission_without_group(user, codename):
    if False:
        i = 10
        return i + 15
    permissions = get_user_permissions(user)
    for permission in permissions:
        if permission.codename == codename:
            return True
    return False

@cache_for_request
def get_group_permissions(group):
    if False:
        print('Hello World!')
    return group.permissions.all()

@register.filter
def group_has_configuration_permission(group, codename):
    if False:
        i = 10
        return i + 15
    for permission in get_group_permissions(group):
        if permission.codename == codename:
            return True
    return False

@register.simple_tag
def user_can_clear_peer_review(finding, user):
    if False:
        while True:
            i = 10
    finding_under_review = finding.under_review
    user_requesting_review = user == finding.review_requested_by
    user_is_reviewer = user in finding.reviewers.all()
    return finding_under_review and (user_requesting_review or user_is_reviewer)