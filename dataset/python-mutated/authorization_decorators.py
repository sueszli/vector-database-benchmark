import functools
from django.core.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404
from dojo.authorization.authorization import user_has_global_permission_or_403, user_has_permission_or_403, user_has_configuration_permission

def user_is_authorized(model, permission, arg, lookup='pk', func=None):
    if False:
        while True:
            i = 10
    'Decorator for functions that ensures the user has permission on an object.'
    if func is None:
        return functools.partial(user_is_authorized, model, permission, arg, lookup)

    @functools.wraps(func)
    def _wrapped(request, *args, **kwargs):
        if False:
            print('Hello World!')
        if isinstance(arg, int):
            args = list(args)
            lookup_value = args[arg]
        else:
            lookup_value = kwargs.get(arg)
        obj = get_object_or_404(model.objects.filter(**{lookup: lookup_value}))
        user_has_permission_or_403(request.user, obj, permission)
        return func(request, *args, **kwargs)
    return _wrapped

def user_has_global_permission(permission, func=None):
    if False:
        for i in range(10):
            print('nop')
    'Decorator for functions that ensures the user has a (global) permission'
    if func is None:
        return functools.partial(user_has_global_permission, permission)

    @functools.wraps(func)
    def _wrapped(request, *args, **kwargs):
        if False:
            print('Hello World!')
        user_has_global_permission_or_403(request.user, permission)
        return func(request, *args, **kwargs)
    return _wrapped

def user_is_configuration_authorized(permission, func=None):
    if False:
        print('Hello World!')
    '\n    Decorator for views that checks whether a user has a particular permission enabled.\n    '
    if func is None:
        return functools.partial(user_is_configuration_authorized, permission)

    @functools.wraps(func)
    def _wrapped(request, *args, **kwargs):
        if False:
            print('Hello World!')
        if not user_has_configuration_permission(request.user, permission):
            raise PermissionDenied
        return func(request, *args, **kwargs)
    return _wrapped