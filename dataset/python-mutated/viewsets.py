"""
ViewSets are essentially just a type of class based view, that doesn't provide
any method handlers, such as `get()`, `post()`, etc... but instead has actions,
such as `list()`, `retrieve()`, `create()`, etc...

Actions are only bound to methods at the point of instantiating the views.

    user_list = UserViewSet.as_view({'get': 'list'})
    user_detail = UserViewSet.as_view({'get': 'retrieve'})

Typically, rather than instantiate views from viewsets directly, you'll
register the viewset with a router and let the URL conf be determined
automatically.

    router = DefaultRouter()
    router.register(r'users', UserViewSet, 'user')
    urlpatterns = router.urls
"""
from functools import update_wrapper
from inspect import getmembers
from django.urls import NoReverseMatch
from django.utils.decorators import classonlymethod
from django.views.decorators.csrf import csrf_exempt
from rest_framework import generics, mixins, views
from rest_framework.decorators import MethodMapper
from rest_framework.reverse import reverse

def _is_extra_action(attr):
    if False:
        i = 10
        return i + 15
    return hasattr(attr, 'mapping') and isinstance(attr.mapping, MethodMapper)

def _check_attr_name(func, name):
    if False:
        while True:
            i = 10
    assert func.__name__ == name, 'Expected function (`{func.__name__}`) to match its attribute name (`{name}`). If using a decorator, ensure the inner function is decorated with `functools.wraps`, or that `{func.__name__}.__name__` is otherwise set to `{name}`.'.format(func=func, name=name)
    return func

class ViewSetMixin:
    """
    This is the magic.

    Overrides `.as_view()` so that it takes an `actions` keyword that performs
    the binding of HTTP methods to actions on the Resource.

    For example, to create a concrete view binding the 'GET' and 'POST' methods
    to the 'list' and 'create' actions...

    view = MyViewSet.as_view({'get': 'list', 'post': 'create'})
    """

    @classonlymethod
    def as_view(cls, actions=None, **initkwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Because of the way class based views create a closure around the\n        instantiated view, we need to totally reimplement `.as_view`,\n        and slightly modify the view function that is created and returned.\n        '
        cls.name = None
        cls.description = None
        cls.suffix = None
        cls.detail = None
        cls.basename = None
        if not actions:
            raise TypeError("The `actions` argument must be provided when calling `.as_view()` on a ViewSet. For example `.as_view({'get': 'list'})`")
        for key in initkwargs:
            if key in cls.http_method_names:
                raise TypeError("You tried to pass in the %s method name as a keyword argument to %s(). Don't do that." % (key, cls.__name__))
            if not hasattr(cls, key):
                raise TypeError('%s() received an invalid keyword %r' % (cls.__name__, key))
        if 'name' in initkwargs and 'suffix' in initkwargs:
            raise TypeError('%s() received both `name` and `suffix`, which are mutually exclusive arguments.' % cls.__name__)

        def view(request, *args, **kwargs):
            if False:
                print('Hello World!')
            self = cls(**initkwargs)
            if 'get' in actions and 'head' not in actions:
                actions['head'] = actions['get']
            self.action_map = actions
            for (method, action) in actions.items():
                handler = getattr(self, action)
                setattr(self, method, handler)
            self.request = request
            self.args = args
            self.kwargs = kwargs
            return self.dispatch(request, *args, **kwargs)
        update_wrapper(view, cls, updated=())
        update_wrapper(view, cls.dispatch, assigned=())
        view.cls = cls
        view.initkwargs = initkwargs
        view.actions = actions
        return csrf_exempt(view)

    def initialize_request(self, request, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Set the `.action` attribute on the view, depending on the request method.\n        '
        request = super().initialize_request(request, *args, **kwargs)
        method = request.method.lower()
        if method == 'options':
            self.action = 'metadata'
        else:
            self.action = self.action_map.get(method)
        return request

    def reverse_action(self, url_name, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Reverse the action for the given `url_name`.\n        '
        url_name = '%s-%s' % (self.basename, url_name)
        namespace = None
        if self.request and self.request.resolver_match:
            namespace = self.request.resolver_match.namespace
        if namespace:
            url_name = namespace + ':' + url_name
        kwargs.setdefault('request', self.request)
        return reverse(url_name, *args, **kwargs)

    @classmethod
    def get_extra_actions(cls):
        if False:
            while True:
                i = 10
        '\n        Get the methods that are marked as an extra ViewSet `@action`.\n        '
        return [_check_attr_name(method, name) for (name, method) in getmembers(cls, _is_extra_action)]

    def get_extra_action_url_map(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build a map of {names: urls} for the extra actions.\n\n        This method will noop if `detail` was not provided as a view initkwarg.\n        '
        action_urls = {}
        if self.detail is None:
            return action_urls
        actions = [action for action in self.get_extra_actions() if action.detail == self.detail]
        for action in actions:
            try:
                url_name = '%s-%s' % (self.basename, action.url_name)
                namespace = self.request.resolver_match.namespace
                if namespace:
                    url_name = '%s:%s' % (namespace, url_name)
                url = reverse(url_name, self.args, self.kwargs, request=self.request)
                view = self.__class__(**action.kwargs)
                action_urls[view.get_view_name()] = url
            except NoReverseMatch:
                pass
        return action_urls

class ViewSet(ViewSetMixin, views.APIView):
    """
    The base ViewSet class does not provide any actions by default.
    """
    pass

class GenericViewSet(ViewSetMixin, generics.GenericAPIView):
    """
    The GenericViewSet class does not provide any actions by default,
    but does include the base set of generic view behavior, such as
    the `get_object` and `get_queryset` methods.
    """
    pass

class ReadOnlyModelViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, GenericViewSet):
    """
    A viewset that provides default `list()` and `retrieve()` actions.
    """
    pass

class ModelViewSet(mixins.CreateModelMixin, mixins.RetrieveModelMixin, mixins.UpdateModelMixin, mixins.DestroyModelMixin, mixins.ListModelMixin, GenericViewSet):
    """
    A viewset that provides default `create()`, `retrieve()`, `update()`,
    `partial_update()`, `destroy()` and `list()` actions.
    """
    pass