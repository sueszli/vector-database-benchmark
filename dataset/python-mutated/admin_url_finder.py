from django.contrib.admin.utils import quote
from django.core.exceptions import ImproperlyConfigured
from django.urls import reverse
from wagtail.hooks import search_for_hooks
from wagtail.utils.registry import ObjectTypeRegistry
'\nA mechanism for finding the admin edit URL for an arbitrary object instance, optionally applying\npermission checks.\n\n    url_finder = AdminURLFinder(request.user)\n    url_finder.get_edit_url(some_page)  # => "/admin/pages/123/edit/"\n    url_finder.get_edit_url(some_image)  # => "/admin/images/456/"\n    url_finder.get_edit_url(some_site)  # => None (user does not have edit permission for sites)\n\nIf the user parameter is omitted, edit URLs are returned without considering permissions.\n\nHandlers for new models can be registered via register_admin_url_finder:\n\n    class SprocketAdminURLFinder(ModelAdminURLFinder):\n        edit_url_name = \'wagtailsprockets:edit\'\n\n    register_admin_url_finder(Sprocket, SprocketAdminURLFinder)\n'

class ModelAdminURLFinder:
    """
    Handles admin edit URL lookups for an individual model
    """
    edit_url_name = None
    permission_policy = None

    def __init__(self, user=None):
        if False:
            for i in range(10):
                print('nop')
        self.user = user

    def construct_edit_url(self, instance):
        if False:
            return 10
        '\n        Return the edit URL for the given instance - regardless of whether the user can access it -\n        or None if no edit URL is available.\n        '
        if self.edit_url_name is None:
            raise ImproperlyConfigured('%r must define edit_url_name or override construct_edit_url' % type(self))
        return reverse(self.edit_url_name, args=(quote(instance.pk),))

    def get_edit_url(self, instance):
        if False:
            i = 10
            return i + 15
        '\n        Return the edit URL for the given instance if one exists and the user has permission for it,\n        or None otherwise.\n        '
        if self.user and self.permission_policy and (not self.permission_policy.user_has_permission_for_instance(self.user, 'change', instance)):
            return None
        else:
            return self.construct_edit_url(instance)

class NullAdminURLFinder:
    """
    A dummy AdminURLFinder that always returns None
    """

    def __init__(self, user=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_edit_url(self, instance):
        if False:
            i = 10
            return i + 15
        return None
finder_classes = ObjectTypeRegistry()

def register_admin_url_finder(model, handler):
    if False:
        while True:
            i = 10
    finder_classes.register(model, value=handler)

class AdminURLFinder:
    """
    The 'main' admin URL finder, which searches across all registered models
    """

    def __init__(self, user=None):
        if False:
            while True:
                i = 10
        search_for_hooks()
        self.user = user
        self.finders_by_model = {}

    def get_edit_url(self, instance):
        if False:
            i = 10
            return i + 15
        model = type(instance)
        try:
            finder = self.finders_by_model[model]
        except KeyError:
            finder_class = finder_classes.get(instance) or NullAdminURLFinder
            finder = finder_class(self.user)
            self.finders_by_model[model] = finder
        return finder.get_edit_url(instance)