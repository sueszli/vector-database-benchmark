from copy import deepcopy
from django.contrib import admin
from django.contrib.admin import site
from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin
from django.contrib.sites.models import Site
from django.db import OperationalError
from django.utils.translation import gettext_lazy as _
from cms.admin.forms import GlobalPagePermissionAdminForm, PagePermissionInlineAdminForm, ViewRestrictionInlineAdminForm
from cms.exceptions import NoPermissionsException
from cms.models import GlobalPagePermission, PagePermission
from cms.utils import page_permissions, permissions
from cms.utils.conf import get_cms_setting
from cms.utils.helpers import classproperty
PERMISSION_ADMIN_INLINES = []
user_model = get_user_model()
admin_class = UserAdmin
for (model, admin_instance) in site._registry.items():
    if model == user_model:
        admin_class = admin_instance.__class__

class TabularInline(admin.TabularInline):
    pass

def users_exceed_threshold():
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if the number of users exceeds the configured threshold. Only bother\n    counting the users when using an integer threshold, otherwise return the\n    truthy value of the setting to avoid a potentially expensive DB query.\n    '
    threshold = get_cms_setting('RAW_ID_USERS')
    if threshold is True or threshold is False or (not isinstance(threshold, int)):
        return threshold
    try:
        return get_user_model().objects.count() > threshold
    except OperationalError:
        return False

class PagePermissionInlineAdmin(TabularInline):
    model = PagePermission
    form = PagePermissionInlineAdminForm
    classes = ['collapse', 'collapsed']
    extra = 0
    show_with_view_permissions = False

    def has_change_permission(self, request, obj=None):
        if False:
            return 10
        if not obj:
            return False
        return page_permissions.user_can_change_page_permissions(request.user, page=obj, site=obj.node.site)

    def has_add_permission(self, request, obj=None):
        if False:
            i = 10
            return i + 15
        return self.has_change_permission(request, obj)

    @classproperty
    def raw_id_fields(cls):
        if False:
            while True:
                i = 10
        return ['user'] if users_exceed_threshold() else []

    def get_queryset(self, request):
        if False:
            print('Hello World!')
        "\n        Queryset change, so user with global change permissions can see\n        all permissions. Otherwise user can see only permissions for\n        peoples which are under him (he can't see his permissions, because\n        this will lead to violation, when he can add more power to himself)\n        "
        site = Site.objects.get_current(request)
        try:
            qs = self.model.objects.subordinate_to_user(request.user, site)
        except NoPermissionsException:
            return self.model.objects.none()
        return qs.filter(can_view=self.show_with_view_permissions)

    def get_formset(self, request, obj=None, **kwargs):
        if False:
            print('Hello World!')
        "\n        Some fields may be excluded here. User can change only\n        permissions which are available for them. E.g. if user does not have\n        can_publish flag, they can't change assign can_publish permissions.\n        "
        exclude = self.exclude or []
        if obj:
            user = request.user
            if not obj.has_add_permission(user):
                exclude.append('can_add')
            if not obj.has_delete_permission(user):
                exclude.append('can_delete')
            if not obj.has_publish_permission(user):
                exclude.append('can_publish')
            if not obj.has_advanced_settings_permission(user):
                exclude.append('can_change_advanced_settings')
            if not obj.has_move_page_permission(user):
                exclude.append('can_move_page')
        kwargs['exclude'] = exclude
        formset_cls = super().get_formset(request, obj=obj, **kwargs)
        qs = self.get_queryset(request)
        if obj is not None:
            qs = qs.filter(page=obj)
        formset_cls._queryset = qs
        return formset_cls

class ViewRestrictionInlineAdmin(PagePermissionInlineAdmin):
    extra = 0
    form = ViewRestrictionInlineAdminForm
    verbose_name = _('View restriction')
    verbose_name_plural = _('View restrictions')
    show_with_view_permissions = True

class GlobalPagePermissionAdmin(admin.ModelAdmin):
    list_display = ['user', 'group', 'can_change', 'can_delete', 'can_publish', 'can_change_permissions']
    list_filter = ['sites', 'user', 'group', 'can_change', 'can_delete', 'can_publish', 'can_change_permissions']
    list_display_links = ['user', 'group']
    form = GlobalPagePermissionAdminForm
    search_fields = []
    for field in admin_class.search_fields:
        search_fields.append('user__%s' % field)
    search_fields.append('group__name')
    list_display.append('can_change_advanced_settings')
    list_filter.append('can_change_advanced_settings')

    def get_list_filter(self, request):
        if False:
            while True:
                i = 10
        filter_copy = deepcopy(self.list_filter)
        if users_exceed_threshold():
            filter_copy.remove('user')
        return filter_copy

    def has_add_permission(self, request):
        if False:
            return 10
        site = Site.objects.get_current(request)
        return permissions.user_can_add_global_permissions(request.user, site)

    def has_change_permission(self, request, obj=None):
        if False:
            return 10
        site = Site.objects.get_current(request)
        return permissions.user_can_change_global_permissions(request.user, site)

    def has_delete_permission(self, request, obj=None):
        if False:
            i = 10
            return i + 15
        site = Site.objects.get_current(request)
        return permissions.user_can_delete_global_permissions(request.user, site)

    @classproperty
    def raw_id_fields(cls):
        if False:
            i = 10
            return i + 15
        return ['user'] if users_exceed_threshold() else []
if get_cms_setting('PERMISSION'):
    admin.site.register(GlobalPagePermission, GlobalPagePermissionAdmin)
    PERMISSION_ADMIN_INLINES.extend([ViewRestrictionInlineAdmin, PagePermissionInlineAdmin])