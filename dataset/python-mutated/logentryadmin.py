from django.contrib import admin
from django.contrib.admin.models import DELETION
from django.contrib.contenttypes.models import ContentType
from django.urls import reverse, NoReverseMatch
from django.utils.encoding import force_str
from django.utils.html import escape
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _

class LogEntryAdmin(admin.ModelAdmin):
    list_filter = ['content_type']
    search_fields = ['object_repr', 'change_message']
    list_display_links = ['action_time', 'get_change_message']
    list_display = ['action_time', 'user_link', 'content_type', 'object_link', 'get_change_message']

    def has_add_permission(self, request):
        if False:
            return 10
        return False

    def has_change_permission(self, request, obj=None):
        if False:
            return 10
        return (request.user.is_superuser or request.user.has_perm('admin.change_logentry')) and request.method != 'POST'

    def has_delete_permission(self, request, obj=None):
        if False:
            return 10
        return False

    def object_link(self, obj):
        if False:
            print('Hello World!')
        object_link = escape(obj.object_repr)
        content_type = obj.content_type
        if obj.action_flag != DELETION and content_type is not None:
            try:
                url = reverse('admin:{}_{}_change'.format(content_type.app_label, content_type.model), args=[obj.object_id])
                object_link = '<a href="{}">{}</a>'.format(url, object_link)
            except NoReverseMatch:
                pass
        return mark_safe(object_link)
    object_link.admin_order_field = 'object_repr'
    object_link.short_description = _('object')

    def user_link(self, obj):
        if False:
            while True:
                i = 10
        content_type = ContentType.objects.get_for_model(type(obj.user))
        user_link = escape(force_str(obj.user))
        try:
            url = reverse('admin:{}_{}_change'.format(content_type.app_label, content_type.model), args=[obj.user.pk])
            user_link = '<a href="{}">{}</a>'.format(url, user_link)
        except NoReverseMatch:
            pass
        return mark_safe(user_link)
    user_link.admin_order_field = 'user'
    user_link.short_description = _('user')

    def get_queryset(self, request):
        if False:
            print('Hello World!')
        queryset = super(LogEntryAdmin, self).get_queryset(request)
        return queryset.prefetch_related('content_type')

    def get_actions(self, request):
        if False:
            return 10
        actions = super(LogEntryAdmin, self).get_actions(request)
        if 'delete_selected' in actions:
            del actions['delete_selected']
        return actions