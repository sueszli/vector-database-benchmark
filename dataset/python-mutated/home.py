import itertools
import re
from typing import Any, Mapping, Union
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import permission_required
from django.db import connection
from django.db.models import Exists, IntegerField, Max, OuterRef, Q
from django.db.models.functions import Cast
from django.forms import Media
from django.http import Http404, HttpResponse
from django.template.loader import render_to_string
from django.utils.translation import gettext_lazy
from django.views.generic.base import TemplateView
from wagtail import hooks
from wagtail.admin.navigation import get_site_for_user
from wagtail.admin.site_summary import SiteSummaryPanel
from wagtail.admin.ui.components import Component
from wagtail.admin.views.generic import WagtailAdminTemplateMixin
from wagtail.models import Page, Revision, TaskState, WorkflowState, get_default_page_content_type
from wagtail.permission_policies.pages import PagePermissionPolicy
User = get_user_model()

class UpgradeNotificationPanel(Component):
    name = 'upgrade_notification'
    template_name = 'wagtailadmin/home/upgrade_notification.html'
    order = 100

    def get_upgrade_check_setting(self) -> Union[bool, str]:
        if False:
            for i in range(10):
                print('nop')
        return getattr(settings, 'WAGTAIL_ENABLE_UPDATE_CHECK', True)

    def upgrade_check_lts_only(self) -> bool:
        if False:
            print('Hello World!')
        upgrade_check = self.get_upgrade_check_setting()
        if isinstance(upgrade_check, str) and upgrade_check.lower() == 'lts':
            return True
        return False

    def get_context_data(self, parent_context: Mapping[str, Any]) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return {'lts_only': self.upgrade_check_lts_only()}

    def render_html(self, parent_context: Mapping[str, Any]=None) -> str:
        if False:
            return 10
        if parent_context['request'].user.is_superuser and self.get_upgrade_check_setting():
            return super().render_html(parent_context)
        else:
            return ''

class WhatsNewInWagtailVersionPanel(Component):
    name = 'whats_new_in_wagtail_version'
    template_name = 'wagtailadmin/home/whats_new_in_wagtail_version.html'
    order = 110
    _version = '4'

    def get_whats_new_banner_setting(self) -> Union[bool, str]:
        if False:
            return 10
        return getattr(settings, 'WAGTAIL_ENABLE_WHATS_NEW_BANNER', True)

    def get_dismissible_id(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.name}_{self._version}'

    def get_context_data(self, parent_context: Mapping[str, Any]) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return {'dismissible_id': self.get_dismissible_id(), 'version': self._version}

    def is_shown(self, parent_context: Mapping[str, Any]=None) -> bool:
        if False:
            while True:
                i = 10
        if not self.get_whats_new_banner_setting():
            return False
        profile = getattr(parent_context['request'].user, 'wagtail_userprofile', None)
        if profile and profile.dismissibles.get(self.get_dismissible_id()):
            return False
        return True

    def render_html(self, parent_context: Mapping[str, Any]=None) -> str:
        if False:
            while True:
                i = 10
        if not self.is_shown(parent_context):
            return ''
        return super().render_html(parent_context)

class UserObjectsInWorkflowModerationPanel(Component):
    name = 'user_objects_in_workflow_moderation'
    template_name = 'wagtailadmin/home/user_objects_in_workflow_moderation.html'
    order = 210

    def get_context_data(self, parent_context):
        if False:
            while True:
                i = 10
        request = parent_context['request']
        context = super().get_context_data(parent_context)
        if getattr(settings, 'WAGTAIL_WORKFLOW_ENABLED', True):
            pages_owned_by_user = Q(base_content_type_id=get_default_page_content_type().id) & Exists(Page.objects.filter(owner=request.user, id=Cast(OuterRef('object_id'), output_field=IntegerField())))
            context['workflow_states'] = WorkflowState.objects.active().filter(pages_owned_by_user | Q(requested_by=request.user)).prefetch_related('content_object', 'content_object__latest_revision').select_related('current_task_state', 'current_task_state__task').order_by('-current_task_state__started_at')
        else:
            context['workflow_states'] = WorkflowState.objects.none()
        context['request'] = request
        return context

class WorkflowObjectsToModeratePanel(Component):
    name = 'workflow_objects_to_moderate'
    template_name = 'wagtailadmin/home/workflow_objects_to_moderate.html'
    order = 220

    def get_context_data(self, parent_context):
        if False:
            print('Hello World!')
        request = parent_context['request']
        context = super().get_context_data(parent_context)
        context['states'] = []
        context['request'] = request
        context['csrf_token'] = parent_context['csrf_token']
        if not getattr(settings, 'WAGTAIL_WORKFLOW_ENABLED', True):
            return context
        states = TaskState.objects.reviewable_by(request.user).select_related('revision', 'task', 'revision__user').prefetch_related('revision__content_object', 'revision__content_object__latest_revision', 'revision__content_object__live_revision').order_by('-started_at')
        for state in states:
            obj = state.revision.content_object
            actions = state.task.specific.get_actions(obj, request.user)
            workflow_tasks = state.workflow_state.all_tasks_with_status()
            workflow_action_url_name = 'wagtailadmin_pages:workflow_action'
            workflow_preview_url_name = 'wagtailadmin_pages:workflow_preview'
            revisions_compare_url_name = 'wagtailadmin_pages:revisions_compare'
            if not isinstance(obj, Page):
                viewset = obj.snippet_viewset
                workflow_action_url_name = viewset.get_url_name('workflow_action')
                workflow_preview_url_name = viewset.get_url_name('workflow_preview')
                revisions_compare_url_name = viewset.get_url_name('revisions_compare')
            if not getattr(obj, 'is_previewable', False):
                workflow_preview_url_name = None
            try:
                previous_revision = state.revision.get_previous()
            except Revision.DoesNotExist:
                previous_revision = None
            context['states'].append({'obj': obj, 'revision': state.revision, 'previous_revision': previous_revision, 'live_revision': obj.live_revision, 'task_state': state, 'actions': actions, 'workflow_tasks': workflow_tasks, 'workflow_action_url_name': workflow_action_url_name, 'workflow_preview_url_name': workflow_preview_url_name, 'revisions_compare_url_name': revisions_compare_url_name})
        return context

class LockedPagesPanel(Component):
    name = 'locked_pages'
    template_name = 'wagtailadmin/home/locked_pages.html'
    order = 300

    def get_context_data(self, parent_context):
        if False:
            while True:
                i = 10
        request = parent_context['request']
        context = super().get_context_data(parent_context)
        context.update({'locked_pages': Page.objects.filter(locked=True, locked_by=request.user), 'can_remove_locks': PagePermissionPolicy().user_has_permission(request.user, 'unlock'), 'request': request, 'csrf_token': parent_context['csrf_token']})
        return context

class RecentEditsPanel(Component):
    name = 'recent_edits'
    template_name = 'wagtailadmin/home/recent_edits.html'
    order = 250

    def get_context_data(self, parent_context):
        if False:
            return 10
        request = parent_context['request']
        context = super().get_context_data(parent_context)
        edit_count = getattr(settings, 'WAGTAILADMIN_RECENT_EDITS_LIMIT', 5)
        if connection.vendor == 'mysql':
            last_edits = Revision.objects.raw('\n                SELECT wr.* FROM\n                    wagtailcore_revision wr JOIN (\n                        SELECT max(created_at) AS max_created_at, object_id FROM\n                            wagtailcore_revision WHERE user_id = %s AND base_content_type_id = %s GROUP BY object_id ORDER BY max_created_at DESC LIMIT %s\n                    ) AS max_rev ON max_rev.max_created_at = wr.created_at ORDER BY wr.created_at DESC\n                 ', [User._meta.pk.get_db_prep_value(request.user.pk, connection), get_default_page_content_type().id, edit_count])
        else:
            last_edits_dates = Revision.page_revisions.filter(user=request.user).values('object_id').annotate(latest_date=Max('created_at')).order_by('-latest_date').values('latest_date')[:edit_count]
            last_edits = Revision.page_revisions.filter(created_at__in=last_edits_dates).order_by('-created_at')
        page_keys = [int(pr.object_id) for pr in last_edits]
        pages = Page.objects.specific().in_bulk(page_keys)
        context['last_edits'] = []
        for revision in last_edits:
            page = pages.get(int(revision.object_id))
            if page:
                context['last_edits'].append([revision, page])
        context['request'] = request
        return context

class HomeView(WagtailAdminTemplateMixin, TemplateView):
    template_name = 'wagtailadmin/home.html'
    page_title = gettext_lazy('Dashboard')

    def get_context_data(self, **kwargs):
        if False:
            i = 10
            return i + 15
        context = super().get_context_data(**kwargs)
        panels = self.get_panels()
        site_details = self.get_site_details()
        context['media'] = self.get_media(panels)
        context['panels'] = sorted(panels, key=lambda p: p.order)
        context['user'] = self.request.user
        return {**context, **site_details}

    def get_media(self, panels=[]):
        if False:
            return 10
        media = Media()
        for panel in panels:
            media += panel.media
        return media

    def get_panels(self):
        if False:
            for i in range(10):
                print('nop')
        request = self.request
        panels = [SiteSummaryPanel(request), UpgradeNotificationPanel(), WorkflowObjectsToModeratePanel(), UserObjectsInWorkflowModerationPanel(), RecentEditsPanel(), LockedPagesPanel()]
        for fn in hooks.get_hooks('construct_homepage_panels'):
            fn(request, panels)
        return panels

    def get_site_details(self):
        if False:
            return 10
        request = self.request
        site = get_site_for_user(request.user)
        return {'root_page': site['root_page'], 'root_site': site['root_site'], 'site_name': site['site_name']}

def error_test(request):
    if False:
        for i in range(10):
            print('nop')
    raise Exception('This is a test of the emergency broadcast system.')

@permission_required('wagtailadmin.access_admin', login_url='wagtailadmin_login')
def default(request):
    if False:
        i = 10
        return i + 15
    "\n    Called whenever a request comes in with the correct prefix (eg /admin/) but\n    doesn't actually correspond to a Wagtail view.\n\n    For authenticated users, it'll raise a 404 error. Anonymous users will be\n    redirected to the login page.\n    "
    raise Http404
icon_comment_pattern = re.compile('<!--.*?-->')
_icons_html = None

def icons():
    if False:
        print('Hello World!')
    global _icons_html
    if _icons_html is None:
        icon_hooks = hooks.get_hooks('register_icons')
        all_icons = sorted(itertools.chain.from_iterable((hook([]) for hook in icon_hooks)))
        combined_icon_markup = ''
        for icon in all_icons:
            symbol = render_to_string(icon).replace('xmlns="http://www.w3.org/2000/svg"', '').replace('svg', 'symbol')
            symbol = icon_comment_pattern.sub('', symbol)
            combined_icon_markup += symbol
        _icons_html = render_to_string('wagtailadmin/shared/icons.html', {'icons': combined_icon_markup})
    return _icons_html

def sprite(request):
    if False:
        i = 10
        return i + 15
    return HttpResponse(icons())