import uuid
import warnings
from urllib.parse import parse_qsl, urlparse
from django.contrib.admin.helpers import AdminForm
from django.contrib.admin.utils import get_deleted_objects
from django.core.exceptions import PermissionDenied
from django.db import router, transaction
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden, HttpResponseNotFound, HttpResponseRedirect
from django.shortcuts import get_list_or_404, get_object_or_404, render
from django.template.response import TemplateResponse
from django.urls import re_path
from django.utils import translation
from django.utils.decorators import method_decorator
from django.utils.encoding import force_str
from django.utils.html import conditional_escape
from django.utils.translation import gettext as _
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.views.decorators.http import require_POST
from cms import operations
from cms.admin.forms import PluginAddValidationForm
from cms.constants import SLUG_REGEXP
from cms.exceptions import PluginLimitReached
from cms.models.placeholdermodel import Placeholder
from cms.models.placeholderpluginmodel import PlaceholderReference
from cms.models.pluginmodel import CMSPlugin
from cms.plugin_pool import plugin_pool
from cms.signals import post_placeholder_operation, pre_placeholder_operation
from cms.toolbar.utils import get_plugin_tree_as_json
from cms.utils import copy_plugins, get_current_site
from cms.utils.conf import get_cms_setting
from cms.utils.i18n import get_language_code, get_language_list
from cms.utils.plugins import has_reached_plugin_limit, reorder_plugins
from cms.utils.urlutils import admin_reverse
_no_default = object()

def get_int(int_str, default=_no_default):
    if False:
        return 10
    '\n    For convenience a get-like method for taking the int() of a string.\n    :param int_str: the string to convert to integer\n    :param default: an optional value to return if ValueError is raised.\n    :return: the int() of «int_str» or «default» on exception.\n    '
    if default == _no_default:
        return int(int_str)
    else:
        try:
            return int(int_str)
        except ValueError:
            return default

def _instance_overrides_method(base, instance, method_name):
    if False:
        return 10
    '\n    Returns True if instance overrides a method (method_name)\n    inherited from base.\n    '
    bound_method = getattr(instance.__class__, method_name)
    unbound_method = getattr(base, method_name)
    return unbound_method != bound_method

class FrontendEditableAdminMixin:
    frontend_editable_fields = []

    def get_urls(self):
        if False:
            while True:
                i = 10
        '\n        Register the url for the single field edit view\n        '
        info = f'{self.model._meta.app_label}_{self.model._meta.model_name}'

        def pat(regex, fn):
            if False:
                print('Hello World!')
            return re_path(regex, self.admin_site.admin_view(fn), name=f'{info}_{fn.__name__}')
        url_patterns = [pat('edit-field/(%s)/([a-z\\-]+)/$' % SLUG_REGEXP, self.edit_field)]
        return url_patterns + super().get_urls()

    def _get_object_for_single_field(self, object_id, language):
        if False:
            i = 10
            return i + 15
        try:
            return self.model.objects.language(language).get(pk=object_id)
        except AttributeError:
            return self.model.objects.get(pk=object_id)

    def edit_field(self, request, object_id, language):
        if False:
            return 10
        obj = self._get_object_for_single_field(object_id, language)
        opts = obj.__class__._meta
        saved_successfully = False
        cancel_clicked = request.POST.get('_cancel', False)
        raw_fields = request.GET.get('edit_fields')
        fields = [field for field in raw_fields.split(',') if field in self.frontend_editable_fields]
        if not fields:
            context = {'opts': opts, 'message': force_str(_('Field %s not found')) % raw_fields}
            return render(request, 'admin/cms/page/plugin/error_form.html', context)
        if not request.user.has_perm('{}.change_{}'.format(self.model._meta.app_label, self.model._meta.model_name)):
            context = {'opts': opts, 'message': force_str(_('You do not have permission to edit this item'))}
            return render(request, 'admin/cms/page/plugin/error_form.html', context)
        form_class = self.get_form(request, obj, fields=fields)
        if not cancel_clicked and request.method == 'POST':
            form = form_class(instance=obj, data=request.POST)
            if form.is_valid():
                form.save()
                saved_successfully = True
        else:
            form = form_class(instance=obj)
        admin_form = AdminForm(form, fieldsets=[(None, {'fields': fields})], prepopulated_fields={}, model_admin=self)
        media = self.media + admin_form.media
        context = {'CMS_MEDIA_URL': get_cms_setting('MEDIA_URL'), 'title': opts.verbose_name, 'plugin': None, 'plugin_id': None, 'adminform': admin_form, 'add': False, 'is_popup': True, 'media': media, 'opts': opts, 'change': True, 'save_as': False, 'has_add_permission': False, 'window_close_timeout': 10}
        if cancel_clicked:
            context.update({'cancel': True})
            return render(request, 'admin/cms/page/plugin/confirm_form.html', context)
        if not cancel_clicked and request.method == 'POST' and saved_successfully:
            return render(request, 'admin/cms/page/plugin/confirm_form.html', context)
        return render(request, 'admin/cms/page/plugin/change_form.html', context)

class PlaceholderAdminMixin:

    def _get_attached_admin(self, placeholder):
        if False:
            return 10
        return placeholder._get_attached_admin(admin_site=self.admin_site)

    def _get_operation_language(self, request):
        if False:
            i = 10
            return i + 15
        site = get_current_site()
        parsed_url = urlparse(request.GET['cms_path'])
        queries = dict(parse_qsl(parsed_url.query))
        language = queries.get('language')
        if not language:
            language = translation.get_language_from_path(parsed_url.path)
        return get_language_code(language, site_id=site.pk)

    def _get_operation_origin(self, request):
        if False:
            print('Hello World!')
        return urlparse(request.GET['cms_path']).path

    def _send_pre_placeholder_operation(self, request, operation, **kwargs):
        if False:
            i = 10
            return i + 15
        token = str(uuid.uuid4())
        if not request.GET.get('cms_path'):
            warnings.warn('All custom placeholder admin endpoints require a "cms_path" GET query which points to the path where the request originates from.This backwards compatible shim will be removed on 3.5 and an HttpBadRequest response will be returned instead.', UserWarning)
            return token
        pre_placeholder_operation.send(sender=self.__class__, operation=operation, request=request, language=self._get_operation_language(request), token=token, origin=self._get_operation_origin(request), **kwargs)
        return token

    def _send_post_placeholder_operation(self, request, operation, token, **kwargs):
        if False:
            i = 10
            return i + 15
        if not request.GET.get('cms_path'):
            return
        post_placeholder_operation.send(sender=self.__class__, operation=operation, request=request, language=self._get_operation_language(request), token=token, origin=self._get_operation_origin(request), **kwargs)

    def _get_plugin_from_id(self, plugin_id):
        if False:
            return 10
        queryset = CMSPlugin.objects.values_list('plugin_type', flat=True)
        plugin_type = get_list_or_404(queryset, pk=plugin_id)[0]
        plugin_class = plugin_pool.get_plugin(plugin_type)
        real_queryset = plugin_class.get_render_queryset().select_related('parent', 'placeholder')
        return get_object_or_404(real_queryset, pk=plugin_id)

    def get_urls(self):
        if False:
            print('Hello World!')
        '\n        Register the plugin specific urls (add/edit/copy/remove/move)\n        '
        info = f'{self.model._meta.app_label}_{self.model._meta.model_name}'

        def pat(regex, fn):
            if False:
                i = 10
                return i + 15
            return re_path(regex, self.admin_site.admin_view(fn), name=f'{info}_{fn.__name__}')
        url_patterns = [pat('copy-plugins/$', self.copy_plugins), pat('add-plugin/$', self.add_plugin), pat('edit-plugin/(%s)/$' % SLUG_REGEXP, self.edit_plugin), pat('delete-plugin/(%s)/$' % SLUG_REGEXP, self.delete_plugin), pat('clear-placeholder/(%s)/$' % SLUG_REGEXP, self.clear_placeholder), pat('move-plugin/$', self.move_plugin)]
        return url_patterns + super().get_urls()

    def has_add_plugin_permission(self, request, placeholder, plugin_type):
        if False:
            return 10
        return placeholder.has_add_plugin_permission(request.user, plugin_type)

    def has_change_plugin_permission(self, request, plugin):
        if False:
            i = 10
            return i + 15
        placeholder = plugin.placeholder
        return placeholder.has_change_plugin_permission(request.user, plugin)

    def has_delete_plugin_permission(self, request, plugin):
        if False:
            while True:
                i = 10
        placeholder = plugin.placeholder
        return placeholder.has_delete_plugin_permission(request.user, plugin)

    def has_copy_plugins_permission(self, request, plugins):
        if False:
            print('Hello World!')
        placeholder = request.toolbar.clipboard
        return placeholder.has_add_plugins_permission(request.user, plugins)

    def has_copy_from_clipboard_permission(self, request, placeholder, plugins):
        if False:
            print('Hello World!')
        return placeholder.has_add_plugins_permission(request.user, plugins)

    def has_copy_from_placeholder_permission(self, request, source_placeholder, target_placeholder, plugins):
        if False:
            return 10
        if not source_placeholder.has_add_plugins_permission(request.user, plugins):
            return False
        return target_placeholder.has_add_plugins_permission(request.user, plugins)

    def has_move_plugin_permission(self, request, plugin, target_placeholder):
        if False:
            i = 10
            return i + 15
        placeholder = plugin.placeholder
        return placeholder.has_move_plugin_permission(request.user, plugin, target_placeholder)

    def has_clear_placeholder_permission(self, request, placeholder, language=None):
        if False:
            print('Hello World!')
        if language:
            languages = [language]
        else:
            languages = placeholder.cmsplugin_set.values_list('language', flat=True).distinct().order_by()
        return placeholder.has_clear_permission(request.user, languages)

    def get_placeholder_template(self, request, placeholder):
        if False:
            return 10
        pass

    @xframe_options_sameorigin
    def add_plugin(self, request):
        if False:
            for i in range(10):
                print('nop')
        '\n        Shows the add plugin form and saves it on POST.\n\n        Requires the following GET parameters:\n            - cms_path\n            - placeholder_id\n            - plugin_type\n            - plugin_language\n            - plugin_parent (optional)\n            - plugin_position (optional)\n        '
        form = PluginAddValidationForm(request.GET)
        if not form.is_valid():
            error = list(form.errors.values())[0][0]
            return HttpResponseBadRequest(conditional_escape(force_str(error)))
        plugin_data = form.cleaned_data
        placeholder = plugin_data['placeholder_id']
        plugin_type = plugin_data['plugin_type']
        if not self.has_add_plugin_permission(request, placeholder, plugin_type):
            message = force_str(_('You do not have permission to add a plugin'))
            return HttpResponseForbidden(message)
        parent = plugin_data.get('plugin_parent')
        if parent:
            position = parent.cmsplugin_set.count()
        else:
            position = CMSPlugin.objects.filter(parent__isnull=True, language=plugin_data['plugin_language'], placeholder=placeholder).count()
        plugin_data['position'] = position
        plugin_class = plugin_pool.get_plugin(plugin_type)
        plugin_instance = plugin_class(plugin_class.model, self.admin_site)
        plugin_instance._cms_initial_attributes = {'language': plugin_data['plugin_language'], 'placeholder': plugin_data['placeholder_id'], 'parent': plugin_data.get('plugin_parent', None), 'plugin_type': plugin_data['plugin_type'], 'position': plugin_data['position']}
        response = plugin_instance.add_view(request)
        plugin = getattr(plugin_instance, 'saved_object', None)
        if plugin:
            plugin.placeholder.mark_as_dirty(plugin.language, clear_cache=False)
        if plugin_instance._operation_token:
            tree_order = placeholder.get_plugin_tree_order(plugin.parent_id)
            self._send_post_placeholder_operation(request, operation=operations.ADD_PLUGIN, token=plugin_instance._operation_token, plugin=plugin, placeholder=plugin.placeholder, tree_order=tree_order)
        return response

    @method_decorator(require_POST)
    @xframe_options_sameorigin
    @transaction.atomic
    def copy_plugins(self, request):
        if False:
            while True:
                i = 10
        '\n        POST request should have the following data:\n\n        - cms_path\n        - source_language\n        - source_placeholder_id\n        - source_plugin_id (optional)\n        - target_language\n        - target_placeholder_id\n        - target_plugin_id (deprecated/unused)\n        '
        source_placeholder_id = request.POST['source_placeholder_id']
        target_language = request.POST['target_language']
        target_placeholder_id = request.POST['target_placeholder_id']
        source_placeholder = get_object_or_404(Placeholder, pk=source_placeholder_id)
        target_placeholder = get_object_or_404(Placeholder, pk=target_placeholder_id)
        if not target_language or target_language not in get_language_list():
            return HttpResponseBadRequest(force_str(_('Language must be set to a supported language!')))
        copy_to_clipboard = target_placeholder.pk == request.toolbar.clipboard.pk
        source_plugin_id = request.POST.get('source_plugin_id', None)
        if copy_to_clipboard and source_plugin_id:
            new_plugin = self._copy_plugin_to_clipboard(request, source_placeholder, target_placeholder)
            new_plugins = [new_plugin]
        elif copy_to_clipboard:
            new_plugin = self._copy_placeholder_to_clipboard(request, source_placeholder, target_placeholder)
            new_plugins = [new_plugin]
        else:
            new_plugins = self._add_plugins_from_placeholder(request, source_placeholder, target_placeholder)
        data = get_plugin_tree_as_json(request, new_plugins)
        return HttpResponse(data, content_type='application/json')

    def _copy_plugin_to_clipboard(self, request, source_placeholder, target_placeholder):
        if False:
            while True:
                i = 10
        source_language = request.POST['source_language']
        source_plugin_id = request.POST.get('source_plugin_id')
        target_language = request.POST['target_language']
        source_plugin = get_object_or_404(CMSPlugin, pk=source_plugin_id, language=source_language)
        old_plugins = CMSPlugin.get_tree(parent=source_plugin).filter(placeholder=source_placeholder).order_by('path')
        if not self.has_copy_plugins_permission(request, old_plugins):
            message = _('You do not have permission to copy these plugins.')
            raise PermissionDenied(force_str(message))
        target_placeholder.clear()
        plugin_pairs = copy_plugins.copy_plugins_to(old_plugins, to_placeholder=target_placeholder, to_language=target_language)
        return plugin_pairs[0][0]

    def _copy_placeholder_to_clipboard(self, request, source_placeholder, target_placeholder):
        if False:
            for i in range(10):
                print('nop')
        source_language = request.POST['source_language']
        target_language = request.POST['target_language']
        old_plugins = source_placeholder.get_plugins_list(language=source_language)
        if not self.has_copy_plugins_permission(request, old_plugins):
            message = _('You do not have permission to copy this placeholder.')
            raise PermissionDenied(force_str(message))
        target_placeholder.clear()
        reference = PlaceholderReference.objects.create(name=source_placeholder.get_label(), plugin_type='PlaceholderPlugin', language=target_language, placeholder=target_placeholder)
        copy_plugins.copy_plugins_to(old_plugins, to_placeholder=reference.placeholder_ref, to_language=target_language)
        return reference

    def _add_plugins_from_placeholder(self, request, source_placeholder, target_placeholder):
        if False:
            for i in range(10):
                print('nop')
        source_language = request.POST['source_language']
        target_language = request.POST['target_language']
        old_plugins = source_placeholder.get_plugins_list(language=source_language)
        has_permissions = self.has_copy_from_placeholder_permission(request, source_placeholder, target_placeholder, old_plugins)
        if not has_permissions:
            message = _('You do not have permission to copy these plugins.')
            raise PermissionDenied(force_str(message))
        target_tree_order = target_placeholder.get_plugin_tree_order(language=target_language, parent_id=None)
        operation_token = self._send_pre_placeholder_operation(request, operation=operations.ADD_PLUGINS_FROM_PLACEHOLDER, plugins=old_plugins, source_language=source_language, source_placeholder=source_placeholder, target_language=target_language, target_placeholder=target_placeholder, target_order=target_tree_order)
        copied_plugins = copy_plugins.copy_plugins_to(old_plugins, to_placeholder=target_placeholder, to_language=target_language)
        new_plugin_ids = (new.pk for (new, old) in copied_plugins)
        top_plugins = (pair for pair in copied_plugins if not pair[0].parent_id)
        top_plugins_pks = [p[0].pk for p in sorted(top_plugins, key=lambda pair: pair[1].position)]
        target_tree_order = target_tree_order + top_plugins_pks
        reorder_plugins(target_placeholder, parent_id=None, language=target_language, order=target_tree_order)
        target_placeholder.mark_as_dirty(target_language, clear_cache=False)
        new_plugins = CMSPlugin.objects.filter(pk__in=new_plugin_ids).order_by('path')
        new_plugins = list(new_plugins)
        self._send_post_placeholder_operation(request, operation=operations.ADD_PLUGINS_FROM_PLACEHOLDER, token=operation_token, plugins=new_plugins, source_language=source_language, source_placeholder=source_placeholder, target_language=target_language, target_placeholder=target_placeholder, target_order=target_tree_order)
        return new_plugins

    @xframe_options_sameorigin
    def edit_plugin(self, request, plugin_id):
        if False:
            return 10
        try:
            plugin_id = int(plugin_id)
        except ValueError:
            return HttpResponseNotFound(force_str(_('Plugin not found')))
        obj = self._get_plugin_from_id(plugin_id)
        plugin_instance = obj.get_plugin_class_instance(admin=self.admin_site)
        if not self.has_change_plugin_permission(request, obj):
            return HttpResponseForbidden(force_str(_('You do not have permission to edit this plugin')))
        response = plugin_instance.change_view(request, str(plugin_id))
        plugin = getattr(plugin_instance, 'saved_object', None)
        if plugin:
            plugin.placeholder.mark_as_dirty(plugin.language, clear_cache=False)
        if plugin_instance._operation_token:
            self._send_post_placeholder_operation(request, operation=operations.CHANGE_PLUGIN, token=plugin_instance._operation_token, old_plugin=obj, new_plugin=plugin, placeholder=plugin.placeholder)
        return response

    @method_decorator(require_POST)
    @xframe_options_sameorigin
    @transaction.atomic
    def move_plugin(self, request):
        if False:
            return 10
        '\n        Performs a move or a "paste" operation (when «move_a_copy» is set)\n\n        POST request with following parameters:\n        - plugin_id\n        - placeholder_id\n        - plugin_language (optional)\n        - plugin_parent (optional)\n        - plugin_order (array, optional)\n        - move_a_copy (Boolean, optional) (anything supplied here except a case-\n                                        insensitive "false" is True)\n        NOTE: If move_a_copy is set, the plugin_order should contain an item\n              \'__COPY__\' with the desired destination of the copied plugin.\n        '
        try:
            plugin_id = get_int(request.POST.get('plugin_id'))
        except TypeError:
            raise RuntimeError("'plugin_id' is a required parameter.")
        plugin = self._get_plugin_from_id(plugin_id)
        try:
            placeholder_id = get_int(request.POST.get('placeholder_id'))
        except TypeError:
            raise RuntimeError("'placeholder_id' is a required parameter.")
        except ValueError:
            raise RuntimeError("'placeholder_id' must be an integer string.")
        placeholder = Placeholder.objects.get(pk=placeholder_id)
        parent_id = get_int(request.POST.get('plugin_parent', ''), None)
        target_language = request.POST['target_language']
        move_a_copy = request.POST.get('move_a_copy')
        move_a_copy = move_a_copy and move_a_copy != '0' and (move_a_copy.lower() != 'false')
        move_to_clipboard = placeholder == request.toolbar.clipboard
        source_placeholder = plugin.placeholder
        order = request.POST.getlist('plugin_order[]')
        parent_plugin = None
        if parent_id is not None:
            parent_plugin = self._get_plugin_from_id(parent_id)
        if placeholder != source_placeholder:
            try:
                template = self.get_placeholder_template(request, placeholder)
                has_reached_plugin_limit(placeholder, plugin.plugin_type, target_language, template=template, parent_plugin=parent_plugin)
            except PluginLimitReached as er:
                return HttpResponseBadRequest(er)
        exclude_from_order_check = ['__COPY__', str(plugin.pk)]
        ordered_plugin_ids = [int(pk) for pk in order if pk not in exclude_from_order_check]
        plugins_in_tree_count = placeholder.get_plugins(target_language).filter(parent=parent_id, pk__in=ordered_plugin_ids).count()
        if len(ordered_plugin_ids) != plugins_in_tree_count:
            message = _('order parameter references plugins in different trees')
            return HttpResponseBadRequest(force_str(message))
        move_a_plugin = not move_a_copy and (not move_to_clipboard)
        if parent_id and plugin.parent_id != parent_id:
            target_parent = get_object_or_404(CMSPlugin, pk=parent_id)
            if move_a_plugin and target_parent.placeholder_id != placeholder.pk:
                return HttpResponseBadRequest(force_str(_('parent must be in the same placeholder')))
            if move_a_plugin and target_parent.language != target_language:
                return HttpResponseBadRequest(force_str(_('parent must be in the same language as plugin_language')))
        elif parent_id:
            target_parent = plugin.parent
        else:
            target_parent = None
        new_plugin = None
        fetch_tree = False
        if move_a_copy and plugin.plugin_type == 'PlaceholderPlugin':
            new_plugins = self._paste_placeholder(request, plugin=plugin, target_language=target_language, target_placeholder=placeholder, tree_order=order)
        elif move_a_copy:
            fetch_tree = True
            new_plugin = self._paste_plugin(request, plugin=plugin, target_parent=target_parent, target_language=target_language, target_placeholder=placeholder, tree_order=order)
        elif move_to_clipboard:
            new_plugin = self._cut_plugin(request, plugin=plugin, target_language=target_language, target_placeholder=placeholder)
            new_plugins = [new_plugin]
        else:
            fetch_tree = True
            new_plugin = self._move_plugin(request, plugin=plugin, target_parent=target_parent, target_language=target_language, target_placeholder=placeholder, tree_order=order)
        if new_plugin and fetch_tree:
            root = new_plugin.parent or new_plugin
            new_plugins = [root] + list(root.get_descendants().order_by('path'))
        placeholder.mark_as_dirty(target_language)
        if placeholder != source_placeholder:
            source_placeholder.mark_as_dirty(plugin.language)
        data = get_plugin_tree_as_json(request, new_plugins)
        return HttpResponse(data, content_type='application/json')

    def _paste_plugin(self, request, plugin, target_language, target_placeholder, tree_order, target_parent=None):
        if False:
            return 10
        plugins = CMSPlugin.get_tree(parent=plugin).filter(placeholder=plugin.placeholder_id).order_by('path')
        plugins = list(plugins)
        if not self.has_copy_from_clipboard_permission(request, target_placeholder, plugins):
            message = force_str(_('You have no permission to paste this plugin'))
            raise PermissionDenied(message)
        if target_parent:
            target_parent_id = target_parent.pk
        else:
            target_parent_id = None
        target_tree_order = [int(pk) for pk in tree_order if not pk == '__COPY__']
        action_token = self._send_pre_placeholder_operation(request, operation=operations.PASTE_PLUGIN, plugin=plugin, target_language=target_language, target_placeholder=target_placeholder, target_parent_id=target_parent_id, target_order=target_tree_order)
        plugin_pairs = copy_plugins.copy_plugins_to(plugins, to_placeholder=target_placeholder, to_language=target_language, parent_plugin_id=target_parent_id)
        root_plugin = plugin_pairs[0][0]
        target_tree_order.insert(tree_order.index('__COPY__'), root_plugin.pk)
        reorder_plugins(target_placeholder, parent_id=target_parent_id, language=target_language, order=target_tree_order)
        target_placeholder.mark_as_dirty(target_language, clear_cache=False)
        root_plugin.refresh_from_db()
        self._send_post_placeholder_operation(request, operation=operations.PASTE_PLUGIN, plugin=root_plugin.get_bound_plugin(), token=action_token, target_language=target_language, target_placeholder=target_placeholder, target_parent_id=target_parent_id, target_order=target_tree_order)
        return root_plugin

    def _paste_placeholder(self, request, plugin, target_language, target_placeholder, tree_order):
        if False:
            while True:
                i = 10
        plugins = plugin.placeholder_ref.get_plugins_list()
        if not self.has_copy_from_clipboard_permission(request, target_placeholder, plugins):
            message = force_str(_('You have no permission to paste this placeholder'))
            raise PermissionDenied(message)
        target_tree_order = [int(pk) for pk in tree_order if not pk == '__COPY__']
        action_token = self._send_pre_placeholder_operation(request, operation=operations.PASTE_PLACEHOLDER, plugins=plugins, target_language=target_language, target_placeholder=target_placeholder, target_order=target_tree_order)
        new_plugins = copy_plugins.copy_plugins_to(plugins, to_placeholder=target_placeholder, to_language=target_language)
        new_plugin_ids = (new.pk for (new, old) in new_plugins)
        top_plugins = (pair for pair in new_plugins if not pair[0].parent_id)
        top_plugins_pks = [p[0].pk for p in sorted(top_plugins, key=lambda pair: pair[1].position)]
        target_tree_order[tree_order.index('__COPY__'):0] = top_plugins_pks
        reorder_plugins(target_placeholder, parent_id=None, language=target_language, order=target_tree_order)
        target_placeholder.mark_as_dirty(target_language, clear_cache=False)
        new_plugins = CMSPlugin.objects.filter(pk__in=new_plugin_ids).order_by('path').select_related('placeholder')
        new_plugins = list(new_plugins)
        self._send_post_placeholder_operation(request, operation=operations.PASTE_PLACEHOLDER, token=action_token, plugins=new_plugins, target_language=target_language, target_placeholder=target_placeholder, target_order=target_tree_order)
        return new_plugins

    def _move_plugin(self, request, plugin, target_language, target_placeholder, tree_order, target_parent=None):
        if False:
            for i in range(10):
                print('nop')
        if not self.has_move_plugin_permission(request, plugin, target_placeholder):
            message = force_str(_('You have no permission to move this plugin'))
            raise PermissionDenied(message)
        plugin_data = {'language': target_language, 'placeholder': target_placeholder}
        source_language = plugin.language
        source_placeholder = plugin.placeholder
        source_tree_order = source_placeholder.get_plugin_tree_order(language=source_language, parent_id=plugin.parent_id)
        if target_parent:
            target_parent_id = target_parent.pk
        else:
            target_parent_id = None
        if target_placeholder != source_placeholder:
            target_tree_order = target_placeholder.get_plugin_tree_order(language=target_language, parent_id=target_parent_id)
        else:
            target_tree_order = source_tree_order
        action_token = self._send_pre_placeholder_operation(request, operation=operations.MOVE_PLUGIN, plugin=plugin, source_language=source_language, source_placeholder=source_placeholder, source_parent_id=plugin.parent_id, source_order=source_tree_order, target_language=target_language, target_placeholder=target_placeholder, target_parent_id=target_parent_id, target_order=target_tree_order)
        if target_parent and plugin.parent != target_parent:
            updated_plugin = plugin.update(refresh=True, parent=target_parent, **plugin_data)
            updated_plugin = updated_plugin.move(target_parent, pos='last-child')
        elif target_parent:
            updated_plugin = plugin.update(refresh=True, **plugin_data)
        else:
            target = CMSPlugin.get_last_root_node()
            updated_plugin = plugin.update(refresh=True, parent=None, **plugin_data)
            updated_plugin = updated_plugin.move(target, pos='right')
        updated_plugin.get_descendants().update(**plugin_data)
        new_source_order = list(source_tree_order)
        new_source_order.remove(updated_plugin.pk)
        new_target_order = [int(pk) for pk in tree_order]
        reorder_plugins(target_placeholder, parent_id=target_parent_id, language=target_language, order=new_target_order)
        target_placeholder.mark_as_dirty(target_language, clear_cache=False)
        if source_placeholder != target_placeholder:
            source_placeholder.mark_as_dirty(source_language, clear_cache=False)
        updated_plugin.refresh_from_db()
        self._send_post_placeholder_operation(request, operation=operations.MOVE_PLUGIN, plugin=updated_plugin.get_bound_plugin(), token=action_token, source_language=source_language, source_placeholder=source_placeholder, source_parent_id=plugin.parent_id, source_order=new_source_order, target_language=target_language, target_placeholder=target_placeholder, target_parent_id=target_parent_id, target_order=new_target_order)
        return updated_plugin

    def _cut_plugin(self, request, plugin, target_language, target_placeholder):
        if False:
            while True:
                i = 10
        if not self.has_move_plugin_permission(request, plugin, target_placeholder):
            message = force_str(_('You have no permission to cut this plugin'))
            raise PermissionDenied(message)
        plugin_data = {'language': target_language, 'placeholder': target_placeholder}
        source_language = plugin.language
        source_placeholder = plugin.placeholder
        source_tree_order = source_placeholder.get_plugin_tree_order(language=source_language, parent_id=plugin.parent_id)
        action_token = self._send_pre_placeholder_operation(request, operation=operations.CUT_PLUGIN, plugin=plugin, clipboard=target_placeholder, clipboard_language=target_language, source_language=source_language, source_placeholder=source_placeholder, source_parent_id=plugin.parent_id, source_order=source_tree_order)
        target_placeholder.clear()
        target = CMSPlugin.get_last_root_node()
        updated_plugin = plugin.update(refresh=True, parent=None, **plugin_data)
        updated_plugin = updated_plugin.move(target, pos='right')
        updated_plugin.get_descendants().update(**plugin_data)
        new_source_order = list(source_tree_order)
        new_source_order.remove(updated_plugin.pk)
        source_placeholder.mark_as_dirty(target_language, clear_cache=False)
        self._send_post_placeholder_operation(request, operation=operations.CUT_PLUGIN, token=action_token, plugin=updated_plugin.get_bound_plugin(), clipboard=target_placeholder, clipboard_language=target_language, source_language=source_language, source_placeholder=source_placeholder, source_parent_id=plugin.parent_id, source_order=new_source_order)
        return updated_plugin

    @xframe_options_sameorigin
    def delete_plugin(self, request, plugin_id):
        if False:
            print('Hello World!')
        plugin = self._get_plugin_from_id(plugin_id)
        if not self.has_delete_plugin_permission(request, plugin):
            return HttpResponseForbidden(force_str(_('You do not have permission to delete this plugin')))
        opts = plugin._meta
        router.db_for_write(opts.model)
        get_deleted_objects_additional_kwargs = {'request': request}
        (deleted_objects, __, perms_needed, protected) = get_deleted_objects([plugin], admin_site=self.admin_site, **get_deleted_objects_additional_kwargs)
        if request.POST:
            if perms_needed:
                raise PermissionDenied(_('You do not have permission to delete this plugin'))
            obj_display = force_str(plugin)
            placeholder = plugin.placeholder
            plugin_tree_order = placeholder.get_plugin_tree_order(language=plugin.language, parent_id=plugin.parent_id)
            operation_token = self._send_pre_placeholder_operation(request, operation=operations.DELETE_PLUGIN, plugin=plugin, placeholder=placeholder, tree_order=plugin_tree_order)
            plugin.delete()
            placeholder.mark_as_dirty(plugin.language, clear_cache=False)
            reorder_plugins(placeholder=placeholder, parent_id=plugin.parent_id, language=plugin.language)
            self.log_deletion(request, plugin, obj_display)
            self.message_user(request, _('The %(name)s plugin "%(obj)s" was deleted successfully.') % {'name': force_str(opts.verbose_name), 'obj': force_str(obj_display)})
            new_plugin_tree_order = list(plugin_tree_order)
            new_plugin_tree_order.remove(plugin.pk)
            self._send_post_placeholder_operation(request, operation=operations.DELETE_PLUGIN, token=operation_token, plugin=plugin, placeholder=placeholder, tree_order=new_plugin_tree_order)
            return HttpResponseRedirect(admin_reverse('index', current_app=self.admin_site.name))
        plugin_name = force_str(plugin.get_plugin_class().name)
        if perms_needed or protected:
            title = _('Cannot delete %(name)s') % {'name': plugin_name}
        else:
            title = _('Are you sure?')
        context = {'title': title, 'object_name': plugin_name, 'object': plugin, 'deleted_objects': deleted_objects, 'perms_lacking': perms_needed, 'protected': protected, 'opts': opts, 'app_label': opts.app_label}
        request.current_app = self.admin_site.name
        return TemplateResponse(request, 'admin/cms/page/plugin/delete_confirmation.html', context)

    @xframe_options_sameorigin
    def clear_placeholder(self, request, placeholder_id):
        if False:
            while True:
                i = 10
        placeholder = get_object_or_404(Placeholder, pk=placeholder_id)
        language = request.GET.get('language')
        if placeholder.pk == request.toolbar.clipboard.pk:
            placeholder.clear(language)
            return HttpResponseRedirect(admin_reverse('index', current_app=self.admin_site.name))
        if not self.has_clear_placeholder_permission(request, placeholder, language):
            return HttpResponseForbidden(force_str(_('You do not have permission to clear this placeholder')))
        opts = Placeholder._meta
        router.db_for_write(Placeholder)
        plugins = placeholder.get_plugins_list(language)
        get_deleted_objects_additional_kwargs = {'request': request}
        (deleted_objects, __, perms_needed, protected) = get_deleted_objects(plugins, admin_site=self.admin_site, **get_deleted_objects_additional_kwargs)
        obj_display = force_str(placeholder)
        if request.POST:
            if perms_needed:
                return HttpResponseForbidden(force_str(_('You do not have permission to clear this placeholder')))
            operation_token = self._send_pre_placeholder_operation(request, operation=operations.CLEAR_PLACEHOLDER, plugins=plugins, placeholder=placeholder)
            placeholder.clear(language)
            placeholder.mark_as_dirty(language, clear_cache=False)
            self.log_deletion(request, placeholder, obj_display)
            self.message_user(request, _('The placeholder "%(obj)s" was cleared successfully.') % {'obj': obj_display})
            self._send_post_placeholder_operation(request, operation=operations.CLEAR_PLACEHOLDER, token=operation_token, plugins=plugins, placeholder=placeholder)
            return HttpResponseRedirect(admin_reverse('index', current_app=self.admin_site.name))
        if perms_needed or protected:
            title = _('Cannot delete %(name)s') % {'name': obj_display}
        else:
            title = _('Are you sure?')
        context = {'title': title, 'object_name': _('placeholder'), 'object': placeholder, 'deleted_objects': deleted_objects, 'perms_lacking': perms_needed, 'protected': protected, 'opts': opts, 'app_label': opts.app_label}
        request.current_app = self.admin_site.name
        return TemplateResponse(request, 'admin/cms/page/plugin/delete_confirmation.html', context)