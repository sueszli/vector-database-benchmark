import inspect
import json
import os
import warnings
from datetime import date
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from django.db.models import ManyToManyField, Model
from django.db.models.base import ModelBase
from django.urls import NoReverseMatch
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
from treebeard.mp_tree import MP_Node
from cms.exceptions import DontUsePageAttributeWarning
from cms.models.placeholdermodel import Placeholder
from cms.utils.conf import get_cms_setting
from cms.utils.urlutils import admin_reverse

class BoundRenderMeta:

    def __init__(self, meta):
        if False:
            while True:
                i = 10
        self.index = 0
        self.total = 1
        self.text_enabled = getattr(meta, 'text_enabled', False)

class PluginModelBase(ModelBase):
    """
    Metaclass for all CMSPlugin subclasses. This class should not be used for
    any other type of models.
    """

    def __new__(cls, name, bases, attrs):
        if False:
            return 10
        super_new = super().__new__
        attr_meta = attrs.pop('RenderMeta', None)
        parents = [b for b in bases if isinstance(b, PluginModelBase)]
        if parents and 'cmsplugin_ptr' not in attrs:
            meta = attrs.get('Meta', None)
            proxy = getattr(meta, 'proxy', False)
            field_is_inherited = any((hasattr(parent, 'cmsplugin_ptr') for parent in parents))
            if not proxy and (not field_is_inherited):
                attrs['cmsplugin_ptr'] = models.OneToOneField(to='cms.CMSPlugin', name='cmsplugin_ptr', related_name='%(app_label)s_%(class)s', auto_created=True, parent_link=True, on_delete=models.CASCADE)
        new_class = super_new(cls, name, bases, attrs)
        meta = attr_meta or getattr(new_class, '_render_meta', None)
        treebeard_view_fields = (f for f in new_class._meta.fields if f.name in ('depth', 'numchild', 'path'))
        for field in treebeard_view_fields:
            field.editable = False
        new_class._render_meta = BoundRenderMeta(meta)
        return new_class

class CMSPlugin(MP_Node, metaclass=PluginModelBase):
    """
    The base class for a CMS plugin model. When defining a new custom plugin, you should
    store plugin-instance specific information on a subclass of this class.

    An example for this would be to store the number of pictures to display in a galery.

    Two restrictions apply when subclassing this to use in your own models:
    1. Subclasses of CMSPlugin *cannot be further subclassed*
    2. Subclasses of CMSPlugin cannot define a "text" field.

    """
    placeholder = models.ForeignKey(Placeholder, on_delete=models.CASCADE, editable=False, null=True)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, blank=True, null=True, editable=False)
    position = models.PositiveSmallIntegerField(_('position'), default=0, editable=False)
    language = models.CharField(_('language'), max_length=15, blank=False, db_index=True, editable=False)
    plugin_type = models.CharField(_('plugin_name'), max_length=50, db_index=True, editable=False)
    creation_date = models.DateTimeField(_('creation date'), editable=False, default=timezone.now)
    changed_date = models.DateTimeField(auto_now=True)
    child_plugin_instances = None

    class Meta:
        app_label = 'cms'

    class RenderMeta:
        index = 0
        total = 1
        text_enabled = False

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return force_str(self.pk)

    def __repr__(self):
        if False:
            print('Hello World!')
        display = "<{module}.{class_name} id={id} plugin_type='{plugin_type}' object at {location}>".format(module=self.__module__, class_name=self.__class__.__name__, id=self.pk, plugin_type=self.plugin_type, location=hex(id(self)))
        return display

    def get_plugin_name(self):
        if False:
            print('Hello World!')
        from cms.plugin_pool import plugin_pool
        return plugin_pool.get_plugin(self.plugin_type).name

    def get_short_description(self):
        if False:
            print('Hello World!')
        instance = self.get_plugin_instance()[0]
        if instance is not None:
            return force_str(instance)
        return _('<Empty>')

    def get_plugin_class(self):
        if False:
            return 10
        from cms.plugin_pool import plugin_pool
        return plugin_pool.get_plugin(self.plugin_type)

    def get_plugin_class_instance(self, admin=None):
        if False:
            for i in range(10):
                print('nop')
        plugin_class = self.get_plugin_class()
        return plugin_class(plugin_class.model, admin)

    def get_plugin_instance(self, admin=None):
        if False:
            return 10
        "\n        Given a plugin instance (usually as a CMSPluginBase), this method\n        returns a tuple containing:\n            instance - The instance AS THE APPROPRIATE SUBCLASS OF\n                       CMSPluginBase and not necessarily just 'self', which is\n                       often just a CMSPluginBase,\n            plugin   - the associated plugin class instance (subclass\n                       of CMSPlugin)\n        "
        plugin = self.get_plugin_class_instance(admin)
        try:
            instance = self.get_bound_plugin()
        except ObjectDoesNotExist:
            instance = None
            self._inst = None
        return (instance, plugin)

    def get_bound_plugin(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns an instance of the plugin model\n        configured for this plugin type.\n        '
        if hasattr(self, '_inst'):
            return self._inst
        plugin = self.get_plugin_class()
        if plugin.model != self.__class__:
            self._inst = plugin.model.objects.get(cmsplugin_ptr=self)
            self._inst._render_meta = self._render_meta
        else:
            self._inst = self
        return self._inst

    def get_plugin_info(self, children=None, parents=None):
        if False:
            return 10
        plugin_name = self.get_plugin_name()
        data = {'type': 'plugin', 'placeholder_id': str(self.placeholder_id), 'plugin_name': force_str(plugin_name) or '', 'plugin_type': self.plugin_type, 'plugin_id': str(self.pk), 'plugin_language': self.language or '', 'plugin_parent': str(self.parent_id or ''), 'plugin_restriction': children or [], 'plugin_parent_restriction': parents or [], 'urls': self.get_action_urls()}
        return data

    def refresh_from_db(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().refresh_from_db(*args, **kwargs)
        try:
            del self._inst
        except AttributeError:
            pass

    def get_media_path(self, filename):
        if False:
            while True:
                i = 10
        pages = self.placeholder.page_set.all()
        if pages.exists():
            return pages[0].get_media_path(filename)
        else:
            today = date.today()
            return os.path.join(get_cms_setting('PAGE_MEDIA_PATH'), str(today.year), str(today.month), str(today.day), filename)

    @property
    def page(self):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn("Don't use the page attribute on CMSPlugins! CMSPlugins are not guaranteed to have a page associated with them!", DontUsePageAttributeWarning, stacklevel=2)
        return self.placeholder.page if self.placeholder_id else None

    def get_instance_icon_src(self):
        if False:
            i = 10
            return i + 15
        "\n        Get src URL for instance's icon\n        "
        (instance, plugin) = self.get_plugin_instance()
        return plugin.icon_src(instance) if instance else ''

    def get_instance_icon_alt(self):
        if False:
            print('Hello World!')
        "\n        Get alt text for instance's icon\n        "
        (instance, plugin) = self.get_plugin_instance()
        return force_str(plugin.icon_alt(instance)) if instance else ''

    def update(self, refresh=False, **fields):
        if False:
            for i in range(10):
                print('nop')
        CMSPlugin.objects.filter(pk=self.pk).update(**fields)
        if refresh:
            return self.reload()
        return

    def save(self, no_signals=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if not self.depth:
            if self.parent_id or self.parent:
                self.parent.add_child(instance=self)
            else:
                if not self.position and (not self.position == 0):
                    self.position = CMSPlugin.objects.filter(parent__isnull=True, language=self.language, placeholder_id=self.placeholder_id).count()
                self.add_root(instance=self)
            return
        super().save(*args, **kwargs)

    def reload(self):
        if False:
            while True:
                i = 10
        return CMSPlugin.objects.get(pk=self.pk)

    def move(self, target, pos=None):
        if False:
            return 10
        super().move(target, pos)
        self = self.reload()
        try:
            new_pos = max(CMSPlugin.objects.filter(parent_id=self.parent_id, placeholder_id=self.placeholder_id, language=self.language).exclude(pk=self.pk).order_by('depth', 'path').values_list('position', flat=True)) + 1
        except ValueError:
            new_pos = 0
        return self.update(refresh=True, position=new_pos)

    def set_base_attr(self, plugin):
        if False:
            return 10
        for attr in ['parent_id', 'placeholder', 'language', 'plugin_type', 'creation_date', 'depth', 'path', 'numchild', 'pk', 'position']:
            setattr(plugin, attr, getattr(self, attr))

    def copy_plugin(self, target_placeholder, target_language, parent_cache, no_signals=False):
        if False:
            i = 10
            return i + 15
        "\n        Copy this plugin and return the new plugin.\n\n        The logic of this method is the following:\n\n         # get a new generic plugin instance\n         # assign the position in the plugin tree\n         # save it to let mptt/treebeard calculate the tree attributes\n         # then get a copy of the current plugin instance\n         # assign to it the id of the generic plugin instance above;\n           this will effectively change the generic plugin created above\n           into a concrete one\n         # copy the tree related attributes from the generic plugin to\n           the concrete one\n         # save the concrete plugin\n         # trigger the copy relations\n         # return the generic plugin instance\n\n        This copy logic is required because we don't know what the fields of\n        the real plugin are. By getting another instance of it at step 4 and\n        then overwriting its ID at step 5, the ORM will copy the custom\n        fields for us.\n        "
        warnings.warn(f'{inspect.stack()[0][3]} is deprecated and will be removed in django CMS 4.1. From version 4 on, please use cms.utils.copy_plugins_to_placeholder instead.', DeprecationWarning, stacklevel=2)
        try:
            (plugin_instance, cls) = self.get_plugin_instance()
        except KeyError:
            return
        new_plugin = CMSPlugin()
        new_plugin.placeholder = target_placeholder
        parent_cache[self.pk] = new_plugin
        if self.parent:
            parent = parent_cache[self.parent_id]
            parent = CMSPlugin.objects.get(pk=parent.pk)
            new_plugin.parent_id = parent.pk
            new_plugin.parent = parent
        new_plugin.language = target_language
        new_plugin.plugin_type = self.plugin_type
        if no_signals:
            new_plugin._no_reorder = True
        new_plugin.save()
        if plugin_instance:
            plugin_instance = plugin_instance.__class__.objects.get(pk=plugin_instance.pk)
            plugin_instance.pk = new_plugin.pk
            plugin_instance.id = new_plugin.pk
            plugin_instance.placeholder = target_placeholder
            plugin_instance.cmsplugin_ptr = new_plugin
            plugin_instance.language = target_language
            plugin_instance.parent = new_plugin.parent
            plugin_instance.depth = new_plugin.depth
            plugin_instance.path = new_plugin.path
            plugin_instance.numchild = new_plugin.numchild
            plugin_instance._no_reorder = True
            plugin_instance.save()
            old_instance = plugin_instance.__class__.objects.get(pk=self.pk)
            plugin_instance.copy_relations(old_instance)
        return new_plugin

    @classmethod
    def fix_tree(cls, destructive=False):
        if False:
            i = 10
            return i + 15
        '\n        Fixes the plugin tree by first calling treebeard fix_tree and the\n        recalculating the correct position property for each plugin.\n        '
        from cms.utils.plugins import reorder_plugins
        super().fix_tree(destructive)
        for placeholder in Placeholder.objects.all():
            for (language, __) in settings.LANGUAGES:
                order = CMSPlugin.objects.filter(placeholder_id=placeholder.pk, language=language, parent_id__isnull=True).order_by('position', 'path').values_list('pk', flat=True)
                reorder_plugins(placeholder, None, language, order)
                for plugin in CMSPlugin.objects.filter(placeholder_id=placeholder.pk, language=language).order_by('depth', 'path'):
                    order = CMSPlugin.objects.filter(parent_id=plugin.pk).order_by('position', 'path').values_list('pk', flat=True)
                    reorder_plugins(placeholder, plugin.pk, language, order)

    def post_copy(self, old_instance, new_old_ziplist):
        if False:
            print('Hello World!')
        '\n        Handle more advanced cases (eg Text Plugins) after the original is\n        copied\n        '
        pass

    def copy_relations(self, old_instance):
        if False:
            return 10
        '\n        Handle copying of any relations attached to this plugin. Custom plugins\n        have to do this themselves!\n        '
        pass

    @classmethod
    def _get_related_objects(cls):
        if False:
            i = 10
            return i + 15
        fields = cls._meta._get_fields(forward=False, reverse=True, include_parents=True, include_hidden=False)
        return [obj for obj in fields if not isinstance(obj.field, ManyToManyField)]

    def get_position_in_placeholder(self):
        if False:
            i = 10
            return i + 15
        '\n        1 based position!\n        '
        return self.position + 1

    def get_breadcrumb(self):
        if False:
            print('Hello World!')
        from cms.models import Page
        model = self.placeholder._get_attached_model() or Page
        breadcrumb = []
        for parent in self.get_ancestors():
            try:
                url = force_str(admin_reverse(f'{model._meta.app_label}_{model._meta.model_name}_edit_plugin', args=[parent.pk]))
            except NoReverseMatch:
                url = force_str(admin_reverse(f'{Page._meta.app_label}_{Page._meta.model_name}_edit_plugin', args=[parent.pk]))
            breadcrumb.append({'title': force_str(parent.get_plugin_name()), 'url': url})
        try:
            url = force_str(admin_reverse(f'{model._meta.app_label}_{model._meta.model_name}_edit_plugin', args=[self.pk]))
        except NoReverseMatch:
            url = force_str(admin_reverse(f'{Page._meta.app_label}_{Page._meta.model_name}_edit_plugin', args=[self.pk]))
        breadcrumb.append({'title': force_str(self.get_plugin_name()), 'url': url})
        return breadcrumb

    def get_breadcrumb_json(self):
        if False:
            i = 10
            return i + 15
        result = json.dumps(self.get_breadcrumb())
        result = mark_safe(result)
        return result

    def num_children(self):
        if False:
            return 10
        return self.numchild

    def notify_on_autoadd(self, request, conf):
        if False:
            for i in range(10):
                print('nop')
        '\n        Method called when we auto add this plugin via default_plugins in\n        CMS_PLACEHOLDER_CONF.\n        Some specific plugins may have some special stuff to do when they are\n        auto added.\n        '
        pass

    def notify_on_autoadd_children(self, request, conf, children):
        if False:
            while True:
                i = 10
        '\n        Method called when we auto add children to this plugin via\n        default_plugins/<plugin>/children in CMS_PLACEHOLDER_CONF.\n        Some specific plugins may have some special stuff to do when we add\n        children to them. ie : TextPlugin must update its content to add HTML\n        tags to be able to see his children in WYSIWYG.\n        '
        pass

    def delete(self, no_mp=False, *args, **kwargs):
        if False:
            print('Hello World!')
        if no_mp:
            Model.delete(self, *args, **kwargs)
        else:
            super().delete(*args, **kwargs)

    def get_action_urls(self, js_compat=True):
        if False:
            while True:
                i = 10
        if js_compat:
            data = {'edit_plugin': self.get_edit_url(), 'add_plugin': self.get_add_url(), 'delete_plugin': self.get_delete_url(), 'move_plugin': self.get_move_url(), 'copy_plugin': self.get_copy_url()}
        else:
            data = {'edit_url': self.get_edit_url(), 'add_url': self.get_add_url(), 'delete_url': self.get_delete_url(), 'move_url': self.get_move_url(), 'copy_url': self.get_copy_url()}
        return data

    def get_add_url(self):
        if False:
            while True:
                i = 10
        return self.placeholder.get_add_url()

    def get_edit_url(self):
        if False:
            while True:
                i = 10
        return self.placeholder.get_edit_url(self.pk)

    def get_delete_url(self):
        if False:
            return 10
        return self.placeholder.get_delete_url(self.pk)

    def get_move_url(self):
        if False:
            for i in range(10):
                print('nop')
        return self.placeholder.get_move_url()

    def get_copy_url(self):
        if False:
            print('Hello World!')
        return self.placeholder.get_copy_url()

def get_plugin_media_path(instance, filename):
    if False:
        return 10
    '\n    Django requires that unbound function used in fields\' definitions to be\n    defined outside the parent class.\n     (see https://docs.djangoproject.com/en/dev/topics/migrations/#serializing-values)\n    This function is used within field definition:\n\n        file = models.FileField(_("file"), upload_to=get_plugin_media_path)\n\n    and it invokes the bounded method on the given instance at runtime\n    '
    return instance.get_media_path(filename)