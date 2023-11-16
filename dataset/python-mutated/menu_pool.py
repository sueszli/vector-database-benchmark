from functools import partial
from logging import getLogger
from django.contrib import messages
from django.contrib.sites.models import Site
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.urls import NoReverseMatch
from django.utils.functional import cached_property
from django.utils.module_loading import autodiscover_modules
from django.utils.translation import get_language_from_request
from django.utils.translation import gettext_lazy as _
from cms.utils import get_current_site
from cms.utils.conf import get_cms_setting
from cms.utils.i18n import get_default_language_for_site, is_language_prefix_patterns_used
from cms.utils.moderator import _use_draft as use_draft
from menus.base import Menu
from menus.exceptions import NamespaceAlreadyRegistered
from menus.models import CacheKey
logger = getLogger('menus')

def _build_nodes_inner_for_one_menu(nodes, menu_class_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    This is an easier to test "inner loop" building the menu tree structure\n    for one menu (one language, one site)\n    '
    done_nodes = {}
    final_nodes = []
    list_total_length = len(nodes)
    while nodes:
        should_add_to_final_list = True
        node = nodes.pop(0)
        node._counter = getattr(node, '_counter', 0) + 1
        if not node.namespace:
            node.namespace = menu_class_name
        if node.namespace not in done_nodes:
            done_nodes[node.namespace] = {}
        if node.parent_id in done_nodes[node.namespace]:
            if not node.parent_namespace:
                node.parent_namespace = menu_class_name
            parent = done_nodes[node.namespace][node.parent_id]
            parent.children.append(node)
            node.parent = parent
        elif node.parent_id:
            if node._counter < list_total_length:
                nodes.append(node)
            should_add_to_final_list = False
        if should_add_to_final_list:
            final_nodes.append(node)
            done_nodes[node.namespace][node.id] = node
    return final_nodes

def _get_menu_class_for_instance(menu_class, instance):
    if False:
        while True:
            i = 10
    '\n    Returns a new menu class that subclasses\n    menu_class but is bound to instance.\n    This means it sets the "instance" attribute of the class.\n    '
    attrs = {'instance': instance}
    class_name = menu_class.__name__
    meta_class = type(menu_class)
    return meta_class(class_name, (menu_class,), attrs)

class MenuRenderer:

    def __init__(self, pool, request):
        if False:
            for i in range(10):
                print('nop')
        self.pool = pool
        self.menus = pool.get_registered_menus(for_rendering=True)
        self.request = request
        if is_language_prefix_patterns_used():
            self.request_language = get_language_from_request(request, check_path=True)
        else:
            self.request_language = get_default_language_for_site(get_current_site().pk)
        self.site = Site.objects.get_current(request)

    @property
    def cache_key(self):
        if False:
            print('Hello World!')
        prefix = get_cms_setting('CACHE_PREFIX')
        key = '%smenu_nodes_%s_%s' % (prefix, self.request_language, self.site.pk)
        if self.request.user.is_authenticated:
            key += '_%s_user' % self.request.user.pk
        if self.draft_mode_active:
            key += ':draft'
        else:
            key += ':public'
        return key

    @cached_property
    def draft_mode_active(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            _use_draft = self.request.current_page.publisher_is_draft
        except AttributeError:
            _use_draft = use_draft(self.request)
        return _use_draft

    @cached_property
    def is_cached(self):
        if False:
            while True:
                i = 10
        db_cache_key_lookup = CacheKey.objects.filter(key=self.cache_key, language=self.request_language, site=self.site.pk)
        return db_cache_key_lookup.exists()

    def _build_nodes(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        This is slow. Caching must be used.\n        One menu is built per language and per site.\n\n        Namespaces: they are ID prefixes to avoid node ID clashes when plugging\n        multiple trees together.\n\n        - We iterate on the list of nodes.\n        - We store encountered nodes in a dict (with namespaces):\n            done_nodes[<namespace>][<node's id>] = node\n        - When a node has a parent defined, we lookup that parent in done_nodes\n            if it's found:\n                set the node as the node's parent's child (re-read this)\n            else:\n                the node is put at the bottom of the list\n        "
        key = self.cache_key
        cached_nodes = cache.get(key, None)
        if cached_nodes and self.is_cached:
            return cached_nodes
        final_nodes = []
        toolbar = getattr(self.request, 'toolbar', None)
        for menu_class_name in self.menus:
            menu = self.get_menu(menu_class_name)
            try:
                nodes = menu.get_nodes(self.request)
            except NoReverseMatch:
                nodes = []
                if toolbar and toolbar.is_staff:
                    messages.error(self.request, _('Menu %s cannot be loaded. Please, make sure all its urls exist and can be resolved.') % menu_class_name)
                logger.error('Menu %s could not be loaded.' % menu_class_name, exc_info=True)
            final_nodes += _build_nodes_inner_for_one_menu(nodes, menu_class_name)
        cache.set(key, final_nodes, get_cms_setting('CACHE_DURATIONS')['menus'])
        if not self.is_cached:
            self.__dict__['is_cached'] = True
            CacheKey.objects.create(key=key, language=self.request_language, site=self.site.pk)
        return final_nodes

    def _mark_selected(self, nodes):
        if False:
            print('Hello World!')
        for node in nodes:
            node.selected = node.is_selected(self.request)
        return nodes

    def apply_modifiers(self, nodes, namespace=None, root_id=None, post_cut=False, breadcrumb=False):
        if False:
            while True:
                i = 10
        if not post_cut:
            nodes = self._mark_selected(nodes)
        for cls in self.pool.get_registered_modifiers():
            inst = cls(renderer=self)
            nodes = inst.modify(self.request, nodes, namespace, root_id, post_cut, breadcrumb)
        return nodes

    def get_nodes(self, namespace=None, root_id=None, breadcrumb=False):
        if False:
            for i in range(10):
                print('nop')
        nodes = self._build_nodes()
        nodes = self.apply_modifiers(nodes=nodes, namespace=namespace, root_id=root_id, post_cut=False, breadcrumb=breadcrumb)
        return nodes

    def get_menu(self, menu_name):
        if False:
            for i in range(10):
                print('nop')
        MenuClass = self.menus[menu_name]
        return MenuClass(renderer=self)

class MenuPool:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.menus = {}
        self.modifiers = []
        self.discovered = False

    def get_renderer(self, request):
        if False:
            return 10
        self.discover_menus()
        return MenuRenderer(pool=self, request=request)

    def discover_menus(self):
        if False:
            print('Hello World!')
        if self.discovered:
            return
        autodiscover_modules('cms_menus')
        from menus.modifiers import register
        register()
        self.discovered = True

    def get_registered_menus(self, for_rendering=False):
        if False:
            i = 10
            return i + 15
        "\n        Returns all registered menu classes.\n\n        :param for_rendering: Flag that when True forces us to include\n            all CMSAttachMenu subclasses, even if they're not attached.\n        "
        self.discover_menus()
        registered_menus = {}
        for (menu_class_name, menu_cls) in self.menus.items():
            if isinstance(menu_cls, Menu):
                menu_cls = menu_cls.__class__
            if hasattr(menu_cls, 'get_instances'):
                _get_menu_class = partial(_get_menu_class_for_instance, menu_cls)
                instances = menu_cls.get_instances() or []
                for instance in instances:
                    namespace = '{0}:{1}'.format(menu_class_name, instance.pk)
                    registered_menus[namespace] = _get_menu_class(instance)
                if not instances and (not for_rendering):
                    registered_menus[menu_class_name] = menu_cls
            elif hasattr(menu_cls, 'get_nodes'):
                registered_menus[menu_class_name] = menu_cls
            else:
                raise ValidationError("Something was registered as a menu, but isn't.")
        return registered_menus

    def get_registered_modifiers(self):
        if False:
            while True:
                i = 10
        return self.modifiers

    def clear(self, site_id=None, language=None, all=False):
        if False:
            print('Hello World!')
        '\n        This invalidates the cache for a given menu (site_id and language)\n        '
        if all:
            cache_keys = CacheKey.objects.get_keys()
        else:
            cache_keys = CacheKey.objects.get_keys(site_id, language)
        to_be_deleted = cache_keys.distinct().values_list('key', flat=True)
        if to_be_deleted:
            cache.delete_many(to_be_deleted)
            cache_keys.delete()

    def register_menu(self, menu_cls):
        if False:
            for i in range(10):
                print('nop')
        from menus.base import Menu
        assert issubclass(menu_cls, Menu)
        if menu_cls.__name__ in self.menus:
            raise NamespaceAlreadyRegistered('[{0}] a menu with this name is already registered'.format(menu_cls.__name__))
        self.menus[menu_cls.__name__] = menu_cls

    def register_modifier(self, modifier_class):
        if False:
            while True:
                i = 10
        from menus.base import Modifier
        assert issubclass(modifier_class, Modifier)
        if modifier_class not in self.modifiers:
            self.modifiers.append(modifier_class)

    def get_menus_by_attribute(self, name, value):
        if False:
            i = 10
            return i + 15
        '\n        Returns the list of menus that match the name/value criteria provided.\n        '
        menus = self.get_registered_menus(for_rendering=False)
        return sorted(list(set([(menu.__name__, menu.name) for (menu_class_name, menu) in menus.items() if getattr(menu, name, None) == value])))

    def get_nodes_by_attribute(self, nodes, name, value):
        if False:
            i = 10
            return i + 15
        return [node for node in nodes if node.attr.get(name, None) == value]
menu_pool = MenuPool()