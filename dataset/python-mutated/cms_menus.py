from django.db.models.query import Prefetch, prefetch_related_objects
from django.urls import reverse
from django.utils.functional import SimpleLazyObject
from django.utils.translation import override as force_language
from cms import constants
from cms.api import get_page_draft
from cms.apphook_pool import apphook_pool
from cms.models import EmptyTitle
from cms.utils.conf import get_cms_setting
from cms.utils.i18n import get_fallback_languages, get_public_languages, hide_untranslated, is_valid_site_language
from cms.utils.page import get_page_queryset
from cms.utils.page_permissions import user_can_view_all_pages
from cms.utils.permissions import get_view_restrictions
from menus.base import Menu, Modifier, NavigationNode
from menus.menu_pool import menu_pool

def get_visible_nodes(request, pages, site):
    if False:
        i = 10
        return i + 15
    '\n     This code is basically a many-pages-at-once version of\n     cms.utils.page_permissions.user_can_view_page\n     pages contains all published pages\n    '
    user = request.user
    _get_page_draft = get_page_draft
    public_for = get_cms_setting('PUBLIC_FOR')
    can_see_unrestricted = public_for == 'all' or (public_for == 'staff' and user.is_staff)
    if not user.is_authenticated and (not can_see_unrestricted):
        return []
    if user_can_view_all_pages(user, site):
        return list(pages)
    draft_pages = [_get_page_draft(page) for page in pages]
    restricted_pages = get_view_restrictions(draft_pages)
    if not restricted_pages:
        return list(pages) if can_see_unrestricted else []
    user_id = user.pk
    user_groups = SimpleLazyObject(lambda : frozenset(user.groups.values_list('pk', flat=True)))
    is_auth_user = user.is_authenticated

    def user_can_see_page(page):
        if False:
            for i in range(10):
                print('nop')
        page_id = page.pk if page.publisher_is_draft else page.publisher_public_id
        page_permissions = restricted_pages.get(page_id, [])
        if not page_permissions:
            return can_see_unrestricted
        if not is_auth_user:
            return False
        for perm in page_permissions:
            if perm.user_id == user_id or perm.group_id in user_groups:
                return True
        return False
    return [page for page in pages if user_can_see_page(page)]

def get_menu_node_for_page(renderer, page, language, fallbacks=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Transform a CMS page into a navigation node.\n\n    :param renderer: MenuRenderer instance bound to the request\n    :param page: the page you wish to transform\n    :param language: The current language used to render the menu\n    '
    if fallbacks is None:
        fallbacks = []
    attr = {'is_page': True, 'soft_root': page.soft_root, 'auth_required': page.login_required, 'reverse_id': page.reverse_id}
    if page.limit_visibility_in_menu is constants.VISIBILITY_ALL:
        attr['visible_for_authenticated'] = True
        attr['visible_for_anonymous'] = True
    else:
        attr['visible_for_authenticated'] = page.limit_visibility_in_menu == constants.VISIBILITY_USERS
        attr['visible_for_anonymous'] = page.limit_visibility_in_menu == constants.VISIBILITY_ANONYMOUS
    attr['is_home'] = page.is_home
    extenders = []
    if page.navigation_extenders:
        if page.navigation_extenders in renderer.menus:
            extenders.append(page.navigation_extenders)
        elif f'{page.navigation_extenders}:{page.pk}' in renderer.menus:
            extenders.append(f'{page.navigation_extenders}:{page.pk}')
    if page.title_cache.get(language) and page.application_urls:
        app = apphook_pool.get_apphook(page.application_urls)
        if app:
            extenders += app.get_menus(page, language)
    exts = []
    for ext in extenders:
        if hasattr(ext, 'get_instances'):
            exts.append(f'{ext.__name__}:{page.pk}')
        elif hasattr(ext, '__name__'):
            exts.append(ext.__name__)
        else:
            exts.append(ext)
    if exts:
        attr['navigation_extenders'] = exts
    for lang in [language] + fallbacks:
        translation = page.title_cache[lang]
        if translation:
            attr['redirect_url'] = translation.redirect
            ret_node = CMSNavigationNode(title=translation.menu_title or translation.title, url='', id=page.pk, attr=attr, visible=page.in_navigation, path=translation.path or translation.slug, language=translation.language if translation.language != language else None)
            return ret_node
    else:
        raise RuntimeError('Unable to render cms menu. There is a language misconfiguration.')

class CMSNavigationNode(NavigationNode):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.path = kwargs.pop('path')
        self.language = kwargs.pop('language', None)
        super().__init__(*args, **kwargs)

    def is_selected(self, request):
        if False:
            while True:
                i = 10
        try:
            page_id = request.current_page.pk
        except AttributeError:
            return False
        return page_id == self.id

    def _get_absolute_url(self):
        if False:
            print('Hello World!')
        if self.attr['is_home']:
            return reverse('pages-root')
        return reverse('pages-details-by-slug', kwargs={'slug': self.path})

    def get_absolute_url(self):
        if False:
            print('Hello World!')
        if self.language:
            with force_language(self.language):
                return self._get_absolute_url()
        return self._get_absolute_url()

class CMSMenu(Menu):

    def get_nodes(self, request):
        if False:
            print('Hello World!')
        from cms.models import Title
        site = self.renderer.site
        lang = self.renderer.request_language
        pages = get_page_queryset(site, draft=self.renderer.draft_mode_active, published=not self.renderer.draft_mode_active)
        if is_valid_site_language(lang, site_id=site.pk):
            _valid_language = True
            _hide_untranslated = hide_untranslated(lang, site.pk)
        else:
            _valid_language = False
            _hide_untranslated = False
        if _valid_language:
            if _hide_untranslated:
                fallbacks = []
            else:
                fallbacks = get_fallback_languages(lang, site_id=site.pk)
            languages = [lang] + [_lang for _lang in fallbacks if _lang != lang]
        else:
            languages = get_public_languages(site.pk)
            fallbacks = languages
        pages = pages.filter(title_set__language__in=languages).select_related('node').order_by('node__path').distinct()
        if not self.renderer.draft_mode_active:
            pages = pages.select_related('publisher_public__node')
        pages = get_visible_nodes(request, pages, site)
        if not pages:
            return []
        try:
            homepage = [page for page in pages if page.is_home][0]
        except IndexError:
            homepage = None
        titles = Title.objects.filter(language__in=languages, publisher_is_draft=self.renderer.draft_mode_active)
        lookup = Prefetch('title_set', to_attr='filtered_translations', queryset=titles)
        prefetch_related_objects(pages, lookup)
        blank_title_cache = {language: EmptyTitle(language=language) for language in languages}
        if lang not in blank_title_cache:
            blank_title_cache[lang] = EmptyTitle(language=lang)
        node_id_to_page = {}

        def _page_to_node(page):
            if False:
                while True:
                    i = 10
            page.title_cache = blank_title_cache.copy()
            for trans in page.filtered_translations:
                page.title_cache[trans.language] = trans
            menu_node = get_menu_node_for_page(self.renderer, page, language=lang, fallbacks=fallbacks)
            return menu_node
        menu_nodes = []
        for page in pages:
            node = page.node
            parent_id = node_id_to_page.get(node.parent_id)
            if node.parent_id and (not parent_id):
                continue
            menu_node = _page_to_node(page)
            cut_homepage = homepage and (not homepage.in_navigation)
            if cut_homepage and parent_id == homepage.pk:
                menu_node.parent_id = None
            else:
                menu_node.parent_id = parent_id
            node_id_to_page[node.pk] = page.pk
            menu_nodes.append(menu_node)
        return menu_nodes
menu_pool.register_menu(CMSMenu)

class NavExtender(Modifier):

    def modify(self, request, nodes, namespace, root_id, post_cut, breadcrumb):
        if False:
            print('Hello World!')
        if post_cut:
            return nodes
        home = next((n for n in nodes if n.attr.get('is_home', False)), None)
        exts = []
        for node in nodes:
            extenders = node.attr.get('navigation_extenders', None)
            if extenders:
                for ext in extenders:
                    if ext not in exts:
                        exts.append(ext)
                    for extnode in nodes:
                        if extnode.namespace == ext and (not extnode.parent_id):
                            if node == home and (not node.visible):
                                extnode.parent_namespace = None
                                extnode.parent = None
                            else:
                                extnode.parent_id = node.id
                                extnode.parent_namespace = node.namespace
                                extnode.parent = node
                                node.children.append(extnode)
        removed = []
        for menu in self.renderer.menus.items():
            if hasattr(menu[1], 'cms_enabled') and menu[1].cms_enabled and (menu[0] not in exts):
                for node in nodes:
                    if node.namespace == menu[0]:
                        removed.append(node)
        if breadcrumb:
            if breadcrumb and home and (not home.visible):
                home.visible = True
                if request.path_info == home.get_absolute_url():
                    home.selected = True
                else:
                    home.selected = False
        for node in removed:
            nodes.remove(node)
        return nodes
menu_pool.register_modifier(NavExtender)

class SoftRootCutter(Modifier):
    """
    Ask evildmp/superdmp if you don't understand softroots!

    Softroot description from the docs:

        A soft root is a page that acts as the root for a menu navigation tree.

        Typically, this will be a page that is the root of a significant new
        section on your site.

        When the soft root feature is enabled, the navigation menu for any page
        will start at the nearest soft root, rather than at the real root of
        the site’s page hierarchy.

        This feature is useful when your site has deep page hierarchies (and
        therefore multiple levels in its navigation trees). In such a case, you
        usually don’t want to present site visitors with deep menus of nested
        items.

        For example, you’re on the page -Introduction to Bleeding-?, so the menu
        might look like this:

            School of Medicine
                Medical Education
                Departments
                    Department of Lorem Ipsum
                    Department of Donec Imperdiet
                    Department of Cras Eros
                    Department of Mediaeval Surgery
                        Theory
                        Cures
                        Bleeding
                            Introduction to Bleeding <this is the current page>
                            Bleeding - the scientific evidence
                            Cleaning up the mess
                            Cupping
                            Leaches
                            Maggots
                        Techniques
                        Instruments
                    Department of Curabitur a Purus
                    Department of Sed Accumsan
                    Department of Etiam
                Research
                Administration
                Contact us
                Impressum

        which is frankly overwhelming.

        By making -Department of Mediaeval Surgery-? a soft root, the menu
        becomes much more manageable:

            Department of Mediaeval Surgery
                Theory
                Cures
                    Bleeding
                        Introduction to Bleeding <current page>
                        Bleeding - the scientific evidence
                        Cleaning up the mess
                    Cupping
                    Leaches
                    Maggots
                Techniques
                Instruments
    """

    def modify(self, request, nodes, namespace, root_id, post_cut, breadcrumb):
        if False:
            for i in range(10):
                print('nop')
        if post_cut or root_id:
            return nodes
        selected = None
        root_nodes = []
        for node in nodes:
            if node.selected:
                selected = node
            if not node.parent:
                root_nodes.append(node)
        if selected:
            if selected.attr.get('soft_root', False):
                nodes = selected.get_descendants()
                selected.parent = None
                nodes = [selected] + nodes
            else:
                nodes = self.find_ancestors_and_remove_children(selected, nodes)
        return nodes

    def find_and_remove_children(self, node, nodes):
        if False:
            i = 10
            return i + 15
        for child in node.children:
            if child.attr.get('soft_root', False):
                self.remove_children(child, nodes)
        return nodes

    def remove_children(self, node, nodes):
        if False:
            i = 10
            return i + 15
        for child in node.children:
            nodes.remove(child)
            self.remove_children(child, nodes)
        node.children = []

    def find_ancestors_and_remove_children(self, node, nodes):
        if False:
            while True:
                i = 10
        '\n        Check ancestors of node for soft roots\n        '
        if node.parent:
            if node.parent.attr.get('soft_root', False):
                nodes = node.parent.get_descendants()
                node.parent.parent = None
                nodes = [node.parent] + nodes
            else:
                nodes = self.find_ancestors_and_remove_children(node.parent, nodes)
        else:
            for newnode in nodes:
                if newnode != node and (not newnode.parent):
                    self.find_and_remove_children(newnode, nodes)
        for child in node.children:
            if child != node:
                self.find_and_remove_children(child, nodes)
        return nodes
menu_pool.register_modifier(SoftRootCutter)