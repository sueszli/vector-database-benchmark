from django.contrib.admin.sites import site
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser, Group, Permission
from django.contrib.sites.models import Site
from django.db.models import Q
from django.test.client import RequestFactory
from django.test.utils import override_settings
from cms.admin.forms import save_permissions
from cms.api import add_plugin, assign_user_to_page, create_page, create_page_user, publish_page
from cms.cms_menus import get_visible_nodes
from cms.models import ACCESS_PAGE, CMSPlugin, Page, Title
from cms.models.permissionmodels import ACCESS_DESCENDANTS, ACCESS_PAGE_AND_DESCENDANTS, GlobalPagePermission, PagePermission
from cms.test_utils.testcases import URL_CMS_PAGE_ADD, CMSTestCase
from cms.test_utils.util.fuzzy_int import FuzzyInt
from cms.utils import get_current_site
from cms.utils.page_permissions import user_can_publish_page, user_can_view_page

def fake_tree_attrs(page):
    if False:
        for i in range(10):
            print('nop')
    page.depth = 1
    page.path = '0001'
    page.numchild = 0

@override_settings(CMS_PERMISSION=True)
class PermissionModeratorTests(CMSTestCase):
    """Permissions and moderator together

    Fixtures contains 3 users and 1 published page and some other stuff

    Users:
        1. `super`: superuser
        2. `master`: user with permissions to all applications
        3. `slave`: user assigned to page `slave-home`

    Pages:
        1. `home`:
            - published page
            - master can do anything on its subpages, but not on home!

        2. `master`:
            - published page
            - created by super
            - `master` can do anything on it and its descendants
            - subpages:

        3.       `slave-home`:
                    - not published
                    - assigned slave user which can add/change/delete/
                      move/publish this page and its descendants
                    - `master` user want to moderate this page and all descendants

        4. `pageA`:
            - created by super
            - master can add/change/delete on it and descendants
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.user_super = self._create_user('super', is_staff=True, is_superuser=True)
        self.user_staff = self._create_user('staff', is_staff=True, add_default_permissions=True)
        self.add_permission(self.user_staff, 'publish_page')
        self.user_master = self._create_user('master', is_staff=True, add_default_permissions=True)
        self.add_permission(self.user_master, 'publish_page')
        self.user_slave = self._create_user('slave', is_staff=True, add_default_permissions=True)
        self.user_normal = self._create_user('normal', is_staff=False)
        self.user_normal.user_permissions.add(Permission.objects.get(codename='publish_page'))
        with self.login_user_context(self.user_super):
            self.home_page = create_page('home', 'nav_playground.html', 'en', created_by=self.user_super)
            self.master_page = create_page('master', 'nav_playground.html', 'en')
            self.user_non_global = self._create_user('nonglobal')
            assign_user_to_page(self.home_page, self.user_master, grant_on=ACCESS_PAGE_AND_DESCENDANTS, grant_all=True)
            assign_user_to_page(self.master_page, self.user_master, grant_on=ACCESS_PAGE_AND_DESCENDANTS, grant_all=True)
            self.slave_page = create_page('slave-home', 'col_two.html', 'en', parent=self.master_page, created_by=self.user_super)
            assign_user_to_page(self.slave_page, self.user_slave, grant_all=True)
            page_b = create_page('pageB', 'nav_playground.html', 'en', created_by=self.user_super)
            assign_user_to_page(page_b, self.user_normal, can_view=True)
            page_a = create_page('pageA', 'nav_playground.html', 'en', created_by=self.user_super)
            assign_user_to_page(page_a, self.user_master, can_add=True, can_change=True, can_delete=True, can_publish=True, can_move_page=True)
            publish_page(self.home_page, self.user_super, 'en')
            publish_page(self.master_page, self.user_super, 'en')
            self.page_b = publish_page(page_b, self.user_super, 'en')

    def _add_plugin(self, user, page):
        if False:
            return 10
        '\n        Add a plugin using the test client to check for permissions.\n        '
        with self.login_user_context(user):
            placeholder = page.placeholders.all()[0]
            post_data = {'body': 'Test'}
            endpoint = self.get_add_plugin_uri(placeholder, 'TextPlugin')
            response = self.client.post(endpoint, post_data)
            self.assertEqual(response.status_code, 302)
            return response.content.decode('utf8')

    def test_super_can_add_page_to_root(self):
        if False:
            return 10
        with self.login_user_context(self.user_super):
            response = self.client.get(URL_CMS_PAGE_ADD)
            self.assertEqual(response.status_code, 200)

    def test_master_cannot_add_page_to_root(self):
        if False:
            while True:
                i = 10
        with self.login_user_context(self.user_master):
            response = self.client.get(URL_CMS_PAGE_ADD)
            self.assertEqual(response.status_code, 403)

    def test_slave_cannot_add_page_to_root(self):
        if False:
            return 10
        with self.login_user_context(self.user_slave):
            response = self.client.get(URL_CMS_PAGE_ADD)
            self.assertEqual(response.status_code, 403)

    def test_slave_can_add_page_under_slave_home(self):
        if False:
            print('Hello World!')
        with self.login_user_context(self.user_slave):
            page = create_page('page', 'nav_playground.html', 'en', parent=self.slave_page, created_by=self.user_slave)
            self.assertFalse(page.publisher_public)
            self.assertObjectExist(Title.objects, slug='page')
            self.assertObjectDoesNotExist(Title.objects.public(), slug='page')
            self.assertTrue(user_can_publish_page(self.user_slave, page))
            publish_page(page, self.user_slave, 'en')

    @override_settings(CMS_PLACEHOLDER_CONF={'col_left': {'default_plugins': [{'plugin_type': 'TextPlugin', 'values': {'body': 'Lorem ipsum dolor sit amet, consectetur adipisicing elit. Culpa, repellendus, delectus, quo quasi ullam inventore quod quam aut voluptatum aliquam voluptatibus harum officiis officia nihil minus unde accusamus dolorem repudiandae.'}}]}})
    def test_default_plugins(self):
        if False:
            return 10
        with self.login_user_context(self.user_slave):
            self.assertEqual(CMSPlugin.objects.count(), 0)
            response = self.client.get(self.slave_page.get_absolute_url(), {'edit': 1})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(CMSPlugin.objects.count(), 1)

    def test_page_added_by_slave_can_be_published_by_user_master(self):
        if False:
            while True:
                i = 10
        page = create_page('page', 'nav_playground.html', 'en', parent=self.slave_page, created_by=self.user_slave)
        self.assertFalse(page.publisher_public)
        self.assertTrue(user_can_publish_page(self.user_master, page))
        publish_page(self.slave_page, self.user_master, 'en')
        page = publish_page(page, self.user_master, 'en')
        self.assertTrue(page.publisher_public_id)

    def test_super_can_add_plugin(self):
        if False:
            i = 10
            return i + 15
        self._add_plugin(self.user_super, page=self.slave_page)

    def test_master_can_add_plugin(self):
        if False:
            print('Hello World!')
        self._add_plugin(self.user_master, page=self.slave_page)

    def test_slave_can_add_plugin(self):
        if False:
            for i in range(10):
                print('nop')
        self._add_plugin(self.user_slave, page=self.slave_page)

    def test_subtree_needs_approval(self):
        if False:
            return 10
        page = create_page('parent', 'nav_playground.html', 'en', parent=self.home_page)
        self.assertFalse(page.publisher_public)
        subpage = create_page('subpage', 'nav_playground.html', 'en', parent=page, published=False)
        subpage = publish_page(subpage, self.user_master, 'en')
        self.assertNeverPublished(subpage)
        page = publish_page(page, self.user_master, 'en')
        self.assertPublished(page)
        self.assertNeverPublished(subpage)
        subpage = publish_page(subpage, self.user_master, 'en')
        self.assertPublished(subpage)

    def test_subtree_with_super(self):
        if False:
            return 10
        page = create_page('page', 'nav_playground.html', 'en')
        self.assertFalse(page.publisher_public)
        subpage = create_page('subpage', 'nav_playground.html', 'en', parent=page)
        self.assertFalse(subpage.publisher_public)
        self.assertEqual(page.node.path[0:4], subpage.node.path[0:4])
        page = self.reload(page)
        page = publish_page(page, self.user_super, 'en')
        subpage = self.reload(subpage)
        self.assertEqual(page.node.path[0:4], subpage.node.path[0:4])
        subpage = publish_page(subpage, self.user_super, 'en')
        self.assertEqual(page.node.path[0:4], subpage.node.path[0:4])

    def test_super_add_page_to_root(self):
        if False:
            for i in range(10):
                print('nop')
        'Create page which is not under moderation in root, and check if\n        some properties are correct.\n        '
        page = create_page('page', 'nav_playground.html', 'en')
        self.assertFalse(page.publisher_public)

    def test_plugins_get_published(self):
        if False:
            i = 10
            return i + 15
        page = create_page('page', 'nav_playground.html', 'en')
        placeholder = page.placeholders.all()[0]
        add_plugin(placeholder, 'TextPlugin', 'en', body='test')
        self.assertEqual(CMSPlugin.objects.all().count(), 1)
        publish_page(page, self.user_super, 'en')
        self.assertEqual(CMSPlugin.objects.all().count(), 2)

    def test_remove_plugin_page_under_moderation(self):
        if False:
            return 10
        page = create_page('page', 'nav_playground.html', 'en', parent=self.slave_page)
        placeholder = page.placeholders.all()[0]
        plugin = add_plugin(placeholder, 'TextPlugin', 'en', body='test')
        page = self.reload(page)
        page = publish_page(page, self.user_slave, 'en')
        self.assertEqual(CMSPlugin.objects.all().count(), 1)
        slave_page = self.reload(self.slave_page)
        publish_page(slave_page, self.user_master, 'en')
        page = self.reload(page)
        page = publish_page(page, self.user_master, 'en')
        self.assertEqual(CMSPlugin.objects.all().count(), 2)
        with self.login_user_context(self.user_slave):
            plugin_data = {'plugin_id': plugin.pk}
            endpoint = self.get_delete_plugin_uri(plugin)
            response = self.client.post(endpoint, plugin_data)
            self.assertEqual(response.status_code, 302)
            self.assertEqual(CMSPlugin.objects.all().count(), 1)
            page = self.reload(page)
            publish_page(page, self.user_super, 'en')
            self.assertEqual(CMSPlugin.objects.all().count(), 0)

    def test_superuser_can_view(self):
        if False:
            while True:
                i = 10
        url = self.page_b.get_absolute_url(language='en')
        with self.login_user_context(self.user_super):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)

    def test_staff_can_view(self):
        if False:
            return 10
        url = self.page_b.get_absolute_url(language='en')
        all_view_perms = PagePermission.objects.filter(can_view=True)
        has_perm = False
        for perm in all_view_perms:
            if perm.page == self.page_b:
                if perm.user == self.user_staff:
                    has_perm = True
        self.assertEqual(has_perm, False)
        login_ok = self.client.login(username=getattr(self.user_staff, get_user_model().USERNAME_FIELD), password=getattr(self.user_staff, get_user_model().USERNAME_FIELD))
        self.assertTrue(login_ok)
        self.assertTrue('_auth_user_id' in self.client.session)
        login_user_id = self.client.session.get('_auth_user_id')
        user = get_user_model().objects.get(pk=self.user_staff.pk)
        self.assertEqual(str(login_user_id), str(user.id))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    def test_user_normal_can_view(self):
        if False:
            print('Hello World!')
        url = self.page_b.get_absolute_url(language='en')
        all_view_perms = PagePermission.objects.filter(can_view=True)
        normal_has_perm = False
        for perm in all_view_perms:
            if perm.page == self.page_b:
                if perm.user == self.user_normal:
                    normal_has_perm = True
        self.assertTrue(normal_has_perm)
        with self.login_user_context(self.user_normal):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
        non_global_has_perm = False
        for perm in all_view_perms:
            if perm.page == self.page_b:
                if perm.user == self.user_non_global:
                    non_global_has_perm = True
        self.assertFalse(non_global_has_perm)
        with self.login_user_context(self.user_non_global):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 404)
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    def test_user_globalpermission(self):
        if False:
            i = 10
            return i + 15
        user_global = self._create_user('global')
        with self.login_user_context(self.user_super):
            user_global = create_page_user(user_global, user_global)
            user_global.is_staff = False
            user_global.save()
            global_page = create_page('global', 'nav_playground.html', 'en', published=True)
            assign_user_to_page(global_page, user_global, global_permission=True, can_view=True)
        url = global_page.get_absolute_url('en')
        all_view_perms = PagePermission.objects.filter(can_view=True)
        has_perm = False
        for perm in all_view_perms:
            if perm.page == self.page_b and perm.user == user_global:
                has_perm = True
        self.assertEqual(has_perm, False)
        global_page_perm_q = Q(user=user_global) & Q(can_view=True)
        global_view_perms = GlobalPagePermission.objects.filter(global_page_perm_q).exists()
        self.assertEqual(global_view_perms, True)
        with self.login_user_context(user_global):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
        has_perm = False
        for perm in all_view_perms:
            if perm.page == self.page_b and perm.user == self.user_non_global:
                has_perm = True
        self.assertEqual(has_perm, False)
        global_page_perm_q = Q(user=self.user_non_global) & Q(can_view=True)
        global_view_perms = GlobalPagePermission.objects.filter(global_page_perm_q).exists()
        self.assertEqual(global_view_perms, False)
        with self.login_user_context(self.user_non_global):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 404)

    def test_anonymous_user_public_for_all(self):
        if False:
            return 10
        url = self.page_b.get_absolute_url('en')
        with self.settings(CMS_PUBLIC_FOR='all'):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 404)

    def test_anonymous_user_public_for_none(self):
        if False:
            return 10
        url = self.page_b.get_absolute_url('en')
        with self.settings(CMS_PUBLIC_FOR=None):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 404)

@override_settings(CMS_PERMISSION=True)
class PatricksMoveTest(CMSTestCase):
    """
    Fixtures contains 3 users and 1 published page and some other stuff

    Users:
        1. `super`: superuser
        2. `master`: user with permissions to all applications
        3. `slave`: user assigned to page `slave-home`

    Pages:
        1. `home`:
            - published page
            - master can do anything on its subpages, but not on home!

        2. `master`:
            - published page
            - created by super
            - `master` can do anything on it and its descendants
            - subpages:

        3.       `slave-home`:
                    - not published
                    - assigned slave user which can add/change/delete/
                      move/publish/moderate this page and its descendants
                    - `master` user want to moderate this page and all descendants

        4. `pageA`:
            - created by super
            - master can add/change/delete on it and descendants
    """

    def setUp(self):
        if False:
            return 10
        self.user_super = self._create_user('super', True, True)
        with self.login_user_context(self.user_super):
            self.home_page = create_page('home', 'nav_playground.html', 'en', created_by=self.user_super)
            self.master_page = create_page('master', 'nav_playground.html', 'en')
            self.user_master = self._create_user('master', True)
            self.add_permission(self.user_master, 'change_page')
            self.add_permission(self.user_master, 'publish_page')
            assign_user_to_page(self.home_page, self.user_master, grant_on=ACCESS_DESCENDANTS, grant_all=True)
            assign_user_to_page(self.master_page, self.user_master, grant_all=True)
            self.slave_page = create_page('slave-home', 'nav_playground.html', 'en', parent=self.master_page, created_by=self.user_super)
            slave = self._create_user('slave', True)
            self.user_slave = create_page_user(self.user_super, slave, can_add_page=True, can_change_page=True, can_delete_page=True)
            assign_user_to_page(self.slave_page, self.user_slave, grant_all=True)
            page_a = create_page('pageA', 'nav_playground.html', 'en', created_by=self.user_super)
            assign_user_to_page(page_a, self.user_master, can_add=True, can_change=True, can_delete=True, can_publish=True, can_move_page=True)
            publish_page(self.home_page, self.user_super, 'en')
            publish_page(self.master_page, self.user_super, 'en')
        with self.login_user_context(self.user_slave):
            self.pa = create_page('pa', 'nav_playground.html', 'en', parent=self.slave_page)
            self.pb = create_page('pb', 'nav_playground.html', 'en', parent=self.pa, position='right')
            self.pc = create_page('pc', 'nav_playground.html', 'en', parent=self.pb, position='right')
            self.pd = create_page('pd', 'nav_playground.html', 'en', parent=self.pb)
            self.pe = create_page('pe', 'nav_playground.html', 'en', parent=self.pd, position='right')
            self.pf = create_page('pf', 'nav_playground.html', 'en', parent=self.pe)
            self.pg = create_page('pg', 'nav_playground.html', 'en', parent=self.pf, position='right')
            self.ph = create_page('ph', 'nav_playground.html', 'en', parent=self.pf, position='right')
            self.assertFalse(self.pg.publisher_public)
            self.slave_page = self.slave_page.reload()
            publish_page(self.slave_page, self.user_master, 'en')
            publish_page(self.pa, self.user_master, 'en')
            publish_page(self.pb, self.user_master, 'en')
            publish_page(self.pc, self.user_master, 'en')
            publish_page(self.pd, self.user_master, 'en')
            publish_page(self.pe, self.user_master, 'en')
            publish_page(self.pf, self.user_master, 'en')
            publish_page(self.pg, self.user_master, 'en')
            publish_page(self.ph, self.user_master, 'en')
            self.reload_pages()

    def reload_pages(self):
        if False:
            print('Hello World!')
        self.pa = self.pa.reload()
        self.pb = self.pb.reload()
        self.pc = self.pc.reload()
        self.pd = self.pd.reload()
        self.pe = self.pe.reload()
        self.pf = self.pf.reload()
        self.pg = self.pg.reload()
        self.ph = self.ph.reload()

    def test_patricks_move(self):
        if False:
            print('Hello World!')
        '\n\n        Tests permmod when moving trees of pages.\n\n        1. build following tree (master node is approved and published)\n\n                 slave-home\n                /    |                   A     B     C\n                   /                    D    E\n                    /  |                     F   G   H\n\n        2. perform move operations:\n            1. move G under C\n            2. move E under G\n\n                 slave-home\n                /    |                   A     B     C\n                   /                          D          G\n                                                             E\n                             /                               F     H\n\n        3. approve nodes in following order:\n            1. approve H\n            2. approve G\n            3. approve E\n            4. approve F\n        '
        self.assertEqual(self.pg.node.parent, self.pe.node)
        self.move_page(self.pg, self.pc)
        self.reload_pages()
        self.assertEqual(self.pg.node.parent, self.pc.node)
        self.assertEqual(self.pg.get_absolute_url(), self.pg.publisher_public.get_absolute_url())
        self.move_page(self.pe, self.pg)
        self.reload_pages()
        self.assertEqual(self.pe.node.parent, self.pg.node)
        self.ph = self.ph.reload()
        self.assertEqual(self.pg.publisher_public.get_absolute_url(), self.pg.get_absolute_url())
        self.assertEqual(self.ph.publisher_public.get_absolute_url(), self.ph.get_absolute_url())
        self.assertEqual(self.pg.publisher_public.get_absolute_url(), '%smaster/slave-home/pc/pg/' % self.get_pages_root())
        self.assertEqual(self.ph.publisher_public.get_absolute_url(), '%smaster/slave-home/pc/pg/pe/ph/' % self.get_pages_root())

class ViewPermissionBaseTests(CMSTestCase):

    def setUp(self):
        if False:
            return 10
        self.page = create_page('testpage', 'nav_playground.html', 'en')
        self.site = get_current_site()

    def get_request(self, user=None):
        if False:
            print('Hello World!')
        attrs = {'user': user or AnonymousUser(), 'REQUEST': {}, 'POST': {}, 'GET': {}, 'session': {}}
        return type('Request', (object,), attrs)

    def assertViewAllowed(self, page, user=None):
        if False:
            for i in range(10):
                print('nop')
        if not user:
            user = AnonymousUser()
        self.assertTrue(user_can_view_page(user, page))

    def assertViewNotAllowed(self, page, user=None):
        if False:
            i = 10
            return i + 15
        if not user:
            user = AnonymousUser()
        self.assertFalse(user_can_view_page(user, page))

@override_settings(CMS_PERMISSION=False, CMS_PUBLIC_FOR='staff')
class BasicViewPermissionTests(ViewPermissionBaseTests):
    """
    Test functionality with CMS_PERMISSION set to false, as this is the
    normal use case
    """

    @override_settings(CMS_PUBLIC_FOR='all')
    def test_unauth_public(self):
        if False:
            while True:
                i = 10
        request = self.get_request()
        with self.assertNumQueries(0):
            self.assertViewAllowed(self.page)
        self.assertEqual(get_visible_nodes(request, [self.page], self.site), [self.page])

    def test_unauth_non_access(self):
        if False:
            print('Hello World!')
        request = self.get_request()
        with self.assertNumQueries(0):
            self.assertViewNotAllowed(self.page)
        self.assertEqual(get_visible_nodes(request, [self.page], self.site), [])

    @override_settings(CMS_PUBLIC_FOR='all')
    def test_staff_public_all(self):
        if False:
            for i in range(10):
                print('nop')
        user = self.get_staff_user_with_no_permissions()
        request = self.get_request(user)
        with self.assertNumQueries(0):
            self.assertViewAllowed(self.page, user)
        self.assertEqual(get_visible_nodes(request, [self.page], self.site), [self.page])

    def test_staff_public_staff(self):
        if False:
            print('Hello World!')
        user = self.get_staff_user_with_no_permissions()
        request = self.get_request(user)
        with self.assertNumQueries(0):
            self.assertViewAllowed(self.page, user)
        self.assertEqual(get_visible_nodes(request, [self.page], self.site), [self.page])

    def test_staff_basic_auth(self):
        if False:
            i = 10
            return i + 15
        user = self.get_staff_user_with_no_permissions()
        request = self.get_request(user)
        with self.assertNumQueries(0):
            self.assertViewAllowed(self.page, user)
        self.assertEqual(get_visible_nodes(request, [self.page], self.site), [self.page])

    @override_settings(CMS_PUBLIC_FOR='all')
    def test_normal_basic_auth(self):
        if False:
            return 10
        user = self.get_standard_user()
        request = self.get_request(user)
        with self.assertNumQueries(0):
            self.assertViewAllowed(self.page, user)
        self.assertEqual(get_visible_nodes(request, [self.page], self.site), [self.page])

@override_settings(CMS_PERMISSION=True, CMS_PUBLIC_FOR='none')
class UnrestrictedViewPermissionTests(ViewPermissionBaseTests):
    """
        Test functionality with CMS_PERMISSION set to True but no restrictions
        apply to this specific page
    """

    def test_unauth_non_access(self):
        if False:
            print('Hello World!')
        request = self.get_request()
        with self.assertNumQueries(1):
            '\n            The query is:\n            PagePermission query for the affected page (is the page restricted?)\n            '
            self.assertViewNotAllowed(self.page)
        self.assertEqual(get_visible_nodes(request, [self.page], self.site), [])

    def test_global_access(self):
        if False:
            for i in range(10):
                print('nop')
        user = self.get_standard_user()
        GlobalPagePermission.objects.create(can_view=True, user=user)
        request = self.get_request(user)
        with self.assertNumQueries(4):
            'The queries are:\n            PagePermission query for the affected page (is the page restricted?)\n            Generic django permission lookup\n            content type lookup by permission lookup\n            GlobalPagePermission query for the page site\n            '
            self.assertViewAllowed(self.page, user)
        self.assertEqual(get_visible_nodes(request, [self.page], self.site), [self.page])

    def test_normal_denied(self):
        if False:
            while True:
                i = 10
        user = self.get_standard_user()
        request = self.get_request(user)
        with self.assertNumQueries(4):
            '\n            The queries are:\n            PagePermission query for the affected page (is the page restricted?)\n            GlobalPagePermission query for the page site\n            User permissions query\n            Content type query\n            '
            self.assertViewNotAllowed(self.page, user)
        self.assertEqual(get_visible_nodes(request, [self.page], self.site), [])

@override_settings(CMS_PERMISSION=True, CMS_PUBLIC_FOR='all')
class RestrictedViewPermissionTests(ViewPermissionBaseTests):
    """
    Test functionality with CMS_PERMISSION set to True and view restrictions
    apply to this specific page
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.group = Group.objects.create(name='testgroup')
        self.pages = [self.page]
        self.expected = [self.page]
        PagePermission.objects.create(page=self.page, group=self.group, can_view=True, grant_on=ACCESS_PAGE)

    def test_unauthed(self):
        if False:
            for i in range(10):
                print('nop')
        request = self.get_request()
        with self.assertNumQueries(1):
            'The queries are:\n            PagePermission query for the affected page (is the page restricted?)\n            '
            self.assertViewNotAllowed(self.page)
        self.assertEqual(get_visible_nodes(request, self.pages, self.site), [])

    def test_page_permissions(self):
        if False:
            return 10
        user = self.get_standard_user()
        request = self.get_request(user)
        PagePermission.objects.create(can_view=True, user=user, page=self.page, grant_on=ACCESS_PAGE)
        with self.assertNumQueries(6):
            '\n            The queries are:\n            PagePermission query (is this page restricted)\n            content type lookup (x2)\n            GlobalpagePermission query for user\n            TreeNode lookup\n            PagePermission query for this user\n            '
            self.assertViewAllowed(self.page, user)
        self.assertEqual(get_visible_nodes(request, self.pages, self.site), self.expected)

    def test_page_group_permissions(self):
        if False:
            while True:
                i = 10
        user = self.get_standard_user()
        user.groups.add(self.group)
        request = self.get_request(user)
        with self.assertNumQueries(6):
            '\n                The queries are:\n                PagePermission query (is this page restricted)\n                content type lookup (x2)\n                GlobalpagePermission query for user\n                TreeNode lookup\n                PagePermission query for user\n            '
            self.assertViewAllowed(self.page, user)
        self.assertEqual(get_visible_nodes(request, self.pages, self.site), self.expected)

    def test_global_permission(self):
        if False:
            i = 10
            return i + 15
        user = self.get_standard_user()
        GlobalPagePermission.objects.create(can_view=True, user=user)
        request = self.get_request(user)
        with self.assertNumQueries(4):
            '\n            The queries are:\n            PagePermission query (is this page restricted)\n            Generic django permission lookup\n            content type lookup by permission lookup\n            GlobalpagePermission query for user\n            '
            self.assertViewAllowed(self.page, user)
        self.assertEqual(get_visible_nodes(request, self.pages, self.site), self.expected)

    def test_basic_perm_denied(self):
        if False:
            print('Hello World!')
        user = self.get_staff_user_with_no_permissions()
        request = self.get_request(user)
        with self.assertNumQueries(6):
            '\n            The queries are:\n            PagePermission query (is this page restricted)\n            content type lookup x2\n            GlobalpagePermission query for user\n            TreeNode lookup\n            PagePermission query for this user\n            '
            self.assertViewNotAllowed(self.page, user)
        self.assertEqual(get_visible_nodes(request, self.pages, self.site), [])

    def test_basic_perm(self):
        if False:
            for i in range(10):
                print('nop')
        user = self.get_standard_user()
        user.user_permissions.add(Permission.objects.get(codename='view_page'))
        request = self.get_request(user)
        with self.assertNumQueries(3):
            '\n            The queries are:\n            PagePermission query (is this page restricted)\n            Generic django permission lookup\n            content type lookup by permission lookup\n            '
            self.assertViewAllowed(self.page, user)
        self.assertEqual(get_visible_nodes(request, self.pages, self.site), self.expected)

class PublicViewPermissionTests(RestrictedViewPermissionTests):
    """ Run the same tests as before, but on the public page instead. """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.page.publish('en')
        self.pages = [self.page.publisher_public]
        self.expected = [self.page.publisher_public]

class GlobalPermissionTests(CMSTestCase):

    def test_emulate_admin_index(self):
        if False:
            while True:
                i = 10
        " Call methods that emulate the adminsite instance's index.\n        This test was basically the reason for the new manager, in light of the\n        problem highlighted in ticket #1120, which asserts that giving a user\n        no site-specific rights when creating a GlobalPagePermission should\n        allow access to all sites.\n        "
        superuser = self._create_user('super', is_staff=True, is_active=True, is_superuser=True)
        superuser.set_password('super')
        superuser.save()
        site_1 = Site.objects.get(pk=1)
        site_2 = Site.objects.create(domain='example2.com', name='example2.com')
        SITES = [site_1, site_2]
        USERS = [self._create_user('staff', is_staff=True, is_active=True), self._create_user('staff_2', is_staff=True, is_active=True)]
        for user in USERS:
            user.set_password('staff')
            save_permissions({'can_add_page': True, 'can_change_page': True, 'can_delete_page': False}, user)
        GlobalPagePermission.objects.create(can_add=True, can_change=True, can_delete=False, user=USERS[0])
        self.assertEqual(1, GlobalPagePermission.objects.with_user(USERS[0]).count())
        GlobalPagePermission.objects.create(can_add=True, can_change=True, can_delete=False, user=USERS[1]).sites.add(SITES[0])
        self.assertEqual(1, GlobalPagePermission.objects.with_user(USERS[1]).count())
        homepage = create_page(title='master', template='nav_playground.html', language='en', in_navigation=True, slug='/')
        publish_page(page=homepage, user=superuser, language='en')
        with self.settings(CMS_PERMISSION=True):
            request = RequestFactory().get(path='/')
            request.session = {'cms_admin_site': site_1.pk}
            request.current_page = None
            for user in USERS:
                request.user = user
                max_queries = 5
                with self.assertNumQueries(FuzzyInt(3, max_queries)):
                    expected_perms = {'add': True, 'change': True, 'delete': False}
                    expected_perms.update({'view': True})
                    self.assertEqual(expected_perms, site._registry[Page].get_model_perms(request))
            request = RequestFactory().get(path='/')
            request.session = {'cms_admin_site': site_2.pk}
            request.current_page = None
            USERS[0] = self.reload(USERS[0])
            USERS[1] = self.reload(USERS[1])
            with self.assertNumQueries(FuzzyInt(5, 15)):
                request.user = USERS[1]
                expected_perms = {'add': False, 'change': False, 'delete': False}
                expected_perms.update({'view': False})
                self.assertEqual(expected_perms, site._registry[Page].get_model_perms(request))
                request = RequestFactory().get('/', data={'site__exact': site_2.pk})
                request.user = USERS[0]
                request.current_page = None
                request.session = {}
                expected_perms = {'add': True, 'change': True, 'delete': False}
                expected_perms.update({'view': True})
                self.assertEqual(expected_perms, site._registry[Page].get_model_perms(request))

    def test_has_page_add_permission_with_target(self):
        if False:
            return 10
        page = create_page('Test', 'nav_playground.html', 'en')
        user = self._create_user('user')
        request = RequestFactory().get('/', data={'target': page.pk})
        request.session = {}
        request.user = user
        has_perm = site._registry[Page].has_add_permission(request)
        self.assertFalse(has_perm)