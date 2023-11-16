from django.contrib.auth.models import Group, Permission
from django.test import TestCase
from django.urls import reverse
from wagtail.documents.models import Document
from wagtail.models import Collection, GroupCollectionPermission, Page, get_root_collection_id
from wagtail.test.utils import WagtailTestUtils

class TestChooser(WagtailTestUtils, TestCase):
    """Test chooser panel rendered by `wagtaildocs_chooser:choose` view"""
    _NO_DOCS_TEXT = "You haven't uploaded any documents."
    _NO_COLLECTION_DOCS_TEXT = "You haven't uploaded any documents in this collection."
    _UPLOAD_ONE_TEXT = 'upload one now'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.root_page = Page.objects.get(id=2)

    def login_as_superuser(self):
        if False:
            return 10
        self.login()

    def login_as_editor(self):
        if False:
            i = 10
            return i + 15
        editors_group = Group.objects.create(name='The Editors')
        access_admin_perm = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        editors_group.permissions.add(access_admin_perm)
        choose_document_permission = Permission.objects.get(content_type__app_label='wagtaildocs', codename='choose_document')
        GroupCollectionPermission.objects.create(group=editors_group, collection=Collection.objects.get(depth=1), permission=choose_document_permission)
        user = self.create_user(username='editor', password='password')
        user.groups.add(editors_group)
        self.login(user)

    def login_as_baker(self):
        if False:
            print('Hello World!')
        bakers_group = Group.objects.create(name='Bakers')
        access_admin_perm = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        bakers_group.permissions.add(access_admin_perm)
        root = Collection.objects.get(id=get_root_collection_id())
        bakery_collection = root.add_child(instance=Collection(name='Bakery'))
        GroupCollectionPermission.objects.create(group=bakers_group, collection=bakery_collection, permission=Permission.objects.get(content_type__app_label='wagtaildocs', codename='choose_document'))
        root.add_child(instance=Collection(name='Office'))
        user = self.create_user(username='baker', password='password')
        user.groups.add(bakers_group)
        self.login(user)

    def get(self, params=None):
        if False:
            i = 10
            return i + 15
        return self.client.get(reverse('wagtaildocs_chooser:choose'), params or {})

    def test_chooser_docs_exist(self):
        if False:
            i = 10
            return i + 15
        self.login_as_editor()
        doc_title = 'document.pdf'
        Document.objects.create(title=doc_title)
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/chooser/chooser.html')
        self.assertContains(response, doc_title)
        self.assertNotContains(response, self._NO_DOCS_TEXT)
        self.assertNotContains(response, self._NO_COLLECTION_DOCS_TEXT)
        self.assertNotContains(response, self._UPLOAD_ONE_TEXT)

    def test_chooser_only_docs_in_chooseable_collection_appear(self):
        if False:
            i = 10
            return i + 15
        self.login_as_baker()
        bun_recipe_title = 'bun_recipe.pdf'
        Document.objects.create(title=bun_recipe_title, collection=Collection.objects.get(name='Bakery'))
        payroll_title = 'payroll.xlsx'
        Document.objects.create(title=payroll_title, collection=Collection.objects.get(name='Office'))
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/chooser/chooser.html')
        self.assertContains(response, bun_recipe_title)
        self.assertNotContains(response, payroll_title)

    def test_chooser_collection_selector_appears_only_if_multiple_collections_are_choosable(self):
        if False:
            i = 10
            return i + 15
        self.login_as_baker()
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/chooser/chooser.html')
        self.assertNotContains(response, 'Collection')
        GroupCollectionPermission.objects.create(group=Group.objects.get(name='Bakers'), collection=Collection.objects.get(name='Office'), permission=Permission.objects.get(content_type__app_label='wagtaildocs', codename='choose_document'))
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/chooser/chooser.html')
        self.assertContains(response, 'Collection')

    def test_chooser_no_docs_upload_allowed(self):
        if False:
            for i in range(10):
                print('nop')
        self.login_as_superuser()
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/chooser/chooser.html')
        self.assertContains(response, self._NO_DOCS_TEXT)
        self.assertContains(response, self._UPLOAD_ONE_TEXT)

    def test_chooser_no_docs_upload_forbidden(self):
        if False:
            i = 10
            return i + 15
        self.login_as_editor()
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/chooser/chooser.html')
        self.assertContains(response, self._NO_DOCS_TEXT)
        self.assertNotContains(response, self._UPLOAD_ONE_TEXT)

    def test_results_docs_exist(self):
        if False:
            print('Hello World!')
        self.login_as_superuser()
        doc_title = 'document.pdf'
        Document.objects.create(title=doc_title)
        response = self.get({'q': ''})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtaildocs/chooser/results.html')
        self.assertContains(response, doc_title)
        self.assertNotContains(response, self._NO_DOCS_TEXT)
        self.assertNotContains(response, self._NO_COLLECTION_DOCS_TEXT)
        self.assertNotContains(response, self._UPLOAD_ONE_TEXT)

    def test_results_no_docs_upload_allowed(self):
        if False:
            while True:
                i = 10
        self.login_as_superuser()
        response = self.get({'q': ''})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtaildocs/chooser/results.html')
        self.assertContains(response, self._NO_DOCS_TEXT)
        self.assertContains(response, self._UPLOAD_ONE_TEXT)

    def test_results_no_docs_upload_forbidden(self):
        if False:
            i = 10
            return i + 15
        self.login_as_editor()
        response = self.get({'q': ''})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtaildocs/chooser/results.html')
        self.assertContains(response, self._NO_DOCS_TEXT)
        self.assertNotContains(response, self._UPLOAD_ONE_TEXT)

    def test_results_no_collection_docs_upload_allowed(self):
        if False:
            for i in range(10):
                print('nop')
        self.login_as_superuser()
        root_id = get_root_collection_id()
        root = Collection.objects.get(id=root_id)
        empty_collection = Collection(name='Nothing to see here')
        root.add_child(instance=empty_collection)
        doc_title = 'document.pdf'
        Document.objects.create(title=doc_title, collection=root)
        response = self.get({'q': '', 'collection_id': empty_collection.id})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtaildocs/chooser/results.html')
        self.assertContains(response, self._NO_COLLECTION_DOCS_TEXT)
        self.assertContains(response, self._UPLOAD_ONE_TEXT)

    def test_results_no_collection_docs_upload_forbidden(self):
        if False:
            return 10
        self.login_as_editor()
        root_id = get_root_collection_id()
        root = Collection.objects.get(id=root_id)
        empty_collection = Collection(name='Nothing to see here')
        root.add_child(instance=empty_collection)
        Document.objects.create(collection=root)
        response = self.get({'q': '', 'collection_id': empty_collection.id})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtaildocs/chooser/results.html')
        self.assertContains(response, self._NO_COLLECTION_DOCS_TEXT)
        self.assertNotContains(response, self._UPLOAD_ONE_TEXT)