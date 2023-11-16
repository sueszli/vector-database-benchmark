from django.contrib.messages.storage.cookie import CookieStorage
from django.forms.models import model_to_dict
from django.test.utils import override_settings
from cms.models.permissionmodels import PageUserGroup
from cms.test_utils.testcases import CMSTestCase
from cms.utils.urlutils import admin_reverse

class PermissionsOnTestCase(CMSTestCase):

    def _group_exists(self, name=None):
        if False:
            print('Hello World!')
        if not name:
            name = 'Test group'
        return PageUserGroup.objects.filter(name=name).exists()

    def _get_group_data(self, **kwargs):
        if False:
            print('Hello World!')
        data = {'name': 'Test group', 'can_add_page': 'on', 'can_change_page': 'on', 'can_delete_page': 'on'}
        data.update(**kwargs)
        return data

    def _get_group(self, created_by=None):
        if False:
            i = 10
            return i + 15
        if not created_by:
            created_by = self.get_superuser()
        data = {'name': 'Test group', 'created_by': created_by}
        return PageUserGroup.objects.create(**data)

@override_settings(CMS_PERMISSION=True)
class PermissionsOnGlobalTest(PermissionsOnTestCase):

    def test_group_in_admin_index(self):
        if False:
            while True:
                i = 10
        endpoint = admin_reverse('app_list', args=['cms'])
        staff_user = self.get_staff_user_with_no_permissions()
        self.add_permission(staff_user, 'change_pageusergroup')
        self.add_global_permission(staff_user, can_change_permissions=True)
        with self.login_user_context(staff_user):
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, '<a href="/en/admin/cms/pageusergroup/">User groups (page)</a>', html=True)
        endpoint = self.get_admin_url(PageUserGroup, 'changelist')
        with self.login_user_context(staff_user):
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 200)

    def test_group_not_in_admin_index(self):
        if False:
            return 10
        endpoint = admin_reverse('app_list', args=['cms'])
        staff_user = self.get_staff_user_with_no_permissions()
        self.add_permission(staff_user, 'change_pageusergroup')
        self.add_global_permission(staff_user, can_change_permissions=False)
        with self.login_user_context(staff_user):
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 404)
        endpoint = self.get_admin_url(PageUserGroup, 'changelist')
        with self.login_user_context(staff_user):
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 403)

    def test_user_can_add_group(self):
        if False:
            for i in range(10):
                print('nop')
        endpoint = self.get_admin_url(PageUserGroup, 'add')
        redirect_to = admin_reverse('index')
        staff_user = self.get_staff_user_with_no_permissions()
        data = self._get_group_data()
        self.add_permission(staff_user, 'add_pageusergroup')
        self.add_global_permission(staff_user, can_change_permissions=True)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertRedirects(response, redirect_to)
            self.assertTrue(self._group_exists())

    def test_user_cant_add_group(self):
        if False:
            while True:
                i = 10
        endpoint = self.get_admin_url(PageUserGroup, 'add')
        staff_user = self.get_staff_user_with_no_permissions()
        data = self._get_group_data()
        self.add_permission(staff_user, 'add_pageusergroup')
        self.add_global_permission(staff_user, can_change_permissions=False)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertEqual(response.status_code, 403)
            self.assertFalse(self._group_exists())

    def test_user_can_change_group(self):
        if False:
            while True:
                i = 10
        group = self._get_group()
        endpoint = self.get_admin_url(PageUserGroup, 'change', group.pk)
        redirect_to = self.get_admin_url(PageUserGroup, 'changelist')
        staff_user = self.get_staff_user_with_no_permissions()
        data = self._get_group_data(name='New test group')
        self.add_permission(staff_user, 'change_pageusergroup')
        self.add_global_permission(staff_user, can_change_permissions=True)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertRedirects(response, redirect_to)
            self.assertTrue(self._group_exists('New test group'))

    def test_user_cant_change_group(self):
        if False:
            for i in range(10):
                print('nop')
        group = self._get_group()
        endpoint = self.get_admin_url(PageUserGroup, 'change', group.pk)
        staff_user = self.get_staff_user_with_no_permissions()
        data = self._get_group_data(name='New test group')
        self.add_permission(staff_user, 'change_pageusergroup')
        self.add_global_permission(staff_user, can_change_permissions=False)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertEqual(response.status_code, 403)
            self.assertTrue(self._group_exists())

    def test_user_can_delete_group(self):
        if False:
            while True:
                i = 10
        group = self._get_group()
        endpoint = self.get_admin_url(PageUserGroup, 'delete', group.pk)
        redirect_to = admin_reverse('index')
        staff_user = self.get_staff_user_with_no_permissions()
        data = {'post': 'yes'}
        self.add_permission(staff_user, 'delete_group')
        self.add_permission(staff_user, 'delete_pageusergroup')
        self.add_global_permission(staff_user, can_change_permissions=True)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertRedirects(response, redirect_to)
            self.assertFalse(self._group_exists())

    def test_user_cant_delete_group(self):
        if False:
            while True:
                i = 10
        group = self._get_group()
        endpoint = self.get_admin_url(PageUserGroup, 'delete', group.pk)
        staff_user = self.get_staff_user_with_no_permissions()
        data = {'post': 'yes'}
        self.add_permission(staff_user, 'delete_group')
        self.add_permission(staff_user, 'delete_pageusergroup')
        self.add_global_permission(staff_user, can_change_permissions=False)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertEqual(response.status_code, 403)
            self.assertTrue(self._group_exists())

@override_settings(CMS_PERMISSION=True)
class PermissionsOnPageTest(PermissionsOnTestCase):
    """
    Uses PagePermission
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._permissions_page = self.get_permissions_test_page()

    def test_group_in_admin_index(self):
        if False:
            return 10
        endpoint = admin_reverse('app_list', args=['cms'])
        staff_user = self.get_staff_user_with_no_permissions()
        self.add_permission(staff_user, 'change_pageusergroup')
        self.add_page_permission(staff_user, self._permissions_page, can_change_permissions=True)
        with self.login_user_context(staff_user):
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, '<a href="/en/admin/cms/pageusergroup/">User groups (page)</a>', html=True)
        endpoint = self.get_admin_url(PageUserGroup, 'changelist')
        with self.login_user_context(staff_user):
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 200)

    def test_group_not_in_admin_index(self):
        if False:
            i = 10
            return i + 15
        endpoint = admin_reverse('app_list', args=['cms'])
        staff_user = self.get_staff_user_with_no_permissions()
        self.add_permission(staff_user, 'change_pageusergroup')
        self.add_page_permission(staff_user, self._permissions_page, can_change_permissions=False)
        with self.login_user_context(staff_user):
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 404)
        endpoint = self.get_admin_url(PageUserGroup, 'changelist')
        with self.login_user_context(staff_user):
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 403)

    def test_user_can_add_group(self):
        if False:
            print('Hello World!')
        '\n        User can add new groups if can_change_permissions\n        is set to True.\n        '
        endpoint = self.get_admin_url(PageUserGroup, 'add')
        redirect_to = admin_reverse('index')
        staff_user = self.get_staff_user_with_no_permissions()
        data = self._get_group_data()
        self.add_permission(staff_user, 'add_pageusergroup')
        self.add_page_permission(staff_user, self._permissions_page, can_change_permissions=True)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertRedirects(response, redirect_to)
            self.assertTrue(self._group_exists())

    def test_user_cant_add_group(self):
        if False:
            return 10
        "\n        User can't add new groups if can_change_permissions\n        is set to False.\n        "
        endpoint = self.get_admin_url(PageUserGroup, 'add')
        staff_user = self.get_staff_user_with_no_permissions()
        data = self._get_group_data()
        self.add_permission(staff_user, 'add_pageusergroup')
        self.add_page_permission(staff_user, self._permissions_page, can_change_permissions=False)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertEqual(response.status_code, 403)
            self.assertFalse(self._group_exists())

    def test_user_can_change_subordinate_group(self):
        if False:
            i = 10
            return i + 15
        '\n        User can change groups he created if can_change_permissions\n        is set to True.\n        '
        staff_user = self.get_staff_user_with_no_permissions()
        group = self._get_group(created_by=staff_user)
        endpoint = self.get_admin_url(PageUserGroup, 'change', group.pk)
        data = model_to_dict(group)
        data['_continue'] = '1'
        data['name'] = 'New test group'
        self.add_permission(staff_user, 'change_pageusergroup')
        self.add_page_permission(staff_user, self._permissions_page, can_change_permissions=True)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertRedirects(response, endpoint)
            self.assertTrue(self._group_exists('New test group'))

    def test_user_cant_change_subordinate_group(self):
        if False:
            return 10
        '\n        User cant change groups he created if can_change_permissions\n        is set to False.\n        '
        staff_user = self.get_staff_user_with_no_permissions()
        group = self._get_group(created_by=staff_user)
        endpoint = self.get_admin_url(PageUserGroup, 'change', group.pk)
        data = model_to_dict(group)
        data['_continue'] = '1'
        data['name'] = 'New test group'
        self.add_permission(staff_user, 'change_pageusergroup')
        self.add_page_permission(staff_user, self._permissions_page, can_change_permissions=False)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertEqual(response.status_code, 403)
            self.assertFalse(self._group_exists('New test group'))

    def test_user_cant_change_own_group(self):
        if False:
            return 10
        "\n        User cant change a group he's a part of,\n        even with can_change_permissions set to True.\n        "
        group = self._get_group()
        staff_user = self.get_staff_user_with_no_permissions()
        staff_user.groups.add(group)
        endpoint = self.get_admin_url(PageUserGroup, 'change', group.pk)
        redirect_to = admin_reverse('index')
        data = model_to_dict(group)
        data['_continue'] = '1'
        data['name'] = 'New test group'
        self.add_permission(staff_user, 'change_pageusergroup')
        self.add_page_permission(staff_user, self._permissions_page, can_change_permissions=True)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertRedirects(response, redirect_to)
            msgs = CookieStorage(response)._decode(response.cookies['messages'].value)
            self.assertTrue(msgs[0], PageUserGroup._meta.verbose_name)
            self.assertTrue(msgs[0], 'ID "%s"' % group.pk)
            self.assertFalse(self._group_exists('New test group'))

    def test_user_cant_change_others_group(self):
        if False:
            while True:
                i = 10
        '\n        User cant change a group created by another user,\n        even with can_change_permissions set to True.\n        '
        admin = self.get_superuser()
        group = self._get_group(created_by=admin)
        staff_user = self.get_staff_user_with_no_permissions()
        endpoint = self.get_admin_url(PageUserGroup, 'change', group.pk)
        redirect_to = admin_reverse('index')
        data = model_to_dict(group)
        data['_continue'] = '1'
        data['name'] = 'New test group'
        self.add_permission(staff_user, 'change_pageusergroup')
        self.add_page_permission(staff_user, self._permissions_page, can_change_permissions=True)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertRedirects(response, redirect_to)
            msgs = CookieStorage(response)._decode(response.cookies['messages'].value)
            self.assertTrue(msgs[0], PageUserGroup._meta.verbose_name)
            self.assertTrue(msgs[0], 'ID "%s"' % group.pk)
            self.assertFalse(self._group_exists('New test group'))

    def test_user_can_delete_subordinate_group(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        User can delete groups he created if can_change_permissions\n        is set to True.\n        '
        staff_user = self.get_staff_user_with_no_permissions()
        group = self._get_group(created_by=staff_user)
        endpoint = self.get_admin_url(PageUserGroup, 'delete', group.pk)
        redirect_to = admin_reverse('index')
        data = {'post': 'yes'}
        self.add_permission(staff_user, 'delete_group')
        self.add_permission(staff_user, 'delete_pageusergroup')
        self.add_page_permission(staff_user, self._permissions_page, can_change_permissions=True)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertRedirects(response, redirect_to)
            self.assertFalse(self._group_exists())

    def test_user_cant_delete_subordinate_group(self):
        if False:
            return 10
        '\n        User cant delete groups he created if can_change_permissions\n        is set to False.\n        '
        staff_user = self.get_staff_user_with_no_permissions()
        group = self._get_group(created_by=staff_user)
        endpoint = self.get_admin_url(PageUserGroup, 'delete', group.pk)
        data = {'post': 'yes'}
        self.add_permission(staff_user, 'delete_group')
        self.add_permission(staff_user, 'delete_pageusergroup')
        self.add_page_permission(staff_user, self._permissions_page, can_change_permissions=False)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertEqual(response.status_code, 403)
            self.assertTrue(self._group_exists())

    def test_user_cant_delete_own_group(self):
        if False:
            return 10
        "\n        User cant delete a group he's a part of,\n        even with can_change_permissions set to True.\n        "
        group = self._get_group()
        staff_user = self.get_staff_user_with_no_permissions()
        staff_user.groups.add(group)
        endpoint = self.get_admin_url(PageUserGroup, 'delete', group.pk)
        redirect_to = admin_reverse('index')
        data = {'post': 'yes'}
        self.add_permission(staff_user, 'delete_group')
        self.add_permission(staff_user, 'delete_pageusergroup')
        self.add_page_permission(staff_user, self._permissions_page, can_change_permissions=True)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertRedirects(response, redirect_to)
            msgs = CookieStorage(response)._decode(response.cookies['messages'].value)
            self.assertTrue(msgs[0], PageUserGroup._meta.verbose_name)
            self.assertTrue(msgs[0], 'ID "%s"' % group.pk)
            self.assertTrue(self._group_exists())

    def test_user_cant_delete_others_group(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        User cant delete a group created by another user,\n        even with can_change_permissions set to True.\n        '
        admin = self.get_superuser()
        group = self._get_group(created_by=admin)
        staff_user = self.get_staff_user_with_no_permissions()
        endpoint = self.get_admin_url(PageUserGroup, 'delete', group.pk)
        redirect_to = admin_reverse('index')
        data = {'post': 'yes'}
        self.add_permission(staff_user, 'delete_group')
        self.add_permission(staff_user, 'delete_pageusergroup')
        self.add_page_permission(staff_user, self._permissions_page, can_change_permissions=True)
        with self.login_user_context(staff_user):
            response = self.client.post(endpoint, data)
            self.assertRedirects(response, redirect_to)
            msgs = CookieStorage(response)._decode(response.cookies['messages'].value)
            self.assertTrue(msgs[0], PageUserGroup._meta.verbose_name)
            self.assertTrue(msgs[0], 'ID "%s"' % group.pk)
            self.assertTrue(self._group_exists())