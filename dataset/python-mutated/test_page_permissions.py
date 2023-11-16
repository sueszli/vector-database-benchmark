import json
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import Client, TestCase, override_settings
from django.utils import timezone
from wagtail.models import GroupApprovalTask, GroupPagePermission, Locale, Page, Workflow, WorkflowTask
from wagtail.permission_policies.pages import PagePermissionPolicy
from wagtail.test.testapp.models import BusinessSubIndex, EventIndex, EventPage, SingletonPageViaMaxCount

class TestPagePermission(TestCase):
    fixtures = ['test.json']

    def create_workflow_and_task(self):
        if False:
            return 10
        workflow = Workflow.objects.create(name='test_workflow')
        task_1 = GroupApprovalTask.objects.create(name='test_task_1')
        task_1.groups.add(Group.objects.get(name='Event moderators'))
        WorkflowTask.objects.create(workflow=workflow, task=task_1.task_ptr, sort_order=1)
        return (workflow, task_1)

    def test_nonpublisher_page_permissions(self):
        if False:
            print('Hello World!')
        event_editor = get_user_model().objects.get(email='eventeditor@example.com')
        homepage = Page.objects.get(url_path='/home/')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        unpublished_event_page = EventPage.objects.get(url_path='/home/events/tentative-unpublished-event/')
        someone_elses_event_page = EventPage.objects.get(url_path='/home/events/someone-elses-event/')
        board_meetings_page = BusinessSubIndex.objects.get(url_path='/home/events/businessy-events/board-meetings/')
        homepage_perms = homepage.permissions_for_user(event_editor)
        christmas_page_perms = christmas_page.permissions_for_user(event_editor)
        unpub_perms = unpublished_event_page.permissions_for_user(event_editor)
        someone_elses_event_perms = someone_elses_event_page.permissions_for_user(event_editor)
        board_meetings_perms = board_meetings_page.permissions_for_user(event_editor)
        self.assertFalse(homepage_perms.can_add_subpage())
        self.assertTrue(christmas_page_perms.can_add_subpage())
        self.assertTrue(unpub_perms.can_add_subpage())
        self.assertTrue(someone_elses_event_perms.can_add_subpage())
        self.assertFalse(homepage_perms.can_edit())
        self.assertTrue(christmas_page_perms.can_edit())
        self.assertTrue(unpub_perms.can_edit())
        self.assertFalse(someone_elses_event_perms.can_edit())
        self.assertFalse(homepage_perms.can_delete())
        self.assertFalse(christmas_page_perms.can_delete())
        self.assertTrue(unpub_perms.can_delete())
        self.assertFalse(someone_elses_event_perms.can_delete())
        self.assertFalse(homepage_perms.can_publish())
        self.assertFalse(christmas_page_perms.can_publish())
        self.assertFalse(unpub_perms.can_publish())
        self.assertFalse(homepage_perms.can_unpublish())
        self.assertFalse(christmas_page_perms.can_unpublish())
        self.assertFalse(unpub_perms.can_unpublish())
        self.assertFalse(homepage_perms.can_publish_subpage())
        self.assertFalse(christmas_page_perms.can_publish_subpage())
        self.assertFalse(unpub_perms.can_publish_subpage())
        self.assertFalse(homepage_perms.can_reorder_children())
        self.assertFalse(christmas_page_perms.can_reorder_children())
        self.assertFalse(unpub_perms.can_reorder_children())
        self.assertFalse(homepage_perms.can_move())
        self.assertFalse(christmas_page_perms.can_move())
        self.assertTrue(unpub_perms.can_move())
        self.assertFalse(someone_elses_event_perms.can_move())
        self.assertFalse(christmas_page_perms.can_move_to(unpublished_event_page))
        self.assertTrue(unpub_perms.can_move_to(christmas_page))
        self.assertFalse(unpub_perms.can_move_to(homepage))
        self.assertFalse(unpub_perms.can_move_to(unpublished_event_page))
        self.assertFalse(unpub_perms.can_move_to(board_meetings_page))
        self.assertTrue(board_meetings_perms.can_move())
        self.assertFalse(board_meetings_perms.can_move_to(christmas_page))

    def test_publisher_page_permissions(self):
        if False:
            print('Hello World!')
        event_moderator = get_user_model().objects.get(email='eventmoderator@example.com')
        homepage = Page.objects.get(url_path='/home/')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        unpublished_event_page = EventPage.objects.get(url_path='/home/events/tentative-unpublished-event/')
        board_meetings_page = BusinessSubIndex.objects.get(url_path='/home/events/businessy-events/board-meetings/')
        homepage_perms = homepage.permissions_for_user(event_moderator)
        christmas_page_perms = christmas_page.permissions_for_user(event_moderator)
        unpub_perms = unpublished_event_page.permissions_for_user(event_moderator)
        board_meetings_perms = board_meetings_page.permissions_for_user(event_moderator)
        self.assertFalse(homepage_perms.can_add_subpage())
        self.assertTrue(christmas_page_perms.can_add_subpage())
        self.assertTrue(unpub_perms.can_add_subpage())
        self.assertFalse(homepage_perms.can_edit())
        self.assertTrue(christmas_page_perms.can_edit())
        self.assertTrue(unpub_perms.can_edit())
        self.assertFalse(homepage_perms.can_delete())
        self.assertTrue(christmas_page_perms.can_delete())
        self.assertTrue(unpub_perms.can_delete())
        self.assertFalse(homepage_perms.can_publish())
        self.assertTrue(christmas_page_perms.can_publish())
        self.assertTrue(unpub_perms.can_publish())
        self.assertFalse(homepage_perms.can_unpublish())
        self.assertTrue(christmas_page_perms.can_unpublish())
        self.assertFalse(unpub_perms.can_unpublish())
        self.assertFalse(homepage_perms.can_publish_subpage())
        self.assertTrue(christmas_page_perms.can_publish_subpage())
        self.assertTrue(unpub_perms.can_publish_subpage())
        self.assertFalse(homepage_perms.can_reorder_children())
        self.assertTrue(christmas_page_perms.can_reorder_children())
        self.assertTrue(unpub_perms.can_reorder_children())
        self.assertFalse(homepage_perms.can_move())
        self.assertTrue(christmas_page_perms.can_move())
        self.assertTrue(unpub_perms.can_move())
        self.assertTrue(christmas_page_perms.can_move_to(unpublished_event_page))
        self.assertTrue(unpub_perms.can_move_to(christmas_page))
        self.assertFalse(unpub_perms.can_move_to(homepage))
        self.assertFalse(unpub_perms.can_move_to(unpublished_event_page))
        self.assertFalse(unpub_perms.can_move_to(board_meetings_page))
        self.assertTrue(board_meetings_perms.can_move())
        self.assertFalse(board_meetings_perms.can_move_to(christmas_page))

    def test_publish_page_permissions_without_edit(self):
        if False:
            print('Hello World!')
        event_moderator = get_user_model().objects.get(email='eventmoderator@example.com')
        GroupPagePermission.objects.filter(group__name='Event moderators', permission__codename='change_page').delete()
        homepage = Page.objects.get(url_path='/home/')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        unpublished_event_page = EventPage.objects.get(url_path='/home/events/tentative-unpublished-event/')
        moderator_event_page = EventPage.objects.get(url_path='/home/events/someone-elses-event/')
        homepage_perms = homepage.permissions_for_user(event_moderator)
        christmas_page_perms = christmas_page.permissions_for_user(event_moderator)
        unpub_perms = unpublished_event_page.permissions_for_user(event_moderator)
        moderator_event_perms = moderator_event_page.permissions_for_user(event_moderator)
        self.assertFalse(homepage_perms.can_add_subpage())
        self.assertTrue(christmas_page_perms.can_add_subpage())
        self.assertFalse(christmas_page_perms.can_edit())
        self.assertTrue(moderator_event_perms.can_edit())
        self.assertTrue(moderator_event_perms.can_delete())
        self.assertFalse(christmas_page_perms.can_delete())
        self.assertFalse(unpub_perms.can_delete())
        self.assertFalse(homepage_perms.can_publish())
        self.assertTrue(christmas_page_perms.can_publish())
        self.assertTrue(unpub_perms.can_publish())
        self.assertFalse(homepage_perms.can_unpublish())
        self.assertTrue(christmas_page_perms.can_unpublish())
        self.assertFalse(unpub_perms.can_unpublish())
        self.assertFalse(homepage_perms.can_publish_subpage())
        self.assertTrue(christmas_page_perms.can_publish_subpage())
        self.assertTrue(unpub_perms.can_publish_subpage())
        self.assertFalse(homepage_perms.can_reorder_children())
        self.assertTrue(christmas_page_perms.can_reorder_children())
        self.assertTrue(unpub_perms.can_reorder_children())
        self.assertFalse(homepage_perms.can_move())
        self.assertFalse(christmas_page_perms.can_move())
        self.assertTrue(moderator_event_perms.can_move())
        self.assertFalse(moderator_event_perms.can_move_to(homepage))
        self.assertTrue(moderator_event_perms.can_move_to(unpublished_event_page))

    def test_cannot_bulk_delete_without_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        event_moderator = get_user_model().objects.get(email='eventmoderator@example.com')
        events_page = EventIndex.objects.get(url_path='/home/events/')
        events_perms = events_page.permissions_for_user(event_moderator)
        self.assertFalse(events_perms.can_delete())

    def test_can_bulk_delete_with_permissions(self):
        if False:
            return 10
        event_moderator = get_user_model().objects.get(email='eventmoderator@example.com')
        events_page = EventIndex.objects.get(url_path='/home/events/')
        event_moderators_group = Group.objects.get(name='Event moderators')
        GroupPagePermission.objects.create(group=event_moderators_group, page=events_page, permission_type='bulk_delete')
        events_perms = events_page.permissions_for_user(event_moderator)
        self.assertTrue(events_perms.can_delete())

    def test_need_delete_permission_to_bulk_delete(self):
        if False:
            return 10
        "\n        Having bulk_delete permission is not in itself sufficient to allow deleting pages -\n        you need actual edit permission on the pages too.\n\n        In this test the event editor is given bulk_delete permission, but since their\n        only other permission is 'add', they cannot delete published pages or pages owned\n        by other users, and therefore the bulk deletion cannot happen.\n        "
        event_editor = get_user_model().objects.get(email='eventeditor@example.com')
        events_page = EventIndex.objects.get(url_path='/home/events/')
        event_editors_group = Group.objects.get(name='Event editors')
        GroupPagePermission.objects.create(group=event_editors_group, page=events_page, permission_type='bulk_delete')
        events_perms = events_page.permissions_for_user(event_editor)
        self.assertFalse(events_perms.can_delete())

    def test_inactive_user_has_no_permissions(self):
        if False:
            while True:
                i = 10
        user = get_user_model().objects.get(email='inactiveuser@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        unpublished_event_page = EventPage.objects.get(url_path='/home/events/tentative-unpublished-event/')
        christmas_page_perms = christmas_page.permissions_for_user(user)
        unpub_perms = unpublished_event_page.permissions_for_user(user)
        self.assertFalse(unpub_perms.can_add_subpage())
        self.assertFalse(unpub_perms.can_edit())
        self.assertFalse(unpub_perms.can_delete())
        self.assertFalse(unpub_perms.can_publish())
        self.assertFalse(christmas_page_perms.can_unpublish())
        self.assertFalse(unpub_perms.can_publish_subpage())
        self.assertFalse(unpub_perms.can_reorder_children())
        self.assertFalse(unpub_perms.can_move())
        self.assertFalse(unpub_perms.can_move_to(christmas_page))

    def test_superuser_has_full_permissions(self):
        if False:
            i = 10
            return i + 15
        user = get_user_model().objects.get(email='superuser@example.com')
        homepage = Page.objects.get(url_path='/home/').specific
        root = Page.objects.get(url_path='/').specific
        unpublished_event_page = EventPage.objects.get(url_path='/home/events/tentative-unpublished-event/')
        board_meetings_page = BusinessSubIndex.objects.get(url_path='/home/events/businessy-events/board-meetings/')
        homepage_perms = homepage.permissions_for_user(user)
        root_perms = root.permissions_for_user(user)
        unpub_perms = unpublished_event_page.permissions_for_user(user)
        board_meetings_perms = board_meetings_page.permissions_for_user(user)
        self.assertTrue(homepage_perms.can_add_subpage())
        self.assertTrue(root_perms.can_add_subpage())
        self.assertTrue(homepage_perms.can_edit())
        self.assertFalse(root_perms.can_edit())
        self.assertTrue(homepage_perms.can_delete())
        self.assertFalse(root_perms.can_delete())
        self.assertTrue(homepage_perms.can_publish())
        self.assertFalse(root_perms.can_publish())
        self.assertTrue(homepage_perms.can_unpublish())
        self.assertFalse(root_perms.can_unpublish())
        self.assertFalse(unpub_perms.can_unpublish())
        self.assertTrue(homepage_perms.can_publish_subpage())
        self.assertTrue(root_perms.can_publish_subpage())
        self.assertTrue(homepage_perms.can_reorder_children())
        self.assertTrue(root_perms.can_reorder_children())
        self.assertTrue(homepage_perms.can_move())
        self.assertFalse(root_perms.can_move())
        self.assertTrue(homepage_perms.can_move_to(root))
        self.assertFalse(homepage_perms.can_move_to(unpublished_event_page))
        self.assertFalse(unpub_perms.can_move_to(board_meetings_page))
        self.assertTrue(board_meetings_perms.can_move())
        self.assertFalse(board_meetings_perms.can_move_to(unpublished_event_page))

    def test_cant_move_pages_between_locales(self):
        if False:
            i = 10
            return i + 15
        user = get_user_model().objects.get(email='superuser@example.com')
        homepage = Page.objects.get(url_path='/home/').specific
        root = Page.objects.get(url_path='/').specific
        fr_locale = Locale.objects.create(language_code='fr')
        fr_page = root.add_child(instance=Page(title='French page', slug='french-page', locale=fr_locale))
        fr_homepage = root.add_child(instance=Page(title='French homepage', slug='french-homepage', locale=fr_locale))
        french_page_perms = fr_page.permissions_for_user(user)
        self.assertFalse(french_page_perms.can_move_to(homepage))
        self.assertTrue(french_page_perms.can_move_to(fr_homepage))
        self.assertTrue(french_page_perms.can_move_to(root))
        events_index = Page.objects.get(url_path='/home/events/')
        events_index_perms = events_index.permissions_for_user(user)
        self.assertTrue(events_index_perms.can_move_to(root))

    def test_editable_pages_for_user_with_add_permission(self):
        if False:
            print('Hello World!')
        event_editor = get_user_model().objects.get(email='eventeditor@example.com')
        homepage = Page.objects.get(url_path='/home/')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        unpublished_event_page = EventPage.objects.get(url_path='/home/events/tentative-unpublished-event/')
        someone_elses_event_page = EventPage.objects.get(url_path='/home/events/someone-elses-event/')
        policy = PagePermissionPolicy()
        editable_pages = policy.instances_user_has_permission_for(event_editor, 'change')
        can_edit_pages = policy.user_has_permission(event_editor, 'change')
        publishable_pages = policy.instances_user_has_permission_for(event_editor, 'publish')
        can_publish_pages = policy.user_has_permission(event_editor, 'publish')
        self.assertFalse(editable_pages.filter(id=homepage.id).exists())
        self.assertTrue(editable_pages.filter(id=christmas_page.id).exists())
        self.assertTrue(editable_pages.filter(id=unpublished_event_page.id).exists())
        self.assertFalse(editable_pages.filter(id=someone_elses_event_page.id).exists())
        self.assertTrue(can_edit_pages)
        self.assertFalse(publishable_pages.filter(id=homepage.id).exists())
        self.assertFalse(publishable_pages.filter(id=christmas_page.id).exists())
        self.assertFalse(publishable_pages.filter(id=unpublished_event_page.id).exists())
        self.assertFalse(publishable_pages.filter(id=someone_elses_event_page.id).exists())
        self.assertFalse(can_publish_pages)

    def test_explorable_pages(self):
        if False:
            print('Hello World!')
        event_editor = get_user_model().objects.get(email='eventeditor@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        unpublished_event_page = EventPage.objects.get(url_path='/home/events/tentative-unpublished-event/')
        someone_elses_event_page = EventPage.objects.get(url_path='/home/events/someone-elses-event/')
        about_us_page = Page.objects.get(url_path='/home/about-us/')
        policy = PagePermissionPolicy()
        explorable_pages = policy.explorable_instances(event_editor)
        self.assertTrue(explorable_pages.filter(id=christmas_page.id).exists())
        self.assertTrue(explorable_pages.filter(id=unpublished_event_page.id).exists())
        self.assertTrue(explorable_pages.filter(id=someone_elses_event_page.id).exists())
        self.assertFalse(explorable_pages.filter(id=about_us_page.id).exists())

    def test_explorable_pages_in_explorer(self):
        if False:
            print('Hello World!')
        event_editor = get_user_model().objects.get(email='eventeditor@example.com')
        client = Client()
        client.force_login(event_editor)
        homepage = Page.objects.get(url_path='/home/')
        explorer_response = client.get(f'/admin/api/main/pages/?child_of={homepage.pk}&for_explorer=1')
        explorer_json = json.loads(explorer_response.content.decode('utf-8'))
        events_page = Page.objects.get(url_path='/home/events/')
        about_us_page = Page.objects.get(url_path='/home/about-us/')
        explorable_titles = [t.get('title') for t in explorer_json.get('items')]
        self.assertIn(events_page.title, explorable_titles)
        self.assertNotIn(about_us_page.title, explorable_titles)

    def test_explorable_pages_with_permission_gap_in_hierarchy(self):
        if False:
            for i in range(10):
                print('nop')
        corporate_editor = get_user_model().objects.get(email='corporateeditor@example.com')
        policy = PagePermissionPolicy()
        about_us_page = Page.objects.get(url_path='/home/about-us/')
        businessy_events = Page.objects.get(url_path='/home/events/businessy-events/')
        events_page = Page.objects.get(url_path='/home/events/')
        explorable_pages = policy.explorable_instances(corporate_editor)
        self.assertTrue(explorable_pages.filter(id=about_us_page.id).exists())
        self.assertTrue(explorable_pages.filter(id=businessy_events.id).exists())
        self.assertTrue(explorable_pages.filter(id=events_page.id).exists())

    def test_editable_pages_for_user_with_edit_permission(self):
        if False:
            for i in range(10):
                print('nop')
        event_moderator = get_user_model().objects.get(email='eventmoderator@example.com')
        homepage = Page.objects.get(url_path='/home/')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        unpublished_event_page = EventPage.objects.get(url_path='/home/events/tentative-unpublished-event/')
        someone_elses_event_page = EventPage.objects.get(url_path='/home/events/someone-elses-event/')
        policy = PagePermissionPolicy()
        editable_pages = policy.instances_user_has_permission_for(event_moderator, 'change')
        can_edit_pages = policy.user_has_permission(event_moderator, 'change')
        publishable_pages = policy.instances_user_has_permission_for(event_moderator, 'publish')
        can_publish_pages = policy.user_has_permission(event_moderator, 'publish')
        self.assertFalse(editable_pages.filter(id=homepage.id).exists())
        self.assertTrue(editable_pages.filter(id=christmas_page.id).exists())
        self.assertTrue(editable_pages.filter(id=unpublished_event_page.id).exists())
        self.assertTrue(editable_pages.filter(id=someone_elses_event_page.id).exists())
        self.assertTrue(can_edit_pages)
        self.assertFalse(publishable_pages.filter(id=homepage.id).exists())
        self.assertTrue(publishable_pages.filter(id=christmas_page.id).exists())
        self.assertTrue(publishable_pages.filter(id=unpublished_event_page.id).exists())
        self.assertTrue(publishable_pages.filter(id=someone_elses_event_page.id).exists())
        self.assertTrue(can_publish_pages)

    def test_editable_pages_for_inactive_user(self):
        if False:
            i = 10
            return i + 15
        user = get_user_model().objects.get(email='inactiveuser@example.com')
        homepage = Page.objects.get(url_path='/home/')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        unpublished_event_page = EventPage.objects.get(url_path='/home/events/tentative-unpublished-event/')
        someone_elses_event_page = EventPage.objects.get(url_path='/home/events/someone-elses-event/')
        policy = PagePermissionPolicy()
        editable_pages = policy.instances_user_has_permission_for(user, 'change')
        can_edit_pages = policy.user_has_permission(user, 'change')
        publishable_pages = policy.instances_user_has_permission_for(user, 'publish')
        can_publish_pages = policy.user_has_permission(user, 'publish')
        self.assertFalse(editable_pages.filter(id=homepage.id).exists())
        self.assertFalse(editable_pages.filter(id=christmas_page.id).exists())
        self.assertFalse(editable_pages.filter(id=unpublished_event_page.id).exists())
        self.assertFalse(editable_pages.filter(id=someone_elses_event_page.id).exists())
        self.assertFalse(can_edit_pages)
        self.assertFalse(publishable_pages.filter(id=homepage.id).exists())
        self.assertFalse(publishable_pages.filter(id=christmas_page.id).exists())
        self.assertFalse(publishable_pages.filter(id=unpublished_event_page.id).exists())
        self.assertFalse(publishable_pages.filter(id=someone_elses_event_page.id).exists())
        self.assertFalse(can_publish_pages)

    def test_editable_pages_for_superuser(self):
        if False:
            for i in range(10):
                print('nop')
        user = get_user_model().objects.get(email='superuser@example.com')
        homepage = Page.objects.get(url_path='/home/')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        unpublished_event_page = EventPage.objects.get(url_path='/home/events/tentative-unpublished-event/')
        someone_elses_event_page = EventPage.objects.get(url_path='/home/events/someone-elses-event/')
        policy = PagePermissionPolicy()
        editable_pages = policy.instances_user_has_permission_for(user, 'change')
        can_edit_pages = policy.user_has_permission(user, 'change')
        publishable_pages = policy.instances_user_has_permission_for(user, 'publish')
        can_publish_pages = policy.user_has_permission(user, 'publish')
        self.assertTrue(editable_pages.filter(id=homepage.id).exists())
        self.assertTrue(editable_pages.filter(id=christmas_page.id).exists())
        self.assertTrue(editable_pages.filter(id=unpublished_event_page.id).exists())
        self.assertTrue(editable_pages.filter(id=someone_elses_event_page.id).exists())
        self.assertTrue(can_edit_pages)
        self.assertTrue(publishable_pages.filter(id=homepage.id).exists())
        self.assertTrue(publishable_pages.filter(id=christmas_page.id).exists())
        self.assertTrue(publishable_pages.filter(id=unpublished_event_page.id).exists())
        self.assertTrue(publishable_pages.filter(id=someone_elses_event_page.id).exists())
        self.assertTrue(can_publish_pages)

    def test_editable_pages_for_non_editing_user(self):
        if False:
            i = 10
            return i + 15
        user = get_user_model().objects.get(email='admin_only_user@example.com')
        homepage = Page.objects.get(url_path='/home/')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        unpublished_event_page = EventPage.objects.get(url_path='/home/events/tentative-unpublished-event/')
        someone_elses_event_page = EventPage.objects.get(url_path='/home/events/someone-elses-event/')
        policy = PagePermissionPolicy()
        editable_pages = policy.instances_user_has_permission_for(user, 'change')
        can_edit_pages = policy.user_has_permission(user, 'change')
        publishable_pages = policy.instances_user_has_permission_for(user, 'publish')
        can_publish_pages = policy.user_has_permission(user, 'publish')
        self.assertFalse(editable_pages.filter(id=homepage.id).exists())
        self.assertFalse(editable_pages.filter(id=christmas_page.id).exists())
        self.assertFalse(editable_pages.filter(id=unpublished_event_page.id).exists())
        self.assertFalse(editable_pages.filter(id=someone_elses_event_page.id).exists())
        self.assertFalse(can_edit_pages)
        self.assertFalse(publishable_pages.filter(id=homepage.id).exists())
        self.assertFalse(publishable_pages.filter(id=christmas_page.id).exists())
        self.assertFalse(publishable_pages.filter(id=unpublished_event_page.id).exists())
        self.assertFalse(publishable_pages.filter(id=someone_elses_event_page.id).exists())
        self.assertFalse(can_publish_pages)

    def test_lock_page_for_superuser(self):
        if False:
            for i in range(10):
                print('nop')
        user = get_user_model().objects.get(email='superuser@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        locked_page = Page.objects.get(url_path='/home/my-locked-page/')
        perms = christmas_page.permissions_for_user(user)
        locked_perms = locked_page.permissions_for_user(user)
        self.assertTrue(perms.can_lock())
        self.assertFalse(locked_perms.can_unpublish())
        self.assertTrue(perms.can_unlock())

    def test_lock_page_for_moderator(self):
        if False:
            return 10
        user = get_user_model().objects.get(email='eventmoderator@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        perms = christmas_page.permissions_for_user(user)
        self.assertTrue(perms.can_lock())
        self.assertTrue(perms.can_unlock())

    def test_lock_page_for_moderator_without_unlock_permission(self):
        if False:
            i = 10
            return i + 15
        user = get_user_model().objects.get(email='eventmoderator@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        GroupPagePermission.objects.filter(group__name='Event moderators', permission__codename='unlock_page').delete()
        perms = christmas_page.permissions_for_user(user)
        self.assertTrue(perms.can_lock())
        self.assertFalse(perms.can_unlock())

    def test_lock_page_for_moderator_whole_locked_page_without_unlock_permission(self):
        if False:
            print('Hello World!')
        user = get_user_model().objects.get(email='eventmoderator@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        christmas_page.locked = True
        christmas_page.locked_by = user
        christmas_page.locked_at = timezone.now()
        christmas_page.save()
        GroupPagePermission.objects.filter(group__name='Event moderators', permission__codename='unlock_page').delete()
        perms = christmas_page.permissions_for_user(user)
        self.assertTrue(perms.can_lock())
        self.assertTrue(perms.can_unlock())

    def test_lock_page_for_editor(self):
        if False:
            i = 10
            return i + 15
        user = get_user_model().objects.get(email='eventeditor@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        perms = christmas_page.permissions_for_user(user)
        self.assertFalse(perms.can_lock())
        self.assertFalse(perms.can_unlock())

    def test_lock_page_for_non_editing_user(self):
        if False:
            for i in range(10):
                print('nop')
        user = get_user_model().objects.get(email='admin_only_user@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        perms = christmas_page.permissions_for_user(user)
        self.assertFalse(perms.can_lock())
        self.assertFalse(perms.can_unlock())

    def test_lock_page_for_editor_with_lock_permission(self):
        if False:
            print('Hello World!')
        user = get_user_model().objects.get(email='eventeditor@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        GroupPagePermission.objects.create(group=Group.objects.get(name='Event editors'), page=christmas_page, permission_type='lock')
        perms = christmas_page.permissions_for_user(user)
        self.assertTrue(perms.can_lock())
        self.assertFalse(perms.can_unlock())

    def test_page_locked_for_unlocked_page(self):
        if False:
            while True:
                i = 10
        user = get_user_model().objects.get(email='eventmoderator@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        perms = christmas_page.permissions_for_user(user)
        self.assertFalse(perms.page_locked())

    def test_page_locked_for_locked_page(self):
        if False:
            while True:
                i = 10
        user = get_user_model().objects.get(email='eventmoderator@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        christmas_page.locked = True
        christmas_page.locked_by = user
        christmas_page.locked_at = timezone.now()
        christmas_page.save()
        perms = christmas_page.permissions_for_user(user)
        self.assertFalse(perms.page_locked())
        other_user = get_user_model().objects.get(email='eventeditor@example.com')
        other_perms = christmas_page.permissions_for_user(other_user)
        self.assertTrue(other_perms.page_locked())

    @override_settings(WAGTAILADMIN_GLOBAL_EDIT_LOCK=True)
    def test_page_locked_for_locked_page_with_global_lock_enabled(self):
        if False:
            print('Hello World!')
        user = get_user_model().objects.get(email='eventmoderator@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        christmas_page.locked = True
        christmas_page.locked_by = user
        christmas_page.locked_at = timezone.now()
        christmas_page.save()
        perms = christmas_page.permissions_for_user(user)
        self.assertTrue(perms.page_locked())
        other_user = get_user_model().objects.get(email='eventeditor@example.com')
        other_perms = christmas_page.permissions_for_user(other_user)
        self.assertTrue(other_perms.page_locked())

    def test_page_locked_in_workflow(self):
        if False:
            while True:
                i = 10
        (workflow, task) = self.create_workflow_and_task()
        editor = get_user_model().objects.get(email='eventeditor@example.com')
        moderator = get_user_model().objects.get(email='eventmoderator@example.com')
        superuser = get_user_model().objects.get(email='superuser@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        christmas_page.save_revision()
        workflow.start(christmas_page, editor)
        moderator_perms = christmas_page.permissions_for_user(moderator)
        self.assertFalse(moderator_perms.page_locked())
        superuser_perms = christmas_page.permissions_for_user(superuser)
        self.assertFalse(superuser_perms.page_locked())
        editor_perms = christmas_page.permissions_for_user(editor)
        self.assertTrue(editor_perms.page_locked())

    def test_page_lock_in_workflow(self):
        if False:
            return 10
        (workflow, task) = self.create_workflow_and_task()
        editor = get_user_model().objects.get(email='eventeditor@example.com')
        moderator = get_user_model().objects.get(email='eventmoderator@example.com')
        christmas_page = EventPage.objects.get(url_path='/home/events/christmas/')
        christmas_page.save_revision()
        workflow.start(christmas_page, editor)
        moderator_perms = christmas_page.permissions_for_user(moderator)
        self.assertTrue(moderator_perms.can_lock())
        self.assertFalse(moderator_perms.can_unlock())
        editor_perms = christmas_page.permissions_for_user(editor)
        self.assertFalse(editor_perms.can_lock())
        self.assertFalse(editor_perms.can_unlock())

class TestPagePermissionTesterCanCopyTo(TestCase):
    """Tests PagePermissionTester.can_copy_to()"""
    fixtures = ['test.json']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.board_meetings_page = BusinessSubIndex.objects.get(url_path='/home/events/businessy-events/board-meetings/')
        self.event_page = EventPage.objects.get(url_path='/home/events/christmas/')
        homepage = Page.objects.get(url_path='/home/')
        self.singleton_page = SingletonPageViaMaxCount(title='there can be only one')
        homepage.add_child(instance=self.singleton_page)

    def test_inactive_user_cannot_copy_any_pages(self):
        if False:
            print('Hello World!')
        user = get_user_model().objects.get(email='inactiveuser@example.com')
        board_meetings_page_perms = self.board_meetings_page.permissions_for_user(user)
        event_page_perms = self.event_page.permissions_for_user(user)
        singleton_page_perms = self.singleton_page.permissions_for_user(user)
        self.assertFalse(event_page_perms.can_copy_to(self.event_page.get_parent()))
        self.assertFalse(board_meetings_page_perms.can_copy_to(self.board_meetings_page.get_parent()))
        self.assertFalse(singleton_page_perms.can_copy_to(self.singleton_page.get_parent()))

    def test_no_permissions_admin_cannot_copy_any_pages(self):
        if False:
            return 10
        user = get_user_model().objects.get(email='admin_only_user@example.com')
        board_meetings_page_perms = self.board_meetings_page.permissions_for_user(user)
        event_page_perms = self.event_page.permissions_for_user(user)
        singleton_page_perms = self.singleton_page.permissions_for_user(user)
        self.assertFalse(event_page_perms.can_copy_to(self.event_page.get_parent()))
        self.assertFalse(board_meetings_page_perms.can_copy_to(self.board_meetings_page.get_parent()))
        self.assertFalse(singleton_page_perms.can_copy_to(self.singleton_page.get_parent()))

    def test_event_moderator_cannot_copy_a_singleton_page(self):
        if False:
            while True:
                i = 10
        user = get_user_model().objects.get(email='eventmoderator@example.com')
        board_meetings_page_perms = self.board_meetings_page.permissions_for_user(user)
        event_page_perms = self.event_page.permissions_for_user(user)
        singleton_page_perms = self.singleton_page.permissions_for_user(user)
        self.assertTrue(event_page_perms.can_copy_to(self.event_page.get_parent()))
        self.assertTrue(board_meetings_page_perms.can_copy_to(self.board_meetings_page.get_parent()))
        self.assertFalse(singleton_page_perms.can_copy_to(self.singleton_page.get_parent()))

    def test_not_even_a_superuser_can_copy_a_singleton_page(self):
        if False:
            return 10
        user = get_user_model().objects.get(email='superuser@example.com')
        board_meetings_page_perms = self.board_meetings_page.permissions_for_user(user)
        event_page_perms = self.event_page.permissions_for_user(user)
        singleton_page_perms = self.singleton_page.permissions_for_user(user)
        self.assertTrue(event_page_perms.can_copy_to(self.event_page.get_parent()))
        self.assertTrue(board_meetings_page_perms.can_copy_to(self.board_meetings_page.get_parent()))
        self.assertFalse(singleton_page_perms.can_copy_to(self.singleton_page.get_parent()))

class TestPagePermissionModel(TestCase):
    fixtures = ['test.json']

    def test_create_with_permission_type_only(self):
        if False:
            print('Hello World!')
        user = get_user_model().objects.get(email='eventmoderator@example.com')
        page = Page.objects.get(url_path='/home/secret-plans/steal-underpants/')
        group_permission = GroupPagePermission.objects.create(group=user.groups.first(), page=page, permission_type='add')
        self.assertEqual(group_permission.permission.codename, 'add_page')