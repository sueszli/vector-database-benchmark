from django.contrib.admin.utils import quote
from django.contrib.auth.models import Permission
from django.contrib.contenttypes.models import ContentType
from django.test import TestCase, override_settings
from django.urls import reverse
from wagtail.models import Workflow, WorkflowContentType, WorkflowState
from wagtail.test.testapp.models import FullFeaturedSnippet, ModeratedModel
from wagtail.test.utils import WagtailTestUtils

class BaseWorkflowsTestCase(WagtailTestUtils, TestCase):
    model = FullFeaturedSnippet

    def setUp(self):
        if False:
            while True:
                i = 10
        self.user = self.login()
        self.object = self.model.objects.create(text="I'm a full-featured snippet!")
        self.object.save_revision().publish()
        self.content_type = ContentType.objects.get_for_model(self.model)
        self.workflow = Workflow.objects.first()
        WorkflowContentType.objects.create(content_type=self.content_type, workflow=self.workflow)

    @property
    def model_name(self):
        if False:
            return 10
        return self.model._meta.verbose_name

    def get_url(self, name, args=None):
        if False:
            print('Hello World!')
        args = args if args is not None else [quote(self.object.pk)]
        return reverse(self.object.snippet_viewset.get_url_name(name), args=args)

class TestCreateView(BaseWorkflowsTestCase):

    def get(self):
        if False:
            return 10
        return self.client.get(self.get_url('add', ()))

    def post(self, post_data):
        if False:
            print('Hello World!')
        return self.client.post(self.get_url('add', ()), post_data)

    def test_get_workflow_buttons_shown(self):
        if False:
            print('Hello World!')
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<button type="submit" name="action-submit" value="Submit to Moderators approval" class="button">', count=1)

    @override_settings(WAGTAIL_WORKFLOW_ENABLED=False)
    def test_get_workflow_buttons_not_shown_when_workflow_disabled(self):
        if False:
            return 10
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'name="action-submit"')

    def test_post_submit_for_moderation(self):
        if False:
            i = 10
            return i + 15
        response = self.post({'text': 'Newly created', 'action-submit': 'Submit'})
        object = self.model.objects.get(text='Newly created')
        self.assertRedirects(response, self.get_url('list', ()))
        self.assertIsInstance(object, self.model)
        self.assertEqual(object.text, 'Newly created')
        self.assertFalse(object.live)
        self.assertFalse(object.first_published_at)
        self.assertEqual(object.current_workflow_state.status, WorkflowState.STATUS_IN_PROGRESS)
        self.assertEqual(object.latest_revision.object_str, 'Newly created')
        self.assertEqual(object.current_workflow_task_state.revision, object.latest_revision)

class TestCreateViewNotLockable(TestCreateView):
    model = ModeratedModel

class TestEditView(BaseWorkflowsTestCase):

    def get(self):
        if False:
            while True:
                i = 10
        return self.client.get(self.get_url('edit'))

    def post(self, post_data):
        if False:
            while True:
                i = 10
        return self.client.post(self.get_url('edit'), post_data)

    def test_get_workflow_buttons_shown(self):
        if False:
            while True:
                i = 10
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<button type="submit" name="action-submit" value="Submit to Moderators approval" class="button">', count=1)

    @override_settings(WAGTAIL_WORKFLOW_ENABLED=False)
    def test_get_workflow_buttons_not_shown_when_workflow_disabled(self):
        if False:
            return 10
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'name="action-submit"')

    def test_post_submit_for_moderation(self):
        if False:
            while True:
                i = 10
        response = self.post({'text': 'Edited!', 'action-submit': 'Submit'})
        self.object.refresh_from_db()
        self.assertRedirects(response, self.get_url('list', ()))
        self.assertIsInstance(self.object, self.model)
        self.assertEqual(self.object.text, "I'm a full-featured snippet!")
        self.assertTrue(self.object.live)
        self.assertTrue(self.object.first_published_at)
        self.assertTrue(self.object.has_unpublished_changes)
        self.assertEqual(self.object.current_workflow_state.status, WorkflowState.STATUS_IN_PROGRESS)
        self.assertEqual(self.object.latest_revision.object_str, 'Edited!')
        self.assertEqual(self.object.current_workflow_task_state.revision, self.object.latest_revision)

class TestEditViewNotLockable(TestEditView):
    model = ModeratedModel

class TestWorkflowHistory(BaseWorkflowsTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.object.text = 'Edited!'
        self.object.save_revision()
        self.workflow_state = self.workflow.start(self.object, self.user)

    def test_get_index(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(self.get_url('workflow_history'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/shared/workflow_history/index.html')
        self.assertContains(response, self.get_url('edit'))
        self.assertContains(response, self.get_url('workflow_history_detail', (quote(self.object.pk), self.workflow_state.id)))
        self.assertContains(response, 'Moderators approval')
        self.assertContains(response, 'In progress')
        self.assertContains(response, 'test@email.com')

    def test_get_index_with_bad_permissions(self):
        if False:
            i = 10
            return i + 15
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.get(self.get_url('workflow_history'))
        self.assertRedirects(response, reverse('wagtailadmin_home'))

    def test_get_detail(self):
        if False:
            return 10
        response = self.client.get(self.get_url('workflow_history_detail', (quote(self.object.pk), self.workflow_state.id)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/shared/workflow_history/detail.html')
        self.assertContains(response, self.get_url('edit'))
        self.assertContains(response, self.get_url('workflow_history'))
        self.assertContains(response, '<div class="w-tabs" data-tabs>')
        self.assertContains(response, '<div class="tab-content">')
        self.assertContains(response, 'Tasks')
        self.assertContains(response, 'Timeline')
        self.assertContains(response, 'Edited!')
        self.assertContains(response, 'Moderators approval')
        self.assertContains(response, 'In progress')
        self.assertContains(response, 'test@email.com')

    def test_get_detail_completed(self):
        if False:
            return 10
        self.workflow_state.current_task_state.approve(user=None)
        response = self.client.get(self.get_url('workflow_history_detail', (quote(self.object.pk), self.workflow_state.id)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/shared/workflow_history/detail.html')
        self.assertContains(response, self.get_url('edit'))
        self.assertContains(response, self.get_url('workflow_history'))
        self.assertContains(response, '<div class="w-tabs" data-tabs>')
        self.assertContains(response, '<div class="tab-content">')
        self.assertContains(response, 'Tasks')
        self.assertContains(response, 'Timeline')
        self.assertContains(response, 'Edited!')
        self.assertContains(response, 'Moderators approval')
        self.assertContains(response, 'Workflow completed')
        self.assertContains(response, 'test@email.com')
        self.assertNotContains(response, 'In progress')

    def test_get_detail_with_bad_permissions(self):
        if False:
            print('Hello World!')
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.get(self.get_url('workflow_history_detail', (quote(self.object.pk), self.workflow_state.id)))
        self.assertRedirects(response, reverse('wagtailadmin_home'))

class TestConfirmWorkflowCancellation(BaseWorkflowsTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.object.text = 'Edited!'
        self.object.save_revision()
        self.workflow_state = self.workflow.start(self.object, self.user)

    def test_get_confirm_workflow_cancellation(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(self.get_url('confirm_workflow_cancellation'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/confirm_workflow_cancellation.html')
        self.assertContains(response, 'Publishing this full-featured snippet will cancel the current workflow.')
        self.assertContains(response, 'Would you still like to publish this full-featured snippet?')

    @override_settings(WAGTAIL_WORKFLOW_CANCEL_ON_PUBLISH=False)
    def test_get_confirm_workflow_cancellation_with_disabled_setting(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(self.get_url('confirm_workflow_cancellation'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateNotUsed(response, 'wagtailadmin/generic/confirm_workflow_cancellation.html')
        self.assertJSONEqual(response.content.decode(), {'step': 'no_confirmation_needed'})