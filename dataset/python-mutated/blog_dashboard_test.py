"""Tests for the blog dashboard page."""
from __future__ import annotations
import os
from core import feconf
from core import utils
from core.constants import constants
from core.domain import blog_services
from core.platform import models
from core.tests import test_utils
MYPY = False
if MYPY:
    from mypy_imports import user_models
(user_models,) = models.Registry.import_models([models.Names.USER])

class BlogDashboardPageTests(test_utils.GenericTestBase):
    """Checks the access to the blog dashboard page and its rendering."""

    def test_blog_dashboard_page_access_without_logging_in(self) -> None:
        if False:
            while True:
                i = 10
        'Tests access to the Blog Dashboard page.'
        self.get_html_response('/blog-dashboard', expected_status_int=302)

    def test_blog_dashboard_page_access_without_having_rights(self) -> None:
        if False:
            while True:
                i = 10
        self.signup(self.VIEWER_EMAIL, self.VIEWER_USERNAME)
        self.login(self.VIEWER_EMAIL)
        self.get_html_response('/blog-dashboard', expected_status_int=401)
        self.logout()

    def test_blog_dashboard_page_access_as_blog_admin(self) -> None:
        if False:
            print('Hello World!')
        self.signup(self.BLOG_ADMIN_EMAIL, self.BLOG_ADMIN_USERNAME)
        self.add_user_role(self.BLOG_ADMIN_USERNAME, feconf.ROLE_ID_BLOG_ADMIN)
        self.login(self.BLOG_ADMIN_EMAIL)
        self.get_html_response('/blog-dashboard', expected_status_int=200)
        self.logout()

    def test_blog_dashboard_page_access_as_blog_post_editor(self) -> None:
        if False:
            print('Hello World!')
        self.signup(self.BLOG_EDITOR_EMAIL, self.BLOG_EDITOR_USERNAME)
        self.add_user_role(self.BLOG_EDITOR_USERNAME, feconf.ROLE_ID_BLOG_POST_EDITOR)
        self.login(self.BLOG_EDITOR_EMAIL)
        self.get_html_response('/blog-dashboard', expected_status_int=200)
        self.logout()

class BlogDashboardDataHandlerTests(test_utils.GenericTestBase):
    username = 'user'
    user_email = 'user@example.com'

    def setUp(self) -> None:
        if False:
            return 10
        'Completes the sign-up process for the various users.'
        super().setUp()
        self.signup(self.BLOG_ADMIN_EMAIL, self.BLOG_ADMIN_USERNAME)
        self.signup(self.BLOG_EDITOR_EMAIL, self.BLOG_EDITOR_USERNAME)
        self.signup(self.user_email, self.username)
        self.add_user_role(self.BLOG_ADMIN_USERNAME, feconf.ROLE_ID_BLOG_ADMIN)
        self.add_user_role(self.BLOG_EDITOR_USERNAME, feconf.ROLE_ID_BLOG_POST_EDITOR)
        self.blog_admin_id = self.get_user_id_from_email(self.BLOG_ADMIN_EMAIL)
        self.blog_editor_id = self.get_user_id_from_email(self.BLOG_EDITOR_EMAIL)

    def test_get_dashboard_page_data(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.login(self.BLOG_EDITOR_EMAIL)
        json_response = self.get_json('%s' % feconf.BLOG_DASHBOARD_DATA_URL)
        self.assertEqual(self.BLOG_EDITOR_USERNAME, json_response['author_details']['displayed_author_name'])
        self.assertEqual(json_response['published_blog_post_summary_dicts'], [])
        self.assertEqual(json_response['draft_blog_post_summary_dicts'], [])
        self.logout()
        self.login(self.BLOG_ADMIN_EMAIL)
        json_response = self.get_json('%s' % feconf.BLOG_DASHBOARD_DATA_URL)
        self.assertEqual(self.BLOG_ADMIN_USERNAME, json_response['username'])
        self.assertEqual(json_response['published_blog_post_summary_dicts'], [])
        self.assertEqual(json_response['draft_blog_post_summary_dicts'], [])
        self.logout()
        self.login(self.user_email)
        json_response = self.get_json('%s' % feconf.BLOG_DASHBOARD_DATA_URL, expected_status_int=401)
        self.logout()
        blog_post = blog_services.create_new_blog_post(self.blog_editor_id)
        change_dict: blog_services.BlogPostChangeDict = {'title': 'Sample Title', 'thumbnail_filename': 'thumbnail.svg', 'content': '<p>Hello Bloggers<p>', 'tags': ['Newsletter', 'Learners']}
        self.login(self.BLOG_EDITOR_EMAIL)
        json_response = self.get_json('%s' % feconf.BLOG_DASHBOARD_DATA_URL)
        self.assertEqual(self.BLOG_EDITOR_USERNAME, json_response['username'])
        self.assertEqual(blog_post.id, json_response['draft_blog_post_summary_dicts'][0]['id'])
        blog_services.update_blog_post(blog_post.id, change_dict)
        blog_services.publish_blog_post(blog_post.id)
        json_response = self.get_json('%s' % feconf.BLOG_DASHBOARD_DATA_URL)
        self.assertEqual(self.BLOG_EDITOR_USERNAME, json_response['username'])
        self.assertEqual(blog_post.id, json_response['published_blog_post_summary_dicts'][0]['id'])
        self.assertEqual(change_dict['title'], json_response['published_blog_post_summary_dicts'][0]['title'])
        self.assertEqual(json_response['draft_blog_post_summary_dicts'], [])

    def test_create_new_blog_post(self) -> None:
        if False:
            return 10
        self.login(self.BLOG_EDITOR_EMAIL)
        csrf_token = self.get_new_csrf_token()
        json_response = self.post_json('%s' % feconf.BLOG_DASHBOARD_DATA_URL, {}, csrf_token=csrf_token)
        blog_post_id = json_response['blog_post_id']
        blog_post_rights = blog_services.get_blog_post_rights(blog_post_id)
        self.assertEqual(blog_post_rights.editor_ids, [self.blog_editor_id])
        self.logout()
        self.login(self.user_email)
        json_response = self.post_json('%s' % feconf.BLOG_DASHBOARD_DATA_URL, {}, csrf_token=csrf_token, expected_status_int=401)
        self.logout()

    def test_put_author_data(self) -> None:
        if False:
            i = 10
            return i + 15
        self.login(self.BLOG_EDITOR_EMAIL)
        csrf_token = self.get_new_csrf_token()
        payload = {'displayed_author_name': 'new user name', 'author_bio': 'general oppia user and blog post author'}
        pre_update_author_details = blog_services.get_blog_author_details(self.blog_editor_id).to_dict()
        self.assertEqual(pre_update_author_details['displayed_author_name'], self.BLOG_EDITOR_USERNAME)
        self.assertEqual(pre_update_author_details['author_bio'], '')
        json_response = self.put_json('%s' % feconf.BLOG_DASHBOARD_DATA_URL, payload, csrf_token=csrf_token)
        self.assertEqual(json_response['author_details']['displayed_author_name'], 'new user name')
        self.assertEqual(json_response['author_details']['author_bio'], 'general oppia user and blog post author')
        self.logout()

    def test_put_author_details_with_invalid_author_name(self) -> None:
        if False:
            return 10
        self.login(self.BLOG_EDITOR_EMAIL)
        csrf_token = self.get_new_csrf_token()
        payload = {'displayed_author_name': 1234, 'author_bio': 'general oppia user and blog post author'}
        pre_update_author_details = blog_services.get_blog_author_details(self.blog_editor_id).to_dict()
        self.assertEqual(pre_update_author_details['displayed_author_name'], self.BLOG_EDITOR_USERNAME)
        self.put_json('%s' % feconf.BLOG_DASHBOARD_DATA_URL, payload, csrf_token=csrf_token, expected_status_int=400)

    def test_put_author_details_with_invalid_author_bio(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.login(self.BLOG_EDITOR_EMAIL)
        csrf_token = self.get_new_csrf_token()
        payload = {'displayed_author_name': 'new user', 'author_bio': 1234}
        pre_update_author_details = blog_services.get_blog_author_details(self.blog_editor_id).to_dict()
        self.assertEqual(pre_update_author_details['author_bio'], '')
        self.put_json('%s' % feconf.BLOG_DASHBOARD_DATA_URL, payload, csrf_token=csrf_token, expected_status_int=400)

class BlogPostHandlerTests(test_utils.GenericTestBase):
    username = 'user'
    user_email = 'user@example.com'

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Completes the sign-up process for the various users.'
        super().setUp()
        self.signup(self.BLOG_ADMIN_EMAIL, self.BLOG_ADMIN_USERNAME)
        self.signup(self.BLOG_EDITOR_EMAIL, self.BLOG_EDITOR_USERNAME)
        self.signup(self.user_email, self.username)
        self.add_user_role(self.BLOG_ADMIN_USERNAME, feconf.ROLE_ID_BLOG_ADMIN)
        self.add_user_role(self.BLOG_EDITOR_USERNAME, feconf.ROLE_ID_BLOG_POST_EDITOR)
        self.blog_admin_id = self.get_user_id_from_email(self.BLOG_ADMIN_EMAIL)
        self.blog_editor_id = self.get_user_id_from_email(self.BLOG_EDITOR_EMAIL)
        self.blog_post = blog_services.create_new_blog_post(self.blog_editor_id)

    def test_get_blog_post_editor_page_data(self) -> None:
        if False:
            i = 10
            return i + 15
        self.login(self.BLOG_EDITOR_EMAIL)
        json_response = self.get_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id))
        self.assertEqual(self.BLOG_EDITOR_USERNAME, json_response['username'])
        assert self.blog_post.last_updated is not None
        expected_blog_post_dict = {'id': u'%s' % self.blog_post.id, 'displayed_author_name': self.BLOG_EDITOR_USERNAME, 'title': '', 'content': '', 'tags': [], 'thumbnail_filename': None, 'url_fragment': '', 'published_on': None, 'last_updated': u'%s' % utils.convert_naive_datetime_to_string(self.blog_post.last_updated)}
        self.assertEqual(expected_blog_post_dict, json_response['blog_post_dict'])
        self.assertEqual(10, json_response['max_no_of_tags'])
        self.logout()
        self.login(self.BLOG_ADMIN_EMAIL)
        json_response = self.get_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id))
        self.assertEqual(self.BLOG_EDITOR_USERNAME, json_response['displayed_author_name'])
        expected_blog_post_dict = {'id': u'%s' % self.blog_post.id, 'displayed_author_name': self.BLOG_EDITOR_USERNAME, 'title': '', 'content': '', 'tags': [], 'thumbnail_filename': None, 'url_fragment': '', 'published_on': None, 'last_updated': u'%s' % utils.convert_naive_datetime_to_string(self.blog_post.last_updated)}
        self.assertEqual(expected_blog_post_dict, json_response['blog_post_dict'])
        self.assertEqual(10, json_response['max_no_of_tags'])
        self.logout()
        self.login(self.user_email)
        json_response = self.get_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), expected_status_int=401)
        self.logout()
        self.set_curriculum_admins([self.username])
        self.login(self.user_email)
        json_response = self.get_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), expected_status_int=401)
        self.logout()

    def test_get_blog_post_data_by_invalid_blog_post_id(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.login(self.BLOG_EDITOR_EMAIL)
        self.get_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, '123'), expected_status_int=400)
        self.get_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, '123' * constants.BLOG_POST_ID_LENGTH), expected_status_int=400)
        blog_services.delete_blog_post(self.blog_post.id)
        self.get_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), expected_status_int=404)
        self.logout()

    def test_get_blog_post_data_with_author_account_deleted_by_blog_admin(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        blog_services.create_blog_author_details_model(self.blog_editor_id)
        blog_services.update_blog_author_details(self.blog_editor_id, 'new author name', 'general user bio')
        blog_editor_model = user_models.UserSettingsModel.get_by_id(self.blog_editor_id)
        blog_editor_model.deleted = True
        blog_editor_model.update_timestamps()
        blog_editor_model.put()
        self.login(self.BLOG_ADMIN_EMAIL)
        json_response = self.get_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id))
        self.assertEqual('new author name', json_response['displayed_author_name'])
        assert self.blog_post.last_updated is not None
        expected_blog_post_dict = {'id': u'%s' % self.blog_post.id, 'displayed_author_name': 'new author name', 'title': '', 'content': '', 'tags': [], 'thumbnail_filename': None, 'url_fragment': '', 'published_on': None, 'last_updated': u'%s' % utils.convert_naive_datetime_to_string(self.blog_post.last_updated)}
        self.assertEqual(expected_blog_post_dict, json_response['blog_post_dict'])
        self.assertEqual(10, json_response['max_no_of_tags'])
        self.logout()

    def test_put_blog_post_data(self) -> None:
        if False:
            print('Hello World!')
        self.login(self.BLOG_EDITOR_EMAIL)
        csrf_token = self.get_new_csrf_token()
        payload = {'change_dict': {'title': 'Sample Title', 'content': '<p>Hello<p>', 'tags': ['New lessons', 'Learners'], 'thumbnail_filename': 'file.svg'}, 'new_publish_status': False}
        json_response = self.put_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), payload, csrf_token=csrf_token)
        self.assertEqual(json_response['blog_post']['title'], 'Sample Title')
        blog_post = blog_services.get_blog_post_by_id(self.blog_post.id)
        self.assertEqual(blog_post.thumbnail_filename, 'file.svg')
        self.logout()

    def test_put_blog_post_data_by_invalid_blog_post_id(self) -> None:
        if False:
            i = 10
            return i + 15
        self.login(self.BLOG_EDITOR_EMAIL)
        csrf_token = self.get_new_csrf_token()
        payload = {'change_dict': {'title': 'Sample Title'}, 'new_publish_status': False}
        self.put_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, 123), payload, csrf_token=csrf_token, expected_status_int=400)
        blog_services.delete_blog_post(self.blog_post.id)
        csrf_token = self.get_new_csrf_token()
        self.put_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), payload, csrf_token=csrf_token, expected_status_int=404)

    def test_update_blog_post_with_invalid_change_dict(self) -> None:
        if False:
            i = 10
            return i + 15
        self.login(self.BLOG_EDITOR_EMAIL)
        csrf_token = self.get_new_csrf_token()
        payload = {'change_dict': {'title': 1234}, 'new_publish_status': False}
        response = self.put_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), payload, csrf_token=csrf_token, expected_status_int=400)
        self.assertEqual(response['error'], "Schema validation for 'change_dict' failed: Title should be a string.")

    def test_publishing_unpublishing_blog_post(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.login(self.BLOG_EDITOR_EMAIL)
        csrf_token = self.get_new_csrf_token()
        payload = {'change_dict': {'title': 'Sample Title', 'content': '<p>Hello<p>', 'tags': ['New lessons', 'Learners'], 'thumbnail_filename': 'file.svg'}, 'new_publish_status': True}
        self.put_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), payload, csrf_token=csrf_token)
        blog_post_rights = blog_services.get_blog_post_rights(self.blog_post.id)
        self.assertTrue(blog_post_rights.blog_post_is_published)
        csrf_token = self.get_new_csrf_token()
        payload = {'change_dict': {}, 'new_publish_status': False}
        self.put_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), payload, csrf_token=csrf_token)
        blog_post_rights = blog_services.get_blog_post_rights(self.blog_post.id)
        self.assertFalse(blog_post_rights.blog_post_is_published)

    def test_uploading_thumbnail_with_valid_image(self) -> None:
        if False:
            while True:
                i = 10
        self.login(self.BLOG_EDITOR_EMAIL)
        csrf_token = self.get_new_csrf_token()
        payload = {'thumbnail_filename': 'test_svg.svg'}
        with utils.open_file(os.path.join(feconf.TESTS_DATA_DIR, 'test_svg.svg'), 'rb', encoding=None) as f:
            raw_image = f.read()
        self.post_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), payload, csrf_token=csrf_token, upload_files=[('image', 'unused_filename', raw_image)], expected_status_int=200)
        self.logout()

    def test_updating_blog_post_fails_with_invalid_image(self) -> None:
        if False:
            i = 10
            return i + 15
        self.login(self.BLOG_EDITOR_EMAIL)
        csrf_token = self.get_new_csrf_token()
        payload = {'thumbnail_filename': 'cafe.flac'}
        with utils.open_file(os.path.join(feconf.TESTS_DATA_DIR, 'dummy_large_image.jpg'), 'rb', encoding=None) as f:
            raw_image = f.read()
        json_response = self.post_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), payload, csrf_token=csrf_token, upload_files=[('image', 'unused_filename', raw_image)], expected_status_int=400)
        self.assertEqual(json_response['error'], 'Image exceeds file size limit of 1024 KB.')

    def test_guest_can_not_delete_blog_post(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        response = self.delete_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), expected_status_int=401)
        self.assertEqual(response['error'], 'You must be logged in to access this resource.')

    def test_cannot_delete_invalid_blog_post(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.login(self.BLOG_ADMIN_EMAIL)
        self.delete_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, 123456), expected_status_int=400)
        self.logout()
        self.login(self.BLOG_ADMIN_EMAIL)
        self.delete_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, 'abc123efgH34'), expected_status_int=404)
        self.logout()

    def test_blog_post_handler_delete_by_admin(self) -> None:
        if False:
            return 10
        self.login(self.BLOG_ADMIN_EMAIL)
        self.delete_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), expected_status_int=200)
        self.logout()

    def test_blog_post_handler_delete_by_blog_editor(self) -> None:
        if False:
            return 10
        self.login(self.BLOG_EDITOR_EMAIL)
        self.delete_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), expected_status_int=200)
        self.logout()

    def test_cannot_delete_post_by_blog_editor(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.add_user_role(self.username, feconf.ROLE_ID_BLOG_POST_EDITOR)
        self.login(self.user_email)
        self.delete_json('%s/%s' % (feconf.BLOG_EDITOR_DATA_URL_PREFIX, self.blog_post.id), expected_status_int=401)
        self.logout()

class BlogPostTitleHandlerTest(test_utils.GenericTestBase):
    """Tests for BlogPostTitleHandler."""

    def setUp(self) -> None:
        if False:
            return 10
        'Complete the setup process for testing.'
        super().setUp()
        self.signup(self.BLOG_ADMIN_EMAIL, self.BLOG_ADMIN_USERNAME)
        self.blog_admin_id = self.get_user_id_from_email(self.BLOG_ADMIN_EMAIL)
        self.add_user_role(self.BLOG_ADMIN_USERNAME, feconf.ROLE_ID_BLOG_ADMIN)
        blog_post = blog_services.create_new_blog_post(self.blog_admin_id)
        self.change_dict: blog_services.BlogPostChangeDict = {'title': 'Sample Title', 'thumbnail_filename': 'thumbnail.svg', 'content': '<p>Hello Bloggers<p>', 'tags': ['Newsletter', 'Learners']}
        self.blog_post_id = blog_post.id
        blog_services.update_blog_post(blog_post.id, self.change_dict)
        blog_services.publish_blog_post(blog_post.id)
        self.new_blog_post_id = blog_services.create_new_blog_post(self.blog_admin_id).id

    def test_blog_post_title_handler_when_unique(self) -> None:
        if False:
            return 10
        self.login(self.BLOG_ADMIN_EMAIL)
        params = {'title': 'Sample'}
        json_response = self.get_json('%s/%s' % (feconf.BLOG_TITLE_HANDLER, self.new_blog_post_id), params=params)
        self.assertEqual(json_response['blog_post_exists'], False)

    def test_blog_post_title_handler_when_duplicate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.login(self.BLOG_ADMIN_EMAIL)
        params = {'title': 'Sample Title'}
        json_response = self.get_json('%s/%s' % (feconf.BLOG_TITLE_HANDLER, self.new_blog_post_id), params=params)
        self.assertEqual(json_response['blog_post_exists'], True)