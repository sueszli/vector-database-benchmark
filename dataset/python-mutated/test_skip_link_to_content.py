from django.contrib.admin.tests import AdminSeleniumTestCase
from django.contrib.auth.models import User
from django.test import override_settings
from django.urls import reverse

@override_settings(ROOT_URLCONF='admin_views.urls')
class SeleniumTests(AdminSeleniumTestCase):
    available_apps = ['admin_views'] + AdminSeleniumTestCase.available_apps

    def setUp(self):
        if False:
            print('Hello World!')
        self.superuser = User.objects.create_superuser(username='super', password='secret', email='super@example.com')

    def test_use_skip_link_to_content(self):
        if False:
            print('Hello World!')
        from selenium.webdriver.common.action_chains import ActionChains
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        self.admin_login(username='super', password='secret', login_url=reverse('admin:index'))
        skip_link = self.selenium.find_element(By.CLASS_NAME, 'skip-to-content-link')
        self.assertFalse(skip_link.is_displayed())
        body = self.selenium.find_element(By.TAG_NAME, 'body')
        body.send_keys(Keys.TAB)
        self.assertTrue(skip_link.is_displayed())
        skip_link.send_keys(Keys.RETURN)
        self.assertFalse(skip_link.is_displayed())
        keys = [Keys.TAB, Keys.TAB]
        if self.browser == 'firefox':
            keys.remove(Keys.TAB)
        body.send_keys(keys)
        actors_a_tag = self.selenium.find_element(By.LINK_TEXT, 'Actors')
        self.assertEqual(self.selenium.switch_to.active_element, actors_a_tag)
        with self.wait_page_loaded():
            actors_a_tag.send_keys(Keys.RETURN)
        body = self.selenium.find_element(By.TAG_NAME, 'body')
        body.send_keys(Keys.TAB)
        skip_link = self.selenium.find_element(By.CLASS_NAME, 'skip-to-content-link')
        self.assertTrue(skip_link.is_displayed())
        ActionChains(self.selenium).send_keys(Keys.RETURN, Keys.TAB).perform()
        actors_add_url = reverse('admin:admin_views_actor_add')
        actors_a_tag = self.selenium.find_element(By.CSS_SELECTOR, f"#content [href='{actors_add_url}']")
        self.assertEqual(self.selenium.switch_to.active_element, actors_a_tag)
        with self.wait_page_loaded():
            actors_a_tag.send_keys(Keys.RETURN)
        first_input = self.selenium.find_element(By.ID, 'id_name')
        self.assertEqual(self.selenium.switch_to.active_element, first_input)

    def test_dont_use_skip_link_to_content(self):
        if False:
            for i in range(10):
                print('nop')
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        self.admin_login(username='super', password='secret', login_url=reverse('admin:index'))
        skip_link = self.selenium.find_element(By.CLASS_NAME, 'skip-to-content-link')
        self.assertFalse(skip_link.is_displayed())
        body = self.selenium.find_element(By.TAG_NAME, 'body')
        body.send_keys(Keys.TAB)
        self.assertTrue(skip_link.is_displayed())
        body.send_keys(Keys.TAB)
        django_administration_title = self.selenium.find_element(By.LINK_TEXT, 'Django administration')
        self.assertFalse(skip_link.is_displayed())
        self.assertEqual(self.selenium.switch_to.active_element, django_administration_title)

    def test_skip_link_with_RTL_language_doesnt_create_horizontal_scrolling(self):
        if False:
            for i in range(10):
                print('nop')
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        with override_settings(LANGUAGE_CODE='ar'):
            self.admin_login(username='super', password='secret', login_url=reverse('admin:index'))
            skip_link = self.selenium.find_element(By.CLASS_NAME, 'skip-to-content-link')
            body = self.selenium.find_element(By.TAG_NAME, 'body')
            body.send_keys(Keys.TAB)
            self.assertTrue(skip_link.is_displayed())
            is_vertical_scrolleable = self.selenium.execute_script('return arguments[0].scrollHeight > arguments[0].offsetHeight;', body)
            is_horizontal_scrolleable = self.selenium.execute_script('return arguments[0].scrollWeight > arguments[0].offsetWeight;', body)
            self.assertTrue(is_vertical_scrolleable)
            self.assertFalse(is_horizontal_scrolleable)