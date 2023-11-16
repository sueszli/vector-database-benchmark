import warnings
from contextlib import contextmanager
from typing import Union
from bs4 import BeautifulSoup
from django import VERSION as DJANGO_VERSION
from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import override_settings as django_override_settings
from django.test.testcases import assert_and_parse_html

def override_settings(**kwargs):
    if False:
        print('Hello World!')
    '\n    Decorator that temporarily overrides Django settings,\n    with compatibility shims for old and newer Django versions.\n    '
    DEFAULT_FILE_STORAGE = kwargs.get('DEFAULT_FILE_STORAGE')
    storages = settings.STORAGES
    if DEFAULT_FILE_STORAGE is not None and DJANGO_VERSION >= (4, 2):
        kwargs.pop('DEFAULT_FILE_STORAGE')
        kwargs['STORAGES'] = {**storages, 'default': {'BACKEND': DEFAULT_FILE_STORAGE}}
    STATICFILES_STORAGE = kwargs.get('STATICFILES_STORAGE')
    if STATICFILES_STORAGE is not None and DJANGO_VERSION >= (4, 2):
        kwargs.pop('STATICFILES_STORAGE')
        kwargs['STORAGES'] = {**storages, 'staticfiles': {'BACKEND': STATICFILES_STORAGE}}
    return django_override_settings(**kwargs)

class WagtailTestUtils:

    @staticmethod
    def get_soup(markup: Union[str, bytes], parser='html.parser') -> BeautifulSoup:
        if False:
            return 10
        return BeautifulSoup(markup, parser)

    @staticmethod
    def create_test_user():
        if False:
            for i in range(10):
                print('nop')
        '\n        Override this method to return an instance of your custom user model\n        '
        user_model = get_user_model()
        user_data = {user_model.USERNAME_FIELD: 'test@email.com', 'email': 'test@email.com', 'password': 'password'}
        for field in user_model.REQUIRED_FIELDS:
            if field not in user_data:
                user_data[field] = field
        return user_model.objects.create_superuser(**user_data)

    def login(self, user=None, username=None, password='password'):
        if False:
            for i in range(10):
                print('nop')
        user_model = get_user_model()
        if username is None:
            if user is None:
                user = self.create_test_user()
            username = getattr(user, user_model.USERNAME_FIELD)
        if user_model.USERNAME_FIELD == 'email' and '@' not in username:
            username = '%s@example.com' % username
        self.assertTrue(self.client.login(password=password, **{user_model.USERNAME_FIELD: username}))
        return user

    @staticmethod
    def create_user(username, email=None, password=None, **kwargs):
        if False:
            while True:
                i = 10
        User = get_user_model()
        kwargs['email'] = email or '%s@example.com' % username
        kwargs['password'] = password
        if User.USERNAME_FIELD != 'email':
            kwargs[User.USERNAME_FIELD] = username
        return User.objects.create_user(**kwargs)

    @staticmethod
    def create_superuser(username, email=None, password=None, **kwargs):
        if False:
            print('Hello World!')
        User = get_user_model()
        kwargs['email'] = email or '%s@example.com' % username
        kwargs['password'] = password
        if User.USERNAME_FIELD != 'email':
            kwargs[User.USERNAME_FIELD] = username
        return User.objects.create_superuser(**kwargs)

    @staticmethod
    @contextmanager
    def ignore_deprecation_warnings():
        if False:
            while True:
                i = 10
        with warnings.catch_warnings(record=True) as warning_list:
            yield
        for w in warning_list:
            if not issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning)):
                warnings.showwarning(message=w.message, category=w.category, filename=w.filename, lineno=w.lineno, file=w.file, line=w.line)

    @contextmanager
    def register_hook(self, hook_name, fn, order=0):
        if False:
            while True:
                i = 10
        from wagtail import hooks
        hooks.register(hook_name, fn, order)
        try:
            yield
        finally:
            hooks._hooks[hook_name].remove((fn, order))

    def _tag_is_equal(self, tag1, tag2):
        if False:
            while True:
                i = 10
        if not hasattr(tag1, 'name') or not hasattr(tag2, 'name'):
            return False
        if tag1.name != tag2.name:
            return False
        if len(tag1.attributes) != len(tag2.attributes):
            return False
        if tag1.attributes != tag2.attributes:
            for i in range(len(tag1.attributes)):
                (attr, value) = tag1.attributes[i]
                (other_attr, other_value) = tag2.attributes[i]
                if value is None:
                    value = attr
                if other_value is None:
                    other_value = other_attr
                if attr != other_attr or value != other_value:
                    return False
        return True

    def _tag_matches_with_extra_attrs(self, thin_tag, fat_tag):
        if False:
            return 10
        if not hasattr(thin_tag, 'name') or not hasattr(fat_tag, 'name'):
            return False
        if thin_tag.name != fat_tag.name:
            return False
        for (attr, value) in thin_tag.attributes:
            if value is None:
                if (attr, None) not in fat_tag.attributes and (attr, attr) not in fat_tag.attributes:
                    return False
            elif (attr, value) not in fat_tag.attributes:
                return False
        return True

    def _count_tag_occurrences(self, needle, haystack, allow_extra_attrs=False):
        if False:
            i = 10
            return i + 15
        count = 0
        if allow_extra_attrs:
            if self._tag_matches_with_extra_attrs(needle, haystack):
                count += 1
        elif self._tag_is_equal(needle, haystack):
            count += 1
        if hasattr(haystack, 'children'):
            count += sum((self._count_tag_occurrences(needle, child, allow_extra_attrs=allow_extra_attrs) for child in haystack.children))
        return count

    def _tag_is_template_script(self, tag):
        if False:
            for i in range(10):
                print('nop')
        if tag.name != 'script':
            return False
        return any((attr == ('type', 'text/template') for attr in tag.attributes))

    def _find_template_script_tags(self, haystack):
        if False:
            while True:
                i = 10
        if not hasattr(haystack, 'name'):
            return
        if self._tag_is_template_script(haystack):
            yield haystack
        else:
            for child in haystack.children:
                yield from self._find_template_script_tags(child)

    def assertTagInHTML(self, needle, haystack, count=None, msg_prefix='', allow_extra_attrs=False):
        if False:
            print('Hello World!')
        needle = assert_and_parse_html(self, needle, None, 'First argument is not valid HTML:')
        haystack = assert_and_parse_html(self, haystack, None, 'Second argument is not valid HTML:')
        real_count = self._count_tag_occurrences(needle, haystack, allow_extra_attrs=allow_extra_attrs)
        if count is not None:
            self.assertEqual(real_count, count, msg_prefix + "Found %d instances of '%s' in response (expected %d)" % (real_count, needle, count))
        else:
            self.assertNotEqual(real_count, 0, msg_prefix + "Couldn't find '%s' in response" % needle)

    def assertNotInHTML(self, needle, haystack, msg_prefix=''):
        if False:
            while True:
                i = 10
        self.assertInHTML(needle, haystack, count=0, msg_prefix=msg_prefix)

    def assertTagInTemplateScript(self, needle, haystack, count=None, msg_prefix=''):
        if False:
            return 10
        needle = assert_and_parse_html(self, needle, None, 'First argument is not valid HTML:')
        haystack = assert_and_parse_html(self, haystack, None, 'Second argument is not valid HTML:')
        real_count = 0
        for script_tag in self._find_template_script_tags(haystack):
            if script_tag.children:
                self.assertEqual(len(script_tag.children), 1)
                script_html = assert_and_parse_html(self, script_tag.children[0], None, 'Script tag content is not valid HTML:')
                real_count += self._count_tag_occurrences(needle, script_html)
        if count is not None:
            self.assertEqual(real_count, count, msg_prefix + "Found %d instances of '%s' in template script (expected %d)" % (real_count, needle, count))
        else:
            self.assertNotEqual(real_count, 0, msg_prefix + "Couldn't find '%s' in template script" % needle)

    def assertFormError(self, response, form, field, errors, msg_prefix=''):
        if False:
            i = 10
            return i + 15
        if DJANGO_VERSION >= (4, 1):
            form = response.context[form]
            return super().assertFormError(form, field, errors, msg_prefix)
        return super().assertFormError(response, form, field, errors, msg_prefix)

    def assertFormsetError(self, response, formset, form_index, field, errors, msg_prefix=''):
        if False:
            print('Hello World!')
        if DJANGO_VERSION >= (4, 1):
            formset = response.context[formset]
            if DJANGO_VERSION >= (4, 2):
                return super().assertFormSetError(formset, form_index, field, errors, msg_prefix)
            return super().assertFormsetError(formset, form_index, field, errors, msg_prefix)
        return super().assertFormsetError(response, formset, form_index, field, errors, msg_prefix)

    def assertQuerysetEqual(self, qs, values, transform=None, ordered=True, msg=None):
        if False:
            return 10
        if DJANGO_VERSION >= (4, 2):
            return super().assertQuerySetEqual(qs, values, transform, ordered, msg)
        return super().assertQuerysetEqual(qs, values, transform, ordered, msg)