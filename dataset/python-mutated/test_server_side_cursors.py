import operator
import unittest
from collections import namedtuple
from contextlib import contextmanager
from django.db import connection, models
from django.test import TestCase
from ..models import Person

@unittest.skipUnless(connection.vendor == 'postgresql', 'PostgreSQL tests')
class ServerSideCursorsPostgres(TestCase):
    cursor_fields = 'name, statement, is_holdable, is_binary, is_scrollable, creation_time'
    PostgresCursor = namedtuple('PostgresCursor', cursor_fields)

    @classmethod
    def setUpTestData(cls):
        if False:
            for i in range(10):
                print('nop')
        Person.objects.create(first_name='a', last_name='a')
        Person.objects.create(first_name='b', last_name='b')

    def inspect_cursors(self):
        if False:
            print('Hello World!')
        with connection.cursor() as cursor:
            cursor.execute('SELECT {fields} FROM pg_cursors;'.format(fields=self.cursor_fields))
            cursors = cursor.fetchall()
        return [self.PostgresCursor._make(cursor) for cursor in cursors]

    @contextmanager
    def override_db_setting(self, **kwargs):
        if False:
            print('Hello World!')
        for setting in kwargs:
            original_value = connection.settings_dict.get(setting)
            if setting in connection.settings_dict:
                self.addCleanup(operator.setitem, connection.settings_dict, setting, original_value)
            else:
                self.addCleanup(operator.delitem, connection.settings_dict, setting)
            connection.settings_dict[setting] = kwargs[setting]
            yield

    def assertUsesCursor(self, queryset, num_expected=1):
        if False:
            for i in range(10):
                print('nop')
        next(queryset)
        cursors = self.inspect_cursors()
        self.assertEqual(len(cursors), num_expected)
        for cursor in cursors:
            self.assertIn('_django_curs_', cursor.name)
            self.assertFalse(cursor.is_scrollable)
            self.assertFalse(cursor.is_holdable)
            self.assertFalse(cursor.is_binary)

    def asserNotUsesCursor(self, queryset):
        if False:
            while True:
                i = 10
        self.assertUsesCursor(queryset, num_expected=0)

    def test_server_side_cursor(self):
        if False:
            return 10
        self.assertUsesCursor(Person.objects.iterator())

    def test_values(self):
        if False:
            while True:
                i = 10
        self.assertUsesCursor(Person.objects.values('first_name').iterator())

    def test_values_list(self):
        if False:
            i = 10
            return i + 15
        self.assertUsesCursor(Person.objects.values_list('first_name').iterator())

    def test_values_list_flat(self):
        if False:
            print('Hello World!')
        self.assertUsesCursor(Person.objects.values_list('first_name', flat=True).iterator())

    def test_values_list_fields_not_equal_to_names(self):
        if False:
            print('Hello World!')
        expr = models.Count('id')
        self.assertUsesCursor(Person.objects.annotate(id__count=expr).values_list(expr, 'id__count').iterator())

    def test_server_side_cursor_many_cursors(self):
        if False:
            while True:
                i = 10
        persons = Person.objects.iterator()
        persons2 = Person.objects.iterator()
        next(persons)
        self.assertUsesCursor(persons2, num_expected=2)

    def test_closed_server_side_cursor(self):
        if False:
            i = 10
            return i + 15
        persons = Person.objects.iterator()
        next(persons)
        del persons
        cursors = self.inspect_cursors()
        self.assertEqual(len(cursors), 0)

    def test_server_side_cursors_setting(self):
        if False:
            for i in range(10):
                print('nop')
        with self.override_db_setting(DISABLE_SERVER_SIDE_CURSORS=False):
            persons = Person.objects.iterator()
            self.assertUsesCursor(persons)
            del persons
        with self.override_db_setting(DISABLE_SERVER_SIDE_CURSORS=True):
            self.asserNotUsesCursor(Person.objects.iterator())