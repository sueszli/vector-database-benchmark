from django.db import connection
from . import PostgreSQLTestCase
from .models import CharFieldModel, TextFieldModel

class UnaccentTest(PostgreSQLTestCase):
    Model = CharFieldModel

    @classmethod
    def setUpTestData(cls):
        if False:
            i = 10
            return i + 15
        cls.Model.objects.bulk_create([cls.Model(field='àéÖ'), cls.Model(field='aeO'), cls.Model(field='aeo')])

    def test_unaccent(self):
        if False:
            while True:
                i = 10
        self.assertQuerySetEqual(self.Model.objects.filter(field__unaccent='aeO'), ['àéÖ', 'aeO'], transform=lambda instance: instance.field, ordered=False)

    def test_unaccent_chained(self):
        if False:
            while True:
                i = 10
        '\n        Unaccent can be used chained with a lookup (which should be the case\n        since unaccent implements the Transform API)\n        '
        self.assertQuerySetEqual(self.Model.objects.filter(field__unaccent__iexact='aeO'), ['àéÖ', 'aeO', 'aeo'], transform=lambda instance: instance.field, ordered=False)
        self.assertQuerySetEqual(self.Model.objects.filter(field__unaccent__endswith='éÖ'), ['àéÖ', 'aeO'], transform=lambda instance: instance.field, ordered=False)

    def test_unaccent_with_conforming_strings_off(self):
        if False:
            print('Hello World!')
        'SQL is valid when standard_conforming_strings is off.'
        with connection.cursor() as cursor:
            cursor.execute('SHOW standard_conforming_strings')
            disable_conforming_strings = cursor.fetchall()[0][0] == 'on'
            if disable_conforming_strings:
                cursor.execute('SET standard_conforming_strings TO off')
            try:
                self.assertQuerySetEqual(self.Model.objects.filter(field__unaccent__endswith='éÖ'), ['àéÖ', 'aeO'], transform=lambda instance: instance.field, ordered=False)
            finally:
                if disable_conforming_strings:
                    cursor.execute('SET standard_conforming_strings TO on')

    def test_unaccent_accentuated_needle(self):
        if False:
            print('Hello World!')
        self.assertQuerySetEqual(self.Model.objects.filter(field__unaccent='aéÖ'), ['àéÖ', 'aeO'], transform=lambda instance: instance.field, ordered=False)

class UnaccentTextFieldTest(UnaccentTest):
    """
    TextField should have the exact same behavior as CharField
    regarding unaccent lookups.
    """
    Model = TextFieldModel