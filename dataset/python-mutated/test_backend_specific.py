from unittest import mock
from django.core.checks import Error
from django.db import connections, models
from django.test import SimpleTestCase
from django.test.utils import isolate_apps

def dummy_allow_migrate(db, app_label, **hints):
    if False:
        for i in range(10):
            print('nop')
    return db == 'default'

@isolate_apps('invalid_models_tests')
class BackendSpecificChecksTests(SimpleTestCase):

    @mock.patch('django.db.models.fields.router.allow_migrate', new=dummy_allow_migrate)
    def test_check_field(self):
        if False:
            while True:
                i = 10
        'Test if backend specific checks are performed.'
        error = Error('an error')

        class Model(models.Model):
            field = models.IntegerField()
        field = Model._meta.get_field('field')
        with mock.patch.object(connections['default'].validation, 'check_field', return_value=[error]):
            self.assertEqual(field.check(databases={'default'}), [error])