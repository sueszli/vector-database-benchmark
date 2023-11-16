from django.db import models
from django.test import SimpleTestCase
from .models import AutoModel, BigAutoModel, SmallAutoModel
from .test_integerfield import BigIntegerFieldTests, IntegerFieldTests, SmallIntegerFieldTests

class AutoFieldTests(IntegerFieldTests):
    model = AutoModel
    rel_db_type_class = models.IntegerField

class BigAutoFieldTests(BigIntegerFieldTests):
    model = BigAutoModel
    rel_db_type_class = models.BigIntegerField

class SmallAutoFieldTests(SmallIntegerFieldTests):
    model = SmallAutoModel
    rel_db_type_class = models.SmallIntegerField

class AutoFieldInheritanceTests(SimpleTestCase):

    def test_isinstance_of_autofield(self):
        if False:
            for i in range(10):
                print('nop')
        for field in (models.BigAutoField, models.SmallAutoField):
            with self.subTest(field.__name__):
                self.assertIsInstance(field(), models.AutoField)

    def test_issubclass_of_autofield(self):
        if False:
            return 10

        class MyBigAutoField(models.BigAutoField):
            pass

        class MySmallAutoField(models.SmallAutoField):
            pass
        tests = [MyBigAutoField, MySmallAutoField, models.BigAutoField, models.SmallAutoField]
        for field in tests:
            with self.subTest(field.__name__):
                self.assertTrue(issubclass(field, models.AutoField))