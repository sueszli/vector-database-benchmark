from django.db import transaction
from django.test import TestCase
from .models import FloatModel

class TestFloatField(TestCase):

    def test_float_validates_object(self):
        if False:
            i = 10
            return i + 15
        instance = FloatModel(size=2.5)
        instance.size = instance
        with transaction.atomic():
            with self.assertRaises(TypeError):
                instance.save()
        instance.size = 2.5
        instance.save()
        self.assertTrue(instance.id)
        instance.size = instance
        msg = 'Tried to update field model_fields.FloatModel.size with a model instance, %r. Use a value compatible with FloatField.' % instance
        with transaction.atomic():
            with self.assertRaisesMessage(TypeError, msg):
                instance.save()
        obj = FloatModel.objects.get(pk=instance.id)
        obj.size = obj
        with self.assertRaisesMessage(TypeError, msg):
            obj.save()

    def test_invalid_value(self):
        if False:
            for i in range(10):
                print('nop')
        tests = [(TypeError, ()), (TypeError, []), (TypeError, {}), (TypeError, set()), (TypeError, object()), (TypeError, complex()), (ValueError, 'non-numeric string'), (ValueError, b'non-numeric byte-string')]
        for (exception, value) in tests:
            with self.subTest(value):
                msg = "Field 'size' expected a number but got %r." % (value,)
                with self.assertRaisesMessage(exception, msg):
                    FloatModel.objects.create(size=value)