from django.db import IntegrityError, connection
from django.db.models import CharField, F, FloatField, GeneratedField, IntegerField, Model
from django.db.models.functions import Lower
from django.test import SimpleTestCase, TestCase, skipUnlessDBFeature
from django.test.utils import isolate_apps
from .models import GeneratedModel, GeneratedModelNull, GeneratedModelNullVirtual, GeneratedModelOutputFieldDbCollation, GeneratedModelOutputFieldDbCollationVirtual, GeneratedModelParams, GeneratedModelParamsVirtual, GeneratedModelVirtual

class BaseGeneratedFieldTests(SimpleTestCase):

    def test_editable_unsupported(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesMessage(ValueError, 'GeneratedField cannot be editable.'):
            GeneratedField(expression=Lower('name'), output_field=CharField(max_length=255), editable=True, db_persist=False)

    def test_blank_unsupported(self):
        if False:
            return 10
        with self.assertRaisesMessage(ValueError, 'GeneratedField must be blank.'):
            GeneratedField(expression=Lower('name'), output_field=CharField(max_length=255), blank=False, db_persist=False)

    def test_default_unsupported(self):
        if False:
            return 10
        msg = 'GeneratedField cannot have a default.'
        with self.assertRaisesMessage(ValueError, msg):
            GeneratedField(expression=Lower('name'), output_field=CharField(max_length=255), default='', db_persist=False)

    def test_database_default_unsupported(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'GeneratedField cannot have a database default.'
        with self.assertRaisesMessage(ValueError, msg):
            GeneratedField(expression=Lower('name'), output_field=CharField(max_length=255), db_default='', db_persist=False)

    def test_db_persist_required(self):
        if False:
            print('Hello World!')
        msg = 'GeneratedField.db_persist must be True or False.'
        with self.assertRaisesMessage(ValueError, msg):
            GeneratedField(expression=Lower('name'), output_field=CharField(max_length=255))
        with self.assertRaisesMessage(ValueError, msg):
            GeneratedField(expression=Lower('name'), output_field=CharField(max_length=255), db_persist=None)

    def test_deconstruct(self):
        if False:
            for i in range(10):
                print('nop')
        field = GeneratedField(expression=F('a') + F('b'), output_field=IntegerField(), db_persist=True)
        (_, path, args, kwargs) = field.deconstruct()
        self.assertEqual(path, 'django.db.models.GeneratedField')
        self.assertEqual(args, [])
        self.assertEqual(kwargs['db_persist'], True)
        self.assertEqual(kwargs['expression'], F('a') + F('b'))
        self.assertEqual(kwargs['output_field'].deconstruct(), IntegerField().deconstruct())

    @isolate_apps('model_fields')
    def test_get_col(self):
        if False:
            for i in range(10):
                print('nop')

        class Square(Model):
            side = IntegerField()
            area = GeneratedField(expression=F('side') * F('side'), output_field=IntegerField(), db_persist=True)
        col = Square._meta.get_field('area').get_col('alias')
        self.assertIsInstance(col.output_field, IntegerField)

        class FloatSquare(Model):
            side = IntegerField()
            area = GeneratedField(expression=F('side') * F('side'), db_persist=True, output_field=FloatField())
        col = FloatSquare._meta.get_field('area').get_col('alias')
        self.assertIsInstance(col.output_field, FloatField)

    @isolate_apps('model_fields')
    def test_cached_col(self):
        if False:
            print('Hello World!')

        class Sum(Model):
            a = IntegerField()
            b = IntegerField()
            total = GeneratedField(expression=F('a') + F('b'), output_field=IntegerField(), db_persist=True)
        field = Sum._meta.get_field('total')
        cached_col = field.cached_col
        self.assertIs(field.get_col(Sum._meta.db_table), cached_col)
        self.assertIs(field.get_col(Sum._meta.db_table, field), cached_col)
        self.assertIsNot(field.get_col('alias'), cached_col)
        self.assertIsNot(field.get_col(Sum._meta.db_table, IntegerField()), cached_col)
        self.assertIs(cached_col.target, field)
        self.assertIsInstance(cached_col.output_field, IntegerField)

class GeneratedFieldTestMixin:

    def _refresh_if_needed(self, m):
        if False:
            return 10
        if not connection.features.can_return_columns_from_insert:
            m.refresh_from_db()
        return m

    def test_unsaved_error(self):
        if False:
            print('Hello World!')
        m = self.base_model(a=1, b=2)
        msg = 'Cannot read a generated field from an unsaved model.'
        with self.assertRaisesMessage(AttributeError, msg):
            m.field

    def test_create(self):
        if False:
            for i in range(10):
                print('nop')
        m = self.base_model.objects.create(a=1, b=2)
        m = self._refresh_if_needed(m)
        self.assertEqual(m.field, 3)

    def test_non_nullable_create(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(IntegrityError):
            self.base_model.objects.create()

    def test_save(self):
        if False:
            while True:
                i = 10
        m = self.base_model(a=2, b=4)
        m.save()
        m = self._refresh_if_needed(m)
        self.assertEqual(m.field, 6)
        m.a = 4
        m.save()
        m.refresh_from_db()
        self.assertEqual(m.field, 8)

    def test_update(self):
        if False:
            print('Hello World!')
        m = self.base_model.objects.create(a=1, b=2)
        self.base_model.objects.update(b=3)
        m = self.base_model.objects.get(pk=m.pk)
        self.assertEqual(m.field, 4)

    def test_bulk_create(self):
        if False:
            print('Hello World!')
        m = self.base_model(a=3, b=4)
        (m,) = self.base_model.objects.bulk_create([m])
        if not connection.features.can_return_rows_from_bulk_insert:
            m = self.base_model.objects.get()
        self.assertEqual(m.field, 7)

    def test_bulk_update(self):
        if False:
            for i in range(10):
                print('nop')
        m = self.base_model.objects.create(a=1, b=2)
        m.a = 3
        self.base_model.objects.bulk_update([m], fields=['a'])
        m = self.base_model.objects.get(pk=m.pk)
        self.assertEqual(m.field, 5)

    def test_output_field_lookups(self):
        if False:
            print('Hello World!')
        'Lookups from the output_field are available on GeneratedFields.'
        internal_type = IntegerField().get_internal_type()
        (min_value, max_value) = connection.ops.integer_field_range(internal_type)
        if min_value is None:
            self.skipTest("Backend doesn't define an integer min value.")
        if max_value is None:
            self.skipTest("Backend doesn't define an integer max value.")
        does_not_exist = self.base_model.DoesNotExist
        underflow_value = min_value - 1
        with self.assertNumQueries(0), self.assertRaises(does_not_exist):
            self.base_model.objects.get(field=underflow_value)
        with self.assertNumQueries(0), self.assertRaises(does_not_exist):
            self.base_model.objects.get(field__lt=underflow_value)
        with self.assertNumQueries(0), self.assertRaises(does_not_exist):
            self.base_model.objects.get(field__lte=underflow_value)
        overflow_value = max_value + 1
        with self.assertNumQueries(0), self.assertRaises(does_not_exist):
            self.base_model.objects.get(field=overflow_value)
        with self.assertNumQueries(0), self.assertRaises(does_not_exist):
            self.base_model.objects.get(field__gt=overflow_value)
        with self.assertNumQueries(0), self.assertRaises(does_not_exist):
            self.base_model.objects.get(field__gte=overflow_value)

    def test_output_field_db_collation(self):
        if False:
            print('Hello World!')
        collation = connection.features.test_collations['virtual']
        m = self.output_field_db_collation_model.objects.create(name='NAME')
        field = m._meta.get_field('lower_name')
        db_parameters = field.db_parameters(connection)
        self.assertEqual(db_parameters['collation'], collation)
        self.assertEqual(db_parameters['type'], field.output_field.db_type(connection))
        self.assertNotEqual(db_parameters['type'], field._resolved_expression.output_field.db_type(connection))

    def test_db_type_parameters(self):
        if False:
            while True:
                i = 10
        db_type_parameters = self.output_field_db_collation_model._meta.get_field('lower_name').db_type_parameters(connection)
        self.assertEqual(db_type_parameters['max_length'], 11)

    def test_model_with_params(self):
        if False:
            while True:
                i = 10
        m = self.params_model.objects.create()
        m = self._refresh_if_needed(m)
        self.assertEqual(m.field, 'Constant')

    def test_nullable(self):
        if False:
            while True:
                i = 10
        m1 = self.nullable_model.objects.create()
        m1 = self._refresh_if_needed(m1)
        none_val = '' if connection.features.interprets_empty_strings_as_nulls else None
        self.assertEqual(m1.lower_name, none_val)
        m2 = self.nullable_model.objects.create(name='NaMe')
        m2 = self._refresh_if_needed(m2)
        self.assertEqual(m2.lower_name, 'name')

@skipUnlessDBFeature('supports_stored_generated_columns')
class StoredGeneratedFieldTests(GeneratedFieldTestMixin, TestCase):
    base_model = GeneratedModel
    nullable_model = GeneratedModelNull
    output_field_db_collation_model = GeneratedModelOutputFieldDbCollation
    params_model = GeneratedModelParams

@skipUnlessDBFeature('supports_virtual_generated_columns')
class VirtualGeneratedFieldTests(GeneratedFieldTestMixin, TestCase):
    base_model = GeneratedModelVirtual
    nullable_model = GeneratedModelNullVirtual
    output_field_db_collation_model = GeneratedModelOutputFieldDbCollationVirtual
    params_model = GeneratedModelParamsVirtual