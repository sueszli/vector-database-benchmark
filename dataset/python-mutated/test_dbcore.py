"""Tests for the DBCore database abstraction.
"""
import os
import shutil
import sqlite3
import unittest
from tempfile import mkstemp
from test import _common
from beets import dbcore

class SortFixture(dbcore.query.FieldSort):
    pass

class QueryFixture(dbcore.query.NamedQuery):

    def __init__(self, pattern):
        if False:
            for i in range(10):
                print('nop')
        self.pattern = pattern

    def clause(self):
        if False:
            for i in range(10):
                print('nop')
        return (None, ())

    def match(self):
        if False:
            return 10
        return True

class ModelFixture1(dbcore.Model):
    _table = 'test'
    _flex_table = 'testflex'
    _fields = {'id': dbcore.types.PRIMARY_ID, 'field_one': dbcore.types.INTEGER, 'field_two': dbcore.types.STRING}
    _types = {'some_float_field': dbcore.types.FLOAT}
    _sorts = {'some_sort': SortFixture}
    _queries = {'some_query': QueryFixture}

    @classmethod
    def _getters(cls):
        if False:
            for i in range(10):
                print('nop')
        return {}

    def _template_funcs(self):
        if False:
            while True:
                i = 10
        return {}

class DatabaseFixture1(dbcore.Database):
    _models = (ModelFixture1,)
    pass

class ModelFixture2(ModelFixture1):
    _fields = {'id': dbcore.types.PRIMARY_ID, 'field_one': dbcore.types.INTEGER, 'field_two': dbcore.types.INTEGER}

class DatabaseFixture2(dbcore.Database):
    _models = (ModelFixture2,)
    pass

class ModelFixture3(ModelFixture1):
    _fields = {'id': dbcore.types.PRIMARY_ID, 'field_one': dbcore.types.INTEGER, 'field_two': dbcore.types.INTEGER, 'field_three': dbcore.types.INTEGER}

class DatabaseFixture3(dbcore.Database):
    _models = (ModelFixture3,)
    pass

class ModelFixture4(ModelFixture1):
    _fields = {'id': dbcore.types.PRIMARY_ID, 'field_one': dbcore.types.INTEGER, 'field_two': dbcore.types.INTEGER, 'field_three': dbcore.types.INTEGER, 'field_four': dbcore.types.INTEGER}

class DatabaseFixture4(dbcore.Database):
    _models = (ModelFixture4,)
    pass

class AnotherModelFixture(ModelFixture1):
    _table = 'another'
    _flex_table = 'anotherflex'
    _fields = {'id': dbcore.types.PRIMARY_ID, 'foo': dbcore.types.INTEGER}

class ModelFixture5(ModelFixture1):
    _fields = {'some_string_field': dbcore.types.STRING, 'some_float_field': dbcore.types.FLOAT, 'some_boolean_field': dbcore.types.BOOLEAN}

class DatabaseFixture5(dbcore.Database):
    _models = (ModelFixture5,)
    pass

class DatabaseFixtureTwoModels(dbcore.Database):
    _models = (ModelFixture2, AnotherModelFixture)
    pass

class ModelFixtureWithGetters(dbcore.Model):

    @classmethod
    def _getters(cls):
        if False:
            print('Hello World!')
        return {'aComputedField': lambda s: 'thing'}

    def _template_funcs(self):
        if False:
            while True:
                i = 10
        return {}

@_common.slow_test()
class MigrationTest(unittest.TestCase):
    """Tests the ability to change the database schema between
    versions.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        (handle, cls.orig_libfile) = mkstemp('orig_db')
        os.close(handle)
        old_lib = DatabaseFixture2(cls.orig_libfile)
        old_lib._connection().execute('insert into test (field_one, field_two) values (4, 2)')
        old_lib._connection().commit()
        old_lib._connection().close()
        del old_lib

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        os.remove(cls.orig_libfile)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        (handle, self.libfile) = mkstemp('db')
        os.close(handle)
        shutil.copyfile(self.orig_libfile, self.libfile)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        os.remove(self.libfile)

    def test_open_with_same_fields_leaves_untouched(self):
        if False:
            for i in range(10):
                print('nop')
        new_lib = DatabaseFixture2(self.libfile)
        c = new_lib._connection().cursor()
        c.execute('select * from test')
        row = c.fetchone()
        c.connection.close()
        self.assertEqual(len(row.keys()), len(ModelFixture2._fields))

    def test_open_with_new_field_adds_column(self):
        if False:
            print('Hello World!')
        new_lib = DatabaseFixture3(self.libfile)
        c = new_lib._connection().cursor()
        c.execute('select * from test')
        row = c.fetchone()
        c.connection.close()
        self.assertEqual(len(row.keys()), len(ModelFixture3._fields))

    def test_open_with_fewer_fields_leaves_untouched(self):
        if False:
            while True:
                i = 10
        new_lib = DatabaseFixture1(self.libfile)
        c = new_lib._connection().cursor()
        c.execute('select * from test')
        row = c.fetchone()
        c.connection.close()
        self.assertEqual(len(row.keys()), len(ModelFixture2._fields))

    def test_open_with_multiple_new_fields(self):
        if False:
            while True:
                i = 10
        new_lib = DatabaseFixture4(self.libfile)
        c = new_lib._connection().cursor()
        c.execute('select * from test')
        row = c.fetchone()
        c.connection.close()
        self.assertEqual(len(row.keys()), len(ModelFixture4._fields))

    def test_extra_model_adds_table(self):
        if False:
            for i in range(10):
                print('nop')
        new_lib = DatabaseFixtureTwoModels(self.libfile)
        try:
            c = new_lib._connection()
            c.execute('select * from another')
            c.close()
        except sqlite3.OperationalError:
            self.fail('select failed')

class TransactionTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.db = DatabaseFixture1(':memory:')

    def tearDown(self):
        if False:
            print('Hello World!')
        self.db._connection().close()

    def test_mutate_increase_revision(self):
        if False:
            while True:
                i = 10
        old_rev = self.db.revision
        with self.db.transaction() as tx:
            tx.mutate('INSERT INTO {} (field_one) VALUES (?);'.format(ModelFixture1._table), (111,))
        self.assertGreater(self.db.revision, old_rev)

    def test_query_no_increase_revision(self):
        if False:
            return 10
        old_rev = self.db.revision
        with self.db.transaction() as tx:
            tx.query('PRAGMA table_info(%s)' % ModelFixture1._table)
        self.assertEqual(self.db.revision, old_rev)

class ModelTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.db = DatabaseFixture1(':memory:')

    def tearDown(self):
        if False:
            print('Hello World!')
        self.db._connection().close()

    def test_add_model(self):
        if False:
            while True:
                i = 10
        model = ModelFixture1()
        model.add(self.db)
        rows = self.db._connection().execute('select * from test').fetchall()
        self.assertEqual(len(rows), 1)

    def test_store_fixed_field(self):
        if False:
            return 10
        model = ModelFixture1()
        model.add(self.db)
        model.field_one = 123
        model.store()
        row = self.db._connection().execute('select * from test').fetchone()
        self.assertEqual(row['field_one'], 123)

    def test_revision(self):
        if False:
            for i in range(10):
                print('nop')
        old_rev = self.db.revision
        model = ModelFixture1()
        model.add(self.db)
        model.store()
        self.assertEqual(model._revision, self.db.revision)
        self.assertGreater(self.db.revision, old_rev)
        mid_rev = self.db.revision
        model2 = ModelFixture1()
        model2.add(self.db)
        model2.store()
        self.assertGreater(model2._revision, mid_rev)
        self.assertGreater(self.db.revision, model._revision)
        model.load()
        self.assertEqual(model._revision, self.db.revision)
        mod2_old_rev = model2._revision
        model2.load()
        self.assertEqual(model2._revision, mod2_old_rev)

    def test_retrieve_by_id(self):
        if False:
            return 10
        model = ModelFixture1()
        model.add(self.db)
        other_model = self.db._get(ModelFixture1, model.id)
        self.assertEqual(model.id, other_model.id)

    def test_store_and_retrieve_flexattr(self):
        if False:
            print('Hello World!')
        model = ModelFixture1()
        model.add(self.db)
        model.foo = 'bar'
        model.store()
        other_model = self.db._get(ModelFixture1, model.id)
        self.assertEqual(other_model.foo, 'bar')

    def test_delete_flexattr(self):
        if False:
            while True:
                i = 10
        model = ModelFixture1()
        model['foo'] = 'bar'
        self.assertTrue('foo' in model)
        del model['foo']
        self.assertFalse('foo' in model)

    def test_delete_flexattr_via_dot(self):
        if False:
            for i in range(10):
                print('nop')
        model = ModelFixture1()
        model['foo'] = 'bar'
        self.assertTrue('foo' in model)
        del model.foo
        self.assertFalse('foo' in model)

    def test_delete_flexattr_persists(self):
        if False:
            for i in range(10):
                print('nop')
        model = ModelFixture1()
        model.add(self.db)
        model.foo = 'bar'
        model.store()
        model = self.db._get(ModelFixture1, model.id)
        del model['foo']
        model.store()
        model = self.db._get(ModelFixture1, model.id)
        self.assertFalse('foo' in model)

    def test_delete_non_existent_attribute(self):
        if False:
            while True:
                i = 10
        model = ModelFixture1()
        with self.assertRaises(KeyError):
            del model['foo']

    def test_delete_fixed_attribute(self):
        if False:
            print('Hello World!')
        model = ModelFixture5()
        model.some_string_field = 'foo'
        model.some_float_field = 1.23
        model.some_boolean_field = True
        for (field, type_) in model._fields.items():
            self.assertNotEqual(model[field], type_.null)
        for (field, type_) in model._fields.items():
            del model[field]
            self.assertEqual(model[field], type_.null)

    def test_null_value_normalization_by_type(self):
        if False:
            for i in range(10):
                print('nop')
        model = ModelFixture1()
        model.field_one = None
        self.assertEqual(model.field_one, 0)

    def test_null_value_stays_none_for_untyped_field(self):
        if False:
            i = 10
            return i + 15
        model = ModelFixture1()
        model.foo = None
        self.assertEqual(model.foo, None)

    def test_normalization_for_typed_flex_fields(self):
        if False:
            return 10
        model = ModelFixture1()
        model.some_float_field = None
        self.assertEqual(model.some_float_field, 0.0)

    def test_load_deleted_flex_field(self):
        if False:
            for i in range(10):
                print('nop')
        model1 = ModelFixture1()
        model1['flex_field'] = True
        model1.add(self.db)
        model2 = self.db._get(ModelFixture1, model1.id)
        self.assertIn('flex_field', model2)
        del model1['flex_field']
        model1.store()
        model2.load()
        self.assertNotIn('flex_field', model2)

    def test_check_db_fails(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'no database'):
            dbcore.Model()._check_db()
        with self.assertRaisesRegex(ValueError, 'no id'):
            ModelFixture1(self.db)._check_db()
        dbcore.Model(self.db)._check_db(need_id=False)

    def test_missing_field(self):
        if False:
            return 10
        with self.assertRaises(AttributeError):
            ModelFixture1(self.db).nonExistingKey

    def test_computed_field(self):
        if False:
            print('Hello World!')
        model = ModelFixtureWithGetters()
        self.assertEqual(model.aComputedField, 'thing')
        with self.assertRaisesRegex(KeyError, 'computed field .+ deleted'):
            del model.aComputedField

    def test_items(self):
        if False:
            while True:
                i = 10
        model = ModelFixture1(self.db)
        model.id = 5
        self.assertEqual({('id', 5), ('field_one', 0), ('field_two', '')}, set(model.items()))

    def test_delete_internal_field(self):
        if False:
            for i in range(10):
                print('nop')
        model = dbcore.Model()
        del model._db
        with self.assertRaises(AttributeError):
            model._db

    def test_parse_nonstring(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, 'must be a string'):
            dbcore.Model._parse(None, 42)

class FormatTest(unittest.TestCase):

    def test_format_fixed_field_integer(self):
        if False:
            print('Hello World!')
        model = ModelFixture1()
        model.field_one = 155
        value = model.formatted().get('field_one')
        self.assertEqual(value, '155')

    def test_format_fixed_field_integer_normalized(self):
        if False:
            print('Hello World!')
        'The normalize method of the Integer class rounds floats'
        model = ModelFixture1()
        model.field_one = 142.432
        value = model.formatted().get('field_one')
        self.assertEqual(value, '142')
        model.field_one = 142.863
        value = model.formatted().get('field_one')
        self.assertEqual(value, '143')

    def test_format_fixed_field_string(self):
        if False:
            for i in range(10):
                print('nop')
        model = ModelFixture1()
        model.field_two = 'café'
        value = model.formatted().get('field_two')
        self.assertEqual(value, 'café')

    def test_format_flex_field(self):
        if False:
            while True:
                i = 10
        model = ModelFixture1()
        model.other_field = 'café'
        value = model.formatted().get('other_field')
        self.assertEqual(value, 'café')

    def test_format_flex_field_bytes(self):
        if False:
            while True:
                i = 10
        model = ModelFixture1()
        model.other_field = 'café'.encode()
        value = model.formatted().get('other_field')
        self.assertTrue(isinstance(value, str))
        self.assertEqual(value, 'café')

    def test_format_unset_field(self):
        if False:
            return 10
        model = ModelFixture1()
        value = model.formatted().get('other_field')
        self.assertEqual(value, '')

    def test_format_typed_flex_field(self):
        if False:
            return 10
        model = ModelFixture1()
        model.some_float_field = 3.14159265358979
        value = model.formatted().get('some_float_field')
        self.assertEqual(value, '3.1')

class FormattedMappingTest(unittest.TestCase):

    def test_keys_equal_model_keys(self):
        if False:
            i = 10
            return i + 15
        model = ModelFixture1()
        formatted = model.formatted()
        self.assertEqual(set(model.keys(True)), set(formatted.keys()))

    def test_get_unset_field(self):
        if False:
            return 10
        model = ModelFixture1()
        formatted = model.formatted()
        with self.assertRaises(KeyError):
            formatted['other_field']

    def test_get_method_with_default(self):
        if False:
            for i in range(10):
                print('nop')
        model = ModelFixture1()
        formatted = model.formatted()
        self.assertEqual(formatted.get('other_field'), '')

    def test_get_method_with_specified_default(self):
        if False:
            i = 10
            return i + 15
        model = ModelFixture1()
        formatted = model.formatted()
        self.assertEqual(formatted.get('other_field', 'default'), 'default')

class ParseTest(unittest.TestCase):

    def test_parse_fixed_field(self):
        if False:
            print('Hello World!')
        value = ModelFixture1._parse('field_one', '2')
        self.assertIsInstance(value, int)
        self.assertEqual(value, 2)

    def test_parse_flex_field(self):
        if False:
            i = 10
            return i + 15
        value = ModelFixture1._parse('some_float_field', '2')
        self.assertIsInstance(value, float)
        self.assertEqual(value, 2.0)

    def test_parse_untyped_field(self):
        if False:
            for i in range(10):
                print('nop')
        value = ModelFixture1._parse('field_nine', '2')
        self.assertEqual(value, '2')

class QueryParseTest(unittest.TestCase):

    def pqp(self, part):
        if False:
            return 10
        return dbcore.queryparse.parse_query_part(part, {'year': dbcore.query.NumericQuery}, {':': dbcore.query.RegexpQuery})[:-1]

    def test_one_basic_term(self):
        if False:
            print('Hello World!')
        q = 'test'
        r = (None, 'test', dbcore.query.SubstringQuery)
        self.assertEqual(self.pqp(q), r)

    def test_one_keyed_term(self):
        if False:
            i = 10
            return i + 15
        q = 'test:val'
        r = ('test', 'val', dbcore.query.SubstringQuery)
        self.assertEqual(self.pqp(q), r)

    def test_colon_at_end(self):
        if False:
            return 10
        q = 'test:'
        r = ('test', '', dbcore.query.SubstringQuery)
        self.assertEqual(self.pqp(q), r)

    def test_one_basic_regexp(self):
        if False:
            return 10
        q = ':regexp'
        r = (None, 'regexp', dbcore.query.RegexpQuery)
        self.assertEqual(self.pqp(q), r)

    def test_keyed_regexp(self):
        if False:
            return 10
        q = 'test::regexp'
        r = ('test', 'regexp', dbcore.query.RegexpQuery)
        self.assertEqual(self.pqp(q), r)

    def test_escaped_colon(self):
        if False:
            return 10
        q = 'test\\:val'
        r = (None, 'test:val', dbcore.query.SubstringQuery)
        self.assertEqual(self.pqp(q), r)

    def test_escaped_colon_in_regexp(self):
        if False:
            return 10
        q = ':test\\:regexp'
        r = (None, 'test:regexp', dbcore.query.RegexpQuery)
        self.assertEqual(self.pqp(q), r)

    def test_single_year(self):
        if False:
            for i in range(10):
                print('nop')
        q = 'year:1999'
        r = ('year', '1999', dbcore.query.NumericQuery)
        self.assertEqual(self.pqp(q), r)

    def test_multiple_years(self):
        if False:
            i = 10
            return i + 15
        q = 'year:1999..2010'
        r = ('year', '1999..2010', dbcore.query.NumericQuery)
        self.assertEqual(self.pqp(q), r)

    def test_empty_query_part(self):
        if False:
            return 10
        q = ''
        r = (None, '', dbcore.query.SubstringQuery)
        self.assertEqual(self.pqp(q), r)

class QueryFromStringsTest(unittest.TestCase):

    def qfs(self, strings):
        if False:
            for i in range(10):
                print('nop')
        return dbcore.queryparse.query_from_strings(dbcore.query.AndQuery, ModelFixture1, {':': dbcore.query.RegexpQuery}, strings)

    def test_zero_parts(self):
        if False:
            i = 10
            return i + 15
        q = self.qfs([])
        self.assertIsInstance(q, dbcore.query.AndQuery)
        self.assertEqual(len(q.subqueries), 1)
        self.assertIsInstance(q.subqueries[0], dbcore.query.TrueQuery)

    def test_two_parts(self):
        if False:
            while True:
                i = 10
        q = self.qfs(['foo', 'bar:baz'])
        self.assertIsInstance(q, dbcore.query.AndQuery)
        self.assertEqual(len(q.subqueries), 2)
        self.assertIsInstance(q.subqueries[0], dbcore.query.AnyFieldQuery)
        self.assertIsInstance(q.subqueries[1], dbcore.query.SubstringQuery)

    def test_parse_fixed_type_query(self):
        if False:
            i = 10
            return i + 15
        q = self.qfs(['field_one:2..3'])
        self.assertIsInstance(q.subqueries[0], dbcore.query.NumericQuery)

    def test_parse_flex_type_query(self):
        if False:
            return 10
        q = self.qfs(['some_float_field:2..3'])
        self.assertIsInstance(q.subqueries[0], dbcore.query.NumericQuery)

    def test_empty_query_part(self):
        if False:
            return 10
        q = self.qfs([''])
        self.assertIsInstance(q.subqueries[0], dbcore.query.TrueQuery)

    def test_parse_named_query(self):
        if False:
            return 10
        q = self.qfs(['some_query:foo'])
        self.assertIsInstance(q.subqueries[0], QueryFixture)

class SortFromStringsTest(unittest.TestCase):

    def sfs(self, strings):
        if False:
            i = 10
            return i + 15
        return dbcore.queryparse.sort_from_strings(ModelFixture1, strings)

    def test_zero_parts(self):
        if False:
            i = 10
            return i + 15
        s = self.sfs([])
        self.assertIsInstance(s, dbcore.query.NullSort)
        self.assertEqual(s, dbcore.query.NullSort())

    def test_one_parts(self):
        if False:
            while True:
                i = 10
        s = self.sfs(['field+'])
        self.assertIsInstance(s, dbcore.query.Sort)

    def test_two_parts(self):
        if False:
            print('Hello World!')
        s = self.sfs(['field+', 'another_field-'])
        self.assertIsInstance(s, dbcore.query.MultipleSort)
        self.assertEqual(len(s.sorts), 2)

    def test_fixed_field_sort(self):
        if False:
            for i in range(10):
                print('nop')
        s = self.sfs(['field_one+'])
        self.assertIsInstance(s, dbcore.query.FixedFieldSort)
        self.assertEqual(s, dbcore.query.FixedFieldSort('field_one'))

    def test_flex_field_sort(self):
        if False:
            return 10
        s = self.sfs(['flex_field+'])
        self.assertIsInstance(s, dbcore.query.SlowFieldSort)
        self.assertEqual(s, dbcore.query.SlowFieldSort('flex_field'))

    def test_special_sort(self):
        if False:
            return 10
        s = self.sfs(['some_sort+'])
        self.assertIsInstance(s, SortFixture)

class ParseSortedQueryTest(unittest.TestCase):

    def psq(self, parts):
        if False:
            while True:
                i = 10
        return dbcore.parse_sorted_query(ModelFixture1, parts.split())

    def test_and_query(self):
        if False:
            while True:
                i = 10
        (q, s) = self.psq('foo bar')
        self.assertIsInstance(q, dbcore.query.AndQuery)
        self.assertIsInstance(s, dbcore.query.NullSort)
        self.assertEqual(len(q.subqueries), 2)

    def test_or_query(self):
        if False:
            for i in range(10):
                print('nop')
        (q, s) = self.psq('foo , bar')
        self.assertIsInstance(q, dbcore.query.OrQuery)
        self.assertIsInstance(s, dbcore.query.NullSort)
        self.assertEqual(len(q.subqueries), 2)

    def test_no_space_before_comma_or_query(self):
        if False:
            print('Hello World!')
        (q, s) = self.psq('foo, bar')
        self.assertIsInstance(q, dbcore.query.OrQuery)
        self.assertIsInstance(s, dbcore.query.NullSort)
        self.assertEqual(len(q.subqueries), 2)

    def test_no_spaces_or_query(self):
        if False:
            print('Hello World!')
        (q, s) = self.psq('foo,bar')
        self.assertIsInstance(q, dbcore.query.AndQuery)
        self.assertIsInstance(s, dbcore.query.NullSort)
        self.assertEqual(len(q.subqueries), 1)

    def test_trailing_comma_or_query(self):
        if False:
            return 10
        (q, s) = self.psq('foo , bar ,')
        self.assertIsInstance(q, dbcore.query.OrQuery)
        self.assertIsInstance(s, dbcore.query.NullSort)
        self.assertEqual(len(q.subqueries), 3)

    def test_leading_comma_or_query(self):
        if False:
            for i in range(10):
                print('nop')
        (q, s) = self.psq(', foo , bar')
        self.assertIsInstance(q, dbcore.query.OrQuery)
        self.assertIsInstance(s, dbcore.query.NullSort)
        self.assertEqual(len(q.subqueries), 3)

    def test_only_direction(self):
        if False:
            return 10
        (q, s) = self.psq('-')
        self.assertIsInstance(q, dbcore.query.AndQuery)
        self.assertIsInstance(s, dbcore.query.NullSort)
        self.assertEqual(len(q.subqueries), 1)

class ResultsIteratorTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.db = DatabaseFixture1(':memory:')
        model = ModelFixture1()
        model['foo'] = 'baz'
        model.add(self.db)
        model = ModelFixture1()
        model['foo'] = 'bar'
        model.add(self.db)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.db._connection().close()

    def test_iterate_once(self):
        if False:
            return 10
        objs = self.db._fetch(ModelFixture1)
        self.assertEqual(len(list(objs)), 2)

    def test_iterate_twice(self):
        if False:
            while True:
                i = 10
        objs = self.db._fetch(ModelFixture1)
        list(objs)
        self.assertEqual(len(list(objs)), 2)

    def test_concurrent_iterators(self):
        if False:
            i = 10
            return i + 15
        results = self.db._fetch(ModelFixture1)
        it1 = iter(results)
        it2 = iter(results)
        next(it1)
        list(it2)
        self.assertEqual(len(list(it1)), 1)

    def test_slow_query(self):
        if False:
            while True:
                i = 10
        q = dbcore.query.SubstringQuery('foo', 'ba', False)
        objs = self.db._fetch(ModelFixture1, q)
        self.assertEqual(len(list(objs)), 2)

    def test_slow_query_negative(self):
        if False:
            for i in range(10):
                print('nop')
        q = dbcore.query.SubstringQuery('foo', 'qux', False)
        objs = self.db._fetch(ModelFixture1, q)
        self.assertEqual(len(list(objs)), 0)

    def test_iterate_slow_sort(self):
        if False:
            i = 10
            return i + 15
        s = dbcore.query.SlowFieldSort('foo')
        res = self.db._fetch(ModelFixture1, sort=s)
        objs = list(res)
        self.assertEqual(objs[0].foo, 'bar')
        self.assertEqual(objs[1].foo, 'baz')

    def test_unsorted_subscript(self):
        if False:
            return 10
        objs = self.db._fetch(ModelFixture1)
        self.assertEqual(objs[0].foo, 'baz')
        self.assertEqual(objs[1].foo, 'bar')

    def test_slow_sort_subscript(self):
        if False:
            for i in range(10):
                print('nop')
        s = dbcore.query.SlowFieldSort('foo')
        objs = self.db._fetch(ModelFixture1, sort=s)
        self.assertEqual(objs[0].foo, 'bar')
        self.assertEqual(objs[1].foo, 'baz')

    def test_length(self):
        if False:
            while True:
                i = 10
        objs = self.db._fetch(ModelFixture1)
        self.assertEqual(len(objs), 2)

    def test_out_of_range(self):
        if False:
            return 10
        objs = self.db._fetch(ModelFixture1)
        with self.assertRaises(IndexError):
            objs[100]

    def test_no_results(self):
        if False:
            while True:
                i = 10
        self.assertIsNone(self.db._fetch(ModelFixture1, dbcore.query.FalseQuery()).get())

def suite():
    if False:
        i = 10
        return i + 15
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')