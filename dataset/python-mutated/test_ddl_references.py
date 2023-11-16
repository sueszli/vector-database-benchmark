from django.db import connection
from django.db.backends.ddl_references import Columns, Expressions, ForeignKeyName, IndexName, Statement, Table
from django.db.models import ExpressionList, F
from django.db.models.functions import Upper
from django.db.models.indexes import IndexExpression
from django.db.models.sql import Query
from django.test import SimpleTestCase, TransactionTestCase
from .models import Person

class TableTests(SimpleTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.reference = Table('table', lambda table: table.upper())

    def test_references_table(self):
        if False:
            print('Hello World!')
        self.assertIs(self.reference.references_table('table'), True)
        self.assertIs(self.reference.references_table('other'), False)

    def test_rename_table_references(self):
        if False:
            print('Hello World!')
        self.reference.rename_table_references('other', 'table')
        self.assertIs(self.reference.references_table('table'), True)
        self.assertIs(self.reference.references_table('other'), False)
        self.reference.rename_table_references('table', 'other')
        self.assertIs(self.reference.references_table('table'), False)
        self.assertIs(self.reference.references_table('other'), True)

    def test_repr(self):
        if False:
            while True:
                i = 10
        self.assertEqual(repr(self.reference), "<Table 'TABLE'>")

    def test_str(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(str(self.reference), 'TABLE')

class ColumnsTests(TableTests):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.reference = Columns('table', ['first_column', 'second_column'], lambda column: column.upper())

    def test_references_column(self):
        if False:
            print('Hello World!')
        self.assertIs(self.reference.references_column('other', 'first_column'), False)
        self.assertIs(self.reference.references_column('table', 'third_column'), False)
        self.assertIs(self.reference.references_column('table', 'first_column'), True)

    def test_rename_column_references(self):
        if False:
            for i in range(10):
                print('nop')
        self.reference.rename_column_references('other', 'first_column', 'third_column')
        self.assertIs(self.reference.references_column('table', 'first_column'), True)
        self.assertIs(self.reference.references_column('table', 'third_column'), False)
        self.assertIs(self.reference.references_column('other', 'third_column'), False)
        self.reference.rename_column_references('table', 'third_column', 'first_column')
        self.assertIs(self.reference.references_column('table', 'first_column'), True)
        self.assertIs(self.reference.references_column('table', 'third_column'), False)
        self.reference.rename_column_references('table', 'first_column', 'third_column')
        self.assertIs(self.reference.references_column('table', 'first_column'), False)
        self.assertIs(self.reference.references_column('table', 'third_column'), True)

    def test_repr(self):
        if False:
            return 10
        self.assertEqual(repr(self.reference), "<Columns 'FIRST_COLUMN, SECOND_COLUMN'>")

    def test_str(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(str(self.reference), 'FIRST_COLUMN, SECOND_COLUMN')

class IndexNameTests(ColumnsTests):

    def setUp(self):
        if False:
            while True:
                i = 10

        def create_index_name(table_name, column_names, suffix):
            if False:
                return 10
            return ', '.join(('%s_%s_%s' % (table_name, column_name, suffix) for column_name in column_names))
        self.reference = IndexName('table', ['first_column', 'second_column'], 'suffix', create_index_name)

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(repr(self.reference), "<IndexName 'table_first_column_suffix, table_second_column_suffix'>")

    def test_str(self):
        if False:
            return 10
        self.assertEqual(str(self.reference), 'table_first_column_suffix, table_second_column_suffix')

class ForeignKeyNameTests(IndexNameTests):

    def setUp(self):
        if False:
            print('Hello World!')

        def create_foreign_key_name(table_name, column_names, suffix):
            if False:
                i = 10
                return i + 15
            return ', '.join(('%s_%s_%s' % (table_name, column_name, suffix) for column_name in column_names))
        self.reference = ForeignKeyName('table', ['first_column', 'second_column'], 'to_table', ['to_first_column', 'to_second_column'], '%(to_table)s_%(to_column)s_fk', create_foreign_key_name)

    def test_references_table(self):
        if False:
            i = 10
            return i + 15
        super().test_references_table()
        self.assertIs(self.reference.references_table('to_table'), True)

    def test_references_column(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_references_column()
        self.assertIs(self.reference.references_column('to_table', 'second_column'), False)
        self.assertIs(self.reference.references_column('to_table', 'to_second_column'), True)

    def test_rename_table_references(self):
        if False:
            i = 10
            return i + 15
        super().test_rename_table_references()
        self.reference.rename_table_references('to_table', 'other_to_table')
        self.assertIs(self.reference.references_table('other_to_table'), True)
        self.assertIs(self.reference.references_table('to_table'), False)

    def test_rename_column_references(self):
        if False:
            print('Hello World!')
        super().test_rename_column_references()
        self.reference.rename_column_references('to_table', 'second_column', 'third_column')
        self.assertIs(self.reference.references_column('table', 'second_column'), True)
        self.assertIs(self.reference.references_column('to_table', 'to_second_column'), True)
        self.reference.rename_column_references('to_table', 'to_first_column', 'to_third_column')
        self.assertIs(self.reference.references_column('to_table', 'to_first_column'), False)
        self.assertIs(self.reference.references_column('to_table', 'to_third_column'), True)

    def test_repr(self):
        if False:
            return 10
        self.assertEqual(repr(self.reference), "<ForeignKeyName 'table_first_column_to_table_to_first_column_fk, table_second_column_to_table_to_first_column_fk'>")

    def test_str(self):
        if False:
            return 10
        self.assertEqual(str(self.reference), 'table_first_column_to_table_to_first_column_fk, table_second_column_to_table_to_first_column_fk')

class MockReference:

    def __init__(self, representation, referenced_tables, referenced_columns):
        if False:
            print('Hello World!')
        self.representation = representation
        self.referenced_tables = referenced_tables
        self.referenced_columns = referenced_columns

    def references_table(self, table):
        if False:
            for i in range(10):
                print('nop')
        return table in self.referenced_tables

    def references_column(self, table, column):
        if False:
            return 10
        return (table, column) in self.referenced_columns

    def rename_table_references(self, old_table, new_table):
        if False:
            while True:
                i = 10
        if old_table in self.referenced_tables:
            self.referenced_tables.remove(old_table)
            self.referenced_tables.add(new_table)

    def rename_column_references(self, table, old_column, new_column):
        if False:
            i = 10
            return i + 15
        column = (table, old_column)
        if column in self.referenced_columns:
            self.referenced_columns.remove(column)
            self.referenced_columns.add((table, new_column))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.representation

class StatementTests(SimpleTestCase):

    def test_references_table(self):
        if False:
            print('Hello World!')
        statement = Statement('', reference=MockReference('', {'table'}, {}), non_reference='')
        self.assertIs(statement.references_table('table'), True)
        self.assertIs(statement.references_table('other'), False)

    def test_references_column(self):
        if False:
            for i in range(10):
                print('nop')
        statement = Statement('', reference=MockReference('', {}, {('table', 'column')}), non_reference='')
        self.assertIs(statement.references_column('table', 'column'), True)
        self.assertIs(statement.references_column('other', 'column'), False)

    def test_rename_table_references(self):
        if False:
            return 10
        reference = MockReference('', {'table'}, {})
        statement = Statement('', reference=reference, non_reference='')
        statement.rename_table_references('table', 'other')
        self.assertEqual(reference.referenced_tables, {'other'})

    def test_rename_column_references(self):
        if False:
            i = 10
            return i + 15
        reference = MockReference('', {}, {('table', 'column')})
        statement = Statement('', reference=reference, non_reference='')
        statement.rename_column_references('table', 'column', 'other')
        self.assertEqual(reference.referenced_columns, {('table', 'other')})

    def test_repr(self):
        if False:
            print('Hello World!')
        reference = MockReference('reference', {}, {})
        statement = Statement('%(reference)s - %(non_reference)s', reference=reference, non_reference='non_reference')
        self.assertEqual(repr(statement), "<Statement 'reference - non_reference'>")

    def test_str(self):
        if False:
            while True:
                i = 10
        reference = MockReference('reference', {}, {})
        statement = Statement('%(reference)s - %(non_reference)s', reference=reference, non_reference='non_reference')
        self.assertEqual(str(statement), 'reference - non_reference')

class ExpressionsTests(TransactionTestCase):
    available_apps = []

    def setUp(self):
        if False:
            i = 10
            return i + 15
        compiler = Person.objects.all().query.get_compiler(connection.alias)
        self.editor = connection.schema_editor()
        self.expressions = Expressions(table=Person._meta.db_table, expressions=ExpressionList(IndexExpression(F('first_name')), IndexExpression(F('last_name').desc()), IndexExpression(Upper('last_name'))).resolve_expression(compiler.query), compiler=compiler, quote_value=self.editor.quote_value)

    def test_references_table(self):
        if False:
            i = 10
            return i + 15
        self.assertIs(self.expressions.references_table(Person._meta.db_table), True)
        self.assertIs(self.expressions.references_table('other'), False)

    def test_references_column(self):
        if False:
            while True:
                i = 10
        table = Person._meta.db_table
        self.assertIs(self.expressions.references_column(table, 'first_name'), True)
        self.assertIs(self.expressions.references_column(table, 'last_name'), True)
        self.assertIs(self.expressions.references_column(table, 'other'), False)

    def test_rename_table_references(self):
        if False:
            for i in range(10):
                print('nop')
        table = Person._meta.db_table
        self.expressions.rename_table_references(table, 'other')
        self.assertIs(self.expressions.references_table(table), False)
        self.assertIs(self.expressions.references_table('other'), True)
        self.assertIn('%s.%s' % (self.editor.quote_name('other'), self.editor.quote_name('first_name')), str(self.expressions))

    def test_rename_table_references_without_alias(self):
        if False:
            for i in range(10):
                print('nop')
        compiler = Query(Person, alias_cols=False).get_compiler(connection=connection)
        table = Person._meta.db_table
        expressions = Expressions(table=table, expressions=ExpressionList(IndexExpression(Upper('last_name')), IndexExpression(F('first_name'))).resolve_expression(compiler.query), compiler=compiler, quote_value=self.editor.quote_value)
        expressions.rename_table_references(table, 'other')
        self.assertIs(expressions.references_table(table), False)
        self.assertIs(expressions.references_table('other'), True)
        expected_str = '(UPPER(%s)), %s' % (self.editor.quote_name('last_name'), self.editor.quote_name('first_name'))
        self.assertEqual(str(expressions), expected_str)

    def test_rename_column_references(self):
        if False:
            print('Hello World!')
        table = Person._meta.db_table
        self.expressions.rename_column_references(table, 'first_name', 'other')
        self.assertIs(self.expressions.references_column(table, 'other'), True)
        self.assertIs(self.expressions.references_column(table, 'first_name'), False)
        self.assertIn('%s.%s' % (self.editor.quote_name(table), self.editor.quote_name('other')), str(self.expressions))

    def test_str(self):
        if False:
            return 10
        table_name = self.editor.quote_name(Person._meta.db_table)
        expected_str = '%s.%s, %s.%s DESC, (UPPER(%s.%s))' % (table_name, self.editor.quote_name('first_name'), table_name, self.editor.quote_name('last_name'), table_name, self.editor.quote_name('last_name'))
        self.assertEqual(str(self.expressions), expected_str)