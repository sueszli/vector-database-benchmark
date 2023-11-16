from pyflink.java_gateway import get_gateway
from pyflink.table import TableSchema, DataTypes
from pyflink.table.catalog import ObjectPath, Catalog, CatalogDatabase, CatalogBaseTable, CatalogFunction, CatalogPartition, CatalogPartitionSpec
from pyflink.testing.test_case_utils import PyFlinkTestCase
from pyflink.util.exceptions import DatabaseNotExistException, FunctionNotExistException, PartitionNotExistException, TableNotExistException, DatabaseAlreadyExistException, FunctionAlreadyExistException, PartitionAlreadyExistsException, PartitionSpecInvalidException, TableNotPartitionedException, TableAlreadyExistException, DatabaseNotEmptyException

class CatalogTestBase(PyFlinkTestCase):
    db1 = 'db1'
    db2 = 'db2'
    non_exist_database = 'non-exist-db'
    t1 = 't1'
    t2 = 't2'
    t3 = 't3'
    test_catalog_name = 'test-catalog'
    test_comment = 'test comment'

    def setUp(self):
        if False:
            while True:
                i = 10
        super(CatalogTestBase, self).setUp()
        gateway = get_gateway()
        self.catalog = Catalog(gateway.jvm.GenericInMemoryCatalog(self.test_catalog_name))
        self.path1 = ObjectPath(self.db1, self.t1)
        self.path2 = ObjectPath(self.db2, self.t2)
        self.path3 = ObjectPath(self.db1, self.t2)
        self.path4 = ObjectPath(self.db1, self.t3)
        self.non_exist_db_path = ObjectPath.from_string('non.exist')
        self.non_exist_object_path = ObjectPath.from_string('db1.nonexist')

    def check_catalog_database_equals(self, cd1, cd2):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(cd1.get_comment(), cd2.get_comment())
        self.assertEqual(cd1.get_properties(), cd2.get_properties())

    def check_catalog_table_equals(self, t1, t2):
        if False:
            i = 10
            return i + 15
        self.assertEqual(t1.get_schema(), t2.get_schema())
        self.assertEqual(t1.get_options(), t2.get_options())
        self.assertEqual(t1.get_comment(), t2.get_comment())

    def check_catalog_view_equals(self, v1, v2):
        if False:
            while True:
                i = 10
        self.assertEqual(v1.get_schema(), v1.get_schema())
        self.assertEqual(v1.get_options(), v2.get_options())
        self.assertEqual(v1.get_comment(), v2.get_comment())

    def check_catalog_function_equals(self, f1, f2):
        if False:
            while True:
                i = 10
        self.assertEqual(f1.get_class_name(), f2.get_class_name())
        self.assertEqual(f1.is_generic(), f2.is_generic())
        self.assertEqual(f1.get_function_language(), f2.get_function_language())

    def check_catalog_partition_equals(self, p1, p2):
        if False:
            return 10
        self.assertEqual(p1.get_properties(), p2.get_properties())

    @staticmethod
    def create_db():
        if False:
            for i in range(10):
                print('nop')
        return CatalogDatabase.create_instance({'k1': 'v1'}, CatalogTestBase.test_comment)

    @staticmethod
    def create_another_db():
        if False:
            i = 10
            return i + 15
        return CatalogDatabase.create_instance({'k2': 'v2'}, 'this is another database.')

    @staticmethod
    def create_table_schema():
        if False:
            while True:
                i = 10
        return TableSchema(['first', 'second', 'third'], [DataTypes.STRING(), DataTypes.INT(), DataTypes.STRING()])

    @staticmethod
    def create_another_table_schema():
        if False:
            i = 10
            return i + 15
        return TableSchema(['first2', 'second', 'third'], [DataTypes.STRING(), DataTypes.STRING(), DataTypes.STRING()])

    @staticmethod
    def get_batch_table_properties():
        if False:
            while True:
                i = 10
        return {'is_streaming': 'false'}

    @staticmethod
    def get_streaming_table_properties():
        if False:
            for i in range(10):
                print('nop')
        return {'is_streaming': 'true'}

    @staticmethod
    def create_partition_keys():
        if False:
            return 10
        return ['second', 'third']

    @staticmethod
    def create_table():
        if False:
            for i in range(10):
                print('nop')
        return CatalogBaseTable.create_table(schema=CatalogTestBase.create_table_schema(), properties=CatalogTestBase.get_batch_table_properties(), comment=CatalogTestBase.test_comment)

    @staticmethod
    def create_another_table():
        if False:
            print('Hello World!')
        return CatalogBaseTable.create_table(schema=CatalogTestBase.create_another_table_schema(), properties=CatalogTestBase.get_batch_table_properties(), comment=CatalogTestBase.test_comment)

    @staticmethod
    def create_stream_table():
        if False:
            i = 10
            return i + 15
        return CatalogBaseTable.create_table(schema=CatalogTestBase.create_table_schema(), properties=CatalogTestBase.get_streaming_table_properties(), comment=CatalogTestBase.test_comment)

    @staticmethod
    def create_partitioned_table():
        if False:
            print('Hello World!')
        return CatalogBaseTable.create_table(schema=CatalogTestBase.create_table_schema(), partition_keys=CatalogTestBase.create_partition_keys(), properties=CatalogTestBase.get_batch_table_properties(), comment=CatalogTestBase.test_comment)

    @staticmethod
    def create_another_partitioned_table():
        if False:
            for i in range(10):
                print('nop')
        return CatalogBaseTable.create_table(schema=CatalogTestBase.create_another_table_schema(), partition_keys=CatalogTestBase.create_partition_keys(), properties=CatalogTestBase.get_batch_table_properties(), comment=CatalogTestBase.test_comment)

    @staticmethod
    def create_view():
        if False:
            while True:
                i = 10
        table_schema = CatalogTestBase.create_table_schema()
        return CatalogBaseTable.create_view('select * from t1', 'select * from test-catalog.db1.t1', table_schema, {}, 'This is a view')

    @staticmethod
    def create_another_view():
        if False:
            while True:
                i = 10
        table_schema = CatalogTestBase.create_another_table_schema()
        return CatalogBaseTable.create_view('select * from t2', 'select * from test-catalog.db2.t2', table_schema, {}, 'This is another view')

    @staticmethod
    def create_function():
        if False:
            return 10
        return CatalogFunction.create_instance('org.apache.flink.table.functions.python.PythonScalarFunction', 'Java')

    @staticmethod
    def create_another_function():
        if False:
            i = 10
            return i + 15
        return CatalogFunction.create_instance('org.apache.flink.table.functions.ScalarFunction', 'Java')

    @staticmethod
    def create_partition_spec():
        if False:
            for i in range(10):
                print('nop')
        return CatalogPartitionSpec({'third': '2000', 'second': 'bob'})

    @staticmethod
    def create_another_partition_spec():
        if False:
            return 10
        return CatalogPartitionSpec({'third': '2010', 'second': 'bob'})

    @staticmethod
    def create_partition():
        if False:
            print('Hello World!')
        return CatalogPartition.create_instance(CatalogTestBase.get_batch_table_properties(), 'catalog partition tests')

    @staticmethod
    def create_partition_spec_subset():
        if False:
            return 10
        return CatalogPartitionSpec({'second': 'bob'})

    @staticmethod
    def create_another_partition_spec_subset():
        if False:
            return 10
        return CatalogPartitionSpec({'third': '2000'})

    @staticmethod
    def create_invalid_partition_spec_subset():
        if False:
            i = 10
            return i + 15
        return CatalogPartitionSpec({'third': '2010'})

    def test_create_db(self):
        if False:
            return 10
        self.assertFalse(self.catalog.database_exists(self.db1))
        catalog_db = self.create_db()
        self.catalog.create_database(self.db1, catalog_db, False)
        self.assertTrue(self.catalog.database_exists(self.db1))
        self.check_catalog_database_equals(catalog_db, self.catalog.get_database(self.db1))

    def test_create_db_database_already_exist_exception(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        with self.assertRaises(DatabaseAlreadyExistException):
            self.catalog.create_database(self.db1, self.create_db(), False)

    def test_create_db_database_already_exist_ignored(self):
        if False:
            for i in range(10):
                print('nop')
        catalog_db = self.create_db()
        self.catalog.create_database(self.db1, catalog_db, False)
        dbs = self.catalog.list_databases()
        self.check_catalog_database_equals(catalog_db, self.catalog.get_database(self.db1))
        self.assertEqual(2, len(dbs))
        self.assertEqual({self.db1, self.catalog.get_default_database()}, set(dbs))
        self.catalog.create_database(self.db1, self.create_another_db(), True)
        self.check_catalog_database_equals(catalog_db, self.catalog.get_database(self.db1))
        self.assertEqual(2, len(dbs))
        self.assertEqual({self.db1, self.catalog.get_default_database()}, set(dbs))

    def test_get_db_database_not_exist_exception(self):
        if False:
            return 10
        with self.assertRaises(DatabaseNotExistException):
            self.catalog.get_database('nonexistent')

    def test_drop_db(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.assertTrue(self.catalog.database_exists(self.db1))
        self.catalog.drop_database(self.db1, False)
        self.assertFalse(self.catalog.database_exists(self.db1))

    def test_drop_db_database_not_exist_exception(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(DatabaseNotExistException):
            self.catalog.drop_database(self.db1, False)

    def test_drop_db_database_not_exist_ignore(self):
        if False:
            while True:
                i = 10
        self.catalog.drop_database(self.db1, True)

    def test_drop_db_database_not_empty_exception(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_table(), False)
        with self.assertRaises(DatabaseNotEmptyException):
            self.catalog.drop_database(self.db1, True)

    def test_alter_db(self):
        if False:
            while True:
                i = 10
        db = self.create_db()
        self.catalog.create_database(self.db1, db, False)
        new_db = self.create_another_db()
        self.catalog.alter_database(self.db1, new_db, False)
        new_properties = self.catalog.get_database(self.db1).get_properties()
        old_properties = db.get_properties()
        self.assertFalse(all((k in new_properties for k in old_properties.keys())))
        self.check_catalog_database_equals(new_db, self.catalog.get_database(self.db1))

    def test_alter_db_database_not_exist_exception(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(DatabaseNotExistException):
            self.catalog.alter_database('nonexistent', self.create_db(), False)

    def test_alter_db_database_not_exist_ignored(self):
        if False:
            return 10
        self.catalog.alter_database('nonexistent', self.create_db(), True)
        self.assertFalse(self.catalog.database_exists('nonexistent'))

    def test_db_exists(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(self.catalog.database_exists('nonexistent'))
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.assertTrue(self.catalog.database_exists(self.db1))

    def test_create_table_streaming(self):
        if False:
            while True:
                i = 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        table = self.create_stream_table()
        self.catalog.create_table(self.path1, table, False)
        self.check_catalog_table_equals(table, self.catalog.get_table(self.path1))

    def test_create_table_batch(self):
        if False:
            return 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        table = self.create_table()
        self.catalog.create_table(self.path1, table, False)
        table_created = self.catalog.get_table(self.path1)
        self.check_catalog_table_equals(table, table_created)
        self.assertEqual(self.test_comment, table_created.get_description())
        tables = self.catalog.list_tables(self.db1)
        self.assertEqual(1, len(tables))
        self.assertEqual(self.path1.get_object_name(), tables[0])
        self.catalog.drop_table(self.path1, False)
        self.table = self.create_partitioned_table()
        self.catalog.create_table(self.path1, table, False)
        self.check_catalog_table_equals(table, self.catalog.get_table(self.path1))
        tables = self.catalog.list_tables(self.db1)
        self.assertEqual(1, len(tables))
        self.assertEqual(self.path1.get_object_name(), tables[0])

    def test_create_table_database_not_exist_exception(self):
        if False:
            return 10
        self.assertFalse(self.catalog.database_exists(self.db1))
        with self.assertRaises(DatabaseNotExistException):
            self.catalog.create_table(self.non_exist_object_path, self.create_table(), False)

    def test_create_table_table_already_exist_exception(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_table(), False)
        with self.assertRaises(TableAlreadyExistException):
            self.catalog.create_table(self.path1, self.create_table(), False)

    def test_create_table_table_already_exist_ignored(self):
        if False:
            i = 10
            return i + 15
        self.catalog.create_database(self.db1, self.create_db(), False)
        table = self.create_table()
        self.catalog.create_table(self.path1, table, False)
        self.check_catalog_table_equals(table, self.catalog.get_table(self.path1))
        self.catalog.create_table(self.path1, self.create_another_table(), True)
        self.check_catalog_table_equals(table, self.catalog.get_table(self.path1))

    def test_get_table_table_not_exist_exception(self):
        if False:
            return 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        with self.assertRaises(TableNotExistException):
            self.catalog.get_table(self.non_exist_object_path)

    def test_get_table_table_not_exist_exception_no_db(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TableNotExistException):
            self.catalog.get_table(self.non_exist_object_path)

    def test_drop_table_non_partitioned_table(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_table(), False)
        self.assertTrue(self.catalog.table_exists(self.path1))
        self.catalog.drop_table(self.path1, False)
        self.assertFalse(self.catalog.table_exists(self.path1))

    def test_drop_table_table_not_exist_exception(self):
        if False:
            return 10
        with self.assertRaises(TableNotExistException):
            self.catalog.drop_table(self.non_exist_db_path, False)

    def test_drop_table_table_not_exist_ignored(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.drop_table(self.non_exist_object_path, True)

    def test_alter_table(self):
        if False:
            i = 10
            return i + 15
        self.catalog.create_database(self.db1, self.create_db(), False)
        table = self.create_table()
        self.catalog.create_table(self.path1, table, False)
        self.check_catalog_table_equals(table, self.catalog.get_table(self.path1))
        new_table = self.create_another_table()
        self.catalog.alter_table(self.path1, new_table, False)
        self.assertNotEqual(table, self.catalog.get_table(self.path1))
        self.check_catalog_table_equals(new_table, self.catalog.get_table(self.path1))
        self.catalog.drop_table(self.path1, False)
        table = self.create_partitioned_table()
        self.catalog.create_table(self.path1, table, False)
        self.check_catalog_table_equals(table, self.catalog.get_table(self.path1))
        new_table = self.create_another_partitioned_table()
        self.catalog.alter_table(self.path1, new_table, False)
        self.check_catalog_table_equals(new_table, self.catalog.get_table(self.path1))
        view = self.create_view()
        self.catalog.create_table(self.path3, view, False)
        self.check_catalog_view_equals(view, self.catalog.get_table(self.path3))
        new_view = self.create_another_view()
        self.catalog.alter_table(self.path3, new_view, False)
        self.assertNotEqual(view, self.catalog.get_table(self.path3))
        self.check_catalog_view_equals(new_view, self.catalog.get_table(self.path3))

    def test_alter_table_table_not_exist_exception(self):
        if False:
            return 10
        with self.assertRaises(TableNotExistException):
            self.catalog.alter_table(self.non_exist_db_path, self.create_table(), False)

    def test_alter_table_table_not_exist_ignored(self):
        if False:
            while True:
                i = 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.alter_table(self.non_exist_object_path, self.create_table(), True)
        self.assertFalse(self.catalog.table_exists(self.non_exist_object_path))

    def test_rename_table_non_partitioned_table(self):
        if False:
            while True:
                i = 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        table = self.create_table()
        self.catalog.create_table(self.path1, table, False)
        self.check_catalog_table_equals(table, self.catalog.get_table(self.path1))
        self.catalog.rename_table(self.path1, self.t2, False)
        self.check_catalog_table_equals(table, self.catalog.get_table(self.path3))
        self.assertFalse(self.catalog.table_exists(self.path1))

    def test_rename_table_table_not_exist_exception(self):
        if False:
            i = 10
            return i + 15
        self.catalog.create_database(self.db1, self.create_db(), False)
        with self.assertRaises(TableNotExistException):
            self.catalog.rename_table(self.path1, self.t2, False)

    def test_rename_table_table_not_exist_exception_ignored(self):
        if False:
            return 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.rename_table(self.path1, self.t2, True)

    def test_rename_table_table_already_exist_exception(self):
        if False:
            while True:
                i = 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        table = self.create_table()
        self.catalog.create_table(self.path1, table, False)
        self.catalog.create_table(self.path3, self.create_another_table(), False)
        with self.assertRaises(TableAlreadyExistException):
            self.catalog.rename_table(self.path1, self.t2, False)

    def test_list_tables(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_table(), False)
        self.catalog.create_table(self.path3, self.create_table(), False)
        self.catalog.create_table(self.path4, self.create_view(), False)
        self.assertEqual(3, len(self.catalog.list_tables(self.db1)))
        self.assertEqual(1, len(self.catalog.list_views(self.db1)))

    def test_table_exists(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.assertFalse(self.catalog.table_exists(self.path1))
        self.catalog.create_table(self.path1, self.create_table(), False)
        self.assertTrue(self.catalog.table_exists(self.path1))

    def test_create_view(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.assertFalse(self.catalog.table_exists(self.path1))
        view = self.create_view()
        self.catalog.create_table(self.path1, view, False)
        self.check_catalog_view_equals(view, self.catalog.get_table(self.path1))

    def test_create_view_database_not_exist_exception(self):
        if False:
            return 10
        self.assertFalse(self.catalog.database_exists(self.db1))
        with self.assertRaises(DatabaseNotExistException):
            self.catalog.create_table(self.non_exist_object_path, self.create_view(), False)

    def test_create_view_table_already_exist_exception(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_view(), False)
        with self.assertRaises(TableAlreadyExistException):
            self.catalog.create_table(self.path1, self.create_view(), False)

    def test_create_view_table_already_exist_ignored(self):
        if False:
            i = 10
            return i + 15
        self.catalog.create_database(self.db1, self.create_db(), False)
        view = self.create_view()
        self.catalog.create_table(self.path1, view, False)
        self.check_catalog_view_equals(view, self.catalog.get_table(self.path1))
        self.catalog.create_table(self.path1, self.create_another_view(), True)
        self.check_catalog_view_equals(view, self.catalog.get_table(self.path1))

    def test_drop_view(self):
        if False:
            return 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_view(), False)
        self.assertTrue(self.catalog.table_exists(self.path1))
        self.catalog.drop_table(self.path1, False)
        self.assertFalse(self.catalog.table_exists(self.path1))

    def test_alter_view(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        view = self.create_view()
        self.catalog.create_table(self.path1, view, False)
        self.check_catalog_view_equals(view, self.catalog.get_table(self.path1))
        new_view = self.create_another_view()
        self.catalog.alter_table(self.path1, new_view, False)
        self.check_catalog_view_equals(new_view, self.catalog.get_table(self.path1))

    def test_alter_view_table_not_exist_exception(self):
        if False:
            return 10
        with self.assertRaises(TableNotExistException):
            self.catalog.alter_table(self.non_exist_db_path, self.create_table(), False)

    def test_alter_view_table_not_exist_ignored(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.alter_table(self.non_exist_object_path, self.create_view(), True)
        self.assertFalse(self.catalog.table_exists(self.non_exist_object_path))

    def test_list_view(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.assertTrue(0 == len(self.catalog.list_tables(self.db1)))
        self.catalog.create_table(self.path1, self.create_view(), False)
        self.catalog.create_table(self.path3, self.create_table(), False)
        self.assertEqual(2, len(self.catalog.list_tables(self.db1)))
        self.assertEqual({self.path1.get_object_name(), self.path3.get_object_name()}, set(self.catalog.list_tables(self.db1)))
        self.assertEqual([self.path1.get_object_name()], self.catalog.list_views(self.db1))

    def test_rename_view(self):
        if False:
            i = 10
            return i + 15
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_view(), False)
        self.assertTrue(self.catalog.table_exists(self.path1))
        self.catalog.rename_table(self.path1, self.t2, False)
        self.assertFalse(self.catalog.table_exists(self.path1))
        self.assertTrue(self.catalog.table_exists(self.path3))

    def test_create_function(self):
        if False:
            return 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.assertFalse(self.catalog.function_exists(self.path1))
        self.catalog.create_function(self.path1, self.create_function(), False)
        self.assertTrue(self.catalog.function_exists(self.path1))

    def test_create_function_database_not_exist_exception(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.catalog.database_exists(self.db1))
        with self.assertRaises(DatabaseNotExistException):
            self.catalog.create_function(self.path1, self.create_function(), False)

    def test_create_functin_function_already_exist_function(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_function(self.path1, self.create_function(), False)
        self.assertTrue(self.catalog.function_exists(self.path1))
        self.catalog.create_function(self.path1, self.create_another_function(), True)
        with self.assertRaises(FunctionAlreadyExistException):
            self.catalog.create_function(self.path1, self.create_function(), False)

    def test_alter_function(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        func = self.create_function()
        self.catalog.create_function(self.path1, func, False)
        self.check_catalog_function_equals(func, self.catalog.get_function(self.path1))
        new_func = self.create_another_function()
        self.catalog.alter_function(self.path1, new_func, False)
        actual = self.catalog.get_function(self.path1)
        self.assertFalse(func.get_class_name() == actual.get_class_name())
        self.check_catalog_function_equals(new_func, actual)

    def test_alter_function_function_not_exist_exception(self):
        if False:
            return 10
        with self.assertRaises(FunctionNotExistException):
            self.catalog.alter_function(self.non_exist_object_path, self.create_function(), False)

    def test_alter_function_function_not_exist_ignored(self):
        if False:
            while True:
                i = 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.alter_function(self.non_exist_object_path, self.create_function(), True)
        self.assertFalse(self.catalog.function_exists(self.non_exist_object_path))

    def test_list_functions(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        func = self.create_function()
        self.catalog.create_function(self.path1, func, False)
        self.assertEqual(self.path1.get_object_name(), self.catalog.list_functions(self.db1)[0])

    def test_list_functions_database_not_exist_exception(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(DatabaseNotExistException):
            self.catalog.list_functions(self.db1)

    def test_get_function_function_not_exist_exception(self):
        if False:
            while True:
                i = 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        with self.assertRaises(FunctionNotExistException):
            self.catalog.get_function(self.non_exist_object_path)

    def test_get_function_function_not_exist_exception_no_db(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(FunctionNotExistException):
            self.catalog.get_function(self.non_exist_object_path)

    def test_drop_function(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_function(self.path1, self.create_function(), False)
        self.assertTrue(self.catalog.function_exists(self.path1))
        self.catalog.drop_function(self.path1, False)
        self.assertFalse(self.catalog.function_exists(self.path1))

    def test_drop_function_function_not_exist_exception(self):
        if False:
            print('Hello World!')
        with self.assertRaises(FunctionNotExistException):
            self.catalog.drop_function(self.non_exist_db_path, False)

    def test_drop_function_function_not_exist_ignored(self):
        if False:
            return 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.drop_function(self.non_exist_object_path, True)
        self.catalog.drop_database(self.db1, False)

    def test_create_partition(self):
        if False:
            return 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_partitioned_table(), False)
        self.assertTrue(0 == len(self.catalog.list_partitions(self.path1)))
        self.catalog.create_partition(self.path1, self.create_partition_spec(), self.create_partition(), False)
        self.check_catalog_partition_equals(self.create_partition(), self.catalog.get_partition(self.path1, self.create_partition_spec()))
        self.catalog.create_partition(self.path1, self.create_another_partition_spec(), self.create_partition(), False)
        self.check_catalog_partition_equals(self.create_partition(), self.catalog.get_partition(self.path1, self.create_another_partition_spec()))

    def test_create_partition_table_not_exist_exception(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        with self.assertRaises(TableNotExistException):
            self.catalog.create_partition(self.path1, self.create_partition_spec(), self.create_partition(), False)

    def test_create_partition_table_not_partitoned_exception(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_table(), False)
        with self.assertRaises(TableNotPartitionedException):
            self.catalog.create_partition(self.path1, self.create_partition_spec(), self.create_partition(), False)

    def test_create_partition_partition_spec_invalid_exception(self):
        if False:
            i = 10
            return i + 15
        self.catalog.create_database(self.db1, self.create_db(), False)
        table = self.create_partitioned_table()
        self.catalog.create_table(self.path1, table, False)
        partition_spec = self.create_invalid_partition_spec_subset()
        with self.assertRaises(PartitionSpecInvalidException):
            self.catalog.create_partition(self.path1, partition_spec, self.create_partition(), False)

    def test_create_partition_partition_already_exists_exception(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_partitioned_table(), False)
        partition = self.create_partition()
        self.catalog.create_partition(self.path1, self.create_partition_spec(), partition, False)
        partition_spec = self.create_partition_spec()
        with self.assertRaises(PartitionAlreadyExistsException):
            self.catalog.create_partition(self.path1, partition_spec, self.create_partition(), False)

    def test_create_partition_partition_already_exists_ignored(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_partitioned_table(), False)
        partition_spec = self.create_partition_spec()
        self.catalog.create_partition(self.path1, partition_spec, self.create_partition(), False)
        self.catalog.create_partition(self.path1, partition_spec, self.create_partition(), True)

    def test_drop_partition(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_partitioned_table(), False)
        self.catalog.create_partition(self.path1, self.create_partition_spec(), self.create_partition(), False)
        self.catalog.drop_partition(self.path1, self.create_partition_spec(), False)
        self.assertEqual([], self.catalog.list_partitions(self.path1))

    def test_drop_partition_table_not_exist(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        partition_spec = self.create_partition_spec()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.drop_partition(self.path1, partition_spec, False)

    def test_drop_partition_table_not_partitioned(self):
        if False:
            i = 10
            return i + 15
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_table(), False)
        partition_spec = self.create_partition_spec()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.drop_partition(self.path1, partition_spec, False)

    def test_drop_partition_partition_spec_invalid(self):
        if False:
            while True:
                i = 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        table = self.create_partitioned_table()
        self.catalog.create_table(self.path1, table, False)
        partition_spec = self.create_invalid_partition_spec_subset()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.drop_partition(self.path1, partition_spec, False)

    def test_drop_partition_patition_not_exist(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_partitioned_table(), False)
        partition_spec = self.create_partition_spec()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.drop_partition(self.path1, partition_spec, False)

    def test_drop_partition_patition_not_exist_ignored(self):
        if False:
            return 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_partitioned_table(), False)
        self.catalog.drop_partition(self.path1, self.create_partition_spec(), True)

    def test_alter_partition(self):
        if False:
            return 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_partitioned_table(), False)
        self.catalog.create_partition(self.path1, self.create_partition_spec(), self.create_partition(), False)
        cp = self.catalog.get_partition(self.path1, self.create_partition_spec())
        self.check_catalog_partition_equals(self.create_partition(), cp)
        self.assertIsNone(cp.get_properties().get('k'))
        another = CatalogPartition.create_instance({'is_streaming': 'false', 'k': 'v'}, 'catalog partition')
        self.catalog.alter_partition(self.path1, self.create_partition_spec(), another, False)
        cp = self.catalog.get_partition(self.path1, self.create_partition_spec())
        self.check_catalog_partition_equals(another, cp)
        self.assertEqual('v', cp.get_properties().get('k'))

    def test_alter_partition_table_not_exist(self):
        if False:
            return 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        partition_spec = self.create_partition_spec()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.alter_partition(self.path1, partition_spec, self.create_partition(), False)

    def test_alter_partition_table_not_partitioned(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_table(), False)
        partition_spec = self.create_partition_spec()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.alter_partition(self.path1, partition_spec, self.create_partition(), False)

    def test_alter_partition_partition_spec_invalid(self):
        if False:
            i = 10
            return i + 15
        self.catalog.create_database(self.db1, self.create_db(), False)
        table = self.create_partitioned_table()
        self.catalog.create_table(self.path1, table, False)
        partition_spec = self.create_invalid_partition_spec_subset()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.alter_partition(self.path1, partition_spec, self.create_partition(), False)

    def test_alter_partition_partition_not_exist(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_partitioned_table(), False)
        partition_spec = self.create_partition_spec()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.alter_partition(self.path1, partition_spec, self.create_partition(), False)

    def test_alter_partition_partition_not_exist_ignored(self):
        if False:
            while True:
                i = 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_partitioned_table(), False)
        self.catalog.alter_partition(self.path1, self.create_partition_spec(), self.create_partition(), True)

    def test_get_partition_table_not_exists(self):
        if False:
            i = 10
            return i + 15
        partition_spec = self.create_partition_spec()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.get_partition(self.path1, partition_spec)

    def test_get_partition_table_not_partitioned(self):
        if False:
            while True:
                i = 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_table(), False)
        partition_spec = self.create_partition_spec()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.get_partition(self.path1, partition_spec)

    def test_get_partition_partition_spec_invalid_invalid_partition_spec(self):
        if False:
            for i in range(10):
                print('nop')
        self.catalog.create_database(self.db1, self.create_db(), False)
        table = self.create_partitioned_table()
        self.catalog.create_table(self.path1, table, False)
        partition_spec = self.create_invalid_partition_spec_subset()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.get_partition(self.path1, partition_spec)

    def test_get_partition_partition_spec_invalid_size_not_equal(self):
        if False:
            return 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        table = self.create_partitioned_table()
        self.catalog.create_table(self.path1, table, False)
        partition_spec = self.create_partition_spec_subset()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.get_partition(self.path1, partition_spec)

    def test_get_partition_partition_not_exist(self):
        if False:
            i = 10
            return i + 15
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_partitioned_table(), False)
        partition_spec = self.create_partition_spec()
        with self.assertRaises(PartitionNotExistException):
            self.catalog.get_partition(self.path1, partition_spec)

    def test_partition_exists(self):
        if False:
            return 10
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_partitioned_table(), False)
        self.catalog.create_partition(self.path1, self.create_partition_spec(), self.create_partition(), False)
        self.assertTrue(self.catalog.partition_exists(self.path1, self.create_partition_spec()))
        self.assertFalse(self.catalog.partition_exists(self.path2, self.create_partition_spec()))
        self.assertFalse(self.catalog.partition_exists(ObjectPath.from_string('non.exist'), self.create_partition_spec()))

    def test_list_partition_partial_spec(self):
        if False:
            print('Hello World!')
        self.catalog.create_database(self.db1, self.create_db(), False)
        self.catalog.create_table(self.path1, self.create_partitioned_table(), False)
        self.catalog.create_partition(self.path1, self.create_partition_spec(), self.create_partition(), False)
        self.catalog.create_partition(self.path1, self.create_another_partition_spec(), self.create_partition(), False)
        self.assertEqual(2, len(self.catalog.list_partitions(self.path1, self.create_partition_spec_subset())))
        self.assertEqual(1, len(self.catalog.list_partitions(self.path1, self.create_another_partition_spec_subset())))