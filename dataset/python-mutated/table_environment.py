import atexit
import os
import sys
import tempfile
import warnings
from typing import Union, List, Tuple, Iterable
from py4j.java_gateway import get_java_class, get_method
from pyflink.common.configuration import Configuration
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table.sources import TableSource
from pyflink.common.typeinfo import TypeInformation
from pyflink.datastream.data_stream import DataStream
from pyflink.java_gateway import get_gateway
from pyflink.serializers import BatchedSerializer, PickleSerializer
from pyflink.table import Table, EnvironmentSettings, Expression, ExplainDetail, Module, ModuleEntry, TableSink, Schema, ChangelogMode
from pyflink.table.catalog import Catalog
from pyflink.table.serializers import ArrowSerializer
from pyflink.table.statement_set import StatementSet
from pyflink.table.table_config import TableConfig
from pyflink.table.table_descriptor import TableDescriptor
from pyflink.table.table_result import TableResult
from pyflink.table.types import _create_type_verifier, RowType, DataType, _infer_schema_from_data, _create_converter, from_arrow_type, RowField, create_arrow_schema, _to_java_data_type
from pyflink.table.udf import UserDefinedFunctionWrapper, AggregateFunction, udaf, udtaf, TableAggregateFunction
from pyflink.table.utils import to_expression_jarray
from pyflink.util import java_utils
from pyflink.util.java_utils import get_j_env_configuration, is_local_deployment, load_java_class, to_j_explain_detail_arr, to_jarray, get_field
__all__ = ['StreamTableEnvironment', 'TableEnvironment']

class TableEnvironment(object):
    """
    A table environment is the base class, entry point, and central context for creating Table
    and SQL API programs.

    It is unified for bounded and unbounded data processing.

    A table environment is responsible for:

        - Connecting to external systems.
        - Registering and retrieving :class:`~pyflink.table.Table` and other meta objects from a
          catalog.
        - Executing SQL statements.
        - Offering further configuration options.

    The path in methods such as :func:`create_temporary_view`
    should be a proper SQL identifier. The syntax is following
    [[catalog-name.]database-name.]object-name, where the catalog name and database are optional.
    For path resolution see :func:`use_catalog` and :func:`use_database`. All keywords or other
    special characters need to be escaped.

    Example: `cat.1`.`db`.`Table` resolves to an object named 'Table' (table is a reserved
    keyword, thus must be escaped) in a catalog named 'cat.1' and database named 'db'.

    .. note::

        This environment is meant for pure table programs. If you would like to convert from or to
        other Flink APIs, it might be necessary to use one of the available language-specific table
        environments in the corresponding bridging modules.

    """

    def __init__(self, j_tenv, serializer=PickleSerializer()):
        if False:
            for i in range(10):
                print('nop')
        self._j_tenv = j_tenv
        self._serializer = serializer
        self._set_python_executable_for_local_executor()
        self._config_chaining_optimization()
        self._open()

    @staticmethod
    def create(environment_settings: Union[EnvironmentSettings, Configuration]) -> 'TableEnvironment':
        if False:
            return 10
        '\n        Creates a table environment that is the entry point and central context for creating Table\n        and SQL API programs.\n\n        :param environment_settings: The configuration or environment settings used to instantiate\n            the :class:`~pyflink.table.TableEnvironment`, the name is for backward compatibility.\n        :return: The :class:`~pyflink.table.TableEnvironment`.\n        '
        gateway = get_gateway()
        if isinstance(environment_settings, Configuration):
            environment_settings = EnvironmentSettings.new_instance().with_configuration(environment_settings).build()
        elif not isinstance(environment_settings, EnvironmentSettings):
            raise TypeError('argument should be EnvironmentSettings or Configuration')
        j_tenv = gateway.jvm.TableEnvironment.create(environment_settings._j_environment_settings)
        return TableEnvironment(j_tenv)

    def from_table_source(self, table_source: 'TableSource') -> 'Table':
        if False:
            while True:
                i = 10
        "\n        Creates a table from a table source.\n\n        Example:\n        ::\n\n            >>> csv_table_source = CsvTableSource(\n            ...     csv_file_path, ['a', 'b'], [DataTypes.STRING(), DataTypes.BIGINT()])\n            >>> table_env.from_table_source(csv_table_source)\n\n        :param table_source: The table source used as table.\n        :return: The result table.\n        "
        warnings.warn('Deprecated in 1.11.', DeprecationWarning)
        return Table(self._j_tenv.fromTableSource(table_source._j_table_source), self)

    def register_catalog(self, catalog_name: str, catalog: Catalog):
        if False:
            i = 10
            return i + 15
        '\n        Registers a :class:`~pyflink.table.catalog.Catalog` under a unique name.\n        All tables registered in the :class:`~pyflink.table.catalog.Catalog` can be accessed.\n\n        :param catalog_name: The name under which the catalog will be registered.\n        :param catalog: The catalog to register.\n        '
        self._j_tenv.registerCatalog(catalog_name, catalog._j_catalog)

    def get_catalog(self, catalog_name: str) -> Catalog:
        if False:
            print('Hello World!')
        '\n        Gets a registered :class:`~pyflink.table.catalog.Catalog` by name.\n\n        :param catalog_name: The name to look up the :class:`~pyflink.table.catalog.Catalog`.\n        :return: The requested catalog, None if there is no\n                 registered catalog with given name.\n        '
        catalog = self._j_tenv.getCatalog(catalog_name)
        if catalog.isPresent():
            return Catalog(catalog.get())
        else:
            return None

    def load_module(self, module_name: str, module: Module):
        if False:
            print('Hello World!')
        '\n        Loads a :class:`~pyflink.table.Module` under a unique name. Modules will be kept\n        in the loaded order.\n        ValidationException is thrown when there is already a module with the same name.\n\n        :param module_name: Name of the :class:`~pyflink.table.Module`.\n        :param module: The module instance.\n\n        .. versionadded:: 1.12.0\n        '
        self._j_tenv.loadModule(module_name, module._j_module)

    def unload_module(self, module_name: str):
        if False:
            i = 10
            return i + 15
        '\n        Unloads a :class:`~pyflink.table.Module` with given name.\n        ValidationException is thrown when there is no module with the given name.\n\n        :param module_name: Name of the :class:`~pyflink.table.Module`.\n\n        .. versionadded:: 1.12.0\n        '
        self._j_tenv.unloadModule(module_name)

    def use_modules(self, *module_names: str):
        if False:
            return 10
        '\n        Use an array of :class:`~pyflink.table.Module` with given names.\n        ValidationException is thrown when there is duplicate name or no module with the given name.\n\n        :param module_names: Names of the modules to be used.\n\n        .. versionadded:: 1.13.0\n        '
        j_module_names = to_jarray(get_gateway().jvm.String, module_names)
        self._j_tenv.useModules(j_module_names)

    def create_java_temporary_system_function(self, name: str, function_class_name: str):
        if False:
            while True:
                i = 10
        '\n        Registers a java user defined function class as a temporary system function.\n\n        Compared to .. seealso:: :func:`create_java_temporary_function`, system functions are\n        identified by a global name that is independent of the current catalog and current\n        database. Thus, this method allows to extend the set of built-in system functions like\n        TRIM, ABS, etc.\n\n        Temporary functions can shadow permanent ones. If a permanent function under a given name\n        exists, it will be inaccessible in the current session. To make the permanent function\n        available again one can drop the corresponding temporary system function.\n\n        Example:\n        ::\n\n            >>> table_env.create_java_temporary_system_function("func",\n            ...     "java.user.defined.function.class.name")\n\n        :param name: The name under which the function will be registered globally.\n        :param function_class_name: The java full qualified class name of the function class\n                                    containing the implementation. The function must have a\n                                    public no-argument constructor and can be founded in current\n                                    Java classloader.\n\n        .. versionadded:: 1.12.0\n        '
        gateway = get_gateway()
        java_function = gateway.jvm.Thread.currentThread().getContextClassLoader().loadClass(function_class_name)
        self._j_tenv.createTemporarySystemFunction(name, java_function)

    def create_temporary_system_function(self, name: str, function: Union[UserDefinedFunctionWrapper, AggregateFunction]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Registers a python user defined function class as a temporary system function.\n\n        Compared to .. seealso:: :func:`create_temporary_function`, system functions are identified\n        by a global name that is independent of the current catalog and current database. Thus,\n        this method allows to extend the set of built-in system functions like TRIM, ABS, etc.\n\n        Temporary functions can shadow permanent ones. If a permanent function under a given name\n        exists, it will be inaccessible in the current session. To make the permanent function\n        available again one can drop the corresponding temporary system function.\n\n        Example:\n        ::\n\n            >>> table_env.create_temporary_system_function(\n            ...     "add_one", udf(lambda i: i + 1, result_type=DataTypes.BIGINT()))\n\n            >>> @udf(result_type=DataTypes.BIGINT())\n            ... def add(i, j):\n            ...     return i + j\n            >>> table_env.create_temporary_system_function("add", add)\n\n            >>> class SubtractOne(ScalarFunction):\n            ...     def eval(self, i):\n            ...         return i - 1\n            >>> table_env.create_temporary_system_function(\n            ...     "subtract_one", udf(SubtractOne(), result_type=DataTypes.BIGINT()))\n\n        :param name: The name under which the function will be registered globally.\n        :param function: The function class containing the implementation. The function must have a\n                         public no-argument constructor and can be founded in current Java\n                         classloader.\n\n        .. versionadded:: 1.12.0\n        '
        function = self._wrap_aggregate_function_if_needed(function)
        java_function = function._java_user_defined_function()
        self._j_tenv.createTemporarySystemFunction(name, java_function)

    def drop_temporary_system_function(self, name: str) -> bool:
        if False:
            while True:
                i = 10
        '\n        Drops a temporary system function registered under the given name.\n\n        If a permanent function with the given name exists, it will be used from now on for any\n        queries that reference this name.\n\n        :param name: The name under which the function has been registered globally.\n        :return: true if a function existed under the given name and was removed.\n\n        .. versionadded:: 1.12.0\n        '
        return self._j_tenv.dropTemporarySystemFunction(name)

    def create_java_function(self, path: str, function_class_name: str, ignore_if_exists: bool=None):
        if False:
            while True:
                i = 10
        '\n        Registers a java user defined function class as a catalog function in the given path.\n\n        Compared to system functions with a globally defined name, catalog functions are always\n        (implicitly or explicitly) identified by a catalog and database.\n\n        There must not be another function (temporary or permanent) registered under the same path.\n\n        Example:\n        ::\n\n            >>> table_env.create_java_function("func", "java.user.defined.function.class.name")\n\n        :param path: The path under which the function will be registered.\n                     See also the :class:`~pyflink.table.TableEnvironment` class description for\n                     the format of the path.\n        :param function_class_name: The java full qualified class name of the function class\n                                    containing the implementation. The function must have a\n                                    public no-argument constructor and can be founded in current\n                                    Java classloader.\n        :param ignore_if_exists: If a function exists under the given path and this flag is set,\n                                 no operation is executed. An exception is thrown otherwise.\n\n        .. versionadded:: 1.12.0\n        '
        gateway = get_gateway()
        java_function = gateway.jvm.Thread.currentThread().getContextClassLoader().loadClass(function_class_name)
        if ignore_if_exists is None:
            self._j_tenv.createFunction(path, java_function)
        else:
            self._j_tenv.createFunction(path, java_function, ignore_if_exists)

    def drop_function(self, path: str) -> bool:
        if False:
            while True:
                i = 10
        '\n        Drops a catalog function registered in the given path.\n\n        :param path: The path under which the function will be registered.\n                     See also the :class:`~pyflink.table.TableEnvironment` class description for\n                     the format of the path.\n        :return: true if a function existed in the given path and was removed.\n\n        .. versionadded:: 1.12.0\n        '
        return self._j_tenv.dropFunction(path)

    def create_java_temporary_function(self, path: str, function_class_name: str):
        if False:
            while True:
                i = 10
        '\n        Registers a java user defined function class as a temporary catalog function.\n\n        Compared to .. seealso:: :func:`create_java_temporary_system_function` with a globally\n        defined name, catalog functions are always (implicitly or explicitly) identified by a\n        catalog and database.\n\n        Temporary functions can shadow permanent ones. If a permanent function under a given name\n        exists, it will be inaccessible in the current session. To make the permanent function\n        available again one can drop the corresponding temporary function.\n\n        Example:\n        ::\n\n            >>> table_env.create_java_temporary_function("func",\n            ...     "java.user.defined.function.class.name")\n\n        :param path: The path under which the function will be registered.\n                     See also the :class:`~pyflink.table.TableEnvironment` class description for\n                     the format of the path.\n        :param function_class_name: The java full qualified class name of the function class\n                                    containing the implementation. The function must have a\n                                    public no-argument constructor and can be founded in current\n                                    Java classloader.\n\n        .. versionadded:: 1.12.0\n        '
        gateway = get_gateway()
        java_function = gateway.jvm.Thread.currentThread().getContextClassLoader().loadClass(function_class_name)
        self._j_tenv.createTemporaryFunction(path, java_function)

    def create_temporary_function(self, path: str, function: Union[UserDefinedFunctionWrapper, AggregateFunction]):
        if False:
            print('Hello World!')
        '\n        Registers a python user defined function class as a temporary catalog function.\n\n        Compared to .. seealso:: :func:`create_temporary_system_function` with a globally defined\n        name, catalog functions are always (implicitly or explicitly) identified by a catalog and\n        database.\n\n        Temporary functions can shadow permanent ones. If a permanent function under a given name\n        exists, it will be inaccessible in the current session. To make the permanent function\n        available again one can drop the corresponding temporary function.\n\n        Example:\n        ::\n\n            >>> table_env.create_temporary_function(\n            ...     "add_one", udf(lambda i: i + 1, result_type=DataTypes.BIGINT()))\n\n            >>> @udf(result_type=DataTypes.BIGINT())\n            ... def add(i, j):\n            ...     return i + j\n            >>> table_env.create_temporary_function("add", add)\n\n            >>> class SubtractOne(ScalarFunction):\n            ...     def eval(self, i):\n            ...         return i - 1\n            >>> table_env.create_temporary_function(\n            ...     "subtract_one", udf(SubtractOne(), result_type=DataTypes.BIGINT()))\n\n        :param path: The path under which the function will be registered.\n                     See also the :class:`~pyflink.table.TableEnvironment` class description for\n                     the format of the path.\n        :param function: The function class containing the implementation. The function must have a\n                         public no-argument constructor and can be founded in current Java\n                         classloader.\n\n        .. versionadded:: 1.12.0\n        '
        function = self._wrap_aggregate_function_if_needed(function)
        java_function = function._java_user_defined_function()
        self._j_tenv.createTemporaryFunction(path, java_function)

    def drop_temporary_function(self, path: str) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Drops a temporary system function registered under the given name.\n\n        If a permanent function with the given name exists, it will be used from now on for any\n        queries that reference this name.\n\n        :param path: The path under which the function will be registered.\n                     See also the :class:`~pyflink.table.TableEnvironment` class description for\n                     the format of the path.\n        :return: true if a function existed in the given path and was removed.\n\n        .. versionadded:: 1.12.0\n        '
        return self._j_tenv.dropTemporaryFunction(path)

    def create_temporary_table(self, path: str, descriptor: TableDescriptor):
        if False:
            return 10
        '\n        Registers the given :class:`~pyflink.table.TableDescriptor` as a temporary catalog table.\n\n        The TableDescriptor is converted into a CatalogTable and stored in the catalog.\n\n        Temporary objects can shadow permanent ones. If a permanent object in a given path exists,\n        it will be inaccessible in the current session. To make the permanent object available again\n        one can drop the corresponding temporary object.\n\n        Examples:\n        ::\n\n            >>> table_env.create_temporary_table("MyTable", TableDescriptor.for_connector("datagen")\n            ...     .schema(Schema.new_builder()\n            ...         .column("f0", DataTypes.STRING())\n            ...         .build())\n            ...     .option("rows-per-second", 10)\n            ...     .option("fields.f0.kind", "random")\n            ...     .build())\n\n        :param path: The path under which the table will be registered.\n        :param descriptor: Template for creating a CatalogTable instance.\n\n        .. versionadded:: 1.14.0\n        '
        self._j_tenv.createTemporaryTable(path, descriptor._j_table_descriptor)

    def create_table(self, path: str, descriptor: TableDescriptor):
        if False:
            for i in range(10):
                print('nop')
        '\n        Registers the given :class:`~pyflink.table.TableDescriptor` as a catalog table.\n\n        The TableDescriptor is converted into a CatalogTable and stored in the catalog.\n\n        If the table should not be permanently stored in a catalog, use\n        :func:`create_temporary_table` instead.\n\n        Examples:\n        ::\n\n            >>> table_env.create_table("MyTable", TableDescriptor.for_connector("datagen")\n            ...     .schema(Schema.new_builder()\n            ...                   .column("f0", DataTypes.STRING())\n            ...                   .build())\n            ...     .option("rows-per-second", 10)\n            ...     .option("fields.f0.kind", "random")\n            ...     .build())\n\n        :param path: The path under which the table will be registered.\n        :param descriptor: Template for creating a CatalogTable instance.\n\n        .. versionadded:: 1.14.0\n        '
        self._j_tenv.createTable(path, descriptor._j_table_descriptor)

    def register_table(self, name: str, table: Table):
        if False:
            while True:
                i = 10
        '\n        Registers a :class:`~pyflink.table.Table` under a unique name in the TableEnvironment\'s\n        catalog. Registered tables can be referenced in SQL queries.\n\n        Example:\n        ::\n\n            >>> tab = table_env.from_elements([(1, \'Hi\'), (2, \'Hello\')], [\'a\', \'b\'])\n            >>> table_env.register_table("source", tab)\n\n        :param name: The name under which the table will be registered.\n        :param table: The table to register.\n\n        .. note:: Deprecated in 1.10. Use :func:`create_temporary_view` instead.\n        '
        warnings.warn('Deprecated in 1.10. Use create_temporary_view instead.', DeprecationWarning)
        self._j_tenv.registerTable(name, table._j_table)

    def register_table_source(self, name: str, table_source: TableSource):
        if False:
            for i in range(10):
                print('nop')
        '\n        Registers an external :class:`~pyflink.table.TableSource` in this\n        :class:`~pyflink.table.TableEnvironment`\'s catalog. Registered tables can be referenced in\n        SQL queries.\n\n        Example:\n        ::\n\n            >>> table_env.register_table_source("source",\n            ...                                 CsvTableSource("./1.csv",\n            ...                                                ["a", "b"],\n            ...                                                [DataTypes.INT(),\n            ...                                                 DataTypes.STRING()]))\n\n        :param name: The name under which the table source is registered.\n        :param table_source: The table source to register.\n\n        .. note:: Deprecated in 1.10. Use :func:`execute_sql` instead.\n        '
        warnings.warn('Deprecated in 1.10. Use create_table instead.', DeprecationWarning)
        self._j_tenv.registerTableSourceInternal(name, table_source._j_table_source)

    def register_table_sink(self, name: str, table_sink: TableSink):
        if False:
            return 10
        '\n        Registers an external :class:`~pyflink.table.TableSink` with given field names and types in\n        this :class:`~pyflink.table.TableEnvironment`\'s catalog. Registered sink tables can be\n        referenced in SQL DML statements.\n\n        Example:\n        ::\n\n            >>> table_env.register_table_sink("sink",\n            ...                               CsvTableSink(["a", "b"],\n            ...                                            [DataTypes.INT(),\n            ...                                             DataTypes.STRING()],\n            ...                                            "./2.csv"))\n\n        :param name: The name under which the table sink is registered.\n        :param table_sink: The table sink to register.\n\n        .. note:: Deprecated in 1.10. Use :func:`execute_sql` instead.\n        '
        warnings.warn('Deprecated in 1.10. Use create_table instead.', DeprecationWarning)
        self._j_tenv.registerTableSinkInternal(name, table_sink._j_table_sink)

    def scan(self, *table_path: str) -> Table:
        if False:
            return 10
        '\n        Scans a registered table and returns the resulting :class:`~pyflink.table.Table`.\n        A table to scan must be registered in the TableEnvironment. It can be either directly\n        registered or be an external member of a :class:`~pyflink.table.catalog.Catalog`.\n\n        See the documentation of :func:`~pyflink.table.TableEnvironment.use_database` or\n        :func:`~pyflink.table.TableEnvironment.use_catalog` for the rules on the path resolution.\n\n        Examples:\n\n        Scanning a directly registered table\n        ::\n\n            >>> tab = table_env.scan("tableName")\n\n        Scanning a table from a registered catalog\n        ::\n\n            >>> tab = table_env.scan("catalogName", "dbName", "tableName")\n\n        :param table_path: The path of the table to scan.\n        :throws: Exception if no table is found using the given table path.\n        :return: The resulting table.\n\n        .. note:: Deprecated in 1.10. Use :func:`from_path` instead.\n        '
        warnings.warn('Deprecated in 1.10. Use from_path instead.', DeprecationWarning)
        gateway = get_gateway()
        j_table_paths = java_utils.to_jarray(gateway.jvm.String, table_path)
        j_table = self._j_tenv.scan(j_table_paths)
        return Table(j_table, self)

    def from_path(self, path: str) -> Table:
        if False:
            while True:
                i = 10
        '\n        Reads a registered table and returns the resulting :class:`~pyflink.table.Table`.\n\n        A table to scan must be registered in the :class:`~pyflink.table.TableEnvironment`.\n\n        See the documentation of :func:`use_database` or :func:`use_catalog` for the rules on the\n        path resolution.\n\n        Examples:\n\n        Reading a table from default catalog and database.\n        ::\n\n            >>> tab = table_env.from_path("tableName")\n\n        Reading a table from a registered catalog.\n        ::\n\n            >>> tab = table_env.from_path("catalogName.dbName.tableName")\n\n        Reading a table from a registered catalog with escaping. (`Table` is a reserved keyword).\n        Dots in e.g. a database name also must be escaped.\n        ::\n\n            >>> tab = table_env.from_path("catalogName.`db.Name`.`Table`")\n\n        :param path: The path of a table API object to scan.\n        :return: Either a table or virtual table (=view).\n\n        .. seealso:: :func:`use_catalog`\n        .. seealso:: :func:`use_database`\n        .. versionadded:: 1.10.0\n        '
        return Table(get_method(self._j_tenv, 'from')(path), self)

    def from_descriptor(self, descriptor: TableDescriptor) -> Table:
        if False:
            i = 10
            return i + 15
        '\n        Returns a Table backed by the given TableDescriptor.\n\n        The TableDescriptor is registered as an inline (i.e. anonymous) temporary table\n        (see :func:`create_temporary_table`) using a unique identifier and then read. Note that\n        calling this method multiple times, even with the same descriptor, results in multiple\n        temporary tables. In such cases, it is recommended to register it under a name using\n        :func:`create_temporary_table` and reference it via :func:`from_path`\n\n        Examples:\n        ::\n\n            >>> table_env.from_descriptor(TableDescriptor.for_connector("datagen")\n            ...     .schema(Schema.new_builder()\n            ...         .column("f0", DataTypes.STRING())\n            ...         .build())\n            ...     .build()\n\n        Note that the returned Table is an API object and only contains a pipeline description.\n        It actually corresponds to a <i>view</i> in SQL terms. Call :func:`execute` in Table to\n        trigger an execution.\n\n        :return: The Table object describing the pipeline for further transformations.\n\n        .. versionadded:: 1.14.0\n        '
        return Table(get_method(self._j_tenv, 'from')(descriptor._j_table_descriptor), self)

    def list_catalogs(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        '\n        Gets the names of all catalogs registered in this environment.\n\n        :return: List of catalog names.\n        '
        j_catalog_name_array = self._j_tenv.listCatalogs()
        return [item for item in j_catalog_name_array]

    def list_modules(self) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        Gets the names of all modules used in this environment.\n\n        :return: List of module names.\n\n        .. versionadded:: 1.10.0\n        '
        j_module_name_array = self._j_tenv.listModules()
        return [item for item in j_module_name_array]

    def list_full_modules(self) -> List[ModuleEntry]:
        if False:
            while True:
                i = 10
        '\n        Gets the names and statuses of all modules loaded in this environment.\n\n        :return: List of module names and use statuses.\n\n        .. versionadded:: 1.13.0\n        '
        j_module_entry_array = self._j_tenv.listFullModules()
        return [ModuleEntry(entry.name(), entry.used()) for entry in j_module_entry_array]

    def list_databases(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        '\n        Gets the names of all databases in the current catalog.\n\n        :return: List of database names in the current catalog.\n        '
        j_database_name_array = self._j_tenv.listDatabases()
        return [item for item in j_database_name_array]

    def list_tables(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        '\n        Gets the names of all tables and views in the current database of the current catalog.\n        It returns both temporary and permanent tables and views.\n\n        :return: List of table and view names in the current database of the current catalog.\n        '
        j_table_name_array = self._j_tenv.listTables()
        return [item for item in j_table_name_array]

    def list_views(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        '\n        Gets the names of all views in the current database of the current catalog.\n        It returns both temporary and permanent views.\n\n        :return: List of view names in the current database of the current catalog.\n\n        .. versionadded:: 1.11.0\n        '
        j_view_name_array = self._j_tenv.listViews()
        return [item for item in j_view_name_array]

    def list_user_defined_functions(self) -> List[str]:
        if False:
            return 10
        '\n        Gets the names of all user defined functions registered in this environment.\n\n        :return: List of the names of all user defined functions registered in this environment.\n        '
        j_udf_name_array = self._j_tenv.listUserDefinedFunctions()
        return [item for item in j_udf_name_array]

    def list_functions(self) -> List[str]:
        if False:
            print('Hello World!')
        '\n        Gets the names of all functions in this environment.\n\n        :return: List of the names of all functions in this environment.\n\n        .. versionadded:: 1.10.0\n        '
        j_function_name_array = self._j_tenv.listFunctions()
        return [item for item in j_function_name_array]

    def list_temporary_tables(self) -> List[str]:
        if False:
            print('Hello World!')
        '\n        Gets the names of all temporary tables and views available in the current namespace\n        (the current database of the current catalog).\n\n        :return: A list of the names of all registered temporary tables and views in the current\n                 database of the current catalog.\n\n        .. seealso:: :func:`list_tables`\n        .. versionadded:: 1.10.0\n        '
        j_table_name_array = self._j_tenv.listTemporaryTables()
        return [item for item in j_table_name_array]

    def list_temporary_views(self) -> List[str]:
        if False:
            return 10
        '\n        Gets the names of all temporary views available in the current namespace (the current\n        database of the current catalog).\n\n        :return: A list of the names of all registered temporary views in the current database\n                 of the current catalog.\n\n        .. seealso:: :func:`list_tables`\n        .. versionadded:: 1.10.0\n        '
        j_view_name_array = self._j_tenv.listTemporaryViews()
        return [item for item in j_view_name_array]

    def drop_temporary_table(self, table_path: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Drops a temporary table registered in the given path.\n\n        If a permanent table with a given path exists, it will be used\n        from now on for any queries that reference this path.\n\n        :param table_path: The path of the registered temporary table.\n        :return: True if a table existed in the given path and was removed.\n\n        .. versionadded:: 1.10.0\n        '
        return self._j_tenv.dropTemporaryTable(table_path)

    def drop_temporary_view(self, view_path: str) -> bool:
        if False:
            while True:
                i = 10
        '\n        Drops a temporary view registered in the given path.\n\n        If a permanent table or view with a given path exists, it will be used\n        from now on for any queries that reference this path.\n\n        :return: True if a view existed in the given path and was removed.\n\n        .. versionadded:: 1.10.0\n        '
        return self._j_tenv.dropTemporaryView(view_path)

    def explain_sql(self, stmt: str, *extra_details: ExplainDetail) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns the AST of the specified statement and the execution plan.\n\n        :param stmt: The statement for which the AST and execution plan will be returned.\n        :param extra_details: The extra explain details which the explain result should include,\n                              e.g. estimated cost, changelog mode for streaming\n        :return: The statement for which the AST and execution plan will be returned.\n\n        .. versionadded:: 1.11.0\n        '
        JExplainFormat = get_gateway().jvm.org.apache.flink.table.api.ExplainFormat
        j_extra_details = to_j_explain_detail_arr(extra_details)
        return self._j_tenv.explainSql(stmt, JExplainFormat.TEXT, j_extra_details)

    def sql_query(self, query: str) -> Table:
        if False:
            while True:
                i = 10
        '\n        Evaluates a SQL query on registered tables and retrieves the result as a\n        :class:`~pyflink.table.Table`.\n\n        All tables referenced by the query must be registered in the TableEnvironment.\n\n        A :class:`~pyflink.table.Table` is automatically registered when its\n        :func:`~Table.__str__` method is called, for example when it is embedded into a String.\n\n        Hence, SQL queries can directly reference a :class:`~pyflink.table.Table` as follows:\n        ::\n\n            >>> table = ...\n            # the table is not registered to the table environment\n            >>> table_env.sql_query("SELECT * FROM %s" % table)\n\n        :param query: The sql query string.\n        :return: The result table.\n        '
        j_table = self._j_tenv.sqlQuery(query)
        return Table(j_table, self)

    def execute_sql(self, stmt: str) -> TableResult:
        if False:
            for i in range(10):
                print('nop')
        '\n        Execute the given single statement, and return the execution result.\n\n        The statement can be DDL/DML/DQL/SHOW/DESCRIBE/EXPLAIN/USE.\n        For DML and DQL, this method returns TableResult once the job has been submitted.\n        For DDL and DCL statements, TableResult is returned once the operation has finished.\n\n        :return content for DQL/SHOW/DESCRIBE/EXPLAIN,\n                the affected row count for `DML` (-1 means unknown),\n                or a string message ("OK") for other statements.\n\n        .. versionadded:: 1.11.0\n        '
        self._before_execute()
        return TableResult(self._j_tenv.executeSql(stmt))

    def create_statement_set(self) -> StatementSet:
        if False:
            while True:
                i = 10
        '\n        Create a StatementSet instance which accepts DML statements or Tables,\n        the planner can optimize all added statements and Tables together\n        and then submit as one job.\n\n        :return statement_set instance\n\n        .. versionadded:: 1.11.0\n        '
        _j_statement_set = self._j_tenv.createStatementSet()
        return StatementSet(_j_statement_set, self)

    def get_current_catalog(self) -> str:
        if False:
            print('Hello World!')
        '\n        Gets the current default catalog name of the current session.\n\n        :return: The current default catalog name that is used for the path resolution.\n\n        .. seealso:: :func:`~pyflink.table.TableEnvironment.use_catalog`\n        '
        return self._j_tenv.getCurrentCatalog()

    def use_catalog(self, catalog_name: str):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the current catalog to the given value. It also sets the default\n        database to the catalog's default one.\n        See also :func:`~TableEnvironment.use_database`.\n\n        This is used during the resolution of object paths. Both the catalog and database are\n        optional when referencing catalog objects such as tables, views etc. The algorithm looks for\n        requested objects in following paths in that order:\n\n        * ``[current-catalog].[current-database].[requested-path]``\n        * ``[current-catalog].[requested-path]``\n        * ``[requested-path]``\n\n        Example:\n\n        Given structure with default catalog set to ``default_catalog`` and default database set to\n        ``default_database``. ::\n\n            root:\n              |- default_catalog\n                  |- default_database\n                      |- tab1\n                  |- db1\n                      |- tab1\n              |- cat1\n                  |- db1\n                      |- tab1\n\n        The following table describes resolved paths:\n\n        +----------------+-----------------------------------------+\n        | Requested path |             Resolved path               |\n        +================+=========================================+\n        | tab1           | default_catalog.default_database.tab1   |\n        +----------------+-----------------------------------------+\n        | db1.tab1       | default_catalog.db1.tab1                |\n        +----------------+-----------------------------------------+\n        | cat1.db1.tab1  | cat1.db1.tab1                           |\n        +----------------+-----------------------------------------+\n\n        :param catalog_name: The name of the catalog to set as the current default catalog.\n        :throws: :class:`~pyflink.util.exceptions.CatalogException` thrown if a catalog with given\n                 name could not be set as the default one.\n\n        .. seealso:: :func:`~pyflink.table.TableEnvironment.use_database`\n        "
        self._j_tenv.useCatalog(catalog_name)

    def get_current_database(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Gets the current default database name of the running session.\n\n        :return: The name of the current database of the current catalog.\n\n        .. seealso:: :func:`~pyflink.table.TableEnvironment.use_database`\n        '
        return self._j_tenv.getCurrentDatabase()

    def use_database(self, database_name: str):
        if False:
            while True:
                i = 10
        '\n        Sets the current default database. It has to exist in the current catalog. That path will\n        be used as the default one when looking for unqualified object names.\n\n        This is used during the resolution of object paths. Both the catalog and database are\n        optional when referencing catalog objects such as tables, views etc. The algorithm looks for\n        requested objects in following paths in that order:\n\n        * ``[current-catalog].[current-database].[requested-path]``\n        * ``[current-catalog].[requested-path]``\n        * ``[requested-path]``\n\n        Example:\n\n        Given structure with default catalog set to ``default_catalog`` and default database set to\n        ``default_database``. ::\n\n            root:\n              |- default_catalog\n                  |- default_database\n                      |- tab1\n                  |- db1\n                      |- tab1\n              |- cat1\n                  |- db1\n                      |- tab1\n\n        The following table describes resolved paths:\n\n        +----------------+-----------------------------------------+\n        | Requested path |             Resolved path               |\n        +================+=========================================+\n        | tab1           | default_catalog.default_database.tab1   |\n        +----------------+-----------------------------------------+\n        | db1.tab1       | default_catalog.db1.tab1                |\n        +----------------+-----------------------------------------+\n        | cat1.db1.tab1  | cat1.db1.tab1                           |\n        +----------------+-----------------------------------------+\n\n        :throws: :class:`~pyflink.util.exceptions.CatalogException` thrown if the given catalog and\n                 database could not be set as the default ones.\n\n        .. seealso:: :func:`~pyflink.table.TableEnvironment.use_catalog`\n\n        :param database_name: The name of the database to set as the current database.\n        '
        self._j_tenv.useDatabase(database_name)

    def get_config(self) -> TableConfig:
        if False:
            print('Hello World!')
        '\n        Returns the table config to define the runtime behavior of the Table API.\n\n        :return: Current table config.\n        '
        if not hasattr(self, 'table_config'):
            table_config = TableConfig()
            table_config._j_table_config = self._j_tenv.getConfig()
            setattr(self, 'table_config', table_config)
        return getattr(self, 'table_config')

    def register_java_function(self, name: str, function_class_name: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Registers a java user defined function under a unique name. Replaces already existing\n        user-defined functions under this name. The acceptable function type contains\n        **ScalarFunction**, **TableFunction** and **AggregateFunction**.\n\n        Example:\n        ::\n\n            >>> table_env.register_java_function("func1", "java.user.defined.function.class.name")\n\n        :param name: The name under which the function is registered.\n        :param function_class_name: The java full qualified class name of the function to register.\n                                    The function must have a public no-argument constructor and can\n                                    be founded in current Java classloader.\n\n        .. note:: Deprecated in 1.12. Use :func:`create_java_temporary_system_function` instead.\n        '
        warnings.warn('Deprecated in 1.12. Use :func:`create_java_temporary_system_function` instead.', DeprecationWarning)
        gateway = get_gateway()
        java_function = gateway.jvm.Thread.currentThread().getContextClassLoader().loadClass(function_class_name).newInstance()
        if not isinstance(self, StreamTableEnvironment) or self.__class__ == TableEnvironment:
            if self._is_table_function(java_function):
                self._register_table_function(name, java_function)
            elif self._is_aggregate_function(java_function):
                self._register_aggregate_function(name, java_function)
            else:
                self._j_tenv.registerFunction(name, java_function)
        else:
            self._j_tenv.registerFunction(name, java_function)

    def register_function(self, name: str, function: UserDefinedFunctionWrapper):
        if False:
            return 10
        '\n        Registers a python user-defined function under a unique name. Replaces already existing\n        user-defined function under this name.\n\n        Example:\n        ::\n\n            >>> table_env.register_function(\n            ...     "add_one", udf(lambda i: i + 1, result_type=DataTypes.BIGINT()))\n\n            >>> @udf(result_type=DataTypes.BIGINT())\n            ... def add(i, j):\n            ...     return i + j\n            >>> table_env.register_function("add", add)\n\n            >>> class SubtractOne(ScalarFunction):\n            ...     def eval(self, i):\n            ...         return i - 1\n            >>> table_env.register_function(\n            ...     "subtract_one", udf(SubtractOne(), result_type=DataTypes.BIGINT()))\n\n        :param name: The name under which the function is registered.\n        :param function: The python user-defined function to register.\n\n        .. versionadded:: 1.10.0\n\n        .. note:: Deprecated in 1.12. Use :func:`create_temporary_system_function` instead.\n        '
        warnings.warn('Deprecated in 1.12. Use :func:`create_temporary_system_function` instead.', DeprecationWarning)
        function = self._wrap_aggregate_function_if_needed(function)
        java_function = function._java_user_defined_function()
        if self.__class__ == TableEnvironment:
            if self._is_table_function(java_function):
                self._register_table_function(name, java_function)
            elif self._is_aggregate_function(java_function):
                self._register_aggregate_function(name, java_function)
            else:
                self._j_tenv.registerFunction(name, java_function)
        else:
            self._j_tenv.registerFunction(name, java_function)

    def create_temporary_view(self, view_path: str, table_or_data_stream: Union[Table, DataStream], *fields_or_schema: Union[str, Expression, Schema]):
        if False:
            for i in range(10):
                print('nop')
        '\n        1. When table_or_data_stream is a :class:`~pyflink.table.Table`:\n\n            Registers a :class:`~pyflink.table.Table` API object as a temporary view similar to SQL\n            temporary views.\n\n            Temporary objects can shadow permanent ones. If a permanent object in a given path\n            exists, it will be inaccessible in the current session. To make the permanent object\n            available again you can drop the corresponding temporary object.\n\n        2. When table_or_data_stream is a :class:`~pyflink.datastream.DataStream`:\n\n            2.1 When fields_or_schema is a str or a sequence of :class:`~pyflink.table.Expression`:\n\n                Creates a view from the given {@link DataStream} in a given path with specified\n                field names. Registered views can be referenced in SQL queries.\n\n                1. Reference input fields by name: All fields in the schema definition are\n                referenced by name (and possibly renamed using an alias (as). Moreover, we can\n                define proctime and rowtime attributes at arbitrary positions using arbitrary names\n                (except those that exist in the result schema). In this mode, fields can be\n                reordered and projected out. This mode can be used for any input type, including\n                POJOs.\n\n                Example:\n                ::\n\n                    >>> stream = ...\n                    # reorder the fields, rename the original \'f0\' field to \'name\' and add\n                    # event-time attribute named \'rowtime\'\n\n                    # use str\n                    >>> table_env.create_temporary_view(\n                    ...     "cat.db.myTable",\n                    ...     stream,\n                    ...     "f1, rowtime.rowtime, f0 as \'name\'")\n\n                    # or use a sequence of expression\n                    >>> table_env.create_temporary_view(\n                    ...     "cat.db.myTable",\n                    ...     stream,\n                    ...     col("f1"),\n                    ...     col("rowtime").rowtime,\n                    ...     col("f0").alias(\'name\'))\n\n                2. Reference input fields by position: In this mode, fields are simply renamed.\n                Event-time attributes can replace the field on their position in the input data\n                (if it is of correct type) or be appended at the end. Proctime attributes must be\n                appended at the end. This mode can only be used if the input type has a defined\n                field order (tuple, case class, Row) and none of the {@code fields} references a\n                field of the input type.\n\n                Example:\n                ::\n\n                    >>> stream = ...\n                    # rename the original fields to \'a\' and \'b\' and extract the internally attached\n                    # timestamp into an event-time attribute named \'rowtime\'\n\n                    # use str\n                    >>> table_env.create_temporary_view(\n                    ...     "cat.db.myTable", stream, "a, b, rowtime.rowtime")\n\n                    # or use a sequence of expressions\n                    >>> table_env.create_temporary_view(\n                    ...     "cat.db.myTable",\n                    ...     stream,\n                    ...     col("a"),\n                    ...     col("b"),\n                    ...     col("rowtime").rowtime)\n\n                Temporary objects can shadow permanent ones. If a permanent object in a given path\n                exists, it will be inaccessible in the current session. To make the permanent object\n                available again you can drop the corresponding temporary object.\n\n            2.2 When fields_or_schema is a :class:`~pyflink.table.Schema`:\n\n                Creates a view from the given {@link DataStream} in a given path. Registered views\n                can be referenced in SQL queries.\n\n                See :func:`from_data_stream` for more information on how a\n                :class:`~pyflink.datastream.DataStream` is translated into a table.\n\n                Temporary objects can shadow permanent ones. If a permanent object in a given path\n                exists, it will be inaccessible in the current session. To make the permanent object\n                available again you can drop the corresponding temporary object.\n\n                .. note:: create_temporary_view by providing a Schema (case 2.) was added from flink\n                    1.14.0.\n\n        :param view_path: The path under which the view will be registered. See also the\n                          :class:`~pyflink.table.TableEnvironment` class description for the format\n                          of the path.\n        :param table_or_data_stream: The Table or DataStream out of which to create the view.\n        :param fields_or_schema: The fields expressions(str) to map original fields of the\n                        DataStream to the fields of the View or the customized schema for the final\n                        table.\n\n        .. versionadded:: 1.10.0\n        '
        if isinstance(table_or_data_stream, Table):
            self._j_tenv.createTemporaryView(view_path, table_or_data_stream._j_table)
        else:
            j_data_stream = table_or_data_stream._j_data_stream
            JPythonConfigUtil = get_gateway().jvm.org.apache.flink.python.util.PythonConfigUtil
            JPythonConfigUtil.configPythonOperator(j_data_stream.getExecutionEnvironment())
            if len(fields_or_schema) == 0:
                self._j_tenv.createTemporaryView(view_path, j_data_stream)
            elif len(fields_or_schema) == 1 and isinstance(fields_or_schema[0], str):
                self._j_tenv.createTemporaryView(view_path, j_data_stream, fields_or_schema[0])
            elif len(fields_or_schema) == 1 and isinstance(fields_or_schema[0], Schema):
                self._j_tenv.createTemporaryView(view_path, j_data_stream, fields_or_schema[0]._j_schema)
            elif len(fields_or_schema) > 0 and all((isinstance(elem, Expression) for elem in fields_or_schema)):
                self._j_tenv.createTemporaryView(view_path, j_data_stream, to_expression_jarray(fields_or_schema))
            else:
                raise ValueError("Invalid arguments for 'fields': %r" % ','.join([repr(item) for item in fields_or_schema]))

    def add_python_file(self, file_path: str):
        if False:
            i = 10
            return i + 15
        '\n        Adds a python dependency which could be python files, python packages or\n        local directories. They will be added to the PYTHONPATH of the python UDF worker.\n        Please make sure that these dependencies can be imported.\n\n        :param file_path: The path of the python dependency.\n\n        .. versionadded:: 1.10.0\n        '
        jvm = get_gateway().jvm
        python_files = self.get_config().get(jvm.PythonOptions.PYTHON_FILES.key(), None)
        if python_files is not None:
            python_files = jvm.PythonDependencyUtils.FILE_DELIMITER.join([file_path, python_files])
        else:
            python_files = file_path
        self.get_config().set(jvm.PythonOptions.PYTHON_FILES.key(), python_files)

    def set_python_requirements(self, requirements_file_path: str, requirements_cache_dir: str=None):
        if False:
            while True:
                i = 10
        '\n        Specifies a requirements.txt file which defines the third-party dependencies.\n        These dependencies will be installed to a temporary directory and added to the\n        PYTHONPATH of the python UDF worker.\n\n        For the dependencies which could not be accessed in the cluster, a directory which contains\n        the installation packages of these dependencies could be specified using the parameter\n        "requirements_cached_dir". It will be uploaded to the cluster to support offline\n        installation.\n\n        Example:\n        ::\n\n            # commands executed in shell\n            $ echo numpy==1.16.5 > requirements.txt\n            $ pip download -d cached_dir -r requirements.txt --no-binary :all:\n\n            # python code\n            >>> table_env.set_python_requirements("requirements.txt", "cached_dir")\n\n        .. note::\n\n            Please make sure the installation packages matches the platform of the cluster\n            and the python version used. These packages will be installed using pip,\n            so also make sure the version of Pip (version >= 20.3) and the version of\n            SetupTools (version >= 37.0.0).\n\n        :param requirements_file_path: The path of "requirements.txt" file.\n        :param requirements_cache_dir: The path of the local directory which contains the\n                                       installation packages.\n\n        .. versionadded:: 1.10.0\n        '
        jvm = get_gateway().jvm
        python_requirements = requirements_file_path
        if requirements_cache_dir is not None:
            python_requirements = jvm.PythonDependencyUtils.PARAM_DELIMITER.join([python_requirements, requirements_cache_dir])
        self.get_config().set(jvm.PythonOptions.PYTHON_REQUIREMENTS.key(), python_requirements)

    def add_python_archive(self, archive_path: str, target_dir: str=None):
        if False:
            print('Hello World!')
        '\n        Adds a python archive file. The file will be extracted to the working directory of\n        python UDF worker.\n\n        If the parameter "target_dir" is specified, the archive file will be extracted to a\n        directory named ${target_dir}. Otherwise, the archive file will be extracted to a\n        directory with the same name of the archive file.\n\n        If python UDF depends on a specific python version which does not exist in the cluster,\n        this method can be used to upload the virtual environment.\n        Note that the path of the python interpreter contained in the uploaded environment\n        should be specified via the method :func:`pyflink.table.TableConfig.set_python_executable`.\n\n        The files uploaded via this method are also accessible in UDFs via relative path.\n\n        Example:\n        ::\n\n            # command executed in shell\n            # assert the relative path of python interpreter is py_env/bin/python\n            $ zip -r py_env.zip py_env\n\n            # python code\n            >>> table_env.add_python_archive("py_env.zip")\n            >>> table_env.get_config().set_python_executable("py_env.zip/py_env/bin/python")\n\n            # or\n            >>> table_env.add_python_archive("py_env.zip", "myenv")\n            >>> table_env.get_config().set_python_executable("myenv/py_env/bin/python")\n\n            # the files contained in the archive file can be accessed in UDF\n            >>> def my_udf():\n            ...     with open("myenv/py_env/data/data.txt") as f:\n            ...         ...\n\n        .. note::\n\n            Please make sure the uploaded python environment matches the platform that the cluster\n            is running on and that the python version must be 3.5 or higher.\n\n        .. note::\n\n            Currently only zip-format is supported. i.e. zip, jar, whl, egg, etc.\n            The other archive formats such as tar, tar.gz, 7z, rar, etc are not supported.\n\n        :param archive_path: The archive file path.\n        :param target_dir: Optional, the target dir name that the archive file extracted to.\n\n        .. versionadded:: 1.10.0\n        '
        jvm = get_gateway().jvm
        if target_dir is not None:
            archive_path = jvm.PythonDependencyUtils.PARAM_DELIMITER.join([archive_path, target_dir])
        python_archives = self.get_config().get(jvm.PythonOptions.PYTHON_ARCHIVES.key(), None)
        if python_archives is not None:
            python_files = jvm.PythonDependencyUtils.FILE_DELIMITER.join([python_archives, archive_path])
        else:
            python_files = archive_path
        self.get_config().set(jvm.PythonOptions.PYTHON_ARCHIVES.key(), python_files)

    def from_elements(self, elements: Iterable, schema: Union[DataType, List[str]]=None, verify_schema: bool=True) -> Table:
        if False:
            print('Hello World!')
        '\n        Creates a table from a collection of elements.\n        The elements types must be acceptable atomic types or acceptable composite types.\n        All elements must be of the same type.\n        If the elements types are composite types, the composite types must be strictly equal,\n        and its subtypes must also be acceptable types.\n        e.g. if the elements are tuples, the length of the tuples must be equal, the element types\n        of the tuples must be equal in order.\n\n        The built-in acceptable atomic element types contains:\n\n        **int**, **long**, **str**, **unicode**, **bool**,\n        **float**, **bytearray**, **datetime.date**, **datetime.time**, **datetime.datetime**,\n        **datetime.timedelta**, **decimal.Decimal**\n\n        The built-in acceptable composite element types contains:\n\n        **list**, **tuple**, **dict**, **array**, :class:`~pyflink.table.Row`\n\n        If the element type is a composite type, it will be unboxed.\n        e.g. table_env.from_elements([(1, \'Hi\'), (2, \'Hello\')]) will return a table like:\n\n        +----+-------+\n        | _1 |  _2   |\n        +====+=======+\n        | 1  |  Hi   |\n        +----+-------+\n        | 2  | Hello |\n        +----+-------+\n\n        "_1" and "_2" are generated field names.\n\n        Example:\n        ::\n\n            # use the second parameter to specify custom field names\n            >>> table_env.from_elements([(1, \'Hi\'), (2, \'Hello\')], [\'a\', \'b\'])\n            # use the second parameter to specify custom table schema\n            >>> table_env.from_elements([(1, \'Hi\'), (2, \'Hello\')],\n            ...                         DataTypes.ROW([DataTypes.FIELD("a", DataTypes.INT()),\n            ...                                        DataTypes.FIELD("b", DataTypes.STRING())]))\n            # use the third parameter to switch whether to verify the elements against the schema\n            >>> table_env.from_elements([(1, \'Hi\'), (2, \'Hello\')],\n            ...                         DataTypes.ROW([DataTypes.FIELD("a", DataTypes.INT()),\n            ...                                        DataTypes.FIELD("b", DataTypes.STRING())]),\n            ...                         False)\n            # create Table from expressions\n            >>> table_env.from_elements([row(1, \'abc\', 2.0), row(2, \'def\', 3.0)],\n            ...                         DataTypes.ROW([DataTypes.FIELD("a", DataTypes.INT()),\n            ...                                        DataTypes.FIELD("b", DataTypes.STRING()),\n            ...                                        DataTypes.FIELD("c", DataTypes.FLOAT())]))\n\n        :param elements: The elements to create a table from.\n        :param schema: The schema of the table.\n        :param verify_schema: Whether to verify the elements against the schema.\n        :return: The result table.\n        '
        if isinstance(schema, RowType):
            verify_func = _create_type_verifier(schema) if verify_schema else lambda _: True

            def verify_obj(obj):
                if False:
                    while True:
                        i = 10
                verify_func(obj)
                return obj
        elif isinstance(schema, DataType):
            data_type = schema
            schema = RowType().add('value', schema)
            verify_func = _create_type_verifier(data_type, name='field value') if verify_schema else lambda _: True

            def verify_obj(obj):
                if False:
                    while True:
                        i = 10
                verify_func(obj)
                return obj
        else:

            def verify_obj(obj):
                if False:
                    print('Hello World!')
                return obj
        if schema is None or isinstance(schema, (list, tuple)):
            schema = _infer_schema_from_data(elements, names=schema)
            converter = _create_converter(schema)
            elements = map(converter, elements)
        elif not isinstance(schema, RowType):
            raise TypeError('schema should be RowType, list, tuple or None, but got: %s' % schema)
        elements = list(elements)
        if len(elements) > 0 and all((isinstance(elem, Expression) for elem in elements)):
            if schema is None:
                return Table(self._j_tenv.fromValues(to_expression_jarray(elements)), self)
            else:
                return Table(self._j_tenv.fromValues(_to_java_data_type(schema), to_expression_jarray(elements)), self)
        elif any((isinstance(elem, Expression) for elem in elements)):
            raise ValueError("It doesn't support part of the elements are Expression, while the others are not.")
        elements = map(verify_obj, elements)
        elements = [schema.to_sql_type(element) for element in elements]
        return self._from_elements(elements, schema)

    def _from_elements(self, elements: List, schema: DataType) -> Table:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a table from a collection of elements.\n\n        :param elements: The elements to create a table from.\n        :return: The result :class:`~pyflink.table.Table`.\n        '
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=tempfile.mkdtemp())
        serializer = BatchedSerializer(self._serializer)
        try:
            with temp_file:
                serializer.serialize(elements, temp_file)
            j_schema = _to_java_data_type(schema)
            gateway = get_gateway()
            PythonTableUtils = gateway.jvm.org.apache.flink.table.utils.python.PythonTableUtils
            j_table = PythonTableUtils.createTableFromElement(self._j_tenv, temp_file.name, j_schema, True)
            return Table(j_table, self)
        finally:
            atexit.register(lambda : os.unlink(temp_file.name))

    def from_pandas(self, pdf, schema: Union[RowType, List[str], Tuple[str], List[DataType], Tuple[DataType]]=None, splits_num: int=1) -> Table:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a table from a pandas DataFrame.\n\n        Example:\n        ::\n\n            >>> pdf = pd.DataFrame(np.random.rand(1000, 2))\n            # use the second parameter to specify custom field names\n            >>> table_env.from_pandas(pdf, ["a", "b"])\n            # use the second parameter to specify custom field types\n            >>> table_env.from_pandas(pdf, [DataTypes.DOUBLE(), DataTypes.DOUBLE()]))\n            # use the second parameter to specify custom table schema\n            >>> table_env.from_pandas(pdf,\n            ...                       DataTypes.ROW([DataTypes.FIELD("a", DataTypes.DOUBLE()),\n            ...                                      DataTypes.FIELD("b", DataTypes.DOUBLE())]))\n\n        :param pdf: The pandas DataFrame.\n        :param schema: The schema of the converted table.\n        :param splits_num: The number of splits the given Pandas DataFrame will be split into. It\n                           determines the number of parallel source tasks.\n                           If not specified, the default parallelism will be used.\n        :return: The result table.\n\n        .. versionadded:: 1.11.0\n        '
        import pandas as pd
        if not isinstance(pdf, pd.DataFrame):
            raise TypeError('Unsupported type, expected pandas.DataFrame, got %s' % type(pdf))
        import pyarrow as pa
        arrow_schema = pa.Schema.from_pandas(pdf, preserve_index=False)
        if schema is not None:
            if isinstance(schema, RowType):
                result_type = schema
            elif isinstance(schema, (list, tuple)) and isinstance(schema[0], str):
                result_type = RowType([RowField(field_name, from_arrow_type(field.type, field.nullable)) for (field_name, field) in zip(schema, arrow_schema)])
            elif isinstance(schema, (list, tuple)) and isinstance(schema[0], DataType):
                result_type = RowType([RowField(field_name, field_type) for (field_name, field_type) in zip(arrow_schema.names, schema)])
            else:
                raise TypeError('Unsupported schema type, it could only be of RowType, a list of str or a list of DataType, got %s' % schema)
        else:
            result_type = RowType([RowField(field.name, from_arrow_type(field.type, field.nullable)) for field in arrow_schema])
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=tempfile.mkdtemp())
        import pytz
        serializer = ArrowSerializer(create_arrow_schema(result_type.field_names(), result_type.field_types()), result_type, pytz.timezone(self.get_config().get_local_timezone()))
        step = -(-len(pdf) // splits_num)
        pdf_slices = [pdf.iloc[start:start + step] for start in range(0, len(pdf), step)]
        data = [[c for (_, c) in pdf_slice.items()] for pdf_slice in pdf_slices]
        try:
            with temp_file:
                serializer.serialize(data, temp_file)
            jvm = get_gateway().jvm
            data_type = _to_java_data_type(result_type).notNull()
            data_type = data_type.bridgedTo(load_java_class('org.apache.flink.table.data.RowData'))
            j_arrow_table_source = jvm.org.apache.flink.table.runtime.arrow.ArrowUtils.createArrowTableSource(data_type, temp_file.name)
            return Table(self._j_tenv.fromTableSource(j_arrow_table_source), self)
        finally:
            os.unlink(temp_file.name)

    def _set_python_executable_for_local_executor(self):
        if False:
            return 10
        jvm = get_gateway().jvm
        j_config = get_j_env_configuration(self._get_j_env())
        if not j_config.containsKey(jvm.PythonOptions.PYTHON_EXECUTABLE.key()) and is_local_deployment(j_config):
            j_config.setString(jvm.PythonOptions.PYTHON_EXECUTABLE.key(), sys.executable)

    def _add_jars_to_j_env_config(self, config_key):
        if False:
            print('Hello World!')
        jvm = get_gateway().jvm
        jar_urls = self.get_config().get(config_key, None)
        if jar_urls is not None:
            jar_urls_list = []
            for url in jar_urls.split(';'):
                url = url.strip()
                if url != '':
                    jar_urls_list.append(jvm.java.net.URL(url).toString())
            j_configuration = get_j_env_configuration(self._get_j_env())
            if j_configuration.containsKey(config_key):
                for url in j_configuration.getString(config_key, '').split(';'):
                    url = url.strip()
                    if url != '' and url not in jar_urls_list:
                        jar_urls_list.append(url)
            j_configuration.setString(config_key, ';'.join(jar_urls_list))

    def _get_j_env(self):
        if False:
            return 10
        return self._j_tenv.getPlanner().getExecEnv()

    @staticmethod
    def _is_table_function(java_function):
        if False:
            while True:
                i = 10
        java_function_class = java_function.getClass()
        j_table_function_class = get_java_class(get_gateway().jvm.org.apache.flink.table.functions.TableFunction)
        return j_table_function_class.isAssignableFrom(java_function_class)

    @staticmethod
    def _is_aggregate_function(java_function):
        if False:
            while True:
                i = 10
        java_function_class = java_function.getClass()
        j_aggregate_function_class = get_java_class(get_gateway().jvm.org.apache.flink.table.functions.ImperativeAggregateFunction)
        return j_aggregate_function_class.isAssignableFrom(java_function_class)

    def _register_table_function(self, name, table_function):
        if False:
            for i in range(10):
                print('nop')
        function_catalog = self._get_function_catalog()
        gateway = get_gateway()
        helper = gateway.jvm.org.apache.flink.table.functions.UserDefinedFunctionHelper
        result_type = helper.getReturnTypeOfTableFunction(table_function)
        function_catalog.registerTempSystemTableFunction(name, table_function, result_type)

    def _register_aggregate_function(self, name, aggregate_function):
        if False:
            while True:
                i = 10
        function_catalog = self._get_function_catalog()
        gateway = get_gateway()
        helper = gateway.jvm.org.apache.flink.table.functions.UserDefinedFunctionHelper
        result_type = helper.getReturnTypeOfAggregateFunction(aggregate_function)
        acc_type = helper.getAccumulatorTypeOfAggregateFunction(aggregate_function)
        function_catalog.registerTempSystemAggregateFunction(name, aggregate_function, result_type, acc_type)

    def _get_function_catalog(self):
        if False:
            while True:
                i = 10
        function_catalog_field = self._j_tenv.getClass().getDeclaredField('functionCatalog')
        function_catalog_field.setAccessible(True)
        function_catalog = function_catalog_field.get(self._j_tenv)
        return function_catalog

    def _before_execute(self):
        if False:
            for i in range(10):
                print('nop')
        jvm = get_gateway().jvm
        jars_key = jvm.org.apache.flink.configuration.PipelineOptions.JARS.key()
        classpaths_key = jvm.org.apache.flink.configuration.PipelineOptions.CLASSPATHS.key()
        self._add_jars_to_j_env_config(jars_key)
        self._add_jars_to_j_env_config(classpaths_key)

    def _wrap_aggregate_function_if_needed(self, function) -> UserDefinedFunctionWrapper:
        if False:
            while True:
                i = 10
        if isinstance(function, AggregateFunction):
            function = udaf(function, result_type=function.get_result_type(), accumulator_type=function.get_accumulator_type(), name=str(function.__class__.__name__))
        elif isinstance(function, TableAggregateFunction):
            function = udtaf(function, result_type=function.get_result_type(), accumulator_type=function.get_accumulator_type(), name=str(function.__class__.__name__))
        return function

    def _config_chaining_optimization(self):
        if False:
            for i in range(10):
                print('nop')
        JChainingOptimizingExecutor = get_gateway().jvm.org.apache.flink.table.executor.python.ChainingOptimizingExecutor
        exec_env_field = get_field(self._j_tenv.getClass(), 'execEnv')
        exec_env_field.set(self._j_tenv, JChainingOptimizingExecutor(exec_env_field.get(self._j_tenv)))

    def _open(self):
        if False:
            return 10

        def startup_loopback_server():
            if False:
                print('Hello World!')
            from pyflink.fn_execution.beam.beam_worker_pool_service import BeamFnLoopbackWorkerPoolServicer
            self.get_config().set('python.loopback-server.address', BeamFnLoopbackWorkerPoolServicer().start())
        python_worker_execution_mode = os.environ.get('_python_worker_execution_mode')
        if python_worker_execution_mode is None:
            if is_local_deployment(get_j_env_configuration(self._get_j_env())):
                startup_loopback_server()
        elif python_worker_execution_mode == 'loopback':
            if is_local_deployment(get_j_env_configuration(self._get_j_env())):
                startup_loopback_server()
            else:
                raise ValueError("Loopback mode is enabled, however the job wasn't configured to run in local deployment mode")
        elif python_worker_execution_mode != 'process':
            raise ValueError("It only supports to execute the Python worker in 'loopback' mode and 'process' mode, unknown mode '%s' is configured" % python_worker_execution_mode)

class StreamTableEnvironment(TableEnvironment):

    def __init__(self, j_tenv):
        if False:
            while True:
                i = 10
        super(StreamTableEnvironment, self).__init__(j_tenv)

    @staticmethod
    def create(stream_execution_environment: StreamExecutionEnvironment=None, environment_settings: EnvironmentSettings=None) -> 'StreamTableEnvironment':
        if False:
            return 10
        "\n        Creates a :class:`~pyflink.table.StreamTableEnvironment`.\n\n        Example:\n        ::\n\n            # create with StreamExecutionEnvironment.\n            >>> env = StreamExecutionEnvironment.get_execution_environment()\n            >>> table_env = StreamTableEnvironment.create(env)\n            # create with StreamExecutionEnvironment and EnvironmentSettings.\n            >>> configuration = Configuration()\n            >>> configuration.set_string('execution.buffer-timeout', '1 min')\n            >>> environment_settings = EnvironmentSettings \\\n            ...     .new_instance() \\\n            ...     .in_streaming_mode() \\\n            ...     .with_configuration(configuration) \\\n            ...     .build()\n            >>> table_env = StreamTableEnvironment.create(\n            ...     env, environment_settings=environment_settings)\n            # create with EnvironmentSettings.\n            >>> table_env = StreamTableEnvironment.create(environment_settings=environment_settings)\n\n\n        :param stream_execution_environment: The\n                                             :class:`~pyflink.datastream.StreamExecutionEnvironment`\n                                             of the TableEnvironment.\n        :param environment_settings: The environment settings used to instantiate the\n                                     TableEnvironment.\n        :return: The StreamTableEnvironment created from given StreamExecutionEnvironment and\n                 configuration.\n        "
        if stream_execution_environment is None and environment_settings is None:
            raise ValueError("No argument found, the param 'stream_execution_environment' or 'environment_settings' is required.")
        gateway = get_gateway()
        if environment_settings is not None:
            if stream_execution_environment is None:
                j_tenv = gateway.jvm.TableEnvironment.create(environment_settings._j_environment_settings)
            else:
                j_tenv = gateway.jvm.StreamTableEnvironment.create(stream_execution_environment._j_stream_execution_environment, environment_settings._j_environment_settings)
        else:
            j_tenv = gateway.jvm.StreamTableEnvironment.create(stream_execution_environment._j_stream_execution_environment)
        return StreamTableEnvironment(j_tenv)

    def from_data_stream(self, data_stream: DataStream, *fields_or_schema: Union[Expression, Schema]) -> Table:
        if False:
            print('Hello World!')
        '\n        1. When fields_or_schema is a sequence of Expression:\n\n            Converts the given DataStream into a Table with specified field names.\n\n            There are two modes for mapping original fields to the fields of the Table:\n\n                1. Reference input fields by name:\n\n                All fields in the schema definition are referenced by name (and possibly renamed\n                using and alias (as). Moreover, we can define proctime and rowtime attributes at\n                arbitrary positions using arbitrary names (except those that exist in the result\n                schema). In this mode, fields can be reordered and projected out. This mode can be\n                used for any input type.\n\n                2. Reference input fields by position:\n\n                In this mode, fields are simply renamed. Event-time attributes can replace the field\n                on their position in the input data (if it is of correct type) or be appended at the\n                end. Proctime attributes must be appended at the end. This mode can only be used if\n                the input type has a defined field order (tuple, case class, Row) and none of the\n                fields references a field of the input type.\n\n        2. When fields_or_schema is a Schema:\n\n            Converts the given DataStream into a Table.\n\n            Column names and types of the Table are automatically derived from the TypeInformation\n            of the DataStream. If the outermost record\'s TypeInformation is a CompositeType, it will\n            be flattened in the first level. Composite nested fields will not be accessible.\n\n            Since the DataStream API does not support changelog processing natively, this method\n            assumes append-only/insert-only semantics during the stream-to-table conversion. Records\n            of class Row must describe RowKind.INSERT changes.\n\n            By default, the stream record\'s timestamp and watermarks are not propagated unless\n            explicitly declared.\n\n            This method allows to declare a Schema for the resulting table. The declaration is\n            similar to a {@code CREATE TABLE} DDL in SQL and allows to:\n\n                1. enrich or overwrite automatically derived columns with a custom DataType\n                2. reorder columns\n                3. add computed or metadata columns next to the physical columns\n                4. access a stream record\'s timestamp\n                5. declare a watermark strategy or propagate the DataStream watermarks\n\n            It is possible to declare a schema without physical/regular columns. In this case, those\n            columns will be automatically derived and implicitly put at the beginning of the schema\n            declaration.\n\n            The following examples illustrate common schema declarations and their semantics:\n\n            Example:\n            ::\n\n                === EXAMPLE 1 ===\n\n                no physical columns defined, they will be derived automatically,\n                e.g. BigDecimal becomes DECIMAL(38, 18)\n\n                >>> Schema.new_builder()                 ...     .column_by_expression("c1", "f1 + 42")                 ...     .column_by_expression("c2", "f1 - 1")                 ...     .build()\n\n                equal to: CREATE TABLE (f0 STRING, f1 DECIMAL(38, 18), c1 AS f1 + 42, c2 AS f1 - 1)\n\n                === EXAMPLE 2 ===\n\n                physical columns defined, input fields and columns will be mapped by name,\n                columns are reordered and their data type overwritten,\n                all columns must be defined to show up in the final table\'s schema\n\n                >>> Schema.new_builder()                 ...     .column("f1", "DECIMAL(10, 2)")                 ...     .column_by_expression("c", "f1 - 1")                 ...     .column("f0", "STRING")                 ...     .build()\n\n                equal to: CREATE TABLE (f1 DECIMAL(10, 2), c AS f1 - 1, f0 STRING)\n\n                === EXAMPLE 3 ===\n\n                timestamp and watermarks can be added from the DataStream API,\n                physical columns will be derived automatically\n\n                >>> Schema.new_builder()                 ...     .column_by_metadata("rowtime", "TIMESTAMP_LTZ(3)")                 ...     .watermark("rowtime", "SOURCE_WATERMARK()")                 ...     .build()\n\n                equal to:\n                    CREATE TABLE (\n                        f0 STRING,\n                        f1 DECIMAL(38, 18),\n                        rowtime TIMESTAMP(3) METADATA,\n                        WATERMARK FOR rowtime AS SOURCE_WATERMARK()\n                    )\n\n            .. note:: create_temporary_view by providing a Schema (case 2.) was added from flink\n                    1.14.0.\n\n        :param data_stream: The datastream to be converted.\n        :param fields_or_schema: The fields expressions to map original fields of the DataStream to\n            the fields of the Table or the customized schema for the final table.\n        :return: The converted Table.\n\n        .. versionadded:: 1.12.0\n        '
        j_data_stream = data_stream._j_data_stream
        JPythonConfigUtil = get_gateway().jvm.org.apache.flink.python.util.PythonConfigUtil
        JPythonConfigUtil.configPythonOperator(j_data_stream.getExecutionEnvironment())
        if len(fields_or_schema) == 0:
            return Table(j_table=self._j_tenv.fromDataStream(j_data_stream), t_env=self)
        elif all((isinstance(f, Expression) for f in fields_or_schema)):
            return Table(j_table=self._j_tenv.fromDataStream(j_data_stream, to_expression_jarray(fields_or_schema)), t_env=self)
        elif len(fields_or_schema) == 1 and isinstance(fields_or_schema[0], Schema):
            return Table(j_table=self._j_tenv.fromDataStream(j_data_stream, fields_or_schema[0]._j_schema), t_env=self)
        raise ValueError("Invalid arguments for 'fields': %r" % fields_or_schema)

    def from_changelog_stream(self, data_stream: DataStream, schema: Schema=None, changelog_mode: ChangelogMode=None) -> Table:
        if False:
            i = 10
            return i + 15
        "\n        Converts the given DataStream of changelog entries into a Table.\n\n        Compared to :func:`from_data_stream`, this method consumes instances of Row and evaluates\n        the RowKind flag that is contained in every record during runtime. The runtime behavior is\n        similar to that of a DynamicTableSource.\n\n        If you don't specify the changelog_mode, the changelog containing all kinds of changes\n        (enumerated in RowKind) as the default ChangelogMode.\n\n        Column names and types of the Table are automatically derived from the TypeInformation of\n        the DataStream. If the outermost record's TypeInformation is a CompositeType, it will be\n        flattened in the first level. Composite nested fields will not be accessible.\n\n        By default, the stream record's timestamp and watermarks are not propagated unless\n        explicitly declared.\n\n        This method allows to declare a Schema for the resulting table. The declaration is similar\n        to a {@code CREATE TABLE} DDL in SQL and allows to:\n\n            1. enrich or overwrite automatically derived columns with a custom DataType\n            2. reorder columns\n            3. add computed or metadata columns next to the physical columns\n            4. access a stream record's timestamp\n            5. declare a watermark strategy or propagate the DataStream watermarks\n            6. declare a primary key\n\n        See :func:`from_data_stream` for more information and examples of how to declare a Schema.\n\n        :param data_stream: The changelog stream of Row.\n        :param schema: The customized schema for the final table.\n        :param changelog_mode: The expected kinds of changes in the incoming changelog.\n        :return: The converted Table.\n        "
        j_data_stream = data_stream._j_data_stream
        JPythonConfigUtil = get_gateway().jvm.org.apache.flink.python.util.PythonConfigUtil
        JPythonConfigUtil.configPythonOperator(j_data_stream.getExecutionEnvironment())
        if schema is None:
            return Table(self._j_tenv.fromChangelogStream(j_data_stream), t_env=self)
        elif changelog_mode is None:
            return Table(self._j_tenv.fromChangelogStream(j_data_stream, schema._j_schema), t_env=self)
        else:
            return Table(self._j_tenv.fromChangelogStream(j_data_stream, schema._j_schema, changelog_mode._j_changelog_mode), t_env=self)

    def to_data_stream(self, table: Table) -> DataStream:
        if False:
            i = 10
            return i + 15
        "\n        Converts the given Table into a DataStream.\n\n        Since the DataStream API does not support changelog processing natively, this method\n        assumes append-only/insert-only semantics during the table-to-stream conversion. The records\n        of class Row will always describe RowKind#INSERT changes. Updating tables are\n        not supported by this method and will produce an exception.\n\n        Note that the type system of the table ecosystem is richer than the one of the DataStream\n        API. The table runtime will make sure to properly serialize the output records to the first\n        operator of the DataStream API. Afterwards, the Types semantics of the DataStream API\n        need to be considered.\n\n        If the input table contains a single rowtime column, it will be propagated into a stream\n        record's timestamp. Watermarks will be propagated as well.\n\n        :param table: The Table to convert.\n        :return: The converted DataStream.\n        "
        return DataStream(self._j_tenv.toDataStream(table._j_table))

    def to_changelog_stream(self, table: Table, target_schema: Schema=None, changelog_mode: ChangelogMode=None) -> DataStream:
        if False:
            print('Hello World!')
        '\n        Converts the given Table into a DataStream of changelog entries.\n\n        Compared to :func:`to_data_stream`, this method produces instances of Row and sets the\n        RowKind flag that is contained in every record during runtime. The runtime behavior is\n        similar to that of a DynamicTableSink.\n\n        If you don\'t specify the changelog_mode, the changelog containing all kinds of changes\n        (enumerated in RowKind) as the default ChangelogMode.\n\n        The given Schema is used to configure the table runtime to convert columns and internal data\n        structures to the desired representation. The following example shows how to\n        convert a table column into a Row type.\n\n        Example:\n        ::\n\n            >>> table_env.to_changelog_stream(\n            ...     table,\n            ...     Schema.new_builder()             ...         .column("id", DataTypes.BIGINT())\n            ...         .column("payload", DataTypes.ROW(\n            ...                                     [DataTypes.FIELD("name", DataTypes.STRING()),\n            ...                                      DataTypes.FIELD("age", DataTypes.INT())]))\n            ...         .build())\n\n        Note that the type system of the table ecosystem is richer than the one of the DataStream\n        API. The table runtime will make sure to properly serialize the output records to the first\n        operator of the DataStream API. Afterwards, the Types semantics of the DataStream API need\n        to be considered.\n\n        If the input table contains a single rowtime column, it will be propagated into a stream\n        record\'s timestamp. Watermarks will be propagated as well.\n\n        If the rowtime should not be a concrete field in the final Row anymore, or the schema should\n        be symmetrical for both :func:`from_changelog_stream` and :func:`to_changelog_stream`, the\n        rowtime can also be declared as a metadata column that will be propagated into a stream\n        record\'s timestamp. It is possible to declare a schema without physical/regular columns.\n        In this case, those columns will be automatically derived and implicitly put at the\n        beginning of the schema declaration.\n\n        The following examples illustrate common schema declarations and their semantics:\n\n        Example:\n        ::\n\n            given a Table of (id INT, name STRING, my_rowtime TIMESTAMP_LTZ(3))\n\n            === EXAMPLE 1 ===\n\n            no physical columns defined, they will be derived automatically,\n            the last derived physical column will be skipped in favor of the metadata column\n\n            >>> Schema.new_builder()             ...     .column_by_metadata("rowtime", "TIMESTAMP_LTZ(3)")             ...     .build()\n\n            equal to: CREATE TABLE (id INT, name STRING, rowtime TIMESTAMP_LTZ(3) METADATA)\n\n            === EXAMPLE 2 ===\n\n            physical columns defined, all columns must be defined\n\n            >>> Schema.new_builder()             ...     .column("id", "INT")             ...     .column("name", "STRING")             ...     .column_by_metadata("rowtime", "TIMESTAMP_LTZ(3)")             ...     .build()\n\n            equal to: CREATE TABLE (id INT, name STRING, rowtime TIMESTAMP_LTZ(3) METADATA)\n\n        :param table: The Table to convert. It can be updating or insert-only.\n        :param target_schema: The Schema that decides about the final external representation in\n            DataStream records.\n        :param changelog_mode: The required kinds of changes in the result changelog. An exception\n            will be thrown if the given updating table cannot be represented in this changelog mode.\n        :return: The converted changelog stream of Row.\n        '
        if target_schema is None:
            return DataStream(self._j_tenv.toChangelogStream(table._j_table))
        elif changelog_mode is None:
            return DataStream(self._j_tenv.toChangelogStream(table._j_table, target_schema._j_schema))
        else:
            return DataStream(self._j_tenv.toChangelogStream(table._j_table, target_schema._j_schema, changelog_mode._j_changelog_mode))

    def to_append_stream(self, table: Table, type_info: TypeInformation) -> DataStream:
        if False:
            print('Hello World!')
        '\n        Converts the given Table into a DataStream of a specified type. The Table must only have\n        insert (append) changes. If the Table is also modified by update or delete changes, the\n        conversion will fail.\n\n        The fields of the Table are mapped to DataStream as follows: Row and Tuple types: Fields are\n        mapped by position, field types must match.\n\n        :param table: The Table to convert.\n        :param type_info: The TypeInformation that specifies the type of the DataStream.\n        :return: The converted DataStream.\n\n        .. versionadded:: 1.12.0\n        '
        j_data_stream = self._j_tenv.toAppendStream(table._j_table, type_info.get_java_type_info())
        return DataStream(j_data_stream=j_data_stream)

    def to_retract_stream(self, table: Table, type_info: TypeInformation) -> DataStream:
        if False:
            return 10
        '\n        Converts the given Table into a DataStream of add and retract messages. The message will be\n        encoded as Tuple. The first field is a boolean flag, the second field holds the record of\n        the specified type.\n\n        A true flag indicates an add message, a false flag indicates a retract message.\n\n        The fields of the Table are mapped to DataStream as follows: Row and Tuple types: Fields are\n        mapped by position, field types must match.\n\n        :param table: The Table to convert.\n        :param type_info: The TypeInformation of the requested record type.\n        :return: The converted DataStream.\n\n        .. versionadded:: 1.12.0\n        '
        j_data_stream = self._j_tenv.toRetractStream(table._j_table, type_info.get_java_type_info())
        return DataStream(j_data_stream=j_data_stream)