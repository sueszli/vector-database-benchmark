from typing import Union
from pyflink.java_gateway import get_gateway
from pyflink.table import ExplainDetail
from pyflink.table.table_descriptor import TableDescriptor
from pyflink.table.table_result import TableResult
from pyflink.util.java_utils import to_j_explain_detail_arr
__all__ = ['StatementSet']

class StatementSet(object):
    """
    A :class:`~StatementSet` accepts pipelines defined by DML statements or :class:`~Table` objects.
    The planner can optimize all added statements together and then submit them as one job.

    The added statements will be cleared when calling the :func:`~StatementSet.execute` method.

    .. versionadded:: 1.11.0
    """

    def __init__(self, _j_statement_set, t_env):
        if False:
            print('Hello World!')
        self._j_statement_set = _j_statement_set
        self._t_env = t_env

    def add_insert_sql(self, stmt: str) -> 'StatementSet':
        if False:
            while True:
                i = 10
        '\n        add insert statement to the set.\n\n        :param stmt: The statement to be added.\n        :return: current StatementSet instance.\n\n        .. versionadded:: 1.11.0\n        '
        self._j_statement_set.addInsertSql(stmt)
        return self

    def attach_as_datastream(self):
        if False:
            return 10
        '\n        Optimizes all statements as one entity and adds them as transformations to the underlying\n        StreamExecutionEnvironment.\n\n        Use :func:`~pyflink.datastream.StreamExecutionEnvironment.execute` to execute them.\n\n        The added statements will be cleared after calling this method.\n\n        .. versionadded:: 1.16.0\n        '
        self._j_statement_set.attachAsDataStream()

    def add_insert(self, target_path_or_descriptor: Union[str, TableDescriptor], table, overwrite: bool=False) -> 'StatementSet':
        if False:
            while True:
                i = 10
        '\n        Adds a statement that the pipeline defined by the given Table object should be written to a\n        table (backed by a DynamicTableSink) that was registered under the specified path or\n        expressed via the given TableDescriptor.\n\n        1. When target_path_or_descriptor is a tale path:\n\n            See the documentation of :func:`~TableEnvironment.use_database` or\n            :func:`~TableEnvironment.use_catalog` for the rules on the path resolution.\n\n        2. When target_path_or_descriptor is a table descriptor:\n\n            The given TableDescriptor is registered as an inline (i.e. anonymous) temporary catalog\n            table (see :func:`~TableEnvironment.create_temporary_table`).\n\n            Then a statement is added to the statement set that inserts the Table object\'s pipeline\n            into that temporary table.\n\n            This method allows to declare a Schema for the sink descriptor. The declaration is\n            similar to a {@code CREATE TABLE} DDL in SQL and allows to:\n\n                1. overwrite automatically derived columns with a custom DataType\n                2. add metadata columns next to the physical columns\n                3. declare a primary key\n\n            It is possible to declare a schema without physical/regular columns. In this case, those\n            columns will be automatically derived and implicitly put at the beginning of the schema\n            declaration.\n\n            Examples:\n            ::\n\n                >>> stmt_set = table_env.create_statement_set()\n                >>> source_table = table_env.from_path("SourceTable")\n                >>> sink_descriptor = TableDescriptor.for_connector("blackhole") \\\n                ...     .schema(Schema.new_builder()\n                ...         .build()) \\\n                ...     .build()\n                >>> stmt_set.add_insert(sink_descriptor, source_table)\n\n            .. note:: add_insert for a table descriptor (case 2.) was added from\n                flink 1.14.0.\n\n        :param target_path_or_descriptor: The path of the registered\n            :class:`~pyflink.table.TableSink` or the descriptor describing the sink table into which\n            data should be inserted to which the :class:`~pyflink.table.Table` is written.\n        :param table: The Table to add.\n        :type table: pyflink.table.Table\n        :param overwrite: Indicates whether the insert should overwrite existing data or not.\n        :return: current StatementSet instance.\n\n        .. versionadded:: 1.11.0\n        '
        if isinstance(target_path_or_descriptor, str):
            self._j_statement_set.addInsert(target_path_or_descriptor, table._j_table, overwrite)
        else:
            self._j_statement_set.addInsert(target_path_or_descriptor._j_table_descriptor, table._j_table, overwrite)
        return self

    def explain(self, *extra_details: ExplainDetail) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        returns the AST and the execution plan of all statements and Tables.\n\n        :param extra_details: The extra explain details which the explain result should include,\n                              e.g. estimated cost, changelog mode for streaming\n        :return: All statements and Tables for which the AST and execution plan will be returned.\n\n        .. versionadded:: 1.11.0\n        '
        TEXT = get_gateway().jvm.org.apache.flink.table.api.ExplainFormat.TEXT
        j_extra_details = to_j_explain_detail_arr(extra_details)
        return self._j_statement_set.explain(TEXT, j_extra_details)

    def execute(self) -> TableResult:
        if False:
            print('Hello World!')
        '\n        execute all statements and Tables as a batch.\n\n        .. note::\n            The added statements and Tables will be cleared when executing this method.\n\n        :return: execution result.\n\n        .. versionadded:: 1.11.0\n        '
        self._t_env._before_execute()
        return TableResult(self._j_statement_set.execute())