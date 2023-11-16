from typing import Optional
from py4j.java_gateway import get_method
from pyflink.common.types import RowKind
from pyflink.common import Row
from pyflink.common.job_client import JobClient
from pyflink.java_gateway import get_gateway
from pyflink.table.result_kind import ResultKind
from pyflink.table.table_schema import TableSchema
from pyflink.table.types import _from_java_data_type
from pyflink.table.utils import pickled_bytes_to_python_converter
__all__ = ['TableResult', 'CloseableIterator']

class TableResult(object):
    """
    A :class:`~pyflink.table.TableResult` is the representation of the statement execution result.

    .. versionadded:: 1.11.0
    """

    def __init__(self, j_table_result):
        if False:
            print('Hello World!')
        self._j_table_result = j_table_result

    def get_job_client(self) -> Optional[JobClient]:
        if False:
            for i in range(10):
                print('nop')
        '\n        For DML and DQL statement, return the JobClient which associates the submitted Flink job.\n        For other statements (e.g.  DDL, DCL) return empty.\n\n        :return: The job client, optional.\n        :rtype: pyflink.common.JobClient\n\n        .. versionadded:: 1.11.0\n        '
        job_client = self._j_table_result.getJobClient()
        if job_client.isPresent():
            return JobClient(job_client.get())
        else:
            return None

    def wait(self, timeout_ms: int=None):
        if False:
            print('Hello World!')
        '\n        Wait if necessary for at most the given time (milliseconds) for the data to be ready.\n\n        For a select operation, this method will wait until the first row can be accessed locally.\n        For an insert operation, this method will wait for the job to finish,\n        because the result contains only one row.\n        For other operations, this method will return immediately,\n        because the result is already available locally.\n\n        .. versionadded:: 1.12.0\n        '
        if timeout_ms:
            TimeUnit = get_gateway().jvm.java.util.concurrent.TimeUnit
            get_method(self._j_table_result, 'await')(timeout_ms, TimeUnit.MILLISECONDS)
        else:
            get_method(self._j_table_result, 'await')()

    def get_table_schema(self) -> TableSchema:
        if False:
            while True:
                i = 10
        '\n        Get the schema of result.\n\n        The schema of DDL, USE, EXPLAIN:\n        ::\n\n            +-------------+-------------+----------+\n            | column name | column type | comments |\n            +-------------+-------------+----------+\n            | result      | STRING      |          |\n            +-------------+-------------+----------+\n\n        The schema of SHOW:\n        ::\n\n            +---------------+-------------+----------+\n            |  column name  | column type | comments |\n            +---------------+-------------+----------+\n            | <object name> | STRING      |          |\n            +---------------+-------------+----------+\n            The column name of `SHOW CATALOGS` is "catalog name",\n            the column name of `SHOW DATABASES` is "database name",\n            the column name of `SHOW TABLES` is "table name",\n            the column name of `SHOW VIEWS` is "view name",\n            the column name of `SHOW FUNCTIONS` is "function name".\n\n        The schema of DESCRIBE:\n        ::\n\n            +------------------+-------------+-------------------------------------------------+\n            | column name      | column type |                 comments                        |\n            +------------------+-------------+-------------------------------------------------+\n            | name             | STRING      | field name                                      |\n            +------------------+-------------+-------------------------------------------------+\n            | type             | STRING      | field type expressed as a String                |\n            +------------------+-------------+-------------------------------------------------+\n            | null             | BOOLEAN     | field nullability: true if a field is nullable, |\n            |                  |             | else false                                      |\n            +------------------+-------------+-------------------------------------------------+\n            | key              | BOOLEAN     | key constraint: \'PRI\' for primary keys,         |\n            |                  |             | \'UNQ\' for unique keys, else null                |\n            +------------------+-------------+-------------------------------------------------+\n            | computed column  | STRING      | computed column: string expression              |\n            |                  |             | if a field is computed column, else null        |\n            +------------------+-------------+-------------------------------------------------+\n            | watermark        | STRING      | watermark: string expression if a field is      |\n            |                  |             | watermark, else null                            |\n            +------------------+-------------+-------------------------------------------------+\n\n        The schema of INSERT: (one column per one sink)\n        ::\n\n            +----------------------------+-------------+-----------------------+\n            | column name                | column type | comments              |\n            +----------------------------+-------------+-----------------------+\n            | (name of the insert table) | BIGINT      | the insert table name |\n            +----------------------------+-------------+-----------------------+\n\n        The schema of SELECT is the selected field names and types.\n\n        :return: The schema of result.\n        :rtype: pyflink.table.TableSchema\n\n        .. versionadded:: 1.11.0\n        '
        return TableSchema(j_table_schema=self._get_java_table_schema())

    def get_result_kind(self) -> ResultKind:
        if False:
            print('Hello World!')
        '\n        Return the ResultKind which represents the result type.\n\n        For DDL operation and USE operation, the result kind is always SUCCESS.\n        For other operations, the result kind is always SUCCESS_WITH_CONTENT.\n\n        :return: The result kind.\n\n        .. versionadded:: 1.11.0\n        '
        return ResultKind._from_j_result_kind(self._j_table_result.getResultKind())

    def collect(self) -> 'CloseableIterator':
        if False:
            print('Hello World!')
        '\n        Get the result contents as a closeable row iterator.\n\n        Note:\n\n        For SELECT operation, the job will not be finished unless all result data has been\n        collected. So we should actively close the job to avoid resource leak through\n        CloseableIterator#close method. Calling CloseableIterator#close method will cancel the job\n        and release related resources.\n\n        For DML operation, Flink does not support getting the real affected row count now. So the\n        affected row count is always -1 (unknown) for every sink, and them will be returned until\n        the job is finished.\n        Calling CloseableIterator#close method will cancel the job.\n\n        For other operations, no flink job will be submitted (get_job_client() is always empty), and\n        the result is bounded. Do noting when calling CloseableIterator#close method.\n\n        Recommended code to call CloseableIterator#close method looks like:\n\n        >>> table_result = t_env.execute("select ...")\n        >>> with table_result.collect() as results:\n        >>>    for result in results:\n        >>>        ...\n\n        In order to fetch result to local, you can call either collect() and print(). But, they can\n        not be called both on the same TableResult instance.\n\n        :return: A CloseableIterator.\n\n        .. versionadded:: 1.12.0\n        '
        field_data_types = self._get_java_table_schema().getFieldDataTypes()
        j_iter = self._j_table_result.collect()
        return CloseableIterator(j_iter, field_data_types)

    def print(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Print the result contents as tableau form to client console.\n\n        This method has slightly different behaviors under different checkpointing settings.\n\n            - For batch jobs or streaming jobs without checkpointing,\n              this method has neither exactly-once nor at-least-once guarantee.\n              Query results are immediately accessible by the clients once they're produced,\n              but exceptions will be thrown when the job fails and restarts.\n            - For streaming jobs with exactly-once checkpointing,\n              this method guarantees an end-to-end exactly-once record delivery.\n              A result will be accessible by clients only after its corresponding checkpoint\n              completes.\n            - For streaming jobs with at-least-once checkpointing,\n              this method guarantees an end-to-end at-least-once record delivery.\n              Query results are immediately accessible by the clients once they're produced,\n              but it is possible for the same result to be delivered multiple times.\n\n        .. versionadded:: 1.11.0\n        "
        self._j_table_result.print()

    def _get_java_table_schema(self):
        if False:
            while True:
                i = 10
        TableSchema = get_gateway().jvm.org.apache.flink.table.api.TableSchema
        return TableSchema.fromResolvedSchema(self._j_table_result.getResolvedSchema())

class CloseableIterator(object):
    """
    Representing an Iterator that is also auto closeable.
    """

    def __init__(self, j_closeable_iterator, field_data_types):
        if False:
            i = 10
            return i + 15
        self._j_closeable_iterator = j_closeable_iterator
        self._j_field_data_types = field_data_types
        self._data_types = [_from_java_data_type(j_field_data_type) for j_field_data_type in self._j_field_data_types]

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        if not self._j_closeable_iterator.hasNext():
            raise StopIteration('No more data.')
        gateway = get_gateway()
        pickle_bytes = gateway.jvm.PythonBridgeUtils.getPickledBytesFromRow(self._j_closeable_iterator.next(), self._j_field_data_types)
        row_kind = RowKind(int.from_bytes(pickle_bytes[0], byteorder='big', signed=False))
        pickle_bytes = list(pickle_bytes[1:])
        field_data = zip(pickle_bytes, self._data_types)
        fields = []
        for (data, field_type) in field_data:
            if len(data) == 0:
                fields.append(None)
            else:
                fields.append(pickled_bytes_to_python_converter(data, field_type))
        result_row = Row(*fields)
        result_row.set_row_kind(row_kind)
        return result_row

    def next(self):
        if False:
            while True:
                i = 10
        return self.__next__()

    def close(self):
        if False:
            return 10
        self._j_closeable_iterator.close()

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        self.close()