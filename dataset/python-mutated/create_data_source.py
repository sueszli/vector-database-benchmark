import inspect
import os
import sys
from typing import IO, List
from pyspark.accumulators import _accumulatorRegistry
from pyspark.errors import PySparkAssertionError, PySparkRuntimeError, PySparkTypeError
from pyspark.java_gateway import local_connect_and_auth
from pyspark.serializers import read_bool, read_int, write_int, write_with_length, SpecialLengths
from pyspark.sql.datasource import DataSource
from pyspark.sql.types import _parse_datatype_json_string, StructType
from pyspark.util import handle_worker_exception
from pyspark.worker_util import check_python_version, read_command, pickleSer, send_accumulator_updates, setup_broadcasts, setup_memory_limits, setup_spark_files, utf8_deserializer

def main(infile: IO, outfile: IO) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Main method for creating a Python data source instance.\n\n    This process is invoked from the `UserDefinedPythonDataSourceRunner.runInPython` method\n    in JVM. This process is responsible for creating a `DataSource` object and send the\n    information needed back to the JVM.\n\n    The JVM sends the following information to this process:\n    - a `DataSource` class representing the data source to be created.\n    - a provider name in string.\n    - a list of paths in string.\n    - an optional user-specified schema in json string.\n    - a dictionary of options in string.\n\n    This process then creates a `DataSource` instance using the above information and\n    sends the pickled instance as well as the schema back to the JVM.\n    '
    try:
        check_python_version(infile)
        memory_limit_mb = int(os.environ.get('PYSPARK_PLANNER_MEMORY_MB', '-1'))
        setup_memory_limits(memory_limit_mb)
        setup_spark_files(infile)
        setup_broadcasts(infile)
        _accumulatorRegistry.clear()
        data_source_cls = read_command(pickleSer, infile)
        if not (isinstance(data_source_cls, type) and issubclass(data_source_cls, DataSource)):
            raise PySparkAssertionError(error_class='PYTHON_DATA_SOURCE_TYPE_MISMATCH', message_parameters={'expected': 'a subclass of DataSource', 'actual': f"'{type(data_source_cls).__name__}'"})
        if not inspect.ismethod(data_source_cls.name):
            raise PySparkTypeError(error_class='PYTHON_DATA_SOURCE_TYPE_MISMATCH', message_parameters={'expected': "'name()' method to be a classmethod", 'actual': f"'{type(data_source_cls.name).__name__}'"})
        provider = utf8_deserializer.loads(infile)
        if provider.lower() != data_source_cls.name().lower():
            raise PySparkAssertionError(error_class='PYTHON_DATA_SOURCE_TYPE_MISMATCH', message_parameters={'expected': f'provider with name {data_source_cls.name()}', 'actual': f"'{provider}'"})
        num_paths = read_int(infile)
        paths: List[str] = []
        for _ in range(num_paths):
            paths.append(utf8_deserializer.loads(infile))
        user_specified_schema = None
        if read_bool(infile):
            user_specified_schema = _parse_datatype_json_string(utf8_deserializer.loads(infile))
            if not isinstance(user_specified_schema, StructType):
                raise PySparkAssertionError(error_class='PYTHON_DATA_SOURCE_TYPE_MISMATCH', message_parameters={'expected': "the user-defined schema to be a 'StructType'", 'actual': f"'{type(data_source_cls).__name__}'"})
        options = dict()
        num_options = read_int(infile)
        for _ in range(num_options):
            key = utf8_deserializer.loads(infile)
            value = utf8_deserializer.loads(infile)
            options[key] = value
        try:
            data_source = data_source_cls(paths=paths, userSpecifiedSchema=user_specified_schema, options=options)
        except Exception as e:
            raise PySparkRuntimeError(error_class='PYTHON_DATA_SOURCE_CREATE_ERROR', message_parameters={'type': 'instance', 'error': str(e)})
        is_ddl_string = False
        if user_specified_schema is None:
            try:
                schema = data_source.schema()
                if isinstance(schema, str):
                    is_ddl_string = True
            except NotImplementedError:
                raise PySparkRuntimeError(error_class='PYTHON_DATA_SOURCE_METHOD_NOT_IMPLEMENTED', message_parameters={'type': 'instance', 'method': 'schema'})
        else:
            schema = user_specified_schema
        assert schema is not None
        pickleSer._write_with_length(data_source, outfile)
        write_int(int(is_ddl_string), outfile)
        if is_ddl_string:
            write_with_length(schema.encode('utf-8'), outfile)
        else:
            write_with_length(schema.json().encode('utf-8'), outfile)
    except BaseException as e:
        handle_worker_exception(e, outfile)
        sys.exit(-1)
    send_accumulator_updates(outfile)
    if read_int(infile) == SpecialLengths.END_OF_STREAM:
        write_int(SpecialLengths.END_OF_STREAM, outfile)
    else:
        write_int(SpecialLengths.END_OF_DATA_SECTION, outfile)
        sys.exit(-1)
if __name__ == '__main__':
    java_port = int(os.environ['PYTHON_WORKER_FACTORY_PORT'])
    auth_secret = os.environ['PYTHON_WORKER_FACTORY_SECRET']
    (sock_file, _) = local_connect_and_auth(java_port, auth_secret)
    main(sock_file, sock_file)