import inspect
import os
import sys
from typing import Dict, List, IO, Tuple
from pyspark.accumulators import _accumulatorRegistry
from pyspark.errors import PySparkRuntimeError, PySparkValueError
from pyspark.java_gateway import local_connect_and_auth
from pyspark.serializers import read_bool, read_int, write_int, write_with_length, SpecialLengths
from pyspark.sql.types import _parse_datatype_json_string
from pyspark.sql.udtf import AnalyzeArgument, AnalyzeResult
from pyspark.util import handle_worker_exception
from pyspark.worker_util import check_python_version, read_command, pickleSer, send_accumulator_updates, setup_broadcasts, setup_memory_limits, setup_spark_files, utf8_deserializer

def read_udtf(infile: IO) -> type:
    if False:
        for i in range(10):
            print('nop')
    'Reads the Python UDTF and checks if its valid or not.'
    handler = read_command(pickleSer, infile)
    if not isinstance(handler, type):
        raise PySparkRuntimeError(f"Invalid UDTF handler type. Expected a class (type 'type'), but got an instance of {type(handler).__name__}.")
    if not hasattr(handler, 'analyze') or not isinstance(inspect.getattr_static(handler, 'analyze'), staticmethod):
        raise PySparkRuntimeError("Failed to execute the user defined table function because it has not implemented the 'analyze' static method or specified a fixed return type during registration time. Please add the 'analyze' static method or specify the return type, and try the query again.")
    return handler

def read_arguments(infile: IO) -> Tuple[List[AnalyzeArgument], Dict[str, AnalyzeArgument]]:
    if False:
        while True:
            i = 10
    'Reads the arguments for `analyze` static method.'
    num_args = read_int(infile)
    args: List[AnalyzeArgument] = []
    kwargs: Dict[str, AnalyzeArgument] = {}
    for _ in range(num_args):
        dt = _parse_datatype_json_string(utf8_deserializer.loads(infile))
        if read_bool(infile):
            value = pickleSer._read_with_length(infile)
            if dt.needConversion():
                value = dt.fromInternal(value)
        else:
            value = None
        is_table = read_bool(infile)
        argument = AnalyzeArgument(dataType=dt, value=value, isTable=is_table)
        is_named_arg = read_bool(infile)
        if is_named_arg:
            name = utf8_deserializer.loads(infile)
            kwargs[name] = argument
        else:
            args.append(argument)
    return (args, kwargs)

def main(infile: IO, outfile: IO) -> None:
    if False:
        print('Hello World!')
    "\n    Runs the Python UDTF's `analyze` static method.\n\n    This process will be invoked from `UserDefinedPythonTableFunctionAnalyzeRunner.runInPython`\n    in JVM and receive the Python UDTF and its arguments for the `analyze` static method,\n    and call the `analyze` static method, and send back a AnalyzeResult as a result of the method.\n    "
    try:
        check_python_version(infile)
        memory_limit_mb = int(os.environ.get('PYSPARK_PLANNER_MEMORY_MB', '-1'))
        setup_memory_limits(memory_limit_mb)
        setup_spark_files(infile)
        setup_broadcasts(infile)
        _accumulatorRegistry.clear()
        handler = read_udtf(infile)
        (args, kwargs) = read_arguments(infile)
        result = handler.analyze(*args, **kwargs)
        if not isinstance(result, AnalyzeResult):
            raise PySparkValueError(f'Output of `analyze` static method of Python UDTFs expects a pyspark.sql.udtf.AnalyzeResult but got: {type(result)}')
        write_with_length(result.schema.json().encode('utf-8'), outfile)
        pickleSer._write_with_length(result, outfile)
        write_int(1 if result.withSinglePartition else 0, outfile)
        write_int(len(result.partitionBy), outfile)
        for partitioning_col in result.partitionBy:
            write_with_length(partitioning_col.name.encode('utf-8'), outfile)
        write_int(len(result.orderBy), outfile)
        for ordering_col in result.orderBy:
            write_with_length(ordering_col.name.encode('utf-8'), outfile)
            write_int(1 if ordering_col.ascending else 0, outfile)
            if ordering_col.overrideNullsFirst is None:
                write_int(0, outfile)
            elif ordering_col.overrideNullsFirst:
                write_int(1, outfile)
            else:
                write_int(2, outfile)
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