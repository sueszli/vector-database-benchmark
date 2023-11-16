from pyspark.sql.connect.utils import check_dependencies
check_dependencies(__name__)
import sys
import pickle
from typing import cast, overload, Callable, Dict, List, Optional, TYPE_CHECKING, Union
from pyspark.serializers import CloudPickleSerializer
from pyspark.sql.connect.plan import DataSource, LogicalPlan, Read, WriteStreamOperation
import pyspark.sql.connect.proto as pb2
from pyspark.sql.connect.readwriter import OptionUtils, to_str
from pyspark.sql.connect.streaming.query import StreamingQuery
from pyspark.sql.streaming.readwriter import DataStreamReader as PySparkDataStreamReader, DataStreamWriter as PySparkDataStreamWriter
from pyspark.sql.connect.utils import get_python_ver
from pyspark.sql.types import Row, StructType
from pyspark.errors import PySparkTypeError, PySparkValueError, PySparkPicklingError
if TYPE_CHECKING:
    from pyspark.sql.connect.session import SparkSession
    from pyspark.sql.connect._typing import OptionalPrimitiveType
    from pyspark.sql.connect.dataframe import DataFrame
    from pyspark.sql._typing import SupportsProcess

class DataStreamReader(OptionUtils):

    def __init__(self, client: 'SparkSession') -> None:
        if False:
            for i in range(10):
                print('nop')
        self._format: Optional[str] = None
        self._schema = ''
        self._client = client
        self._options: Dict[str, str] = {}

    def _df(self, plan: LogicalPlan) -> 'DataFrame':
        if False:
            return 10
        from pyspark.sql.connect.dataframe import DataFrame
        return DataFrame.withPlan(plan, self._client)

    def format(self, source: str) -> 'DataStreamReader':
        if False:
            for i in range(10):
                print('nop')
        self._format = source
        return self
    format.__doc__ = PySparkDataStreamReader.format.__doc__

    def schema(self, schema: Union[StructType, str]) -> 'DataStreamReader':
        if False:
            return 10
        if isinstance(schema, StructType):
            self._schema = schema.json()
        elif isinstance(schema, str):
            self._schema = schema
        else:
            raise PySparkTypeError(error_class='NOT_STR_OR_STRUCT', message_parameters={'arg_name': 'schema', 'arg_type': type(schema).__name__})
        return self
    schema.__doc__ = PySparkDataStreamReader.schema.__doc__

    def option(self, key: str, value: 'OptionalPrimitiveType') -> 'DataStreamReader':
        if False:
            for i in range(10):
                print('nop')
        self._options[key] = str(value)
        return self
    option.__doc__ = PySparkDataStreamReader.option.__doc__

    def options(self, **options: 'OptionalPrimitiveType') -> 'DataStreamReader':
        if False:
            i = 10
            return i + 15
        for k in options:
            self.option(k, to_str(options[k]))
        return self
    options.__doc__ = PySparkDataStreamReader.options.__doc__

    def load(self, path: Optional[str]=None, format: Optional[str]=None, schema: Optional[Union[StructType, str]]=None, **options: 'OptionalPrimitiveType') -> 'DataFrame':
        if False:
            i = 10
            return i + 15
        if format is not None:
            self.format(format)
        if schema is not None:
            self.schema(schema)
        self.options(**options)
        if path is not None and (type(path) != str or len(path.strip()) == 0):
            raise PySparkValueError(error_class='VALUE_NOT_NON_EMPTY_STR', message_parameters={'arg_name': 'path', 'arg_value': str(path)})
        plan = DataSource(format=self._format, schema=self._schema, options=self._options, paths=[path] if path else None, is_streaming=True)
        return self._df(plan)
    load.__doc__ = PySparkDataStreamReader.load.__doc__

    def json(self, path: str, schema: Optional[Union[StructType, str]]=None, primitivesAsString: Optional[Union[bool, str]]=None, prefersDecimal: Optional[Union[bool, str]]=None, allowComments: Optional[Union[bool, str]]=None, allowUnquotedFieldNames: Optional[Union[bool, str]]=None, allowSingleQuotes: Optional[Union[bool, str]]=None, allowNumericLeadingZero: Optional[Union[bool, str]]=None, allowBackslashEscapingAnyCharacter: Optional[Union[bool, str]]=None, mode: Optional[str]=None, columnNameOfCorruptRecord: Optional[str]=None, dateFormat: Optional[str]=None, timestampFormat: Optional[str]=None, multiLine: Optional[Union[bool, str]]=None, allowUnquotedControlChars: Optional[Union[bool, str]]=None, lineSep: Optional[str]=None, locale: Optional[str]=None, dropFieldIfAllNull: Optional[Union[bool, str]]=None, encoding: Optional[str]=None, pathGlobFilter: Optional[Union[bool, str]]=None, recursiveFileLookup: Optional[Union[bool, str]]=None, allowNonNumericNumbers: Optional[Union[bool, str]]=None) -> 'DataFrame':
        if False:
            while True:
                i = 10
        self._set_opts(schema=schema, primitivesAsString=primitivesAsString, prefersDecimal=prefersDecimal, allowComments=allowComments, allowUnquotedFieldNames=allowUnquotedFieldNames, allowSingleQuotes=allowSingleQuotes, allowNumericLeadingZero=allowNumericLeadingZero, allowBackslashEscapingAnyCharacter=allowBackslashEscapingAnyCharacter, mode=mode, columnNameOfCorruptRecord=columnNameOfCorruptRecord, dateFormat=dateFormat, timestampFormat=timestampFormat, multiLine=multiLine, allowUnquotedControlChars=allowUnquotedControlChars, lineSep=lineSep, locale=locale, dropFieldIfAllNull=dropFieldIfAllNull, encoding=encoding, pathGlobFilter=pathGlobFilter, recursiveFileLookup=recursiveFileLookup, allowNonNumericNumbers=allowNonNumericNumbers)
        if isinstance(path, str):
            return self.load(path=path, format='json')
        else:
            raise PySparkTypeError(error_class='NOT_STR', message_parameters={'arg_name': 'path', 'arg_type': type(path).__name__})
    json.__doc__ = PySparkDataStreamReader.json.__doc__

    def orc(self, path: str, mergeSchema: Optional[bool]=None, pathGlobFilter: Optional[Union[bool, str]]=None, recursiveFileLookup: Optional[Union[bool, str]]=None) -> 'DataFrame':
        if False:
            while True:
                i = 10
        self._set_opts(mergeSchema=mergeSchema, pathGlobFilter=pathGlobFilter, recursiveFileLookup=recursiveFileLookup)
        if isinstance(path, str):
            return self.load(path=path, format='orc')
        else:
            raise PySparkTypeError(error_class='NOT_STR', message_parameters={'arg_name': 'path', 'arg_type': type(path).__name__})
    orc.__doc__ = PySparkDataStreamReader.orc.__doc__

    def parquet(self, path: str, mergeSchema: Optional[bool]=None, pathGlobFilter: Optional[Union[bool, str]]=None, recursiveFileLookup: Optional[Union[bool, str]]=None, datetimeRebaseMode: Optional[Union[bool, str]]=None, int96RebaseMode: Optional[Union[bool, str]]=None) -> 'DataFrame':
        if False:
            return 10
        self._set_opts(mergeSchema=mergeSchema, pathGlobFilter=pathGlobFilter, recursiveFileLookup=recursiveFileLookup, datetimeRebaseMode=datetimeRebaseMode, int96RebaseMode=int96RebaseMode)
        self._set_opts(mergeSchema=mergeSchema, pathGlobFilter=pathGlobFilter, recursiveFileLookup=recursiveFileLookup, datetimeRebaseMode=datetimeRebaseMode, int96RebaseMode=int96RebaseMode)
        if isinstance(path, str):
            return self.load(path=path, format='parquet')
        else:
            raise PySparkTypeError(error_class='NOT_STR', message_parameters={'arg_name': 'path', 'arg_type': type(path).__name__})
    parquet.__doc__ = PySparkDataStreamReader.parquet.__doc__

    def text(self, path: str, wholetext: bool=False, lineSep: Optional[str]=None, pathGlobFilter: Optional[Union[bool, str]]=None, recursiveFileLookup: Optional[Union[bool, str]]=None) -> 'DataFrame':
        if False:
            for i in range(10):
                print('nop')
        self._set_opts(wholetext=wholetext, lineSep=lineSep, pathGlobFilter=pathGlobFilter, recursiveFileLookup=recursiveFileLookup)
        if isinstance(path, str):
            return self.load(path=path, format='text')
        else:
            raise PySparkTypeError(error_class='NOT_STR', message_parameters={'arg_name': 'path', 'arg_type': type(path).__name__})
    text.__doc__ = PySparkDataStreamReader.text.__doc__

    def csv(self, path: str, schema: Optional[Union[StructType, str]]=None, sep: Optional[str]=None, encoding: Optional[str]=None, quote: Optional[str]=None, escape: Optional[str]=None, comment: Optional[str]=None, header: Optional[Union[bool, str]]=None, inferSchema: Optional[Union[bool, str]]=None, ignoreLeadingWhiteSpace: Optional[Union[bool, str]]=None, ignoreTrailingWhiteSpace: Optional[Union[bool, str]]=None, nullValue: Optional[str]=None, nanValue: Optional[str]=None, positiveInf: Optional[str]=None, negativeInf: Optional[str]=None, dateFormat: Optional[str]=None, timestampFormat: Optional[str]=None, maxColumns: Optional[Union[int, str]]=None, maxCharsPerColumn: Optional[Union[int, str]]=None, maxMalformedLogPerPartition: Optional[Union[int, str]]=None, mode: Optional[str]=None, columnNameOfCorruptRecord: Optional[str]=None, multiLine: Optional[Union[bool, str]]=None, charToEscapeQuoteEscaping: Optional[Union[bool, str]]=None, enforceSchema: Optional[Union[bool, str]]=None, emptyValue: Optional[str]=None, locale: Optional[str]=None, lineSep: Optional[str]=None, pathGlobFilter: Optional[Union[bool, str]]=None, recursiveFileLookup: Optional[Union[bool, str]]=None, unescapedQuoteHandling: Optional[str]=None) -> 'DataFrame':
        if False:
            for i in range(10):
                print('nop')
        self._set_opts(schema=schema, sep=sep, encoding=encoding, quote=quote, escape=escape, comment=comment, header=header, inferSchema=inferSchema, ignoreLeadingWhiteSpace=ignoreLeadingWhiteSpace, ignoreTrailingWhiteSpace=ignoreTrailingWhiteSpace, nullValue=nullValue, nanValue=nanValue, positiveInf=positiveInf, negativeInf=negativeInf, dateFormat=dateFormat, timestampFormat=timestampFormat, maxColumns=maxColumns, maxCharsPerColumn=maxCharsPerColumn, maxMalformedLogPerPartition=maxMalformedLogPerPartition, mode=mode, columnNameOfCorruptRecord=columnNameOfCorruptRecord, multiLine=multiLine, charToEscapeQuoteEscaping=charToEscapeQuoteEscaping, enforceSchema=enforceSchema, emptyValue=emptyValue, locale=locale, lineSep=lineSep, pathGlobFilter=pathGlobFilter, recursiveFileLookup=recursiveFileLookup, unescapedQuoteHandling=unescapedQuoteHandling)
        if isinstance(path, str):
            return self.load(path=path, format='csv')
        else:
            raise PySparkTypeError(error_class='NOT_STR', message_parameters={'arg_name': 'path', 'arg_type': type(path).__name__})
    csv.__doc__ = PySparkDataStreamReader.csv.__doc__

    def xml(self, path: str, rowTag: Optional[str]=None, schema: Optional[Union[StructType, str]]=None, excludeAttribute: Optional[Union[bool, str]]=None, attributePrefix: Optional[str]=None, valueTag: Optional[str]=None, ignoreSurroundingSpaces: Optional[Union[bool, str]]=None, rowValidationXSDPath: Optional[str]=None, ignoreNamespace: Optional[Union[bool, str]]=None, wildcardColName: Optional[str]=None, encoding: Optional[str]=None, inferSchema: Optional[Union[bool, str]]=None, nullValue: Optional[str]=None, dateFormat: Optional[str]=None, timestampFormat: Optional[str]=None, mode: Optional[str]=None, columnNameOfCorruptRecord: Optional[str]=None, multiLine: Optional[Union[bool, str]]=None, samplingRatio: Optional[Union[float, str]]=None, locale: Optional[str]=None) -> 'DataFrame':
        if False:
            i = 10
            return i + 15
        self._set_opts(rowTag=rowTag, schema=schema, excludeAttribute=excludeAttribute, attributePrefix=attributePrefix, valueTag=valueTag, ignoreSurroundingSpaces=ignoreSurroundingSpaces, rowValidationXSDPath=rowValidationXSDPath, ignoreNamespace=ignoreNamespace, wildcardColName=wildcardColName, encoding=encoding, inferSchema=inferSchema, nullValue=nullValue, dateFormat=dateFormat, timestampFormat=timestampFormat, mode=mode, columnNameOfCorruptRecord=columnNameOfCorruptRecord, multiLine=multiLine, samplingRatio=samplingRatio, locale=locale)
        if isinstance(path, str):
            return self.load(path=path, format='xml')
        else:
            raise PySparkTypeError(error_class='NOT_STR', message_parameters={'arg_name': 'path', 'arg_type': type(path).__name__})
    xml.__doc__ = PySparkDataStreamReader.xml.__doc__

    def table(self, tableName: str) -> 'DataFrame':
        if False:
            return 10
        return self._df(Read(tableName, self._options, is_streaming=True))
    table.__doc__ = PySparkDataStreamReader.table.__doc__
DataStreamReader.__doc__ = PySparkDataStreamReader.__doc__

class DataStreamWriter:

    def __init__(self, plan: 'LogicalPlan', session: 'SparkSession') -> None:
        if False:
            print('Hello World!')
        self._session = session
        self._write_stream = WriteStreamOperation(plan)
        self._write_proto = self._write_stream.write_op

    def outputMode(self, outputMode: str) -> 'DataStreamWriter':
        if False:
            return 10
        self._write_proto.output_mode = outputMode
        return self
    outputMode.__doc__ = PySparkDataStreamWriter.outputMode.__doc__

    def format(self, source: str) -> 'DataStreamWriter':
        if False:
            print('Hello World!')
        self._write_proto.format = source
        return self
    format.__doc__ = PySparkDataStreamWriter.format.__doc__

    def option(self, key: str, value: 'OptionalPrimitiveType') -> 'DataStreamWriter':
        if False:
            print('Hello World!')
        self._write_proto.options[key] = cast(str, to_str(value))
        return self
    option.__doc__ = PySparkDataStreamWriter.option.__doc__

    def options(self, **options: 'OptionalPrimitiveType') -> 'DataStreamWriter':
        if False:
            for i in range(10):
                print('nop')
        for k in options:
            self.option(k, options[k])
        return self
    options.__doc__ = PySparkDataStreamWriter.options.__doc__

    @overload
    def partitionBy(self, *cols: str) -> 'DataStreamWriter':
        if False:
            return 10
        ...

    @overload
    def partitionBy(self, __cols: List[str]) -> 'DataStreamWriter':
        if False:
            for i in range(10):
                print('nop')
        ...

    def partitionBy(self, *cols: str) -> 'DataStreamWriter':
        if False:
            for i in range(10):
                print('nop')
        if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
            cols = cols[0]
        while len(self._write_proto.partitioning_column_names) > 0:
            self._write_proto.partitioning_column_names.pop()
        self._write_proto.partitioning_column_names.extend(cast(List[str], cols))
        return self
    partitionBy.__doc__ = PySparkDataStreamWriter.partitionBy.__doc__

    def queryName(self, queryName: str) -> 'DataStreamWriter':
        if False:
            print('Hello World!')
        self._write_proto.query_name = queryName
        return self
    queryName.__doc__ = PySparkDataStreamWriter.queryName.__doc__

    @overload
    def trigger(self, *, processingTime: str) -> 'DataStreamWriter':
        if False:
            while True:
                i = 10
        ...

    @overload
    def trigger(self, *, once: bool) -> 'DataStreamWriter':
        if False:
            return 10
        ...

    @overload
    def trigger(self, *, continuous: str) -> 'DataStreamWriter':
        if False:
            print('Hello World!')
        ...

    @overload
    def trigger(self, *, availableNow: bool) -> 'DataStreamWriter':
        if False:
            return 10
        ...

    def trigger(self, *, processingTime: Optional[str]=None, once: Optional[bool]=None, continuous: Optional[str]=None, availableNow: Optional[bool]=None) -> 'DataStreamWriter':
        if False:
            return 10
        params = [processingTime, once, continuous, availableNow]
        if params.count(None) == 4:
            raise PySparkValueError(error_class='ONLY_ALLOW_SINGLE_TRIGGER', message_parameters={})
        elif params.count(None) < 3:
            raise PySparkValueError(error_class='ONLY_ALLOW_SINGLE_TRIGGER', message_parameters={})
        if processingTime is not None:
            if type(processingTime) != str or len(processingTime.strip()) == 0:
                raise PySparkValueError(error_class='VALUE_NOT_NON_EMPTY_STR', message_parameters={'arg_name': 'processingTime', 'arg_value': str(processingTime)})
            self._write_proto.processing_time_interval = processingTime.strip()
        elif once is not None:
            if once is not True:
                raise PySparkValueError(error_class='VALUE_NOT_TRUE', message_parameters={'arg_name': 'once', 'arg_value': str(once)})
            self._write_proto.once = True
        elif continuous is not None:
            if type(continuous) != str or len(continuous.strip()) == 0:
                raise PySparkValueError(error_class='VALUE_NOT_NON_EMPTY_STR', message_parameters={'arg_name': 'continuous', 'arg_value': str(continuous)})
            self._write_proto.continuous_checkpoint_interval = continuous.strip()
        else:
            if availableNow is not True:
                raise PySparkValueError(error_class='VALUE_NOT_TRUE', message_parameters={'arg_name': 'availableNow', 'arg_value': str(availableNow)})
            self._write_proto.available_now = True
        return self
    trigger.__doc__ = PySparkDataStreamWriter.trigger.__doc__

    @overload
    def foreach(self, f: Callable[[Row], None]) -> 'DataStreamWriter':
        if False:
            return 10
        ...

    @overload
    def foreach(self, f: 'SupportsProcess') -> 'DataStreamWriter':
        if False:
            for i in range(10):
                print('nop')
        ...

    def foreach(self, f: Union[Callable[[Row], None], 'SupportsProcess']) -> 'DataStreamWriter':
        if False:
            return 10
        from pyspark.serializers import CPickleSerializer, AutoBatchedSerializer
        func = PySparkDataStreamWriter._construct_foreach_function(f)
        serializer = AutoBatchedSerializer(CPickleSerializer())
        command = (func, None, serializer, serializer)
        try:
            self._write_proto.foreach_writer.python_function.command = CloudPickleSerializer().dumps(command)
        except pickle.PicklingError:
            raise PySparkPicklingError(error_class='STREAMING_CONNECT_SERIALIZATION_ERROR', message_parameters={'name': 'foreach'})
        self._write_proto.foreach_writer.python_function.python_ver = '%d.%d' % sys.version_info[:2]
        return self
    foreach.__doc__ = PySparkDataStreamWriter.foreach.__doc__

    def foreachBatch(self, func: Callable[['DataFrame', int], None]) -> 'DataStreamWriter':
        if False:
            for i in range(10):
                print('nop')
        try:
            self._write_proto.foreach_batch.python_function.command = CloudPickleSerializer().dumps(func)
        except pickle.PicklingError:
            raise PySparkPicklingError(error_class='STREAMING_CONNECT_SERIALIZATION_ERROR', message_parameters={'name': 'foreachBatch'})
        self._write_proto.foreach_batch.python_function.python_ver = get_python_ver()
        return self
    foreachBatch.__doc__ = PySparkDataStreamWriter.foreachBatch.__doc__

    def _start_internal(self, path: Optional[str]=None, tableName: Optional[str]=None, format: Optional[str]=None, outputMode: Optional[str]=None, partitionBy: Optional[Union[str, List[str]]]=None, queryName: Optional[str]=None, **options: 'OptionalPrimitiveType') -> StreamingQuery:
        if False:
            return 10
        self.options(**options)
        if outputMode is not None:
            self.outputMode(outputMode)
        if partitionBy is not None:
            self.partitionBy(partitionBy)
        if format is not None:
            self.format(format)
        if queryName is not None:
            self.queryName(queryName)
        if path:
            self._write_proto.path = path
        if tableName:
            self._write_proto.table_name = tableName
        cmd = self._write_stream.command(self._session.client)
        (_, properties) = self._session.client.execute_command(cmd)
        start_result = cast(pb2.WriteStreamOperationStartResult, properties['write_stream_operation_start_result'])
        return StreamingQuery(session=self._session, queryId=start_result.query_id.id, runId=start_result.query_id.run_id, name=start_result.name)

    def start(self, path: Optional[str]=None, format: Optional[str]=None, outputMode: Optional[str]=None, partitionBy: Optional[Union[str, List[str]]]=None, queryName: Optional[str]=None, **options: 'OptionalPrimitiveType') -> StreamingQuery:
        if False:
            i = 10
            return i + 15
        return self._start_internal(path=path, tableName=None, format=format, outputMode=outputMode, partitionBy=partitionBy, queryName=queryName, **options)
    start.__doc__ = PySparkDataStreamWriter.start.__doc__

    def toTable(self, tableName: str, format: Optional[str]=None, outputMode: Optional[str]=None, partitionBy: Optional[Union[str, List[str]]]=None, queryName: Optional[str]=None, **options: 'OptionalPrimitiveType') -> StreamingQuery:
        if False:
            i = 10
            return i + 15
        return self._start_internal(path=None, tableName=tableName, format=format, outputMode=outputMode, partitionBy=partitionBy, queryName=queryName, **options)
    toTable.__doc__ = PySparkDataStreamWriter.toTable.__doc__

def _test() -> None:
    if False:
        return 10
    import sys
    import doctest
    from pyspark.sql import SparkSession as PySparkSession
    import pyspark.sql.connect.streaming.readwriter
    globs = pyspark.sql.connect.readwriter.__dict__.copy()
    globs['spark'] = PySparkSession.builder.appName('sql.connect.streaming.readwriter tests').remote('local[4]').getOrCreate()
    (failure_count, test_count) = doctest.testmod(pyspark.sql.connect.streaming.readwriter, globs=globs, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL)
    globs['spark'].stop()
    if failure_count:
        sys.exit(-1)
if __name__ == '__main__':
    _test()