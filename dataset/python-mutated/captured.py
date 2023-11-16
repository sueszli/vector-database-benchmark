from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, cast
import py4j
from py4j.protocol import Py4JJavaError
from py4j.java_gateway import is_instance_of
from pyspark import SparkContext
from pyspark.errors.exceptions.base import AnalysisException as BaseAnalysisException, IllegalArgumentException as BaseIllegalArgumentException, ArithmeticException as BaseArithmeticException, UnsupportedOperationException as BaseUnsupportedOperationException, ArrayIndexOutOfBoundsException as BaseArrayIndexOutOfBoundsException, DateTimeException as BaseDateTimeException, NumberFormatException as BaseNumberFormatException, ParseException as BaseParseException, PySparkException, PythonException as BasePythonException, QueryExecutionException as BaseQueryExecutionException, SparkRuntimeException as BaseSparkRuntimeException, SparkUpgradeException as BaseSparkUpgradeException, StreamingQueryException as BaseStreamingQueryException, UnknownException as BaseUnknownException

class CapturedException(PySparkException):

    def __init__(self, desc: Optional[str]=None, stackTrace: Optional[str]=None, cause: Optional[Py4JJavaError]=None, origin: Optional[Py4JJavaError]=None):
        if False:
            i = 10
            return i + 15
        assert origin is not None and desc is None and (stackTrace is None) or (origin is None and desc is not None and (stackTrace is not None))
        self._desc = desc if desc is not None else cast(Py4JJavaError, origin).getMessage()
        assert SparkContext._jvm is not None
        self._stackTrace = stackTrace if stackTrace is not None else SparkContext._jvm.org.apache.spark.util.Utils.exceptionString(origin)
        self._cause = convert_exception(cause) if cause is not None else None
        if self._cause is None and origin is not None and (origin.getCause() is not None):
            self._cause = convert_exception(origin.getCause())
        self._origin = origin

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        assert SparkContext._jvm is not None
        jvm = SparkContext._jvm
        debug_enabled = True
        try:
            sql_conf = jvm.org.apache.spark.sql.internal.SQLConf.get()
            debug_enabled = sql_conf.pysparkJVMStacktraceEnabled()
        except BaseException:
            pass
        desc = self._desc
        if debug_enabled:
            desc = desc + '\n\nJVM stacktrace:\n%s' % self._stackTrace
        return str(desc)

    def getErrorClass(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        assert SparkContext._gateway is not None
        gw = SparkContext._gateway
        if self._origin is not None and is_instance_of(gw, self._origin, 'org.apache.spark.SparkThrowable'):
            return self._origin.getErrorClass()
        else:
            return None

    def getMessageParameters(self) -> Optional[Dict[str, str]]:
        if False:
            print('Hello World!')
        assert SparkContext._gateway is not None
        gw = SparkContext._gateway
        if self._origin is not None and is_instance_of(gw, self._origin, 'org.apache.spark.SparkThrowable'):
            return self._origin.getMessageParameters()
        else:
            return None

    def getSqlState(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        assert SparkContext._gateway is not None
        gw = SparkContext._gateway
        if self._origin is not None and is_instance_of(gw, self._origin, 'org.apache.spark.SparkThrowable'):
            return self._origin.getSqlState()
        else:
            return None

def convert_exception(e: Py4JJavaError) -> CapturedException:
    if False:
        for i in range(10):
            print('nop')
    assert e is not None
    assert SparkContext._jvm is not None
    assert SparkContext._gateway is not None
    jvm = SparkContext._jvm
    gw = SparkContext._gateway
    if is_instance_of(gw, e, 'org.apache.spark.sql.catalyst.parser.ParseException'):
        return ParseException(origin=e)
    elif is_instance_of(gw, e, 'org.apache.spark.sql.AnalysisException'):
        return AnalysisException(origin=e)
    elif is_instance_of(gw, e, 'org.apache.spark.sql.streaming.StreamingQueryException'):
        return StreamingQueryException(origin=e)
    elif is_instance_of(gw, e, 'org.apache.spark.sql.execution.QueryExecutionException'):
        return QueryExecutionException(origin=e)
    elif is_instance_of(gw, e, 'java.lang.NumberFormatException'):
        return NumberFormatException(origin=e)
    elif is_instance_of(gw, e, 'java.lang.IllegalArgumentException'):
        return IllegalArgumentException(origin=e)
    elif is_instance_of(gw, e, 'java.lang.ArithmeticException'):
        return ArithmeticException(origin=e)
    elif is_instance_of(gw, e, 'java.lang.UnsupportedOperationException'):
        return UnsupportedOperationException(origin=e)
    elif is_instance_of(gw, e, 'java.lang.ArrayIndexOutOfBoundsException'):
        return ArrayIndexOutOfBoundsException(origin=e)
    elif is_instance_of(gw, e, 'java.time.DateTimeException'):
        return DateTimeException(origin=e)
    elif is_instance_of(gw, e, 'org.apache.spark.SparkRuntimeException'):
        return SparkRuntimeException(origin=e)
    elif is_instance_of(gw, e, 'org.apache.spark.SparkUpgradeException'):
        return SparkUpgradeException(origin=e)
    c: Py4JJavaError = e.getCause()
    stacktrace: str = jvm.org.apache.spark.util.Utils.exceptionString(e)
    if c is not None and (is_instance_of(gw, c, 'org.apache.spark.api.python.PythonException') and any(map(lambda v: 'org.apache.spark.sql.execution.python' in v.toString(), c.getStackTrace()))):
        msg = '\n  An exception was thrown from the Python worker. Please see the stack trace below.\n%s' % c.getMessage()
        return PythonException(msg, stacktrace)
    return UnknownException(desc=e.toString(), stackTrace=stacktrace, cause=c)

def capture_sql_exception(f: Callable[..., Any]) -> Callable[..., Any]:
    if False:
        return 10

    def deco(*a: Any, **kw: Any) -> Any:
        if False:
            i = 10
            return i + 15
        try:
            return f(*a, **kw)
        except Py4JJavaError as e:
            converted = convert_exception(e.java_exception)
            if not isinstance(converted, UnknownException):
                raise converted from None
            else:
                raise
    return deco

@contextmanager
def unwrap_spark_exception() -> Iterator[Any]:
    if False:
        i = 10
        return i + 15
    assert SparkContext._gateway is not None
    gw = SparkContext._gateway
    try:
        yield
    except Py4JJavaError as e:
        je: Py4JJavaError = e.java_exception
        if je is not None and is_instance_of(gw, je, 'org.apache.spark.SparkException'):
            converted = convert_exception(je.getCause())
            if not isinstance(converted, UnknownException):
                raise converted from None
        raise

def install_exception_handler() -> None:
    if False:
        return 10
    "\n    Hook an exception handler into Py4j, which could capture some SQL exceptions in Java.\n\n    When calling Java API, it will call `get_return_value` to parse the returned object.\n    If any exception happened in JVM, the result will be Java exception object, it raise\n    py4j.protocol.Py4JJavaError. We replace the original `get_return_value` with one that\n    could capture the Java exception and throw a Python one (with the same error message).\n\n    It's idempotent, could be called multiple times.\n    "
    original = py4j.protocol.get_return_value
    patched = capture_sql_exception(original)
    py4j.java_gateway.get_return_value = patched

class AnalysisException(CapturedException, BaseAnalysisException):
    """
    Failed to analyze a SQL query plan.
    """

class ParseException(AnalysisException, BaseParseException):
    """
    Failed to parse a SQL command.
    """

class IllegalArgumentException(CapturedException, BaseIllegalArgumentException):
    """
    Passed an illegal or inappropriate argument.
    """

class StreamingQueryException(CapturedException, BaseStreamingQueryException):
    """
    Exception that stopped a :class:`StreamingQuery`.
    """

class QueryExecutionException(CapturedException, BaseQueryExecutionException):
    """
    Failed to execute a query.
    """

class PythonException(CapturedException, BasePythonException):
    """
    Exceptions thrown from Python workers.
    """

class ArithmeticException(CapturedException, BaseArithmeticException):
    """
    Arithmetic exception.
    """

class UnsupportedOperationException(CapturedException, BaseUnsupportedOperationException):
    """
    Unsupported operation exception.
    """

class ArrayIndexOutOfBoundsException(CapturedException, BaseArrayIndexOutOfBoundsException):
    """
    Array index out of bounds exception.
    """

class DateTimeException(CapturedException, BaseDateTimeException):
    """
    Datetime exception.
    """

class NumberFormatException(IllegalArgumentException, BaseNumberFormatException):
    """
    Number format exception.
    """

class SparkRuntimeException(CapturedException, BaseSparkRuntimeException):
    """
    Runtime exception.
    """

class SparkUpgradeException(CapturedException, BaseSparkUpgradeException):
    """
    Exception thrown because of Spark upgrade.
    """

class UnknownException(CapturedException, BaseUnknownException):
    """
    None of the above exceptions.
    """