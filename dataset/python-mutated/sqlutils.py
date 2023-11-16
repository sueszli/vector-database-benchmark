import datetime
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
pandas_requirement_message = None
try:
    from pyspark.sql.pandas.utils import require_minimum_pandas_version
    require_minimum_pandas_version()
except ImportError as e:
    pandas_requirement_message = str(e)
pyarrow_requirement_message = None
try:
    from pyspark.sql.pandas.utils import require_minimum_pyarrow_version
    require_minimum_pyarrow_version()
except ImportError as e:
    pyarrow_requirement_message = str(e)
test_not_compiled_message = None
try:
    from pyspark.sql.utils import require_test_compiled
    require_test_compiled()
except Exception as e:
    test_not_compiled_message = str(e)
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, DoubleType, UserDefinedType, Row
from pyspark.testing.utils import ReusedPySparkTestCase, PySparkErrorTestUtils
have_pandas = pandas_requirement_message is None
have_pyarrow = pyarrow_requirement_message is None
test_compiled = test_not_compiled_message is None

class UTCOffsetTimezone(datetime.tzinfo):
    """
    Specifies timezone in UTC offset
    """

    def __init__(self, offset=0):
        if False:
            for i in range(10):
                print('nop')
        self.ZERO = datetime.timedelta(hours=offset)

    def utcoffset(self, dt):
        if False:
            print('Hello World!')
        return self.ZERO

    def dst(self, dt):
        if False:
            return 10
        return self.ZERO

class ExamplePointUDT(UserDefinedType):
    """
    User-defined type (UDT) for ExamplePoint.
    """

    @classmethod
    def sqlType(cls):
        if False:
            i = 10
            return i + 15
        return ArrayType(DoubleType(), False)

    @classmethod
    def module(cls):
        if False:
            print('Hello World!')
        return 'pyspark.sql.tests'

    @classmethod
    def scalaUDT(cls):
        if False:
            i = 10
            return i + 15
        return 'org.apache.spark.sql.test.ExamplePointUDT'

    def serialize(self, obj):
        if False:
            i = 10
            return i + 15
        return [obj.x, obj.y]

    def deserialize(self, datum):
        if False:
            return 10
        return ExamplePoint(datum[0], datum[1])

class ExamplePoint:
    """
    An example class to demonstrate UDT in Scala, Java, and Python.
    """
    __UDT__ = ExamplePointUDT()

    def __init__(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        self.x = x
        self.y = y

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'ExamplePoint(%s,%s)' % (self.x, self.y)

    def __str__(self):
        if False:
            print('Hello World!')
        return '(%s,%s)' % (self.x, self.y)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, self.__class__) and other.x == self.x and (other.y == self.y)

class PythonOnlyUDT(UserDefinedType):
    """
    User-defined type (UDT) for ExamplePoint.
    """

    @classmethod
    def sqlType(cls):
        if False:
            print('Hello World!')
        return ArrayType(DoubleType(), False)

    @classmethod
    def module(cls):
        if False:
            print('Hello World!')
        return '__main__'

    def serialize(self, obj):
        if False:
            for i in range(10):
                print('nop')
        return [obj.x, obj.y]

    def deserialize(self, datum):
        if False:
            return 10
        return PythonOnlyPoint(datum[0], datum[1])

    @staticmethod
    def foo():
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    def props(self):
        if False:
            for i in range(10):
                print('nop')
        return {}

class PythonOnlyPoint(ExamplePoint):
    """
    An example class to demonstrate UDT in only Python
    """
    __UDT__ = PythonOnlyUDT()

class MyObject:

    def __init__(self, key, value):
        if False:
            i = 10
            return i + 15
        self.key = key
        self.value = value

class SQLTestUtils:
    """
    This util assumes the instance of this to have 'spark' attribute, having a spark session.
    It is usually used with 'ReusedSQLTestCase' class but can be used if you feel sure the
    the implementation of this class has 'spark' attribute.
    """

    @contextmanager
    def sql_conf(self, pairs):
        if False:
            print('Hello World!')
        '\n        A convenient context manager to test some configuration specific logic. This sets\n        `value` to the configuration `key` and then restores it back when it exits.\n        '
        assert isinstance(pairs, dict), 'pairs should be a dictionary.'
        assert hasattr(self, 'spark'), "it should have 'spark' attribute, having a spark session."
        keys = pairs.keys()
        new_values = pairs.values()
        old_values = [self.spark.conf.get(key, None) for key in keys]
        for (key, new_value) in zip(keys, new_values):
            self.spark.conf.set(key, new_value)
        try:
            yield
        finally:
            for (key, old_value) in zip(keys, old_values):
                if old_value is None:
                    self.spark.conf.unset(key)
                else:
                    self.spark.conf.set(key, old_value)

    @contextmanager
    def database(self, *databases):
        if False:
            i = 10
            return i + 15
        '\n        A convenient context manager to test with some specific databases. This drops the given\n        databases if it exists and sets current database to "default" when it exits.\n        '
        assert hasattr(self, 'spark'), "it should have 'spark' attribute, having a spark session."
        try:
            yield
        finally:
            for db in databases:
                self.spark.sql('DROP DATABASE IF EXISTS %s CASCADE' % db)
            self.spark.catalog.setCurrentDatabase('default')

    @contextmanager
    def table(self, *tables):
        if False:
            while True:
                i = 10
        '\n        A convenient context manager to test with some specific tables. This drops the given tables\n        if it exists.\n        '
        assert hasattr(self, 'spark'), "it should have 'spark' attribute, having a spark session."
        try:
            yield
        finally:
            for t in tables:
                self.spark.sql('DROP TABLE IF EXISTS %s' % t)

    @contextmanager
    def tempView(self, *views):
        if False:
            print('Hello World!')
        '\n        A convenient context manager to test with some specific views. This drops the given views\n        if it exists.\n        '
        assert hasattr(self, 'spark'), "it should have 'spark' attribute, having a spark session."
        try:
            yield
        finally:
            for v in views:
                self.spark.catalog.dropTempView(v)

    @contextmanager
    def function(self, *functions):
        if False:
            i = 10
            return i + 15
        '\n        A convenient context manager to test with some specific functions. This drops the given\n        functions if it exists.\n        '
        assert hasattr(self, 'spark'), "it should have 'spark' attribute, having a spark session."
        try:
            yield
        finally:
            for f in functions:
                self.spark.sql('DROP FUNCTION IF EXISTS %s' % f)

    @staticmethod
    def assert_close(a, b):
        if False:
            i = 10
            return i + 15
        c = [j[0] for j in b]
        diff = [abs(v - c[k]) < 1e-06 if math.isfinite(v) else v == c[k] for (k, v) in enumerate(a)]
        assert sum(diff) == len(a), f'sum: {sum(diff)}, len: {len(a)}'

class ReusedSQLTestCase(ReusedPySparkTestCase, SQLTestUtils, PySparkErrorTestUtils):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super(ReusedSQLTestCase, cls).setUpClass()
        cls.spark = SparkSession(cls.sc)
        cls.tempdir = tempfile.NamedTemporaryFile(delete=False)
        os.unlink(cls.tempdir.name)
        cls.testData = [Row(key=i, value=str(i)) for i in range(100)]
        cls.df = cls.spark.createDataFrame(cls.testData)

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        super(ReusedSQLTestCase, cls).tearDownClass()
        cls.spark.stop()
        shutil.rmtree(cls.tempdir.name, ignore_errors=True)