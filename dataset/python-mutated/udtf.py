"""
A simple example demonstrating Python UDTFs in Spark
Run with:
  ./bin/spark-submit examples/src/main/python/sql/udtf.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.pandas.utils import require_minimum_pandas_version, require_minimum_pyarrow_version
require_minimum_pandas_version()
require_minimum_pyarrow_version()

def python_udtf_simple_example(spark: SparkSession) -> None:
    if False:
        i = 10
        return i + 15

    class SquareNumbers:

        def eval(self, start: int, end: int):
            if False:
                print('Hello World!')
            for num in range(start, end + 1):
                yield (num, num * num)
    from pyspark.sql.functions import lit, udtf
    square_num = udtf(SquareNumbers, returnType='num: int, squared: int')
    square_num(lit(1), lit(3)).show()

def python_udtf_decorator_example(spark: SparkSession) -> None:
    if False:
        return 10
    from pyspark.sql.functions import lit, udtf

    @udtf(returnType='num: int, squared: int')
    class SquareNumbers:

        def eval(self, start: int, end: int):
            if False:
                for i in range(10):
                    print('nop')
            for num in range(start, end + 1):
                yield (num, num * num)
    SquareNumbers(lit(1), lit(3)).show()

def python_udtf_registration(spark: SparkSession) -> None:
    if False:
        for i in range(10):
            print('nop')
    from pyspark.sql.functions import udtf

    @udtf(returnType='word: string')
    class WordSplitter:

        def eval(self, text: str):
            if False:
                i = 10
                return i + 15
            for word in text.split(' '):
                yield (word.strip(),)
    spark.udtf.register('split_words', WordSplitter)
    spark.sql("SELECT * FROM split_words('hello world')").show()
    spark.sql("SELECT * FROM VALUES ('Hello World'), ('Apache Spark') t(text), LATERAL split_words(text)").show()

def python_udtf_arrow_example(spark: SparkSession) -> None:
    if False:
        for i in range(10):
            print('nop')
    from pyspark.sql.functions import udtf

    @udtf(returnType='c1: int, c2: int', useArrow=True)
    class PlusOne:

        def eval(self, x: int):
            if False:
                for i in range(10):
                    print('nop')
            yield (x, x + 1)

def python_udtf_date_expander_example(spark: SparkSession) -> None:
    if False:
        while True:
            i = 10
    from datetime import datetime, timedelta
    from pyspark.sql.functions import lit, udtf

    @udtf(returnType='date: string')
    class DateExpander:

        def eval(self, start_date: str, end_date: str):
            if False:
                i = 10
                return i + 15
            current = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            while current <= end:
                yield (current.strftime('%Y-%m-%d'),)
                current += timedelta(days=1)
    DateExpander(lit('2023-02-25'), lit('2023-03-01')).show()

def python_udtf_terminate_example(spark: SparkSession) -> None:
    if False:
        i = 10
        return i + 15
    from pyspark.sql.functions import udtf

    @udtf(returnType='cnt: int')
    class CountUDTF:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.count = 0

        def eval(self, x: int):
            if False:
                print('Hello World!')
            self.count += 1

        def terminate(self):
            if False:
                return 10
            yield (self.count,)
    spark.udtf.register('count_udtf', CountUDTF)
    spark.sql('SELECT * FROM range(0, 10, 1, 1), LATERAL count_udtf(id)').show()
    spark.sql('SELECT * FROM range(0, 10, 1, 2), LATERAL count_udtf(id)').show()

def python_udtf_table_argument(spark: SparkSession) -> None:
    if False:
        for i in range(10):
            print('nop')
    from pyspark.sql.functions import udtf
    from pyspark.sql.types import Row

    @udtf(returnType='id: int')
    class FilterUDTF:

        def eval(self, row: Row):
            if False:
                return 10
            if row['id'] > 5:
                yield (row['id'],)
    spark.udtf.register('filter_udtf', FilterUDTF)
    spark.sql('SELECT * FROM filter_udtf(TABLE(SELECT * FROM range(10)))').show()
if __name__ == '__main__':
    spark = SparkSession.builder.appName('Python UDTF example').getOrCreate()
    print('Running Python UDTF single example')
    python_udtf_simple_example(spark)
    print('Running Python UDTF decorator example')
    python_udtf_decorator_example(spark)
    print('Running Python UDTF registration example')
    python_udtf_registration(spark)
    print('Running Python UDTF arrow example')
    python_udtf_arrow_example(spark)
    print('Running Python UDTF date expander example')
    python_udtf_date_expander_example(spark)
    print('Running Python UDTF terminate example')
    python_udtf_terminate_example(spark)
    print('Running Python UDTF table argument example')
    python_udtf_table_argument(spark)
    spark.stop()