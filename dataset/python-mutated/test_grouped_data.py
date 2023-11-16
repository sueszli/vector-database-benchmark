from pyspark.sql import functions as F
from sqlglot.dataframe.sql import functions as SF
from tests.dataframe.integration.dataframe_validator import DataFrameValidator

class TestDataframeFunc(DataFrameValidator):

    def test_group_by(self):
        if False:
            while True:
                i = 10
        df_employee = self.df_spark_employee.groupBy(self.df_spark_employee.age).agg(F.min(self.df_spark_employee.employee_id))
        dfs_employee = self.df_sqlglot_employee.groupBy(self.df_sqlglot_employee.age).agg(SF.min(self.df_sqlglot_employee.employee_id))
        self.compare_spark_with_sqlglot(df_employee, dfs_employee, skip_schema_compare=True)

    def test_group_by_where_non_aggregate(self):
        if False:
            i = 10
            return i + 15
        df_employee = self.df_spark_employee.groupBy(self.df_spark_employee.age).agg(F.min(self.df_spark_employee.employee_id).alias('min_employee_id')).where(F.col('age') > F.lit(50))
        dfs_employee = self.df_sqlglot_employee.groupBy(self.df_sqlglot_employee.age).agg(SF.min(self.df_sqlglot_employee.employee_id).alias('min_employee_id')).where(SF.col('age') > SF.lit(50))
        self.compare_spark_with_sqlglot(df_employee, dfs_employee)

    def test_group_by_where_aggregate_like_having(self):
        if False:
            i = 10
            return i + 15
        df_employee = self.df_spark_employee.groupBy(self.df_spark_employee.age).agg(F.min(self.df_spark_employee.employee_id).alias('min_employee_id')).where(F.col('min_employee_id') > F.lit(1))
        dfs_employee = self.df_sqlglot_employee.groupBy(self.df_sqlglot_employee.age).agg(SF.min(self.df_sqlglot_employee.employee_id).alias('min_employee_id')).where(SF.col('min_employee_id') > SF.lit(1))
        self.compare_spark_with_sqlglot(df_employee, dfs_employee)

    def test_count(self):
        if False:
            i = 10
            return i + 15
        df = self.df_spark_employee.groupBy(self.df_spark_employee.age).count()
        dfs = self.df_sqlglot_employee.groupBy(self.df_sqlglot_employee.age).count()
        self.compare_spark_with_sqlglot(df, dfs)

    def test_mean(self):
        if False:
            while True:
                i = 10
        df = self.df_spark_employee.groupBy().mean('age', 'store_id')
        dfs = self.df_sqlglot_employee.groupBy().mean('age', 'store_id')
        self.compare_spark_with_sqlglot(df, dfs)

    def test_avg(self):
        if False:
            return 10
        df = self.df_spark_employee.groupBy('age').avg('store_id')
        dfs = self.df_sqlglot_employee.groupBy('age').avg('store_id')
        self.compare_spark_with_sqlglot(df, dfs)

    def test_max(self):
        if False:
            return 10
        df = self.df_spark_employee.groupBy('age').max('store_id')
        dfs = self.df_sqlglot_employee.groupBy('age').max('store_id')
        self.compare_spark_with_sqlglot(df, dfs)

    def test_min(self):
        if False:
            print('Hello World!')
        df = self.df_spark_employee.groupBy('age').min('store_id')
        dfs = self.df_sqlglot_employee.groupBy('age').min('store_id')
        self.compare_spark_with_sqlglot(df, dfs)

    def test_sum(self):
        if False:
            return 10
        df = self.df_spark_employee.groupBy('age').sum('store_id')
        dfs = self.df_sqlglot_employee.groupBy('age').sum('store_id')
        self.compare_spark_with_sqlglot(df, dfs)