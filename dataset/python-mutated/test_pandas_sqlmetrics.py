import unittest
from typing import cast
from pyspark.sql.functions import pandas_udf
from pyspark.testing.sqlutils import ReusedSQLTestCase, have_pandas, have_pyarrow, pandas_requirement_message, pyarrow_requirement_message

@unittest.skipIf(not have_pandas or not have_pyarrow, cast(str, pandas_requirement_message or pyarrow_requirement_message))
class PandasSQLMetrics(ReusedSQLTestCase):

    def test_pandas_sql_metrics_basic(self):
        if False:
            while True:
                i = 10
        python_sql_metrics = ['data sent to Python workers', 'data returned from Python workers', 'number of output rows']

        @pandas_udf('long')
        def test_pandas(col1):
            if False:
                i = 10
                return i + 15
            return col1 * col1
        self.spark.range(10).select(test_pandas('id')).collect()
        statusStore = self.spark._jsparkSession.sharedState().statusStore()
        lastExecId = statusStore.executionsList().last().executionId()
        executionMetrics = statusStore.execution(lastExecId).get().metrics().mkString()
        for metric in python_sql_metrics:
            self.assertIn(metric, executionMetrics)
if __name__ == '__main__':
    from pyspark.sql.tests.test_pandas_sqlmetrics import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)