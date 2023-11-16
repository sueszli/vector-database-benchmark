"""Correlations between variables."""
from typing import Optional
import pandas as pd
import phik
import pyspark
from packaging import version
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
from pyspark.sql.types import ArrayType, DoubleType, StructField, StructType
from ydata_profiling.config import Settings
from ydata_profiling.model.correlations import Cramers, Kendall, Pearson, PhiK, Spearman
SPARK_CORRELATION_PEARSON = 'pearson'
SPARK_CORRELATION_SPEARMAN = 'spearman'

@Spearman.compute.register(Settings, DataFrame, dict)
def spark_spearman_compute(config: Settings, df: DataFrame, summary: dict) -> Optional[pd.DataFrame]:
    if False:
        print('Hello World!')
    (matrix, num_cols) = _compute_spark_corr_natively(df, summary, corr_type=SPARK_CORRELATION_SPEARMAN)
    return pd.DataFrame(matrix, index=num_cols, columns=num_cols)

@Pearson.compute.register(Settings, DataFrame, dict)
def spark_pearson_compute(config: Settings, df: DataFrame, summary: dict) -> Optional[pd.DataFrame]:
    if False:
        print('Hello World!')
    (matrix, num_cols) = _compute_spark_corr_natively(df, summary, corr_type=SPARK_CORRELATION_PEARSON)
    return pd.DataFrame(matrix, index=num_cols, columns=num_cols)

def _compute_spark_corr_natively(df: DataFrame, summary: dict, corr_type: str) -> ArrayType:
    if False:
        return 10
    '\n    This function exists as pearson and spearman correlation computations have the\n    exact same workflow. The syntax is Correlation.corr(dataframe, method="pearson" OR "spearman"),\n    and Correlation is from pyspark.ml.stat\n    '
    variables = {column: description['type'] for (column, description) in summary.items()}
    interval_columns = [column for (column, type_name) in variables.items() if type_name == 'Numeric']
    df = df.select(*interval_columns)
    vector_col = 'corr_features'
    assembler_args = {'inputCols': df.columns, 'outputCol': vector_col}
    if version.parse(pyspark.__version__) >= version.parse('2.4.0'):
        assembler_args['handleInvalid'] = 'skip'
    assembler = VectorAssembler(**assembler_args)
    df_vector = assembler.transform(df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col, method=corr_type).head()[0].toArray()
    return (matrix, interval_columns)

@Kendall.compute.register(Settings, DataFrame, dict)
def spark_kendall_compute(config: Settings, df: DataFrame, summary: dict) -> Optional[pd.DataFrame]:
    if False:
        while True:
            i = 10
    raise NotImplementedError()

@Cramers.compute.register(Settings, DataFrame, dict)
def spark_cramers_compute(config: Settings, df: DataFrame, summary: dict) -> Optional[pd.DataFrame]:
    if False:
        print('Hello World!')
    raise NotImplementedError()

@PhiK.compute.register(Settings, DataFrame, dict)
def spark_phi_k_compute(config: Settings, df: DataFrame, summary: dict) -> Optional[pd.DataFrame]:
    if False:
        i = 10
        return i + 15
    threshold = config.categorical_maximum_correlation_distinct
    intcols = {key for (key, value) in summary.items() if value['type'] == 'Numeric' and 1 < value['n_distinct']}
    supportedcols = {key for (key, value) in summary.items() if value['type'] != 'Unsupported' and 1 < value['n_distinct'] <= threshold}
    selcols = list(supportedcols.union(intcols))
    if len(selcols) <= 1:
        return None
    groupby_df = df.select(selcols).withColumn('groupby', lit(1))
    output_schema_components = []
    for column in selcols:
        output_schema_components.append(StructField(column, DoubleType(), True))
    output_schema = StructType(output_schema_components)

    @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
    def spark_phik(pdf: pd.DataFrame) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        correlation = phik.phik_matrix(df=pdf, interval_cols=list(intcols))
        return correlation
    if len(groupby_df.head(1)) > 0:
        df = pd.DataFrame(groupby_df.groupby('groupby').apply(spark_phik).toPandas().values, columns=selcols, index=selcols)
    else:
        df = pd.DataFrame()
    return df