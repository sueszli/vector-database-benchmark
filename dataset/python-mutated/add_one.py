from pyflink.table import DataTypes
from pyflink.table.udf import udf

@udf(input_types=[DataTypes.BIGINT()], result_type=DataTypes.BIGINT())
def add_one(i):
    if False:
        i = 10
        return i + 15
    import pytest
    return i + 1