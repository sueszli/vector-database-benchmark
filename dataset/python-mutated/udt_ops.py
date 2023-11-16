from pyspark.pandas.data_type_ops.base import DataTypeOps

class UDTOps(DataTypeOps):
    """
    The class for binary operations of pandas-on-Spark objects with Spark type:
    UserDefinedType or its subclasses.
    """

    @property
    def pretty_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'user defined types'