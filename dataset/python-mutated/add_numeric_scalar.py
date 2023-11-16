from woodwork.column_schema import ColumnSchema
from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library

class AddNumericScalar(TransformPrimitive):
    """Adds a scalar to each value in the list.

    Description:
        Given a list of numeric values and a scalar, add
        the given scalar to each value in the list.

    Examples:
        >>> add_numeric_scalar = AddNumericScalar(value=2)
        >>> add_numeric_scalar([3, 1, 2]).tolist()
        [5, 3, 4]
    """
    name = 'add_numeric_scalar'
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    def __init__(self, value=0):
        if False:
            for i in range(10):
                print('nop')
        self.value = value
        self.description_template = 'the sum of {{}} and {}'.format(self.value)

    def get_function(self):
        if False:
            while True:
                i = 10

        def add_scalar(vals):
            if False:
                print('Hello World!')
            return vals + self.value
        return add_scalar

    def generate_name(self, base_feature_names):
        if False:
            while True:
                i = 10
        return '%s + %s' % (base_feature_names[0], str(self.value))